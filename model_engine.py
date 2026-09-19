"""
Real VoxCPM neural TTS engine for the RunPod Serverless Khmer TTS Worker.

This engine runs the genuine OpenBMB VoxCPM model (https://github.com/OpenBMB/VoxCPM)
for Khmer speech generation with true zero-shot voice cloning and voice design.

Synthesis chain (in priority order):
  1. VoxCPM  (real neural model, GPU or CPU) - voice design / voice cloning
  2. Edge-TTS (explicitly labelled fallback, requires internet access to
     Microsoft's public Edge TTS service)
  3. Hard failure - a TTSWorkerError is raised so RunPod marks the job as
     FAILED and the desktop client surfaces the real reason. The legacy
     synthetic "beep" generator has been removed on purpose.

Environment variables:
  VOXCPM_MODEL_ID    HuggingFace repo id or local checkpoint dir
                     (default: openbmb/VoxCPM2)
  VOXCPM_DEVICE      "auto" | "cuda" | "cpu" | "mps" (default: auto)
  VOXCPM_TIMESTEPS   diffusion steps 4-30 (default: 10)
  VOXCPM_DENOISER    "1" to load the reference-audio denoiser (default: "0")
  VOXCPM_OPTIMIZE    "0" disables torch.compile optimizations (default: "1")
  VOXCPM_PRELOAD     "1" loads weights at container start, before the first
                     job arrives (default: "1")
  MODEL_PATH         local checkpoint/volume directory (default: /workspace/models)
"""
import asyncio
import gc
import inspect
import io
import os
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, List, NamedTuple

import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("KhmerTTSModelEngine")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("edge_tts package not found; the labelled Edge-TTS fallback is disabled.")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from voxcpm import VoxCPM
    HAS_VOXCPM = True
except ImportError:
    HAS_VOXCPM = False
    logger.warning(
        "voxcpm package is not installed. The worker will fall back to the "
        "labelled Edge-TTS engine. Install with: pip install voxcpm"
    )


class TTSWorkerError(Exception):
    """Raised when the worker cannot produce audio. Marks the RunPod job FAILED."""


# Canonical engine labels reported back to the desktop client. The client
# surfaces this string to the user, so it must always describe the engine that
# REALLY produced the audio - never merely the engine we hoped to use.
ENGINE_LABEL_VOXCPM = "VoxCPM"
ENGINE_LABEL_EDGE_FALLBACK = "Edge-TTS (fallback)"


class SynthesisOutcome(NamedTuple):
    """
    Result of one synthesis attempt, including truthful engine provenance.

    Fields:
        audio                : float32 mono waveform
        sample_rate          : native sample rate of that waveform
        engine               : ENGINE_LABEL_VOXCPM or ENGINE_LABEL_EDGE_FALLBACK
        voice_cloning_applied: True only when the caller's reference audio was
                               actually used to clone the voice
        fallback_reason      : human-readable reason the primary engine failed
                               (None when VoxCPM produced the audio cleanly)
    """
    audio: np.ndarray
    sample_rate: int
    engine: str
    voice_cloning_applied: bool = False
    fallback_reason: Optional[str] = None



def _discover_model_dir(explicit_dir: Optional[str] = None) -> Path:
    """
    Intelligently discover model storage path:
    1. Explicit dir parameter
    2. MODEL_PATH environment variable
    3. /runpod-volume/models (RunPod default Network Volume mount)
    4. /workspace/models (Common container mount)
    5. Fallback to /workspace/models
    """
    if explicit_dir:
        p = Path(explicit_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    env_dir = os.getenv("MODEL_PATH")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Auto-detect mounted Network Volume paths
    candidates = [
        Path("/runpod-volume/models"),
        Path("/workspace/models"),
        Path("/app/models")
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    # Default fallback
    fallback = Path("/runpod-volume/models") if Path("/runpod-volume").exists() else Path("/workspace/models")
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return fallback


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes")


def _map_temperature_to_cfg(temperature: float) -> float:
    """
    Map the UI 'temperature' (0.0 - 1.5, default 0.7) onto VoxCPM's
    classifier-free-guidance scale (typical range 1.0 - 3.0, default 2.0).
        temperature 0.7 -> cfg ~2.0 (balanced default)
        temperature 0.2 -> cfg ~1.1 (very relaxed / natural)
        temperature 1.0 -> cfg ~2.5 (strict adherence to the text)
    """
    cfg = 0.7 + float(temperature) * 1.8
    return float(np.clip(cfg, 1.0, 3.0))


def _build_style_instruction(
    prompt: Optional[str],
    emotion: Optional[str],
    speed: float,
    include_voice_design: bool = True,
) -> str:
    """
    Build a VoxCPM2 Voice-Design / Controllable-Cloning control instruction.
    The instruction is prepended to the text inside parentheses, e.g.
    "(warm female voice, calm tone, slightly slower pace)".

    include_voice_design=False (clone mode) keeps only delivery hints such as
    emotion and pacing: the timbre comes from the reference audio, and a long
    voice-design description would FIGHT the cloned voice (the model follows
    the instruction instead of the reference).
    """
    parts: List[str] = []

    prompt_text = (prompt or "").strip()
    # The desktop app's prompt presets are long descriptive sentences; they
    # work verbatim as a design instruction. Only hard-cap absurd lengths.
    if include_voice_design and prompt_text:
        parts.append(prompt_text[:300])

    emotion_text = (emotion or "").strip()
    if emotion_text and emotion_text.lower() not in ("neutral", ""):
        parts.append(f"{emotion_text.lower()} tone")

    if speed <= 0.92:
        parts.append("slightly slower, unhurried pace")
    elif speed >= 1.08:
        parts.append("slightly faster pace")

    return ", ".join(parts)


def _split_text_for_voxcpm(text: str, max_chars: int = 350) -> List[str]:
    """
    Split long text into stable segments. VoxCPM (like all diffusion
    autoregressive TTS) can speed up or buzz on very long inputs, so the
    official recommendation is to generate per segment and concatenate.
    Honours Khmer sentence terminators (khan '។', bariyoosan '៕').
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[។៕!?.\n])\s*", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    segments: List[str] = []
    current = ""
    for sentence in sentences:
        # Hard-split any single sentence that is itself too long.
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                segments.append(current)
                current = ""
            segments.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                segments.append(current)
            current = sentence
    if current:
        segments.append(current)
    return segments or [text]


def _accepted_generate_params(model) -> set:
    """
    VoxCPM.generate() is a thin *args/**kwargs wrapper, so its public
    signature carries no parameter names; the real contract lives on the
    internal _generate() (and historically on generate() in some releases).
    Inspect both so unsupported kwargs can be dropped instead of crashing
    (e.g. reference_wav_path on VoxCPM 1.x models).
    """
    names: set = set()
    for fn in (getattr(model, "generate", None), getattr(model, "_generate", None)):
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        for name, param in params.items():
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                names.add(name)
    return names


def _filter_generate_kwargs(kwargs: dict, accepted: set) -> dict:
    """Drop generation params unsupported by the loaded VoxCPM version."""
    if not accepted:
        return kwargs
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        logger.warning(f"Dropped generation params unsupported by this VoxCPM version: {dropped}")
    return filtered


def build_pretrained_kwargs(voxcpm_cls, load_denoiser: bool, optimize: bool, device: str):
    """
    Build the from_pretrained() kwargs the INSTALLED voxcpm release supports.

    The PyPI 'voxcpm' package (1.x) has:
        from_pretrained(model_id, zip_enhanced_conformer=True, optimize_bn=True, load_denoiser=True)
    and does NOT accept 'device' or 'optimize'. Passing unknown kwargs raises
    TypeError only AFTER the multi-GB weights have been downloaded. Newer
    releases accept 'device' and 'optimize' directly. Returns
    (kwargs, is_legacy_api).
    """
    import inspect

    def _sig_info(fn):
        if fn is None:
            return None
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        names = {
            n for n, p in sig.parameters.items()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        return names, has_varkw

    infos = [
        _sig_info(getattr(voxcpm_cls, "from_pretrained", None)),
        _sig_info(getattr(voxcpm_cls, "__init__", None)),
    ]

    def accepts(name: str) -> bool:
        # from_pretrained may be a thin *args/**kwargs forwarder whose public
        # signature hides the real contract; in that case consult __init__.
        # Only an explicit named parameter proves acceptance - a wrong kwarg
        # raises TypeError only AFTER the multi-GB weights have been
        # downloaded, so unknown kwargs must never be forwarded blindly.
        for info in infos:
            if info is None:
                continue
            names, has_varkw = info
            if has_varkw:
                continue  # forwarding wrapper: real contract not visible here
            return name in names
        # Both signatures unknown (or pure *args/**kwargs): optimistically
        # pass through; the load-time retry below drops rejected kwargs.
        return True

    kwargs = {}
    if accepts("load_denoiser"):
        kwargs["load_denoiser"] = load_denoiser
    if accepts("optimize"):
        kwargs["optimize"] = optimize
    elif accepts("optimize_bn"):
        kwargs["optimize_bn"] = optimize
    if device and accepts("device"):
        kwargs["device"] = device

    is_legacy = not (accepts("optimize") or accepts("device"))
    return kwargs, is_legacy


def _is_voxcpm2_family(model_id: str) -> bool:
    """
    True for any VoxCPM2-generation checkpoint id: 'openbmb/VoxCPM2',
    mirrors such as 'Tha456/VoxCPM2', and variants like 'voxcpm-2' or
    'VoxCPM2-1.5B'. The legacy 1.x PyPI package cannot run these weights
    and must be downgraded to 'openbmb/VoxCPM-0.5B' instead.
    """
    return bool(re.search(r"voxcpm[-_]?2", str(model_id), re.IGNORECASE))


def _pretrained_load_with_retry(voxcpm_cls, model_id: str, kwargs: dict):
    """
    Call voxcpm_cls.from_pretrained(model_id, **kwargs), dropping kwargs the
    runtime rejects one at a time and retrying.

    Why: signature introspection cannot see through *args/**kwargs wrappers,
    and a rejected kwarg currently raises TypeError only AFTER the multi-GB
    weights have been downloaded. HuggingFace caches the download, so the
    retry is cheap. Returns (model, kwargs_actually_used).
    """
    current = dict(kwargs)
    while True:
        try:
            return voxcpm_cls.from_pretrained(model_id, **current), current
        except TypeError as ex:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(ex))
            if not m or m.group(1) not in current:
                raise
            dropped = m.group(1)
            current.pop(dropped)
            logger.warning(
                f"from_pretrained rejected kwarg '{dropped}' at runtime; "
                "dropping it and retrying (weights are HF-cached, retry is cheap)."
            )
            if not current:
                raise


def _looks_like_compile_error(ex: Exception) -> bool:
    """
    Detect torch.compile / torch._inductor / torch._dynamo failures. The
    'optimize' path compiles graphs with the inductor backend, which needs a
    C/C++ compiler (gcc/g++) inside the container; without one the model load
    fails at runtime. Such failures are configuration-transient: retrying in
    eager mode (optimizations disabled) loads the model with identical
    synthesis quality, only slower warm-up.
    """
    msg = str(ex).lower()
    markers = (
        "failed to find c compiler",
        "specify via cc environment variable",
        "backend='inductor'",
        "inductor raised",
        "torch._dynamo",
        "torch dynamo",
        "triton",
    )
    return any(marker in msg for marker in markers)


def _consume_generated(generated) -> np.ndarray:
    """Flatten VoxCPM output (array, nested list, or generator) to 1-D float32."""
    if isinstance(generated, np.ndarray):
        return generated.astype(np.float32).ravel()
    if hasattr(generated, "__iter__") and not isinstance(generated, (str, bytes, dict)):
        # Generator / list of chunks (possibly ragged) -> concatenate chunks
        chunks = []
        for chunk in generated:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        if chunks:
            return np.concatenate(chunks).astype(np.float32)
        return np.array([], dtype=np.float32)
    return np.asarray(generated, dtype=np.float32).reshape(-1)


def _polish_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Master the generated waveform: DC removal, 10 ms edge fades (removes
    leading/trailing clicks), and peak normalization to -1 dBFS.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return audio
    audio = audio - np.mean(audio)
    fade = min(int(sample_rate * 0.01), len(audio) // 4)
    if fade > 0:
        audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio * (10 ** (-1.0 / 20) / peak)
    else:
        # (Near-)silence: do not amplify float residue into full-scale noise
        audio = np.zeros_like(audio)
    return np.clip(audio, -1.0, 1.0)


class KhmerTTSModelEngine:
    """
    Real VoxCPM model engine. Manages checkpoint loading, GPU memory
    allocation, and neural speech inference with true voice cloning.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = _discover_model_dir(model_dir)
        self.model_id = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2").strip()
        # Checkpoint id ACTUALLY loaded (differs from model_id when the legacy
        # 1.x package forces a downgrade to VoxCPM-0.5B). Reported to the
        # desktop client as provenance.
        self.effective_model_id = self.model_id

        # Resolve device to concrete string ("cuda" or "cpu")
        raw_device = os.getenv("VOXCPM_DEVICE", "auto").strip().lower() or "auto"
        if raw_device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = raw_device

        if self.device == "cuda" and HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                cuda_ver = getattr(torch.version, "cuda", "unknown")
                logger.info(
                    f"GPU Acceleration ACTIVE: {gpu_name} ({vram_gb:.2f} GB VRAM) | CUDA: {cuda_ver}"
                )
            except Exception as ex:
                logger.info(f"GPU Acceleration ACTIVE (CUDA enabled): {ex}")
        else:
            logger.info(f"Engine running on device: {self.device}")

        self.inference_timesteps = max(4, min(30, int(os.getenv("VOXCPM_TIMESTEPS", "10"))))
        self.load_denoiser = _env_flag("VOXCPM_DENOISER", False)
        self.optimize = _env_flag("VOXCPM_OPTIMIZE", True)
        self.sample_rate = 48000  # VoxCPM2 native output; updated after model load
        self._legacy_api = False  # True when the installed voxcpm is the 1.x API
        # VoxCPM2's LM has an 8192-token KV cache; long reference audios
        # overflow it during prompt prefill, so cap how much prompt we keep.
        self.max_reference_seconds = float(os.getenv("MAX_REFERENCE_AUDIO_SECONDS", "10"))
        self.accepted_generate_kwargs: set = set()

        self.model = None
        self.model_loaded = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _resolve_local_checkpoint(self) -> Optional[Path]:
        """Prefer a local checkpoint on disk (or a Network Volume) over HF download."""
        if not self.model_dir.exists() or not self.model_dir.is_dir():
            return None

        def _has_weights(d: Path) -> bool:
            return any(d.glob("*.safetensors")) or any(d.glob("*.bin"))

        if _has_weights(self.model_dir):
            return self.model_dir
        # Common layout: /workspace/models/<ModelName>/
        for child in sorted(self.model_dir.iterdir()):
            if child.is_dir() and _has_weights(child):
                return child
        return None

    def _load_model(self) -> None:
        if not HAS_VOXCPM:
            logger.warning(
                "VoxCPM engine unavailable (package 'voxcpm' not installed). "
                "Worker will use the labelled Edge-TTS fallback engine."
            )
            return

        logger.info(
            f"Loading VoxCPM model | id={self.model_id} | device={self.device} | "
            f"denoiser={self.load_denoiser} | optimize={self.optimize} | timesteps={self.inference_timesteps}"
        )
        try:
            self._load_model_once(self.optimize)
            return
        except Exception as primary_ex:
            if not self.optimize or not _looks_like_compile_error(primary_ex):
                self.model = None
                self.model_loaded = False
                logger.error(f"Failed to load VoxCPM model: {primary_ex}", exc_info=True)
                raise TTSWorkerError(f"VoxCPM model failed to load: {primary_ex}")

            # torch.compile (inductor) failed - typically a missing C/C++
            # compiler in the container. Retry in eager mode: identical
            # synthesis quality, slower warm-up, but the model LOADS and
            # jobs succeed instead of failing.
            logger.warning(
                "Model load failed inside torch.compile/inductor "
                f"({type(primary_ex).__name__}: {str(primary_ex)[:200]}). "
                "Retrying with optimizations disabled (eager mode)..."
            )
            try:
                self._load_model_once(False)
                self.optimize = False  # reflect what is actually running
                logger.warning(
                    "VoxCPM loaded WITHOUT torch.compile optimizations (eager "
                    "mode). For full optimization, install a C/C++ toolchain "
                    "(e.g. 'build-essential') in the container image."
                )
            except Exception as fallback_ex:
                self.model = None
                self.model_loaded = False
                logger.error(
                    f"Failed to load VoxCPM model (eager retry): {fallback_ex}",
                    exc_info=True,
                )
                raise TTSWorkerError(
                    f"VoxCPM model failed to load (also without optimizations): {fallback_ex}"
                ) from fallback_ex

    def _load_model_once(self, optimize: bool) -> None:
        """Single load attempt. Raises on failure; the caller decides retries."""
        # Version compatibility: introspect the installed voxcpm package
        # and pass only supported kwargs (PyPI 1.x rejects device/optimize
        # AFTER the weights download - fatal on serverless cold starts).
        pretrained_kwargs, self._legacy_api = build_pretrained_kwargs(
            VoxCPM, self.load_denoiser, optimize, self.device
        )

        model_id = self.model_id
        if self._legacy_api and _is_voxcpm2_family(model_id):
            # The 1.x package cannot run VoxCPM2 weights - fall back.
            # (Catches 'openbmb/VoxCPM2', mirrors like 'Tha456/VoxCPM2',
            # and variants such as 'voxcpm-2' / 'VoxCPM2-1.5B'.)
            model_id = "openbmb/VoxCPM-0.5B"
            logger.warning(
                "Installed 'voxcpm' package is the 1.x API (no device/optimize "
                "support), which cannot load VoxCPM2 weights. Automatically "
                f"switching model to {model_id}."
            )
        self.effective_model_id = model_id

        local_checkpoint = self._resolve_local_checkpoint()
        if local_checkpoint:
            logger.info(f"Using local VoxCPM checkpoint: {local_checkpoint}")
            self.model, pretrained_kwargs = _pretrained_load_with_retry(
                VoxCPM, str(local_checkpoint), pretrained_kwargs
            )
        else:
            logger.info(
                f"No local checkpoint found; downloading/loading from HuggingFace Hub "
                f"(id={model_id}, kwargs={sorted(pretrained_kwargs) or '(defaults)'})..."
            )
            self.model, pretrained_kwargs = _pretrained_load_with_retry(
                VoxCPM, model_id, pretrained_kwargs
            )

        # Ensure model is on target device if from_pretrained didn't place it
        if self.device == "cuda" and hasattr(self.model, "to"):
            try:
                self.model.to("cuda")
            except Exception as ex:
                logger.debug(f"Model .to('cuda') check: {ex}")

        # Safe sample-rate detection across package versions.
        tts_model = getattr(self.model, "tts_model", None)
        detected_sr = getattr(tts_model, "sample_rate", None) if tts_model is not None else None
        if detected_sr:
            self.sample_rate = int(detected_sr)
        elif self._legacy_api:
            self.sample_rate = 16000  # VoxCPM 0.5B native output rate
            logger.warning("Could not detect sample rate; assuming 16000 Hz for legacy VoxCPM 1.x.")

        self.accepted_generate_kwargs = _accepted_generate_params(self.model)
        self.model_loaded = True
        logger.info(
            f"VoxCPM model loaded successfully | native sample rate: {self.sample_rate} Hz | "
            f"legacy_api={self._legacy_api} | "
            f"accepted generate params: {sorted(self.accepted_generate_kwargs) or 'unknown (*args/**kwargs)'}"
        )

    # ------------------------------------------------------------------
    # Synthesis paths
    # ------------------------------------------------------------------
    def _resolve_clone_params(
        self,
        ref_path: Optional[str],
        ref_text: Optional[str],
    ) -> dict:
        """
        Decide how (and whether) the caller's reference audio can actually be
        used for voice cloning with the loaded voxcpm build/model, returning
        the generate() kwargs that enable cloning ({} = cannot clone).

        Official API contract (OpenBMB/VoxCPM core.py):
          - reference_wav_path: VoxCPM2-only TRUE zero-shot cloning; can be
            used ALONE - no transcript needed.
          - prompt_wav_path + prompt_text: continuation cloning; the API
            validates that BOTH are provided together or both are None
            (a wav alone raises "prompt_wav_path and prompt_text must both
            be provided or both be None").
        """
        if not ref_path:
            return {}
        accepted = self.accepted_generate_kwargs
        if not accepted:
            # Introspection-blind build (thin *args/**kwargs wrapper):
            # optimistically request VoxCPM2-style true cloning; the
            # generate-time shrink-and-retry drops/remaps rejected params.
            return {"reference_wav_path": ref_path}
        if "reference_wav_path" in accepted:
            return {"reference_wav_path": ref_path}
        if "prompt_wav_path" in accepted and (ref_text or "").strip():
            return {"prompt_wav_path": ref_path, "prompt_text": ref_text.strip()}
        logger.warning(
            "Voice reference received but cannot be used by this voxcpm build/model "
            "(reference_wav_path requires a VoxCPM2-capable build; legacy prompt "
            "cloning additionally requires the reference transcript). "
            "Synthesizing WITHOUT voice cloning."
        )
        return {}

    def _generate_voxcpm(
        self,
        text: str,
        speed: float,
        style_instruction: str,
        cfg_value: float,
        clone_params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Real VoxCPM neural generation (blocking; run in a worker thread).

        clone_params carries the voice-cloning kwargs decided by
        _resolve_clone_params(); {} means no cloning.

        Returns (audio, sample_rate, voice_cloning_applied). The cloning flag
        is True only if the reference audio was ACTUALLY forwarded to the
        model and survived runtime retries - never guessed.
        """
        segments = _split_text_for_voxcpm(text)
        if len(segments) > 1:
            logger.info(f"VoxCPM: splitting long text into {len(segments)} segment(s)")

        wavs = []
        # Mutable clone state shared across segments: a param dropped in a
        # runtime retry must not reappear in later segments, and a param the
        # runtime already rejected must never be re-added (no ping-pong).
        active_clone_params = dict(clone_params or {})
        rejected_ref_params: set = set()
        ref_source_path = active_clone_params.get("reference_wav_path") or active_clone_params.get("prompt_wav_path")
        ref_source_text = active_clone_params.get("prompt_text")

        for seg in segments:
            # Voice-Design / Controllable-Cloning control instruction is
            # prepended in parentheses before the target text. In clone mode
            # the caller already stripped the voice-design description
            # (delivery hints only), so it cannot fight the reference timbre.
            gen_text = f"({style_instruction}){seg}" if style_instruction else seg
            kwargs = dict(
                cfg_value=cfg_value,
                inference_timesteps=self.inference_timesteps,
                retry_badcase=True,
            )
            if active_clone_params:
                # Controllable Voice Cloning: reference/prompt audio supplies
                # the timbre; the parenthesized instruction steers delivery.
                kwargs.update(active_clone_params)
            kwargs = _filter_generate_kwargs(kwargs, self.accepted_generate_kwargs)

            # Shrink-and-retry: some voxcpm releases wrap generate(*args,
            # **kwargs) so signature introspection cannot see the real
            # contract. If the runtime rejects a kwarg, drop it and retry.
            while True:
                try:
                    wav = self.model.generate(text=gen_text, **kwargs)
                    break
                except TypeError as ex:
                    m = re.search(r"unexpected keyword argument '(\w+)'", str(ex))
                    if not m or m.group(1) not in kwargs:
                        raise
                    dropped = m.group(1)
                    kwargs.pop(dropped)
                    active_clone_params.pop(dropped, None)
                    if dropped in ("reference_wav_path", "prompt_wav_path") and ref_source_path:
                        rejected_ref_params.add(dropped)
                        alt_param = "prompt_wav_path" if dropped == "reference_wav_path" else "reference_wav_path"
                        if alt_param not in kwargs and alt_param not in rejected_ref_params:
                            logger.info(f"Retrying with alternative voice reference param '{alt_param}'...")
                            kwargs[alt_param] = ref_source_path
                            active_clone_params[alt_param] = ref_source_path
                            # The legacy prompt pair REQUIRES the transcript too
                            if alt_param == "prompt_wav_path" and ref_source_text:
                                kwargs["prompt_text"] = ref_source_text
                                active_clone_params["prompt_text"] = ref_source_text
                        else:
                            active_clone_params.pop("prompt_wav_path", None)
                            active_clone_params.pop("reference_wav_path", None)
                            active_clone_params.pop("prompt_text", None)
                            logger.warning(
                                "This voxcpm version does not support reference/prompt voice cloning "
                                "- continuing without cloning."
                            )
                    else:
                        logger.warning(
                            f"Dropped generation param '{dropped}' unsupported by this voxcpm version"
                        )

            wavs.append(_consume_generated(wav))

        wavs = [w for w in wavs if w.size > 0]
        if not wavs:
            raise TTSWorkerError("VoxCPM generation returned no audio.")
        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        audio = np.clip(audio, -1.0, 1.0)

        # Precise speed control via high-quality time-stretch (VoxCPM's
        # parenthesized pace instruction is soft guidance; this guarantees
        # the requested rate).
        if HAS_LIBROSA and abs(speed - 1.0) >= 0.03:
            audio = librosa.effects.time_stretch(audio, rate=float(speed))

        # Memory hygiene: clear CUDA cache and invoke garbage collection
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()

        return audio, self.sample_rate, bool(active_clone_params)

    def _condition_reference_audio(self, ref_path: str) -> str:
        """
        Normalize the caller-supplied reference audio for VoxCPM cloning:
        - load regardless of container format (wav / mp3 / ogg / m4a)
        - convert to mono PCM16 WAV at the model's native sample rate
        - trim to MAX_REFERENCE_AUDIO_SECONDS (VoxCPM2's 8192-token KV cache
          overflows on long prompt prefill)
        Returns the conditioned path, or the original path on failure.
        """
        if not HAS_LIBROSA:
            return ref_path
        try:
            import tempfile
            y, _sr = librosa.load(ref_path, sr=self.sample_rate, mono=True)
            max_samples = int(self.max_reference_seconds * self.sample_rate)
            if len(y) > max_samples:
                y = y[:max_samples]
                logger.info(f"Reference audio trimmed to {self.max_reference_seconds}s")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, y, self.sample_rate, subtype="PCM_16")
            tmp.close()
            logger.info(
                f"Reference audio conditioned: {len(y) / self.sample_rate:.2f}s @ {self.sample_rate}Hz"
            )
            return tmp.name
        except Exception as ex:
            logger.warning(f"Reference audio conditioning failed ({ex}); using original file.")
            return ref_path

    async def _synthesize_edge_fallback(
        self,
        text: str,
        speed: float,
        prompt: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        """Explicitly-labelled Edge-TTS fallback (requires internet access)."""
        prompt_str = (prompt or "").lower()
        if any(w in prompt_str for w in ["female", "ស្រី", "sreymom", "girl", "woman"]):
            voice = "km-KH-SreymomNeural"
        else:
            voice = "km-KH-PisethNeural"

        rate_pct = int(round((speed - 1.0) * 100))
        rate_str = f"{rate_pct:+d}%"

        logger.info(f"[FALLBACK] Generating speech with Edge-TTS: voice={voice} | rate={rate_str}")
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)

        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        mp3_buffer.seek(0)
        y, sr = sf.read(mp3_buffer, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return np.clip(y.astype(np.float32), -1.0, 1.0), int(sr)

    async def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        prompt: Optional[str] = None,
        emotion: Optional[str] = None,
        temperature: float = 0.7,
        voice_ref_path: Optional[str] = None,
        voice_ref_text: Optional[str] = None,
    ) -> SynthesisOutcome:
        """
        Synthesize Khmer speech waveform from normalized text.

        Returns:
            SynthesisOutcome: (audio, sample_rate, engine, voice_cloning_applied,
            fallback_reason). `engine` names the engine that REALLY produced the
            audio, so a fallback is never misreported as VoxCPM.
        Raises:
            TTSWorkerError: when every synthesis path fails (job -> FAILED).
        """
        if not text or not text.strip():
            raise TTSWorkerError("Missing or empty text for synthesis.")

        speed = max(0.5, min(2.0, float(speed)))

        # Self-healing: if model failed to load on cold start, retry once lazily
        if not self.model_loaded and HAS_VOXCPM:
            logger.info("VoxCPM model not loaded; attempting lazy initialization on request...")
            try:
                self._load_model()
            except Exception as ex:
                logger.warning(f"Lazy VoxCPM initialization failed: {ex}")

        # Condition the voice reference (format-agnostic load, mono PCM16 at
        # the model rate, trimmed to avoid KV-cache overflow on long prompts).
        ref_path = voice_ref_path
        conditioned_ref = None
        if self.model_loaded and self.model is not None and voice_ref_path:
            ref_path = self._condition_reference_audio(voice_ref_path)
            if ref_path != voice_ref_path:
                conditioned_ref = ref_path

        # Tracks why VoxCPM did not produce the audio (surfaced to the user).
        voxcpm_failure_reason: Optional[str] = None

        # 1. Primary: REAL VoxCPM neural generation (GPU/CPU, voice cloning)
        if self.model_loaded and self.model is not None:
            # Decide ONCE how (and whether) cloning can actually happen with
            # the loaded voxcpm build/model. In clone mode the timbre comes
            # from the reference audio, so the voice-design prompt is NOT
            # injected (it would fight the clone); only delivery hints
            # (emotion / pacing) are kept.
            clone_params = self._resolve_clone_params(ref_path, voice_ref_text)
            style = _build_style_instruction(
                prompt, emotion, speed, include_voice_design=not clone_params
            )
            cfg_value = _map_temperature_to_cfg(temperature)
            cloning_applied = False

            try:
                audio, sr, cloning_applied = await asyncio.to_thread(
                    self._generate_voxcpm, text, speed, style, cfg_value, clone_params
                )
            except Exception as ex:
                logger.error(f"VoxCPM synthesis failed: {ex}", exc_info=True)
                if voice_ref_path:
                    # Retry once without the reference (timbre may be unusable),
                    # steering with the FULL voice-design prompt instead.
                    logger.warning("Retrying VoxCPM synthesis without voice reference...")
                    style_full = _build_style_instruction(
                        prompt, emotion, speed, include_voice_design=True
                    )
                    try:
                        audio, sr, _unused = await asyncio.to_thread(
                            self._generate_voxcpm, text, speed, style_full, cfg_value, {}
                        )
                        cloning_applied = False
                        voxcpm_failure_reason = (
                            f"Voice reference rejected ({ex}); VoxCPM retried without cloning."
                        )
                    except Exception as ex2:
                        logger.error(f"VoxCPM retry without reference also failed: {ex2}")
                        audio, sr = None, None
                        voxcpm_failure_reason = (
                            f"VoxCPM failed with reference ({ex}) and without it ({ex2})."
                        )
                else:
                    audio, sr = None, None
                    voxcpm_failure_reason = f"VoxCPM generation error: {ex}"

            if audio is not None:
                logger.info(
                    f"VoxCPM synthesis OK | cfg={cfg_value} | timesteps={self.inference_timesteps} | "
                    f"ref_audio={'yes' if ref_path else 'no'} | duration={len(audio)/sr:.2f}s"
                )
                # Mastering: DC removal, edge fades, peak normalize to -1 dBFS
                audio = _polish_audio(audio, sr)
                if conditioned_ref:
                    try:
                        os.remove(conditioned_ref)
                    except OSError:
                        pass
                return SynthesisOutcome(
                    audio=audio.astype(np.float32),
                    sample_rate=int(sr),
                    engine=ENGINE_LABEL_VOXCPM,
                    voice_cloning_applied=bool(cloning_applied),
                    fallback_reason=voxcpm_failure_reason,
                )

            if conditioned_ref:
                try:
                    os.remove(conditioned_ref)
                except OSError:
                    pass
        else:
            voxcpm_failure_reason = (
                "VoxCPM model is not loaded (the 'voxcpm' package is missing or the "
                "weights failed to load)."
            )

        # 2. Explicitly-labelled Edge-TTS fallback
        if HAS_EDGE_TTS:
            try:
                audio, sr = await self._synthesize_edge_fallback(text, speed, prompt)
                reason = voxcpm_failure_reason or "VoxCPM unavailable; Edge-TTS fallback used."
                logger.warning(
                    "[FALLBACK] Audio produced by Edge-TTS (NOT VoxCPM). "
                    "Voice cloning and emotion control were NOT applied. "
                    f"Reason: {reason} | duration={len(audio)/sr:.2f}s"
                )
                return SynthesisOutcome(
                    audio=np.clip(audio.astype(np.float32), -1.0, 1.0),
                    sample_rate=int(sr),
                    engine=ENGINE_LABEL_EDGE_FALLBACK,
                    voice_cloning_applied=False,
                    fallback_reason=reason,
                )
            except Exception as ex:
                logger.error(f"Edge-TTS fallback failed: {ex}")
                raise TTSWorkerError(
                    "All synthesis engines failed. VoxCPM is not loaded and the "
                    f"Edge-TTS fallback is unavailable (offline or blocked): {ex}"
                ) from ex

        # 3. Hard failure - no fake beeps.
        raise TTSWorkerError(
            "All synthesis engines failed. VoxCPM is not loaded and the "
            "Edge-TTS fallback is unavailable (offline or blocked)."
        )


# Singleton instance for worker warm-start efficiency
_engine_instance: Optional[KhmerTTSModelEngine] = None


def get_model_engine() -> "KhmerTTSModelEngine":
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = KhmerTTSModelEngine()
    return _engine_instance


def preload_model() -> None:
    """Optionally load weights at container start (serverless cold start)."""
    if _env_flag("VOXCPM_PRELOAD", True):
        try:
            engine = get_model_engine()
            if engine.model_loaded:
                logger.info("VoxCPM model preloaded and warm.")
        except TTSWorkerError as ex:
            logger.error(f"Model preload failed (job requests will surface this error): {ex}")