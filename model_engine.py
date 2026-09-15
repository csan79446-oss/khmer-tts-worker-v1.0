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
import io
import os
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("KhmerTTSModelEngine")

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
) -> str:
    """
    Build a VoxCPM2 Voice-Design / Controllable-Cloning control instruction.
    The instruction is prepended to the text inside parentheses, e.g.
    "(warm female voice, calm tone, slightly slower pace)".
    """
    parts: List[str] = []

    prompt_text = (prompt or "").strip()
    # The desktop app's prompt presets are long descriptive sentences; they
    # work verbatim as a design instruction. Only hard-cap absurd lengths.
    if prompt_text:
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
class KhmerTTSModelEngine:
    """
    Real VoxCPM model engine. Manages checkpoint loading, GPU memory
    allocation, and neural speech inference with true voice cloning.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or os.getenv("MODEL_PATH", "/workspace/models"))
        self.model_id = os.getenv("VOXCPM_MODEL_ID", "openbmb/VoxCPM2").strip()
        self.device = os.getenv("VOXCPM_DEVICE", "auto").strip() or "auto"
        self.inference_timesteps = max(4, min(30, int(os.getenv("VOXCPM_TIMESTEPS", "10"))))
        self.load_denoiser = _env_flag("VOXCPM_DENOISER", False)
        self.optimize = _env_flag("VOXCPM_OPTIMIZE", True)
        self.sample_rate = 48000  # VoxCPM2 native output; updated after model load

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
            local_checkpoint = self._resolve_local_checkpoint()
            if local_checkpoint:
                logger.info(f"Using local VoxCPM checkpoint: {local_checkpoint}")
                self.model = VoxCPM.from_pretrained(
                    str(local_checkpoint),
                    load_denoiser=self.load_denoiser,
                    optimize=self.optimize,
                    device=self.device,
                )
            else:
                logger.info("No local checkpoint found; downloading/loading from HuggingFace Hub...")
                self.model = VoxCPM.from_pretrained(
                    self.model_id,
                    load_denoiser=self.load_denoiser,
                    optimize=self.optimize,
                    device=self.device,
                )
            self.sample_rate = int(self.model.tts_model.sample_rate)
            self.model_loaded = True
            logger.info(
                f"VoxCPM model loaded successfully | native sample rate: {self.sample_rate} Hz"
            )
        except Exception as ex:
            self.model = None
            self.model_loaded = False
            logger.error(f"Failed to load VoxCPM model: {ex}", exc_info=True)
            raise TTSWorkerError(f"VoxCPM model failed to load: {ex}")

    # ------------------------------------------------------------------
    # Synthesis paths
    # ------------------------------------------------------------------
    def _generate_voxcpm(
        self,
        text: str,
        speed: float,
        style_instruction: str,
        cfg_value: float,
        voice_ref_path: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        """Real VoxCPM neural generation (blocking; run in a worker thread)."""
        segments = _split_text_for_voxcpm(text)
        if len(segments) > 1:
            logger.info(f"VoxCPM: splitting long text into {len(segments)} segment(s)")

        wavs = []
        for seg in segments:
            # Voice-Design / Controllable-Cloning control instruction is
            # prepended in parentheses before the target text.
            gen_text = f"({style_instruction}){seg}" if style_instruction else seg
            kwargs = dict(
                cfg_value=cfg_value,
                inference_timesteps=self.inference_timesteps,
                retry_badcase=True,
            )
            if voice_ref_path:
                # Controllable Voice Cloning: reference_wav_path supplies the
                # timbre; the parenthesized instruction steers style. No
                # transcript of the reference is required (VoxCPM2).
                kwargs["reference_wav_path"] = voice_ref_path
            wav = self.model.generate(text=gen_text, **kwargs)
            wavs.append(np.asarray(wav, dtype=np.float32).squeeze())

        audio = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        audio = np.clip(audio, -1.0, 1.0)

        # Precise speed control via high-quality time-stretch (VoxCPM's
        # parenthesized pace instruction is soft guidance; this guarantees
        # the requested rate).
        if HAS_LIBROSA and abs(speed - 1.0) >= 0.03:
            audio = librosa.effects.time_stretch(audio, rate=float(speed))

        return audio, self.sample_rate

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
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize Khmer speech waveform from normalized text.

        Returns:
            Tuple[np.ndarray, int]: (audio_waveform_float32, sample_rate)
        Raises:
            TTSWorkerError: when every synthesis path fails (job -> FAILED).
        """
        if not text or not text.strip():
            raise TTSWorkerError("Missing or empty text for synthesis.")

        speed = max(0.5, min(2.0, float(speed)))

        # 1. Primary: REAL VoxCPM neural generation (GPU/CPU, voice cloning)
        if self.model_loaded and self.model is not None:
            style = _build_style_instruction(prompt, emotion, speed)
            cfg_value = _map_temperature_to_cfg(temperature)

            try:
                audio, sr = await asyncio.to_thread(
                    self._generate_voxcpm, text, speed, style, cfg_value, voice_ref_path
                )
            except Exception as ex:
                logger.error(f"VoxCPM synthesis failed: {ex}", exc_info=True)
                if voice_ref_path:
                    # Retry once without the reference (timbre may be unusable)
                    logger.warning("Retrying VoxCPM synthesis without voice reference...")
                    try:
                        audio, sr = await asyncio.to_thread(
                            self._generate_voxcpm, text, speed, style, cfg_value, None
                        )
                    except Exception as ex2:
                        logger.error(f"VoxCPM retry without reference also failed: {ex2}")
                        audio, sr = None, None
                else:
                    audio, sr = None, None

            if audio is not None:
                logger.info(
                    f"VoxCPM synthesis OK | cfg={cfg_value} | timesteps={self.inference_timesteps} | "
                    f"ref_audio={'yes' if voice_ref_path else 'no'} | duration={len(audio)/sr:.2f}s"
                )
                # Mild peak normalization (app caps any further boost)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = np.clip((audio / peak) * 0.92, -1.0, 1.0)
                return audio.astype(np.float32), sr

        # 2. Explicitly-labelled Edge-TTS fallback
        if HAS_EDGE_TTS:
            try:
                audio, sr = await self._synthesize_edge_fallback(text, speed, prompt)
                logger.warning(
                    "[FALLBACK] Audio produced by Edge-TTS (NOT VoxCPM). "
                    "Voice cloning and emotion control were NOT applied. "
                    f"duration={len(audio)/sr:.2f}s"
                )
                return audio, sr
            except Exception as ex:
                logger.error(f"Edge-TTS fallback failed: {ex}")

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