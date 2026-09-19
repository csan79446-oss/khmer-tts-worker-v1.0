"""
RunPod Serverless Async Handler for Khmer TTS / VoxCPM AI Engine.
Complies with RunPod Serverless SDK v1.6+ specification.

Error contract: this handler RAISES on failure so the RunPod job is marked
FAILED with the real error message (surfaced by the desktop client). Success
returns: {"audio_base64": ..., "sample_rate": ..., "duration": ..., "status": "success"}
"""
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from typing import Dict, Any
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [Worker] %(message)s")
logger = logging.getLogger("KhmerTTSHandler")

# Ensure worker directory is on sys.path regardless of execution CWD
_worker_dir = os.path.dirname(os.path.abspath(__file__))
if _worker_dir not in sys.path:
    sys.path.insert(0, _worker_dir)

from model_engine import get_model_engine, preload_model, TTSWorkerError, ENGINE_LABEL_VOXCPM


def normalize_khmer_text(text: str) -> str:
    """NFC normalize, strip zero-width/invisible characters, collapse whitespace."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    zero_width = "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad"
    text = "".join(ch for ch in text if ch not in zero_width)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless async worker entry point.
    Receives incoming job with payload: {"input": {...}}
    Input fields:
        text (required), speed, prompt, emotion, temperature,
        voice_reference | reference_audio_base64 (Base64 WAV/MP3, optional)
    Returns:
        {
            "audio_base64": "<base64_encoded_wav>",
            "sample_rate": 48000,
            "duration": 4.52,
            "status": "success",
            "engine": "VoxCPM" | "Edge-TTS (fallback)",
            "voice_cloning_applied": true | false,
            "fallback_reason": "<why the primary engine failed>"  # omitted on clean VoxCPM runs
        }

        The "engine" field always names the engine that REALLY produced the
        audio. A silent degradation to Edge-TTS is reported as
        "Edge-TTS (fallback)" - never as "VoxCPM".
    Raises:
        ValueError / TTSWorkerError -> RunPod marks the job FAILED.
    """
    job_id = job.get("id", "unknown_job")
    job_input = job.get("input") or {}

    logger.info(f"Received job {job_id} | Input keys: {list(job_input.keys())}")

    if not isinstance(job_input, dict):
        raise ValueError("Job 'input' must be a JSON object.")

    # Validate input (NFC normalize + strip invisible characters)
    text = normalize_khmer_text(str(job_input.get("text", "")))
    if not text:
        raise ValueError("Missing or empty 'text' in job input.")

    try:
        speed = float(job_input.get("speed", 1.0))
        temperature = float(job_input.get("temperature", 0.7))
    except (TypeError, ValueError) as ex:
        raise ValueError(f"Invalid numeric parameter: {ex}")

    prompt = job_input.get("prompt", None)
    emotion = job_input.get("emotion", None)

    # Check for voice cloning reference audio (Base64)
    ref_b64 = job_input.get("voice_reference") or job_input.get("reference_audio_base64")
    temp_ref_path = None

    if ref_b64:
        try:
            # Handle data URL prefix if present (e.g. data:audio/wav;base64,...)
            if "," in ref_b64:
                ref_b64 = ref_b64.split(",", 1)[1]

            audio_bytes = base64.b64decode(ref_b64)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.write(audio_bytes)
            temp_file.close()
            temp_ref_path = temp_file.name
            logger.info(f"Decoded voice reference audio ({len(audio_bytes)} bytes) to {temp_ref_path}")
        except Exception as ex:
            logger.warning(f"Failed to decode voice reference audio: {ex}")

    # Optional transcript of the reference audio (what is spoken in it).
    # VoxCPM2 clones via reference_wav_path alone; legacy 1.x builds can only
    # clone via the prompt_wav_path + prompt_text pair, which needs this.
    ref_text_raw = (
        job_input.get("reference_text")
        or job_input.get("reference_prompt_text")
        or job_input.get("reference_transcript")
    )
    ref_text = str(ref_text_raw).strip() if ref_text_raw else None
    if ref_text:
        logger.info(f"Reference transcript received ({len(ref_text)} chars)")

    try:
        # Obtain model engine singleton
        start_t = time.time()
        engine = get_model_engine()

        # Asynchronously synthesize audio (raises TTSWorkerError on total failure).
        # The outcome names the engine that ACTUALLY produced the audio, so a
        # silent fallback is never reported as "VoxCPM".
        outcome = await engine.synthesize(
            text=text,
            speed=speed,
            prompt=prompt,
            emotion=emotion,
            temperature=temperature,
            voice_ref_path=temp_ref_path,
            voice_ref_text=ref_text,
        )
        audio_array = outcome.audio
        sample_rate = outcome.sample_rate
        used_engine = outcome.engine

        duration = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0
        elapsed = time.time() - start_t
        logger.info(
            f"Synthesis finished in {elapsed:.2f}s | engine={used_engine} | "
            f"voice_cloning={outcome.voice_cloning_applied} | "
            f"Audio duration: {duration:.2f}s"
        )
        if used_engine != ENGINE_LABEL_VOXCPM:
            logger.warning(
                f"Job {job_id} degraded to '{used_engine}' - audio was NOT produced by "
                f"VoxCPM (voice cloning, emotion and temperature were not applied). "
                f"Reason: {outcome.fallback_reason}"
            )

        # Encode generated PCM float32 array to 16-bit WAV PCM in memory
        bio = io.BytesIO()
        sf.write(bio, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = bio.getvalue()
        b64_output = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "audio_base64": b64_output,
            "sample_rate": sample_rate,
            "duration": round(duration, 3),
            "status": "success",
            "engine": used_engine,
            "voice_cloning_applied": bool(outcome.voice_cloning_applied),
            # VoxCPM checkpoint actually configured/loaded for this job
            # (differs from VOXCPM_MODEL_ID when the legacy 1.x package
            # forced a downgrade). The desktop client surfaces this as
            # worker provenance in the generation diagnostics.
            "model_id": getattr(engine, "effective_model_id", None) or "unknown",
            **({"fallback_reason": outcome.fallback_reason[:500]}
               if outcome.fallback_reason else {}),
        }

    finally:
        # Clean up temporary voice reference file
        if temp_ref_path and os.path.exists(temp_ref_path):
            try:
                os.remove(temp_ref_path)
            except OSError:
                pass


if __name__ == "__main__":
    # Check if CLI test mode is requested
    if "--test" in sys.argv or "-t" in sys.argv:
        test_file = os.path.join(os.path.dirname(__file__), "test_input.json")
        logger.info(f"Running in test mode with input file: {test_file}")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_payload = json.load(f)
        else:
            test_payload = {"input": {"text": "ជំរាបសួរ! នេះជាការសាកល្បង Khmer TTS។", "speed": 0.9}}

        result = asyncio.run(handler(test_payload))
        if "audio_base64" in result:
            out_wav = os.path.join(os.path.dirname(__file__), "test_output.wav")
            with open(out_wav, "wb") as f:
                f.write(base64.b64decode(result["audio_base64"]))
            logger.info(
                f"SUCCESS: Generated {out_wav} (duration={result['duration']}s, "
                f"sr={result['sample_rate']}Hz, engine={result.get('engine')})"
            )
        else:
            logger.error(f"FAILED: {result}")
    else:
        # Standard RunPod Serverless startup (natively supports async handler)
        try:
            import runpod
            logger.info("Initializing Khmer TTS AI Worker for RunPod Serverless...")
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(
                        f"Hardware: GPU '{torch.cuda.get_device_name(0)}' detected "
                        f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM) "
                        f"| PyTorch {torch.__version__} | CUDA {getattr(torch.version, 'cuda', 'unknown')}"
                    )
                else:
                    logger.warning("Hardware: No GPU detected! Worker running on CPU.")
            except ImportError:
                pass

            # Warm the model before the first job arrives (cold-start hygiene).
            preload_model()
            logger.info("Starting RunPod Serverless worker listening loop...")
            runpod.serverless.start({"handler": handler})
        except ImportError:
            logger.error("runpod SDK is not installed. To test locally, run: python handler.py --test")
            sys.exit(1)