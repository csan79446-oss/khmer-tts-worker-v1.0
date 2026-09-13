"""
RunPod Serverless Async Handler for Khmer TTS / VoxCPM AI Engine.
Complies with RunPod Serverless SDK v1.6+ specification.
"""
import asyncio
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, Any
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [Worker] %(message)s")
logger = logging.getLogger("KhmerTTSHandler")

from model_engine import get_model_engine


async def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless async worker entry point.
    Receives incoming job with payload: {"input": {...}}
    Returns:
        {
            "audio_base64": "<base64_encoded_wav>",
            "sample_rate": 24000,
            "duration": 4.52,
            "status": "success"
        }
    """
    job_id = job.get("id", "unknown_job")
    job_input = job.get("input", {})

    logger.info(f"Received job {job_id} | Input keys: {list(job_input.keys())}")

    # Validate input
    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "Missing or empty 'text' in job input."}

    speed = float(job_input.get("speed", 1.0))
    prompt = job_input.get("prompt", None)
    emotion = job_input.get("emotion", None)
    temperature = float(job_input.get("temperature", 0.7))

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

    try:
        # Obtain model engine singleton
        start_t = time.time()
        engine = get_model_engine()

        # Asynchronously synthesize audio
        audio_array, sample_rate = await engine.synthesize(
            text=text,
            speed=speed,
            prompt=prompt,
            emotion=emotion,
            temperature=temperature,
            voice_ref_path=temp_ref_path
        )

        duration = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0
        elapsed = time.time() - start_t
        logger.info(f"Synthesis finished in {elapsed:.2f}s | Audio duration: {duration:.2f}s")

        # Encode generated PCM float32 array to 16-bit WAV PCM in memory
        bio = io.BytesIO()
        sf.write(bio, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = bio.getvalue()
        b64_output = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "audio_base64": b64_output,
            "sample_rate": sample_rate,
            "duration": round(duration, 3),
            "status": "success"
        }

    except Exception as exc:
        logger.error(f"Error during TTS synthesis in job {job_id}: {exc}", exc_info=True)
        return {"error": f"Synthesis error: {str(exc)}"}

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
            logger.info(f"SUCCESS: Generated {out_wav} (duration={result['duration']}s, sr={result['sample_rate']}Hz)")
        else:
            logger.error(f"FAILED: {result}")
    else:
        # Standard RunPod Serverless startup (natively supports async handler)
        try:
            import runpod
            logger.info("Starting RunPod Serverless async worker loop...")
            runpod.serverless.start({"handler": handler})
        except ImportError:
            logger.error("runpod SDK is not installed. To test locally, run: python handler.py --test")
            sys.exit(1)
