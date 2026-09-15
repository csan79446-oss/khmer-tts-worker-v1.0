"""
Local verification script for RunPod Worker without running inside Docker.

Requires network access for the Edge-TTS fallback (the real VoxCPM engine is
only active when the 'voxcpm' package and model weights are available).
"""
import asyncio
import base64
import json
import os
import sys
import io
import soundfile as sf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import handler
from model_engine import TTSWorkerError

def run_local_test():
    input_path = os.path.join(os.path.dirname(__file__), "test_input.json")
    print(f"[*] Reading test payload: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    print("[*] Invoking handler(job)...")
    try:
        result = asyncio.run(handler(job))
    except (TTSWorkerError, ValueError) as ex:
        print(f"[!] Worker raised (job would be FAILED on RunPod): {ex}")
        sys.exit(1)

    if "error" in result:
        print(f"[!] Worker returned error: {result['error']}")
        sys.exit(1)

    assert "audio_base64" in result, "Missing audio_base64 in response"
    assert "sample_rate" in result, "Missing sample_rate in response"
    assert "duration" in result, "Missing duration in response"
    assert result.get("status") == "success", "Worker status is not 'success'"

    b64_audio = result["audio_base64"]
    raw_wav = base64.b64decode(b64_audio)
    print(f"[+] Audio decoded successfully! Size: {len(raw_wav):,} bytes")
    print(f"[+] Engine used: {result.get('engine', 'unknown')}")

    with io.BytesIO(raw_wav) as bio:
        data, sr = sf.read(bio)
        print(f"[+] Verified WAV header: Sample Rate={sr}Hz, Channels={1 if data.ndim == 1 else data.shape[1]}, Samples={len(data)}")

    out_file = os.path.join(os.path.dirname(__file__), "test_output.wav")
    with open(out_file, "wb") as f:
        f.write(raw_wav)
    print(f"[+] Saved test audio output to: {out_file}")
    print("[✓] Local worker test PASSED!")

if __name__ == "__main__":
    run_local_test()