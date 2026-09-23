# Khmer TTS AI Worker (RunPod Serverless v2.0 — REAL VoxCPM)

Production-grade RunPod Serverless AI Worker for Khmer Text-to-Speech, running the
**genuine [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM)** neural model
(voice design + true zero-shot voice cloning), designed to serve the
Khmer TTS Studio Desktop Application.

## 🌟 What this worker actually does

| Feature | Implementation |
|---|---|
| **VoxCPM neural synthesis** | Real `voxcpm` package, `VoxCPM.from_pretrained("openbmb/VoxCPM2")` (or a local checkpoint / Network Volume) |
| **Voice Design** | The UI's voice-prompt presets + emotion are compiled into a VoxCPM control instruction, e.g. `(A warm Cambodian female voice, calm tone, slightly slower pace)` |
| **Voice Cloning** | Base64 reference audio is decoded and passed as `reference_wav_path` (VoxCPM2 true zero-shot cloning — used ALONE, no transcript needed). Legacy 1.x builds fall back to the `prompt_wav_path` + `prompt_text` pair (requires the reference transcript from the UI) or skip cloning with truthful provenance |
| **Temperature** | Mapped onto VoxCPM `cfg_value` (0.7 → ≈2.0 balanced default; range 1.0–3.0) |
| **Speed** | Soft style guidance + precise `librosa` time-stretch to guarantee the requested rate |
| **Edge-TTS fallback** | Explicitly labelled and logged as `[FALLBACK]` (used only when VoxCPM weights can't load or the package is missing); **no** synthetic-beep generator exists anymore |
| **Error contract** | The handler **raises** on failure → RunPod marks the job `FAILED` and the desktop client shows the real reason |
| **Output** | Mono WAV PCM-16, Base64, native VoxCPM2 sample rate (48 kHz) |

## 📂 Project Structure

```text
runpod_worker/
├── handler.py          # RunPod serverless entrypoint (raises on failure)
├── model_engine.py     # REAL VoxCPM engine + labelled Edge-TTS fallback
├── Dockerfile          # Official pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime base (CUDA 12.8, Blackwell sm_100)
├── requirements.txt    # Minimal verified runtime deps; VoxCPM 2.0.3 installed separately (--no-deps)
├── .dockerignore
├── test_input.json     # Test payload with Khmer text
├── test_local.py       # Local verification (uses Edge-TTS fallback if voxcpm absent)
├── DEPLOY_GUIDE.md     # Full deployment guide (Khmer & English)
└── models/             # Optional local checkpoint dir / HF cache (Network Volume)
```

## 🔧 Worker Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VOXCPM_MODEL_ID` | `openbmb/VoxCPM2` | HF repo id (the `Tha456/VoxCPM2` mirror works too), or a local checkpoint dir. **Auto-fallback:** if the installed `voxcpm` package is the 1.x PyPI API, the worker detects it at load time and automatically switches to `openbmb/VoxCPM-0.5B` (16 kHz) instead of crashing after the multi-GB download. For true 48 kHz VoxCPM2 output, install a modern build: `pip install git+https://github.com/OpenBMB/VoxCPM.git` |
| `VOXCPM_DEVICE` | `auto` | `auto` (cuda→mps→cpu), `cuda`, `cpu` |
| `VOXCPM_TIMESTEPS` | `10` | Diffusion steps (4–30; more = better quality, slower) |
| `VOXCPM_DENOISER` | `0` | Load reference-audio denoiser (16 kHz pipeline) |
| `VOXCPM_OPTIMIZE` | `1` | torch.compile optimizations ( `0` to disable on issues). If torch.compile/inductor fails at load (e.g. no C compiler in the image), the worker **automatically retries in eager mode** instead of failing the job |
| `MAX_REFERENCE_AUDIO_SECONDS` | `10` | Trim voice-clone reference audio (VoxCPM2's 8192-token KV cache overflows on long prompt prefill) |
| `VOXCPM_PRELOAD` | `1` | Load weights at container start (cold-start hygiene) |
| `MODEL_PATH` | `/workspace/models` | Local checkpoint / volume dir (auto-checks `/runpod-volume/models`) |
| `HF_HOME` | `/workspace/models/hf_cache` | HF download cache (auto-checks `/runpod-volume/hf_cache`) |

## 🚀 Quick Start (Local Testing)

```bash
python runpod_worker/test_local.py
# or
python runpod_worker/handler.py --test
```

> Without the `voxcpm` package installed, this exercises the labelled Edge-TTS
> fallback so the payload/response contract can still be verified.

## 🐳 Docker Build & Push

```bash
docker build -t your-dockerhub-username/khmer-tts-worker:v2.0 .
docker push your-dockerhub-username/khmer-tts-worker:v2.0
```

## ☁️ RunPod Serverless Deployment

1. **Template**: container image above, **Container Disk ≥ 25 GB** (VoxCPM2 weights are multi-GB).
2. **Network Volume** (strongly recommended): mount at `/runpod-volume` (RunPod default) or `/workspace`.
   Worker code is isolated in `/app` so external volume mounts never mask the application code.
3. **GPU**: RTX 4090 / 3090 / A40 (VoxCPM2 is a 2B model; ≥16 GB VRAM recommended).
   Min workers `0`, idle timeout `60 s`.
4. Connect Endpoint URL + API Key in the desktop app's Settings → **VoxCPM & RunPod**.

See [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) for the full guide.

## ⚠️ Language support note

VoxCPM2 officially supports 30 languages; Khmer is **not guaranteed** to be on
that list. Quality on Khmer should be evaluated — if results are poor, options
are (a) LoRA fine-tuning VoxCPM on Khmer data, or (b) using the desktop app's
local Edge-TTS engine. The worker always reports which engine produced the
audio via the `engine` field in its response.

## 📄 License

MIT License (worker code). VoxCPM model weights: Apache-2.0.
