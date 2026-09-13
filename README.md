# Khmer TTS AI Worker (RunPod Serverless v1.0)

Production-grade RunPod Serverless AI Worker for Khmer Text-to-Speech (TTS), voice cloning, and audio synthesis, designed to serve the [Khmer TTS Studio Desktop Application](https://github.com/csan79446-oss).

---

## 🌟 Features

- **RunPod Serverless SDK v1.6+**: Standard handler architecture (`handler.py`) supporting asynchronous execution.
- **Khmer Neural Acoustic Synthesis**: 24,000 Hz high-fidelity mono WAV PCM output.
- **Voice Reference Audio Cloning**: Accepts Base64 encoded WAV/MP3 reference audio to adapt pitch and speaker cadence.
- **Speed, Emotion & Prompt Control**: Fine-grained duration stretching, prompt steering, and emotional variance.
- **Containerized for GPU**: Optimized Docker image with PyTorch 2.2, CUDA 12.1, FFmpeg, and libsndfile.

---

## 📂 Project Structure

```text
khmer-tts-worker/
├── handler.py          # RunPod serverless entrypoint
├── model_engine.py     # Neural acoustic model engine & GPU manager
├── Dockerfile          # CUDA-accelerated container image
├── requirements.txt    # Python dependencies
├── test_input.json     # Test payload with Khmer text
├── test_local.py       # Offline local verification script
├── DEPLOY_GUIDE.md     # Full step-by-step deployment guide (Khmer & English)
├── models/             # Directory for custom model checkpoints (.gitkeep)
└── README.md           # Project documentation
```

---

## 🚀 Quick Start (Local Testing)

Test the worker handler locally on your workstation without Docker:

```bash
python test_local.py
```

Or run directly with the test flag:
```bash
python handler.py --test
```

---

## 🐳 Docker Build & Push

```bash
# 1. Login to Docker Hub
docker login

# 2. Build image
docker build -t your-dockerhub-username/khmer-tts-worker:v1.0 .

# 3. Push to registry
docker push your-dockerhub-username/khmer-tts-worker:v1.0
```

---

## ☁️ RunPod Serverless Deployment

1. Create a **Template** on [RunPod Serverless Console](https://www.runpod.io/console/serverless):
   - **Container Image**: `your-dockerhub-username/khmer-tts-worker:v1.0`
   - **Container Disk**: `20 GB`
2. Create an **Endpoint**:
   - Choose GPU: **RTX 4090**, **RTX 3090**, or **A4000**
   - Active Workers (Min): `0` (cost efficient)
   - Max Workers: `2`
   - Idle Timeout: `60` seconds
3. Connect your Endpoint URL and API Key in **Khmer TTS V2 Studio** Settings.

See [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) for the full guide.

---

## 📄 License

MIT License.
