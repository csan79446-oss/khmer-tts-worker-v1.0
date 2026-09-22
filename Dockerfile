# NVIDIA NGC PyTorch 25.04 — includes PyTorch 2.7+ with CUDA 13.0 + Blackwell (sm_100) support.
# Use this image when RunPod assigns Blackwell GPUs (B200/GB200, compute capability sm_100).
# For older GPUs (RTX 4090/A100/H100), pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime also works.
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (audio codecs, ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
# build-essential: gcc/g++ required by torch._inductor when VOXCPM_OPTIMIZE=1
# (torch.compile compiles generated C/C++ code; without a compiler the model
# load fails with "Failed to find C compiler").

# VoxCPM weight cache & volume mount defaults.
# By default we support /workspace or /runpod-volume (RunPod default).
ENV HF_HOME=/workspace/models/hf_cache
ENV MODEL_PATH=/workspace/models
# Target checkpoint. requirements.txt installs the MODERN voxcpm build from
# the OpenBMB GitHub repo (VoxCPM2 API), so VoxCPM2 runs natively at 48 kHz
# with true zero-shot cloning via reference_wav_path (alone, no transcript).
# If a legacy 1.x build ever ends up installed instead, the worker detects it
# at load time and auto-downgrades to 'openbmb/VoxCPM-0.5B' (16 kHz).
# Mirror repos (e.g. Tha456/VoxCPM2) also work as VOXCPM_MODEL_ID values.
ENV VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_DEVICE=auto
ENV VOXCPM_TIMESTEPS=10
ENV VOXCPM_PRELOAD=1
ENV PYTHONPATH=/app

# Python dependencies:
# NGC base image already includes PyTorch 2.7+ and torchaudio — do NOT reinstall them.
# Only install worker-specific packages (runpod, voxcpm, edge-tts, soundfile, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Application worker code (isolated in /app so Network Volume mounts at
# /workspace or /runpod-volume NEVER mask or overwrite the handler scripts)
COPY . /app/

# Ensure default fallback directories exist
RUN mkdir -p /workspace/models /runpod-volume/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "/app/handler.py" ]
