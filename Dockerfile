# PyTorch 2.5.1 + CUDA 12.4 base image (VoxCPM requires torch >= 2.5.0)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (audio codecs, ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# VoxCPM weight cache & volume mount defaults.
# By default we support /workspace or /runpod-volume (RunPod default).
ENV HF_HOME=/workspace/models/hf_cache
ENV MODEL_PATH=/workspace/models
ENV VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_DEVICE=auto
ENV VOXCPM_TIMESTEPS=10
ENV VOXCPM_PRELOAD=1
ENV PYTHONPATH=/app

# Python dependencies:
# 1. Base image provides torch 2.5.1 + CUDA 12.4.
# 2. Explicitly install torchaudio 2.5.1 with CUDA 12.4 wheels first so pip
#    never downgrades torch or pulls incompatible CPU torchaudio when voxcpm installs.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r /app/requirements.txt

# Application worker code (isolated in /app so Network Volume mounts at
# /workspace or /runpod-volume NEVER mask or overwrite the handler scripts)
COPY . /app/

# Ensure default fallback directories exist
RUN mkdir -p /workspace/models /runpod-volume/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "/app/handler.py" ]
