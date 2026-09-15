# PyTorch 2.5.1 + CUDA 12.4 base image (VoxCPM requires torch >= 2.5.0)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies (audio codecs, ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# VoxCPM weight cache. Point HF_HOME at a RunPod Network Volume mount so the
# multi-GB checkpoint survives cold starts and is downloaded only once.
# (Template env: HF_HOME=/workspace/models/hf_cache, MODEL_PATH=/workspace/models)
ENV HF_HOME=/workspace/models/hf_cache
ENV MODEL_PATH=/workspace/models
ENV VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_DEVICE=auto
ENV VOXCPM_TIMESTEPS=10
ENV VOXCPM_PRELOAD=1

# Python dependencies (torch 2.5.1 is preinstalled in the base image, so the
# voxcpm dependency pin 'torch>=2.5.0' is already satisfied)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application worker code
COPY . /workspace/

RUN mkdir -p /workspace/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "handler.py" ]
