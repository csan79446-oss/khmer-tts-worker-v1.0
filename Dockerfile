# NVIDIA NGC PyTorch 25.04 — PyTorch 2.7.0 (NVIDIA build 2.7.0a0+79aa17489c, Python 3.12)
# with CUDA 12.9 + Blackwell (sm_100) support.
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

# ---------------------------------------------------------------------------
# Python dependencies
#
# The NGC image already ships PyTorch, torchaudio and the whole CUDA stack, so
# this step only adds the worker's own packages. Two traps live here:
#
#   1) Since NGC 25.03 the container exports PIP_CONSTRAINT=/etc/pip/constraint.txt
#      — an exact-pin file listing EVERY package the image was built with. Each
#      pip run inside the image inherits it, which makes current PyPI packages
#      (gradio 6, datasets 3, funasr, modelscope, huggingface-hub, ...)
#      unresolvable and aborts the build with "exit code: 1" (the real pip error
#      scrolls by above that line).
#      -> We REWRITE that file with a minimal constraint set that only pins the
#         two CUDA/ABI-critical packages to the builds already in the image.
#         pip can therefore never silently re-download a multi-GB PyPI
#         torch/torchaudio (the classic cause of these failed builds), while
#         every other package resolves normally.
#
#   2) Never add `pip install torchaudio` (or torch) to this image: PyPI's
#      torchaudio pins `torch==<exact release>`, while this container ships an
#      NVIDIA pre-release build (2.7.0a0+<hash>) that PEP 440 ranks BELOW
#      2.7.0. pip would attempt to replace the entire CUDA stack. The image's
#      own torchaudio already satisfies requirements.txt (torchaudio>=2.5.0).
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN set -eux; \
    python -c "import torch, torchaudio; print('torch==' + torch.__version__); print('torchaudio==' + torchaudio.__version__)" > /etc/pip/constraint.txt; \
    echo '--- pip constraints (replaces the NGC blanket pin file) ---'; \
    cat /etc/pip/constraint.txt; \
    echo '--- installing worker requirements ---'; \
    PIP_CONSTRAINT=/etc/pip/constraint.txt python -m pip install --no-cache-dir --upgrade pip; \
    PIP_CONSTRAINT=/etc/pip/constraint.txt python -m pip install --no-cache-dir -r /app/requirements.txt; \
    echo '--- key package versions now in the image ---'; \
    python -m pip list --format=freeze | grep -Ei '^(torch|torchaudio|torchcodec|voxcpm|transformers|numpy|librosa|soundfile|edge-tts|runpod)='; \
    echo '--- import smoke test: fail the BUILD now, not the first paid job ---'; \
    python -c "import torch, torchaudio, numpy, soundfile, librosa, edge_tts, runpod; print('core imports OK | torch', torch.__version__)"; \
    python -c "import voxcpm; print('voxcpm import OK | version', getattr(voxcpm, '__version__', 'unknown'))"

# Application worker code (isolated in /app so Network Volume mounts at
# /workspace or /runpod-volume NEVER mask or overwrite the handler scripts)
COPY . /app/

# Ensure default fallback directories exist
RUN mkdir -p /workspace/models /runpod-volume/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "/app/handler.py" ]
