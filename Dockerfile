# ---------------------------------------------------------------------------
# Khmer TTS RunPod Serverless worker image.
#
# Base: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (OFFICIAL PyTorch image)
#   Python 3.11 | PyTorch 2.7.1 stable | CUDA 12.8 | cuDNN 9
#   CUDA 12.8 ships Blackwell (sm_100) kernels, so B200/GB200 pods work too.
#
# Design philosophy:
#   - NO constraints.txt. That file contained phantom version numbers that
#     caused every build to fail. pip is allowed to resolve freely.
#   - The ONLY hard pins are the ones that are genuinely known to break:
#       * transformers<5   -- VoxCPM 2.0.3 uses V1 tokenizer APIs removed in 5.x
#       * VoxCPM @ tag     -- pinned to a specific release tag for reproducibility
#   - VoxCPM is installed with --no-deps because its pyproject.toml declares
#     `torchcodec` which has NO pre-built wheel for torch 2.7.x. All other
#     VoxCPM runtime deps are installed explicitly in a later step.
#   - torch / torchaudio ship inside the base image and are never reinstalled.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
#   ffmpeg        -- audio encode/decode used by librosa, soundfile, edge-tts
#   libsndfile1   -- soundfile C library
#   git           -- needed so pip can clone VoxCPM from GitHub
#   build-essential -- gcc/g++ for torch._inductor (torch.compile) if enabled
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Runtime environment variables
# ---------------------------------------------------------------------------
ENV HF_HOME=/workspace/models/hf_cache
ENV MODEL_PATH=/workspace/models
ENV VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_DEVICE=auto
ENV VOXCPM_TIMESTEPS=10
ENV VOXCPM_PRELOAD=1
ENV PYTHONPATH=/app
# Allow pip to install into conda-managed Python (PEP 668 override)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Copy only the files needed at build time (cached separately from app code)
COPY check_imports.py /app/

# ---------------------------------------------------------------------------
# Step 1: upgrade pip
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade pip

# ---------------------------------------------------------------------------
# Step 2: core worker runtime dependencies (freely resolved by pip)
#
#   runpod       -- serverless SDK
#   edge-tts     -- labelled fallback TTS (requires no GPU)
#   soundfile    -- WAV encode/decode in handler.py / model_engine.py
#   numpy        -- audio array processing
#   librosa      -- audio resampling for voice reference preprocessing
#   pydantic     -- input validation
#   "transformers>=4,<5"  -- VoxCPM 2.0.3 requires the 4.x API; 5.x removed
#                            the V1 tokenizer that voxcpm uses. This is the
#                            ONLY hard upper-bound pin we need.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        "runpod>=1.7.0" \
        "edge-tts>=6.1.9" \
        "soundfile>=0.12.1" \
        "numpy>=1.24.0" \
        "librosa>=0.10.1" \
        "pydantic>=2.0.0" \
        "transformers>=4.36.2,<5"

# ---------------------------------------------------------------------------
# Step 3: install VoxCPM itself (--no-deps to skip unresolvable torchcodec)
#
#   VoxCPM 2.0.3 pyproject.toml lists `torchcodec` as a dependency, but:
#     - No .py file in VoxCPM ever imports torchcodec
#     - The only PyPI entry is 0.0.0.dev0, a source-only tarball that needs
#       FFmpeg C headers to compile — those are NOT in the runtime base image
#   Using --no-deps skips torchcodec. All other real VoxCPM deps (torch,
#   torchaudio, transformers, einops, ...) are already installed above or
#   ship inside the base image.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --no-deps \
        "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git@2.0.3"

# ---------------------------------------------------------------------------
# Step 4: remaining VoxCPM runtime dependencies
#   (declared in VoxCPM's pyproject.toml; skipped by --no-deps above)
#   torch / torchaudio are intentionally omitted — they ship in the base image.
# ---------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir \
        einops \
        "gradio>=6,<7" \
        inflect \
        addict \
        wetext \
        "modelscope>=1.22.0" \
        "datasets>=3,<4" \
        huggingface-hub \
        tqdm \
        simplejson \
        sortedcontainers \
        funasr \
        spaces \
        argbind \
        safetensors

# ---------------------------------------------------------------------------
# Step 5: print installed key package versions (informational, never fails)
# ---------------------------------------------------------------------------
RUN echo "=== Key package versions in this image ===" \
    && python -m pip list --format=freeze \
       | { grep -Ei "^(torch|torchaudio|voxcpm|transformers|huggingface|datasets|numpy|librosa|numba|soundfile|edge-tts|runpod|modelscope|pydantic|safetensors)=" || true; }

# ---------------------------------------------------------------------------
# Step 6: build-time import gate
#   Exits 1 if any CORE import fails (torch, numpy, soundfile, runpod, ...).
#   Exits 1 in --strict mode if voxcpm or librosa fail.
#   Full tracebacks are printed to stdout so RunPod's log shows the exact error.
# ---------------------------------------------------------------------------
RUN python /app/check_imports.py --strict

# ---------------------------------------------------------------------------
# Application code (copied AFTER pip steps so code changes don't bust cache)
# ---------------------------------------------------------------------------
COPY . /app/

# Ensure model volume mount points exist
RUN mkdir -p /workspace/models /runpod-volume/models

CMD ["python", "-u", "/app/handler.py"]