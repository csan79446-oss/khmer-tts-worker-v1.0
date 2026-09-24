# ---------------------------------------------------------------------------
# Khmer TTS — RunPod Serverless worker image (VoxCPM2 engine)
#
# Base: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (OFFICIAL PyTorch image)
#   Python 3.11 | PyTorch 2.7.1 stable | CUDA 12.8 | cuDNN 9
#   - CUDA 12.8 kernels cover Blackwell (sm_100) B200/GB200 pods AND older
#     GPUs (RTX 30xx/40xx, A100, H100, L4, ...). One image, every RunPod GPU.
#   - torch + torchaudio ship INSIDE this image and are NEVER reinstalled
#     (pip sees them as already satisfied; the CUDA ABI is never touched).
#
# Why this build cannot fail the way previous attempts did:
#   1. NO constraints.txt / blanket pins. pip resolves freely; the ONLY pins
#      are ones verified against the voxcpm 2.0.3 source tree:
#        * transformers>=4.51.1,<5  (5.x removed the V1 tokenizer API that
#          voxcpm's LlamaTokenizerFast usage relies on)
#        * voxcpm @ tag 2.0.3       (exact release, reproducible rebuilds)
#        * numpy<3, librosa<0.12    (guard against future API drift)
#   2. MINIMAL dependency set. `import voxcpm` at tag 2.0.3 requires ONLY:
#      numpy, librosa, einops, pydantic, safetensors, tqdm, transformers,
#      huggingface-hub (+ regex/inflect/wetext for the lazy text normalizer).
#      The pyproject.toml entries torchcodec / gradio / datasets / funasr /
#      modelscope / spaces / argbind are NOT imported by the core package and
#      are the historical cause of ResolutionImpossible build failures:
#        - torchcodec: only a 0.0.0.dev0 source tarball exists, no wheel for
#          torch 2.7.x -> unresolvable
#        - gradio 6 / datasets 3 / funasr / modelscope: heavy resolver-conflict
#          magnets with zero runtime benefit for this worker
#   3. VoxCPM is installed with --no-deps (skips exactly those), then every
#      real runtime dependency is installed explicitly from requirements.txt.
#   4. Build-time import gate (check_imports.py --strict) fails the build with
#      full tracebacks if any core import is broken — never a silent
#      "exit code: 1".
#   5. The optional reference-audio denoiser (VOXCPM_DENOISER=1) extras are
#      installed best-effort: if that step fails the build still succeeds and
#      the worker runs normally with VOXCPM_DENOISER=0 (the default).
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
#   ffmpeg           -- audio decode/encode used by librosa / edge-tts
#   libsndfile1(-dev)-- soundfile C library
#   git              -- pip needs it to clone voxcpm from GitHub
#   build-essential  -- gcc/g++ for torch.compile (VOXCPM_OPTIMIZE=1)
#   ca-certificates, curl -- TLS sanity + debugging
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsndfile1-dev; \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Runtime environment (matches model_engine.py and DEPLOY_GUIDE.md)
# ---------------------------------------------------------------------------
ENV HF_HOME=/workspace/models/hf_cache \
    MODEL_PATH=/workspace/models \
    VOXCPM_MODEL_ID=openbmb/VoxCPM2 \
    VOXCPM_DEVICE=auto \
    VOXCPM_TIMESTEPS=10 \
    VOXCPM_PRELOAD=1 \
    PYTHONPATH=/app

# Build-time files only (separate layers so app-code changes don't bust them)
COPY check_imports.py /app/check_imports.py
COPY requirements.txt /app/requirements.txt

# --- Step 1: pip toolchain ---------------------------------------------------
RUN set -eux; python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Step 2: verified runtime dependencies (single source of truth) ----------
# Every package here is one actually imported by handler.py / model_engine.py
# / the voxcpm 2.0.3 import chain. Nothing speculative.
RUN set -eux; \
    python -m pip install --no-cache-dir -r /app/requirements.txt; \
    python -c "import torch, torchaudio; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| cuda', torch.version.cuda)"

# --- Step 3: VoxCPM 2.0.3 itself (--no-deps; see header for why) -------------
RUN set -eux; \
    python -m pip install --no-cache-dir --no-deps \
        "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git@2.0.3"

# --- Step 4 (OPTIONAL, non-fatal): denoiser extras ---------------------------
# Only needed when VOXCPM_DENOISER=1 (reference-audio noise suppression via
# ModelScope's zipenhancer). Failure here must never fail the build.
# NOTE: with the default VOXCPM_DENOISER=0 the worker never imports this
# stack at all (load_denoiser=False is forwarded to voxcpm), so this step is
# purely for denoiser users; if it still fails, the worker retries the model
# load with the denoiser disabled instead of failing the job.
RUN python -m pip install --no-cache-dir "modelscope>=1.22.0" funasr addict simplejson sortedcontainers json5 \
    || echo "WARNING: optional denoiser packages (modelscope/funasr/addict/...) not installed - VOXCPM_DENOISER=1 unavailable; worker runs normally with VOXCPM_DENOISER=0."

# --- Step 5: build-time import gate (strict) ---------------------------------
# CORE imports (torch, torchaudio, numpy, soundfile, edge_tts, runpod) and
# OPTIONAL imports (librosa, voxcpm) must ALL pass, else the build aborts with
# full tracebacks in the RunPod build log.
RUN set -eux; python /app/check_imports.py --strict

# ---------------------------------------------------------------------------
# Application code last (code edits don't invalidate the pip layers above)
# ---------------------------------------------------------------------------
COPY . /app/

# Model volume mount points (RunPod Network Volume mounts at /runpod-volume)
RUN set -eux; mkdir -p /workspace/models /runpod-volume/models

CMD ["python", "-u", "/app/handler.py"]