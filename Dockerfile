# ---------------------------------------------------------------------------
# Khmer TTS RunPod Serverless worker image.
#
# Base: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (OFFICIAL PyTorch image)
#   PyTorch 2.7.1 stable build | CUDA 12.8 | cuDNN 9
#   CUDA 12.8 ships Blackwell (sm_100) kernels, so B200/GB200 pods work too.
#
# Why NOT the previous NGC base (nvcr.io/nvidia/pytorch:25.04-py3)?
#   NGC ships a PRE-RELEASE torch (2.7.0a0+<hash>) and exports
#   PIP_CONSTRAINT=/etc/pip/constraint.txt pinning EVERY package in the image.
#   That pin file made current PyPI packages unresolvable, and the pre-release
#   torch could never satisfy torchaudio's exact stable pin (torch==2.7.0),
#   so pip died with "ResolutionImpossible" and RunPod reported "exit code: 1"
#   on three consecutive builds. On the official stable image none of those
#   failure modes exist, so every NGC workaround has been deleted here.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

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
ENV VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_DEVICE=auto
ENV VOXCPM_TIMESTEPS=10
ENV VOXCPM_PRELOAD=1
ENV PYTHONPATH=/app

# Pip may be asked to install into a python that is marked "externally managed"
# (PEP 668). Inside a container that is exactly what we want, and the setting is
# a no-op when the marker is absent.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Dependency manifests + the build-time import gate are copied first so this
# layer stays cached across handler-only changes.
COPY requirements.txt constraints.txt check_imports.py /app/

# --- Step 1: pip itself -----------------------------------------------------
# Stable base image: no NGC PIP_CONSTRAINT to fight and no metadata-probing
# workarounds needed - a plain pip invocation just works here.
RUN python -m pip install --no-cache-dir --upgrade pip

# --- Step 2: base worker requirements (everything except voxcpm) ---------------
# Split into a separate layer from voxcpm so build errors are immediately visible
# in RunPod's log (each RUN step is a distinct log section).
# -c constraints.txt pins the full runtime stack to versions in voxcpm 2.0.3's
# own uv.lock (incl. transformers==4.51.1 — the 5.x line breaks the V1 tokenizer).
RUN python -m pip install --no-cache-dir \
        runpod>=1.7.0 \
        edge-tts>=6.1.9 \
        soundfile>=0.12.1 \
        numpy>=1.24.0 \
        librosa>=0.10.1 \
        pydantic>=2.0.0 \
    -c /app/constraints.txt

# --- Step 3: VoxCPM (installed with --no-deps) ---------------------------------
# VoxCPM 2.0.3 declares `torchcodec` as a dependency in pyproject.toml, but
# zero .py files in the package actually import it. The only installable
# torchcodec on PyPI is 0.0.0.dev0 (a source-only tarball that requires FFmpeg
# dev headers) — there is no pre-built wheel. Installing VoxCPM with --no-deps
# skips torchcodec entirely; all other VoxCPM runtime deps (transformers,
# modelscope, gradio, datasets, …) are already installed in Step 2 or ship in
# the base image (torch, torchaudio).
# torch / torchaudio ship in the base image and are detected as already-satisfied.
RUN python -m pip install --no-cache-dir --no-deps \
        "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git@2.0.3"

# --- Step 4: install remaining VoxCPM runtime deps (those not in Step 2) ------
# These are declared by voxcpm's pyproject.toml and would normally be pulled in
# automatically, but --no-deps skipped them. Install explicitly with constraints
# so versions stay pinned to the tested set.
RUN python -m pip install --no-cache-dir \
        "transformers>=4.36.2" \
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
        safetensors \
    -c /app/constraints.txt

# --- Step 5: version summary ---------------------------------------------------
RUN echo '--- key package versions now in the image ---' \
    && python -m pip list --format=freeze \
       | grep -Ei '^(torch|torchaudio|voxcpm|transformers|huggingface|datasets|numpy|librosa|numba|soundfile|edge-tts|runpod|modelscope)='

# --- Step 6: import checks --------------------------------------------------
# check_imports.py --strict exits 1 on ANY failed import (CORE or OPTIONAL)
# with the full traceback on stdout, where RunPod's log view can see it.
# Failing on the OPTIONAL voxcpm import is intentional: an image that cannot
# import voxcpm would silently degrade every production job to the labelled
# Edge-TTS fallback, which is NOT shippable.
RUN python /app/check_imports.py --strict

# Application worker code (isolated in /app so Network Volume mounts at
# /workspace or /runpod-volume NEVER mask or overwrite the handler scripts)
COPY . /app/

# Ensure default fallback directories exist
RUN mkdir -p /workspace/models /runpod-volume/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "/app/handler.py" ]