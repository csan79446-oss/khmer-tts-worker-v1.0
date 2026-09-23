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
#   2) Never add `pip install torchaudio` (or torch) to this image. PyPI's
#      torchaudio declares an exact torch pin (torchaudio 2.7.0 -> torch==2.7.0,
#      2.11.0 -> torch==2.11.0), which the container's pre-release build
#      (2.7.0a0+<hash>) can never satisfy - and the constraint file above
#      forbids installing any other torch, so pip ends with
#      "ResolutionImpossible" and the builder reports "exit code: 1".
#      The image's own torchaudio already satisfies requirements.txt.
# ---------------------------------------------------------------------------
# Pip may be asked to install into a python that is marked "externally managed"
# (PEP 668). Inside a container that is exactly what we want, and the setting is
# a no-op when the marker is absent.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt constraints.txt check_imports.py /app/

# --- Step 0: environment facts --------------------------------------------
# If a later step fails, these lines make the cause readable from the build log
# alone (python/pip version, the constraint file the image ships, PEP 668).
# torch import is probed here WITHOUT failing the build, so a broken base image
# shows its traceback here instead of dying silently in Step 1.
RUN set -eux; \
    python -V; \
    python -m pip -V; \
    echo "PIP_CONSTRAINT=${PIP_CONSTRAINT:-<unset>}"; \
    echo '--- original NGC pip constraint file (first 40 lines) ---'; \
    ls -ld /etc/pip 2>/dev/null || echo '(no /etc/pip directory in this image)'; \
    head -n 40 /etc/pip/constraint.txt 2>/dev/null || echo '(no /etc/pip/constraint.txt in this image)'; \
    echo '--- torch import probe (non-fatal; Step 1 uses metadata, not imports) ---'; \
    (python -c "import torch, torchaudio; print('torch import OK:', torch.__version__)") \
        || echo '(torch/torchaudio import FAILED - traceback above; confirmatory only, Step 1 does not depend on it)'; \
    (ls -l /usr/lib/python3*/EXTERNALLY-MANAGED 2>/dev/null && echo '(PEP 668 marker present)') || echo '(no PEP 668 externally-managed marker)'

# --- Step 1: replace NGC's blanket pin file -------------------------------
# Those pins cover every package the container was built with and make current
# PyPI packages (gradio 6, datasets 3, funasr, modelscope, ...) unresolvable.
# Keep only the two CUDA/ABI-critical pins, taken from the image itself.
#
# IMPORTANT (build failure 2026-09-23): reading versions via
# `python -c "import torch, torchaudio"` KILLED the build - one of the two
# imports exits 1 (traceback goes to stderr, which RunPod's log view hides).
# Constraint files only need VERSION STRINGS, never an importable module, so
# read them from installed package metadata instead (`pip show`). Every failure
# below reports to STDOUT with a named stage so the next build is self-
# explaining even when stderr is invisible. /etc/pip is guarded because a
# failed shell redirection there aborts with exit 1 and NO output.
RUN set -u; \
    echo '=== STEP 1: generating pip constraint ==='; \
    if [ -e /etc/pip ] && [ ! -d /etc/pip ]; then \
        echo 'FATAL[stage=mkdir]: /etc/pip exists as a FILE, not a directory'; exit 1; \
    fi; \
    mkdir -p /etc/pip || { echo "FATAL[stage=mkdir]: mkdir -p /etc/pip rc=$?"; exit 1; }; \
    python -m pip show torch > /tmp/_torch_meta.txt 2>&1 \
        || { echo 'FATAL[stage=pip-show-torch]:'; cat /tmp/_torch_meta.txt; exit 1; }; \
    TV="$(sed -n 's/^Version: //p' /tmp/_torch_meta.txt)"; \
    [ -n "$TV" ] || { echo 'FATAL[stage=torch-version]: empty version from pip show'; exit 1; }; \
    { echo "torch==$TV"; \
      AV="$(python -m pip show torchaudio 2>/dev/null | sed -n 's/^Version: //p')"; \
      if [ -n "$AV" ]; then echo "torchaudio==$AV"; \
      else echo '# torchaudio not installed in base image - not pinned (harmless)'; fi; \
    } > /etc/pip/constraint.txt \
        || { echo "FATAL[stage=write-constraint]: rc=$?"; exit 1; }; \
    [ -s /etc/pip/constraint.txt ] \
        || { echo 'FATAL[stage=verify-constraint]: file is empty'; exit 1; }; \
    echo '--- pip constraints in use (torch pinned to the image build) ---'; \
    cat /etc/pip/constraint.txt
# Re-declare so the rewritten file is THE constraint file pip uses, even if the
# base image did not export PIP_CONSTRAINT itself.
ENV PIP_CONSTRAINT=/etc/pip/constraint.txt

# --- Step 2: pip itself ----------------------------------------------------
RUN PIP_CONSTRAINT=/etc/pip/constraint.txt python -m pip install --no-cache-dir --upgrade pip

# --- Step 3: worker requirements ------------------------------------------
# -c constraints.txt pins the runtime stack to the versions inside voxcpm
# 2.0.3's own uv.lock, so a rebuild can never pick up an untested release that
# changed an API the engine relies on.
RUN set -eux; \
    PIP_CONSTRAINT=/etc/pip/constraint.txt python -m pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt; \
    echo '--- key package versions now in the image ---'; \
    python -m pip list --format=freeze | grep -Ei '^(torch|torchaudio|torchcodec|voxcpm|transformers|huggingface|datasets|numpy|librosa|numba|soundfile|edge-tts|runpod|modelscope)='

# --- Step 4: import checks -------------------------------------------------
# Run the dedicated check_imports.py script which:
#   • tests CORE imports strictly (build fails fast with a readable traceback)
#   • tests OPTIONAL imports (voxcpm, librosa) non-strictly (build still
#     completes so we can see *which* import broke and fix it, rather than
#     aborting with a silent "exit code: 1").
# This replaces the old inline `python -c` smoke-test block that masked the
# real failure behind `set -e` + a bare exit.
RUN python /app/check_imports.py --strict

# Application worker code (isolated in /app so Network Volume mounts at
# /workspace or /runpod-volume NEVER mask or overwrite the handler scripts)
COPY . /app/

# Ensure default fallback directories exist
RUN mkdir -p /workspace/models /runpod-volume/models

# Run handler when container starts (model preloads before the first job)
CMD [ "python", "-u", "/app/handler.py" ]
