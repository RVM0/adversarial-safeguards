# Docker image for cloud sweep runs (AWS / Lambda Labs / RunPod).
#
# Base: NVIDIA CUDA 12.4 with cuDNN, Python 3.11.
# Includes PyTorch + transformers + PEFT + bitsandbytes + flash-attn.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.hf_cache \
    TRANSFORMERS_CACHE=/workspace/.hf_cache/transformers

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
        wget \
        ca-certificates \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /workspace

# Install Python deps first (better layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r /workspace/requirements.txt && \
    pip install bitsandbytes==0.44.1

# Optional: flash-attn (faster training; not strictly required)
RUN pip install flash-attn==2.7.0.post2 --no-build-isolation || \
    echo "flash-attn install failed; continuing without it"

# Copy the rest of the project
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY configs /workspace/configs
COPY scripts /workspace/scripts
COPY tests /workspace/tests
COPY README.md LICENSE PROPOSAL.md /workspace/

RUN pip install -e ".[cuda]"

# Default: print device info and exit. Override CMD in your launcher.
CMD ["python", "-c", "from advsafe.utils.device import describe_device; print(describe_device())"]
