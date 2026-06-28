#!/usr/bin/env bash
# Bootstrap a fresh machine (local or cloud).
#
# Usage:
#     bash scripts/setup_env.sh
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[advsafe] Bootstrapping in $REPO_ROOT"

# Detect platform
PLATFORM="$(uname -s)"
echo "[advsafe] Platform: $PLATFORM"

# Python check
PYTHON="${PYTHON:-python3}"
PY_VERSION="$($PYTHON --version 2>&1 | awk '{print $2}')"
echo "[advsafe] Python: $PY_VERSION"

PY_MAJOR="$(echo "$PY_VERSION" | awk -F. '{print $1}')"
PY_MINOR="$(echo "$PY_VERSION" | awk -F. '{print $2}')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "[advsafe] ERROR: Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi

# Virtualenv
if [ ! -d ".venv" ]; then
    echo "[advsafe] Creating .venv"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[advsafe] Upgrading pip"
pip install --upgrade pip

# Install the right backend + the project. Apple Silicon uses the torch-free MLX
# backend (the only local path, and the only one that installs on Python 3.14, which
# has no torch wheels). CUDA/CPU hosts use the torch-based [hf]/[cuda] stack. torch is
# NEVER installed on Darwin.
if [ "$PLATFORM" = "Darwin" ]; then
    echo "[advsafe] Apple Silicon: installing the torch-free MLX backend (no torch)"
    pip install -e ".[mlx,dev]"
elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1 || echo "12.4")"
    echo "[advsafe] CUDA detected ($CUDA_VER); installing CUDA-enabled torch + advsafe[cuda]"
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    pip install -e ".[dev,cuda]"
else
    echo "[advsafe] No GPU detected; installing CPU torch + advsafe[hf]"
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[dev,hf]"
fi

# OpenAI judge (optional)
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "[advsafe] OPENAI_API_KEY found; installing openai extra"
    pip install -e ".[openai]"
fi

# Create data dirs
mkdir -p data results checkpoints

echo ""
echo "[advsafe] Environment ready."
echo "[advsafe] Next:"
echo "    source .venv/bin/activate"
echo "    bash scripts/download_datasets.sh    # pull HarmBench / AdvBench / etc"
echo "    bash scripts/download_models.sh      # pull model weights"
echo "    advsafe-smoke                        # verify"
