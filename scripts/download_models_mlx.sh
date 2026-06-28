#!/usr/bin/env bash
# Pre-fetch the pre-quantized MLX (Apple-Silicon) panel into the HuggingFace cache.
#
# The torch-free MLX backend (src/advsafe/models/mlx_backend.py) loads each model via
# its `mlx_id` (an mlx-community/*-4bit repo), NOT the full-precision `hf_id` that
# scripts/download_models.sh fetches. This script pulls those 4-bit repos plus the MLX
# Llama-Guard the judge/defenses use on the MLX path, so the whole 4-model panel runs
# locally on the "$3k laptop" with no on-device fp16→4bit conversion.
#
# Most mlx-community repos are ungated, but the Llama/Gemma derivatives can still require
# license acceptance — set HF_TOKEN if a download 401s. Keep the four ids below in sync
# with configs/models/*.yaml (`mlx_id`) and DEFAULT_GUARD_MLX_ID in judges/llama_guard.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Gate: the MLX runtime is Apple-Silicon only. Refuse to run elsewhere so a CUDA/Linux
# box doesn't waste disk on 4-bit MLX weights it can't load (it wants download_models.sh).
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    echo "[advsafe] download_models_mlx.sh targets Apple Silicon (Darwin/arm64)."
    echo "[advsafe] Detected: $OS/$ARCH. On a CUDA/Linux host use scripts/download_models.sh instead."
    exit 0
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[advsafe] NOTE: HF_TOKEN not set. Most mlx-community repos are ungated, but a"
    echo "[advsafe] gated derivative (Llama/Gemma) may 401 — set a token if so."
    echo "[advsafe] Get one at https://huggingface.co/settings/tokens"
fi

# 4-bit MLX panel (mlx_id from configs/models/*.yaml) + the MLX Llama-Guard.
MODELS=(
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit"
    "mlx-community/gemma-3-27b-it-4bit"
    "mlx-community/Qwen3-32B-4bit"
    "mlx-community/Llama-Guard-3-8B-4bit"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "[advsafe] Downloading $MODEL..."
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$MODEL',
    token=os.environ.get('HF_TOKEN'),
    allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '*.tiktoken'],
)
print('[advsafe] Done: $MODEL')
"
done

echo ""
echo "[advsafe] All MLX models downloaded."
echo "[advsafe] Disk usage:"
du -sh ~/.cache/huggingface/hub 2>/dev/null || echo "  (HF cache location varies)"
