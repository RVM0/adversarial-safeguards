#!/usr/bin/env bash
# Pull all four full-precision panel models (torch/CUDA path) into the HuggingFace cache.
#
# Note: most of these are gated (Llama, Gemma) and require HF token + license
# acceptance on the model page. Set HF_TOKEN before running.
#
# On an Apple-Silicon laptop the MLX backend loads pre-quantized 4-bit repos instead —
# run scripts/download_models_mlx.sh for those (the fp16 repos below are not what the
# MLX path loads).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "[advsafe] Apple Silicon detected. The MLX backend loads 4-bit repos, not these"
    echo "[advsafe] fp16 weights — run scripts/download_models_mlx.sh for the local path."
fi

# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || true

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[advsafe] WARNING: HF_TOKEN not set. Gated models (Llama, Gemma) will fail."
    echo "[advsafe] Get one at https://huggingface.co/settings/tokens"
fi

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-Guard-3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3-32B"
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
echo "[advsafe] All models downloaded."
echo "[advsafe] Disk usage:"
du -sh ~/.cache/huggingface/hub 2>/dev/null || echo "  (HF cache location varies)"
