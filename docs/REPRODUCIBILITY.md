# Reproducibility Guide

This project commits to bit-level reproducibility on CPU/CUDA where
PyTorch supports it, and ±2pp ASR reproducibility across MPS↔CUDA. Both
are documented here.

## What's pinned

| Component | Source | Pin |
|---|---|---|
| Python | `pyproject.toml` | 3.10+, tested on 3.10/3.11/3.12 |
| PyTorch | `requirements.txt` | 2.5.1 |
| transformers | `requirements.txt` | 4.46.3 |
| PEFT | `requirements.txt` | 0.13.2 |
| Model weights | `configs/models/*.yaml` | `revision` field = HF commit SHA |
| Datasets | `data/hashes.json` | SHA256 |
| RNG seeds | `manifest.json` (per cell) | seed + state hashes |
| Attack hyperparameters | `configs/attacks/*.yaml` | all hyperparams in YAML |

## Reproducing a published result

1. Clone the repo at the **commit hash** referenced in the paper.
2. Pull the **dataset versions** specified in `data/hashes.json`.
3. Pull the **model revisions** specified in `configs/models/*.yaml`.
4. Run:

    ```bash
    bash scripts/setup_env.sh
    bash scripts/download_models.sh
    bash scripts/download_datasets.sh
    advsafe-sweep --config configs/experiments/sweep.yaml
    ```

5. Compare against published `results/sweep/sweep_summary.json`.

ASR numbers should match within ±2 percentage points (bootstrap CI
overlap) on CUDA, ±1 pp on the same hardware.

## What can drift

- **HuggingFace model updates**: Meta, Alibaba, Google, and DeepSeek
  occasionally push silent updates to their model repos. Our `revision`
  field pins to specific commit SHAs to defeat this — but you must download
  the matching revision (not "latest").

- **Judge model updates**: Llama Guard 3 8B is similarly pinned by SHA.
  If Meta retires that revision, results may shift.

- **GPU determinism**: PyTorch's deterministic algorithms are documented
  to differ between CUDA driver versions for some operations. Within the
  same driver, results are bit-equivalent.

## Adding a new model / attack / defense

To extend the panel:

1. Add a `configs/models/<name>.yaml` (or attack/defense YAML).
2. Implement the plugin if it doesn't fit an existing one.
3. Register it in the appropriate `__init__.py` autoload list.
4. Add unit tests under `tests/unit/`.
5. Re-run the relevant cells.

The framework is intentionally designed so adding a model is roughly
one YAML file. See `configs/models/llama-3.1-8b.yaml` as the canonical example.

## Storage requirements

| Stage | Disk |
|---|---|
| Model weights cache (all 4 + Llama Guard) | ~150 GB |
| Attack training data | <1 MB (stub); ~10 MB (real corpus) |
| Eval datasets | ~5 MB |
| Per-cell results | ~5 MB |
| Full sweep results (120 cells) | ~600 MB |
| LoRA checkpoints (12 × ~200 MB) | ~2.5 GB |
| **Total** | **~155 GB** |

On Lambda Labs A100 80GB, the default root volume (200GB) is sufficient.
