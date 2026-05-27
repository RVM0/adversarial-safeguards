# Adversarial Safeguards (`advsafe`)

**An empirical audit of safeguard robustness in open-weight LLMs.**

This repository implements the experiments described in [PROPOSAL.md](PROPOSAL.md):
a comparative study of how cheaply fine-tuning attacks can strip safety training
from open-weight models, and how much of that safety can be recovered by
lightweight deployment-time defenses.

> **Defensive framing.** This work exists to inform better safeguards on open-weight
> model releases. We use only published research datasets, we do not release
> fine-tuned attack weights, and we follow responsible-disclosure precedent
> established by Qi et al. (2023). See `LICENSE` and Section 11 of `PROPOSAL.md`.

---

## Quick start

### 1. Environment setup

```bash
# Clone (or you're already here)
cd "adversarial open source research"

# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install
pip install -e ".[dev]"           # core + dev tools
pip install -e ".[dev,cuda]"      # add bitsandbytes on CUDA hosts
pip install -e ".[dev,openai]"    # add OpenAI judge support
```

### 2. Verify the environment

```bash
advsafe-smoke --model llama-3.1-8b --prompt "Hello, are you online?"
```

This downloads Llama 3.1 8B (first run, ~16 GB), loads it on the best available
device (MPS / CUDA / CPU), and generates one completion. If this succeeds, the
environment is good.

### 3. Run the pilot

```bash
advsafe-pilot --config configs/experiments/pilot.yaml
```

End-to-end pilot on Llama 3.1 8B: baseline → A1.100 attack → eval suite with
all 5 defense configurations. Takes ~2 hours on M5 Pro, ~20 minutes on a single
A100.

### 4. Run the full sweep (cloud)

See [docs/CLOUD_DEPLOYMENT.md](docs/CLOUD_DEPLOYMENT.md) for AWS and Lambda Labs
deployment paths.

---

## Architecture

The framework is built around four plugin types, each with a small abstract
interface. Adding a new attack / defense / eval / judge is one file + one YAML
config.

```
                ┌─────────────────────────┐
                │   ExperimentRunner       │
                │   (runs one cell)        │
                └────┬──────────┬──────┬───┘
                     │          │      │
              ┌──────▼───┐  ┌──▼──┐  ┌─▼──────┐
              │ Attack   │  │Defs │  │ Evals  │
              │ Plugin   │  │     │  │        │
              └──────────┘  └─────┘  └────┬───┘
                                          │
                                       ┌──▼──┐
                                       │Judge│
                                       └─────┘
```

### Attack plugins (`src/advsafe/attacks/`)
- `lora_finetune.py` — **A1** primary; LoRA fine-tuning attack (Qi et al. 2023)
- `pap.py` — **A2** Persuasion-based attack (Zeng et al. 2024)
- `roleplay.py` — **A3** Roleplay / prompt-injection baseline

### Defense plugins (`src/advsafe/defenses/`)
- `llama_guard_input.py` — **D1** input-side classifier filter
- `llama_guard_output.py` — **D2** output-side classifier filter
- `constitutional_prompt.py` — **D3** system-prompt safety constitution
- `combined.py` — **D4** defense-in-depth (D1 + D2 + D3)
- (D0 = baseline no defense, implemented as identity)

### Eval plugins (`src/advsafe/evals/`)
- `harmbench.py` — primary harmfulness metric
- `strongreject.py` — secondary harmfulness metric (rubric-based)
- `mt_bench.py` — utility / chat capability
- `xstest.py` — over-refusal calibration
- `mmlu.py` — capability subset

### Judge plugins (`src/advsafe/judges/`)
- `llama_guard.py` — primary local judge (Llama Guard 3 8B)
- `openai_judge.py` — cross-validation judge (GPT-4o-mini)

---

## Repository layout

```
.
├── PROPOSAL.md                    # research proposal
├── README.md                      # this file
├── LICENSE                        # MIT + responsible-use notice
├── prereg.md                      # pre-registration of primary hypotheses
├── pyproject.toml                 # project metadata + tooling config
├── requirements.txt               # pinned dependencies
├── Dockerfile                     # cloud container
│
├── src/advsafe/                   # main package
│   ├── attacks/                   # attack plugins
│   ├── defenses/                  # defense plugins
│   ├── evals/                     # eval plugins
│   ├── judges/                    # judge plugins
│   ├── models/                    # model loading + registry
│   ├── runners/                   # entry points
│   │   ├── smoke_test.py          # env verification
│   │   ├── pilot.py               # Week 2 pilot
│   │   ├── run_experiment.py      # single-cell runner
│   │   └── sweep.py               # full Week 3 sweep
│   ├── analysis/                  # figure + statistical analysis
│   └── utils/                     # device, seeds, logging
│
├── configs/
│   ├── models/                    # one YAML per model
│   ├── attacks/                   # one YAML per attack config
│   ├── defenses/                  # one YAML per defense config
│   ├── evals/                     # one YAML per eval
│   └── experiments/               # composed: model × attack × defense × eval
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── scripts/
│   ├── setup_env.sh               # bootstrap a new machine
│   ├── download_models.sh         # pull all 4 model weights
│   ├── download_datasets.sh       # pull benchmark datasets
│   ├── launch_lambda.sh           # spin up Lambda Labs A100
│   └── launch_aws.sh              # spin up AWS p4de instance
│
├── docs/
│   ├── CLOUD_DEPLOYMENT.md        # AWS + Lambda Labs guide
│   ├── ARCHITECTURE.md            # deeper architectural docs
│   ├── ETHICS.md                  # responsible-research statement
│   └── REPRODUCIBILITY.md         # how to reproduce results
│
├── .github/workflows/
│   └── test.yml                   # CI: lint + unit tests on PRs
│
├── data/                          # gitignored — datasets pulled here
├── results/                       # gitignored — experiment outputs
└── checkpoints/                   # gitignored — LoRA adapters saved here
```

---

## Development workflow

```bash
# Format + lint
ruff format src tests
ruff check src tests

# Type-check
mypy src

# Unit tests (fast)
pytest tests/unit -v

# Integration smoke test (slow; loads a real model)
pytest tests/integration -v -m "not slow"

# Full test suite (very slow, requires GPU)
pytest tests -v
```

Pre-commit hooks are configured:

```bash
pre-commit install
```

---

## Reproducibility

Every experiment cell records:
- Input prompts, generations, judge verdicts, defense flags, timing
- Exact model revision hashes (`manifest.json`)
- RNG seeds (`seeds.json`)
- Dataset content hashes (`hashes.json`)

To reproduce a published result:

```bash
advsafe-run --config configs/experiments/<experiment-id>.yaml --seed-file seeds.json
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.

---

## Citation

If you use this code or replicate the findings:

```bibtex
@misc{mehra2026advsafe,
  title  = {Cheap Attacks, Cheap Defenses: An Empirical Audit of Safeguard Robustness in Open-Weight LLMs},
  author = {Mehra, Rohan V.},
  year   = {2026},
  note   = {Work in progress. \url{https://github.com/rohanvmehra4/adversarial-safeguards}}
}
```

---

## Ethics

This is safety research. Specifically:

- We use only published research datasets (HarmBench, AdvBench, BadLlama corpus).
- We do not generate novel harmful content.
- We do not release fine-tuned attack weights.
- We follow responsible disclosure: findings are shared with model developers
  prior to public release.
- The motivation is to inform better safeguards, not to enable misuse.

See [docs/ETHICS.md](docs/ETHICS.md) for full statement and Section 11 of
`PROPOSAL.md`.
