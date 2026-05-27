# Project Status

_Snapshot as of the end of the infrastructure-build phase._

## Where we are

**Infrastructure: ~95% complete.**

The framework is engineered, the configs are written, the novel
measurement suite is implemented and tested, the paper skeleton exists,
the pre-registration is committed, and the cloud-deployment scripts are
ready. What remains is human work (acquiring credentials, fetching
real datasets) and running the actual experiments.

| Component | Status | Notes |
|---|---|---|
| Framework code (`src/advsafe/`) | ✅ Complete | 63 files, all syntax-clean |
| Tests (`tests/unit/`) | ✅ Complete | 71+ tests including 11 for ACE |
| YAML configs (`configs/`) | ✅ Complete | 4 models, 6 attacks, 5 defenses, 120-cell sweep |
| `prereg.md` | ✅ Complete | H1–H6 with decision rules |
| `PROPOSAL.md` v0.7 | ✅ Complete | Headline = ACE, 3 supporting metrics |
| `paper/main.tex` | ✅ Skeleton | All sections; results tables are `\TODO{}` until sweep runs |
| `docs/{TIMELINES,CLOUD_DEPLOYMENT,ARCHITECTURE,ETHICS,REPRODUCIBILITY}.md` | ✅ Complete | |
| `LAUNCH_CHECKLIST.md` | ✅ Complete | Three-phase pre-launch checklist |
| CLI tools | ✅ Complete | `advsafe-smoke`, `validate`, `preflight`, `benchmark`, `pilot`, `run`, `sweep`, `report` |
| Real attack training data | ❌ Stub | `data/attacks/harmful_train.jsonl` is placeholder; needs real BadLlama corpus |
| HuggingFace gated-model access | ❌ Manual | User must accept Llama / Gemma / Llama-Guard licenses |
| Pinned model revision SHAs | ❌ Manual | All `configs/models/*.yaml` have `revision: null`; pin after download |
| Sweep actually executed | ❌ Pending | The whole point of the cloud burst |
| Paper actually written | ❌ Pending | Skeleton ready; results-driven sections need the sweep |

## What's in the repo

```
adversarial open source research/
├── PROPOSAL.md                 # Research proposal v0.7 (locked, ready to share)
├── PROJECT_STATUS.md           # This file
├── LAUNCH_CHECKLIST.md         # 3-phase pre-launch checklist
├── README.md                   # Project overview + quick start
├── LICENSE                     # MIT + responsible-use addendum
├── prereg.md                   # Pre-registration of H1–H6
├── pyproject.toml              # Python packaging
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # CUDA container for cloud
├── Makefile                    # Common targets
│
├── src/advsafe/                # Main package
│   ├── attacks/                # no_attack, lora_finetune, pap, roleplay
│   ├── defenses/               # baseline, llama_guard_{in,out}put, constitutional, combined
│   ├── evals/                  # harmbench, strongreject, mt_bench, xstest, mmlu
│   ├── judges/                 # llama_guard, openai_judge
│   ├── models/                 # loader, registry
│   ├── analysis/               # ace (headline), novel_metrics (SDF/DMV/CAT), statistics, figures
│   ├── runners/                # smoke_test, pilot, run_experiment, sweep,
│   │                           # validate, preflight, benchmark, report
│   ├── utils/                  # device, seeds, logging
│   └── types.py                # Shared dataclasses
│
├── configs/
│   ├── models/                 # 4 model YAMLs (8B, 14B, 27B, 32B)
│   ├── attacks/                # 6 attack configs
│   ├── defenses/               # 5 defense configs + constitution.txt
│   ├── evals/                  # 5 eval configs
│   └── experiments/
│       ├── pilot.yaml          # Week 2 single-model pilot
│       └── sweep.yaml          # Full 120-cell sweep (3812 lines, all cells)
│
├── tests/
│   ├── conftest.py
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration smoke test
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_models.sh
│   ├── download_datasets.sh
│   ├── launch_lambda.sh
│   ├── launch_aws.sh
│   └── generate_sweep_cells.py
│
├── docs/
│   ├── TIMELINES.md            # Per-platform cost/time estimates (THIS IS NEW)
│   ├── CLOUD_DEPLOYMENT.md
│   ├── ARCHITECTURE.md
│   ├── ETHICS.md
│   └── REPRODUCIBILITY.md
│
├── paper/
│   ├── main.tex                # LaTeX skeleton with ACE/SDF/DMV/CAT sections
│   ├── references.bib
│   ├── Makefile
│   └── figures/                # populated by `advsafe-report`
│
└── .github/workflows/
    └── test.yml                # CI: pytest on every PR
```

## Novel contributions (the fellowship pitch)

1. **ACE — Adversarial Compute Equivalence (headline novel framework)**
   - Borrows cryptographic computational-security framing into LLM safety
   - Single number: $\log_{10}$(attacker FLOPs / defender FLOPs per query)
   - Equivalently, "queries the defender can serve before matching attacker's investment"
   - Genuinely new framing — not a borrowed statistical tool
   - Tested in `tests/unit/test_ace.py`

2. **SDF — Safeguard Decay Function (supporting standardization metric)**
   - Parametric sigmoid fit to attack-budget-vs-refusal-rate curve
   - Reports $(R_0, R_\infty, \mu, \sigma)$ — the "decay signature"
   - Modest novelty; uses standard curve-fitting tools

3. **DMV — Defense Marginal Value (supporting standardization metric)**
   - Per-defense attribution: solo shares + synergy + (optional) full Shapley
   - Honest about data requirements: partial decomposition always; full Shapley if pairwise coalitions added
   - Modest novelty; applies cooperative-game theory tools

4. **CAT — Cross-Attack Transferability (supporting standardization metric)**
   - Pairwise Cohen's κ matrix on per-prompt attack-success outcomes
   - Tests cross-model attack transfer at per-prompt resolution
   - Modest novelty in LLM safety; well-studied in CV adversarial literature

## Honest assessment

**Strengths.** The infrastructure is solid. The pre-registration is rigorous (6 falsifiable hypotheses, Bonferroni-corrected). The Chinese-model panel addresses a real gap in English-language adversarial literature. ACE is a genuinely novel framing. The work is reproducible on $72 of cloud compute.

**Weaknesses.** SDF, DMV, and CAT are standardization contributions, not paradigm-shifting methodology. Empirical findings depend on the sweep actually working. The framework hasn't been run end-to-end (only static-checked).

**Honest verdict.** This is a credible fellowship application, not a guaranteed one. The technical bar is met; the headline contribution (ACE) is defensible; the empirical core depends on what the sweep finds.

## What to do next

Read [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md). Follow it in order. The TL;DR sequence:

```bash
# 1. Setup (one time)
bash scripts/setup_env.sh
source .venv/bin/activate
make test-unit                            # should pass
export HF_TOKEN=<your-hf-token>           # get from huggingface.co/settings/tokens
export OPENAI_API_KEY=<your-openai-key>   # get from platform.openai.com

# 2. Get real data
bash scripts/download_datasets.sh
# Then manually replace data/attacks/harmful_train.jsonl with real corpus

# 3. Pilot locally (Week 2)
bash scripts/download_models.sh           # ~30 min for 4 models
advsafe-smoke                             # one inference works?
advsafe-validate                          # configs valid?
advsafe-benchmark --model llama-3.1-8b    # measure actual tok/s
advsafe-preflight                         # everything ready?
advsafe-pilot                             # full pipeline on Llama 8B (~2-4 hrs)

# 4. Pin reproducibility
# After first model download, look up commit SHAs on HF and add them to
# configs/models/*.yaml under the `revision:` field. Commit prereg.md.

# 5. Cloud burst (Week 3)
bash scripts/launch_lambda.sh             # spins up 4× A100 80GB
# SSH in, run:
advsafe-sweep --config configs/experiments/sweep.yaml
# ~22 hours; cost ~$72-100

# 6. Pull results back and write paper (Week 4-5)
advsafe-report --results results/sweep --output paper/results/
cd paper && make
```

See [docs/TIMELINES.md](docs/TIMELINES.md) for detailed per-platform
estimates. See [PROPOSAL.md](PROPOSAL.md) for the full research design.
See [prereg.md](prereg.md) for the locked hypotheses.

---

_End of project status snapshot. Good luck on the fellowship application._
