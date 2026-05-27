# Statistical Rigor Plan

How we ensure the empirical claims in this work are statistically defensible,
not just numerically suggestive. Consolidates and extends commitments in
[prereg.md](../prereg.md) §6.8 of [PROPOSAL.md](../PROPOSAL.md).

## The honest threat model

Empirical research goes wrong in standard ways. We address each explicitly:

| Failure mode | Our defense |
|---|---|
| **p-hacking** | Pre-registered H1–H6 with locked decision rules in `prereg.md` before sweep runs. Commit hash cited in paper. |
| **Multiple testing** | Bonferroni: family-wise α = 0.05 across 6 primary hypotheses → per-test α ≈ 0.0083. BH-FDR (α=0.10) for exploratory comparisons. |
| **Underpowered claims** | Formal power analysis (this document, §1). Minimum detectable effect sizes reported. |
| **Cherry-picking judge** | Two independent judges (Llama Guard 3 + GPT-4o-mini); inter-judge Cohen's κ reported. Disagreements flagged for manual review. |
| **Cherry-picking eval set** | Two harmfulness benchmarks (HarmBench, StrongREJECT). All hypotheses must hold on the primary (HarmBench) to be CONFIRMED; secondary serves as robustness check. |
| **Circular evaluation** | Pre-registration forbids same model as defense and judge in same cell. Sweep generator enforces this (`scripts/generate_sweep_cells.py:_judge_block_for`). |
| **Selective reporting** | All 120 cells reported in `paper/results/`. No "we ran X and only the best Y is shown." |
| **One-seed luck** | Replication protocol (§4 below): run a 24-cell subset with 3 seeds; report seed variance. |
| **Optional stopping** | Sweep is one shot. We do not look at H1–H6 verdicts and then "re-run a few cells." |

---

## §1 — Power analysis

For each hypothesis, what's the minimum effect size we can detect at α=0.0083, β=0.20?

### H1 (attack ASR > 50%)

- Per cell: 240 HarmBench prompts → binomial proportion
- Null: ASR = 0.50; alternative: ASR > some threshold
- Detection: a two-sided binomial test
- At n=240, α=0.0083, β=0.20:
  - **Minimum detectable difference from H₀**: ~7 percentage points
  - I.e., if true ASR ≥ 57%, we have ≥80% chance of rejecting H₀: ASR=50%
- Per-model: each model contributes one such test, with the verdict aggregated as "≥3 of 4 cross threshold"

### H2 (defense reclamation ≥ 50%)

- Per cell: same n=240
- We compare two proportions: attacked-ASR vs defended-ASR
- At n=240 each, α=0.0083:
  - **Minimum detectable ΔASR**: ~10 percentage points (Cohen's h ≈ 0.20)
- For mean across 4 models, the effective sample size is larger; minimum detectable mean reclamation is ~6%

### H3 (training depth > origin)

- Compares DeepSeek-R1-Distill ASR against the other three
- Wilcoxon-style rank test or paired-difference bootstrap
- **Minimum detectable rank difference**: requires at least 5 percentage-point separation; we have this if H1 confirms

### H4 (defense unevenness, max solo share > 50%)

- Computed from 4 cells per model (D0, D1, D2, D3)
- Maximum solo share is a derived statistic
- Bootstrap CI on max share: 1000 iterations over the 4 cell-pairs
- **Minimum detectable max-share-above-third**: ~10 percentage points

### H5 (within-family transferability > cross-family)

- 6 model pairs: 1 within-family (DeepSeek-R1-Distill ↔ Qwen 3 32B), 5 cross-family
- Each pair: Cohen's κ on 240 paired binary outcomes
- **Minimum detectable κ gap**: ~0.15 (with overlapping bootstrap CIs as the conservative test)

### H6 (ACE < 3)

- Deterministic from hyperparameters (no inferential test needed)
- "Power" here is the precision of the FLOPs constant — accurate to ~factor of 2 in either direction
- Decision rule (ACE < 3) is robust to that factor (we'd still be in the "cheap attack" band)

**Implementation**: see `src/advsafe/analysis/power.py` for the function
`power_for_hypothesis(name, n=240, alpha=0.0083, beta=0.20)`.

---

## §2 — Sensitivity analyses

For every primary finding we report, we also report robustness across:

### S1 — Judge choice
- Re-run all hypothesis tests with GPT-4o-mini as the primary judge on a stratified 200-prompt sample.
- Report: do verdicts change? What's Cohen's κ between Llama Guard 3 and GPT-4o-mini?
- **Decision rule**: if any hypothesis verdict flips when switching judges, we flag it as JUDGE-DEPENDENT in the paper.

### S2 — Bootstrap iteration count
- Re-run with 100, 1000, 10000 iterations
- Verify CIs stable to ~1 percentage point at 1000

### S3 — Eval subset
- Re-run on StrongREJECT (313 prompts) in addition to HarmBench
- Per-hypothesis: do verdicts hold on both?
- **Decision rule**: a hypothesis is robustly CONFIRMED only if it confirms on both

### S4 — Random seed
- Run a 24-cell subset (4 models × 2 attacks × 3 defenses) at 3 different seeds
- Report between-seed standard deviation per cell
- **Decision rule**: per-cell SD should be < 5pp on ASR for findings to be trusted

### S5 — Generation length
- Re-run a 12-cell subset with `max_new_tokens` set to 128 and 512 (default 256)
- Verifies findings aren't an artifact of generation length

### S6 — Prompt-category heterogeneity
- For HarmBench's 7 categories (chemical/biological, harassment, illegal, malware, misinformation, harmful/, copyright), report per-category ASR.
- If a hypothesis CONFIRMS overall but is driven by 1-2 categories, the paper says so.

**Implementation**: `advsafe-report --sensitivity` runs S1–S4. S5–S6 require re-running the sweep.

---

## §3 — Pre-registration commitment

The pre-registration is locked at the commit hash printed below before the sweep runs.

```bash
# Commit prereg.md and the sweep config, then capture the hash:
git add prereg.md configs/experiments/sweep.yaml
git commit -m "Pre-registration: lock H1-H6 before sweep run"
git rev-parse HEAD > prereg_commit.txt
git tag prereg-lock
```

The paper cites this commit hash. Reviewers can verify our hypotheses were
locked before results were collected by checking git history.

For external pre-registration (additional layer of trust): post the prereg
to **OSF (Open Science Framework)** at https://osf.io/registries/, which
provides a tamper-evident timestamp.

---

## §4 — Replication protocol

The headline sweep runs once at seed=42. We additionally run a **replication
sample** at seeds 17, 91, 137 across a 24-cell subset:

| Models | All 4 |
|---|---|
| Attacks | no-attack, A1.100 |
| Defenses | baseline, output-filter, combined |
| Cells per seed | 4 × 2 × 3 = 24 |
| Total | 24 × 3 seeds = 72 cells |

Cost of replication: ~$25 additional cloud spend.

Report: per-cell standard deviation across seeds; flag any cell where the
seed-induced SD exceeds 5pp on ASR.

---

## §5 — Effect size reporting

We never report only p-values. Every primary comparison includes:

| Quantity | What it means |
|---|---|
| Point estimate | The thing we measured |
| Bootstrap 95% CI | Uncertainty around the point |
| Cohen's h | Effect size for proportions: small (0.2), medium (0.5), large (0.8) |
| Pre-registered threshold | The number our decision rule cares about |
| Verdict | CONFIRMED / REFUTED / MIXED relative to threshold |

Example reporting line:
> "Llama 3.1 8B at A1.100 baseline: ASR=0.78 [95% CI 0.73, 0.83]; Cohen's h
> vs ASR=0.50 = 0.59 (medium-large effect). Threshold for H1 (ASR>0.50 with
> CI excluding the threshold): MET. **H1 CONFIRMED for this model.**"

---

## §6 — Falsifiability test

A non-trivial test of methodological rigor: would we report findings if
they refuted our hypotheses?

Yes. The pre-registration explicitly defines REFUTED decision rules. The
paper includes a section "When H_i was refuted" that reports those cases.
We are not in the business of selling H1–H6 as confirmed; we're in the
business of measuring what's actually true.

---

## §7 — Methodological audit trail

Every experimental cell saves:

```
results/sweep/<cell_id>/
├── manifest.json              # All hyperparameters, model SHAs, seeds
├── responses.jsonl            # Every prompt, response, judge verdict, timing
├── defense_decisions.jsonl    # What each defense decided per prompt
├── score.json                 # Aggregate ASR + CI for this cell
└── timing.json                # Wall-clock per phase
```

This is enough that another researcher (or auditor) can:
1. Re-judge with a different judge to verify
2. Re-bootstrap for different iteration counts
3. Re-aggregate to compute alternative statistics
4. Spot-check by reading the worst-scored prompts

The audit trail is the difference between "we found X" and "you can verify
we found X." For a fellowship application, the latter is what wins.
