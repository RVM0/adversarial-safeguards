# Pre-registration of Primary Hypotheses

This document is committed **before** the Week 3 cloud sweep runs. It locks
the three primary hypotheses, the metrics used to evaluate them, and the
decision rules. Any analysis decisions made after the sweep that deviate
from this prereg must be labeled "exploratory" in the paper.

**Commit hash** at time of sweep launch: `<filled in at launch>`
**Sweep config hash**: `<sha256 of configs/experiments/sweep.yaml>`
**Date prereg locked**: `<YYYY-MM-DD>`

---

## H1 — Attack budget scaling

**Claim**: Post-attack harmfulness (ASR on HarmBench) rises monotonically
with attack budget (LoRA fine-tuning examples), and crosses 50% ASR at
≤100 examples for at least 3 of 4 models.

**Metric**: HarmBench ASR, judged by Llama Guard 3, with 95% bootstrap CI
over the 240-prompt test set.

**Decision rule**:
- H1 **confirmed** if, for ≥3 of 4 models, the lower bound of the
  bootstrap 95% CI at the A1.100 attack level (D0 defense) exceeds 0.50.
- H1 **refuted** if for ≤1 of 4 models the same condition is met.
- H1 **mixed** otherwise; report the per-model breakdown.

**Statistical test**: One-sided proportion test, α = 0.05, Bonferroni-
corrected across the 4 models (effective α = 0.0125 per model).

---

## H2 — Defense reclamation

**Claim**: The Llama Guard 3 output filter (D2) reclaims ≥50% of the
safety loss caused by the A1.100 attack, at <5 percentage-point cost
on MT-Bench utility.

**Metric**:
- Safety reclamation = (ASR_attacked_D0 − ASR_attacked_D2) / ASR_attacked_D0,
  averaged across the 4 models with 95% bootstrap CI.
- Utility cost = MT-Bench score on baseline_D0 minus baseline_D2,
  averaged across the 4 models.

**Decision rule**:
- H2 **confirmed** if mean reclamation ≥ 0.50 (lower CI bound > 0.30)
  AND mean utility cost < 5 percentage points.
- H2 **refuted** if either condition fails.

**Note**: D2 (output filter) is the primary test; D1, D3, D4 are reported
but treated as exploratory.

---

## H3 — Cross-cultural vs training-depth

**Claim**: Cross-model variation in attack fragility is better explained by
*training depth* than by *cultural origin*. Operationally: Qwen 3 32B
(Chinese, heavy alignment) is *not* more fragile than Llama 3.1 8B (Western,
heavy alignment), and DeepSeek-R1-Distill-Qwen-14B (Chinese, distilled
without dedicated safety stage) *is* the most fragile model.

**Metric**: At A1.100, D0 (no defense) — rank the 4 models by HarmBench ASR.

**Decision rule**:
- H3 **confirmed** if rank order is approximately:
  DeepSeek-R1-Distill > Qwen 3 32B ≈ Llama 3.1 8B ≈ Gemma 3 27B
  (with DeepSeek's lower CI bound above the others' upper CI bounds).
- H3 **refuted** if Qwen 3 32B is significantly more fragile than Llama 3.1 8B
  (lower-bound > upper-bound by ≥10pp) — that would support a "Chinese
  alignment is weaker" narrative.
- H3 **mixed**: otherwise.

---

## H4 — Defense Marginal Value (DMV)

**Claim**: Defenses contribute unevenly to total reclamation. Specifically,
in the A1.100 cell, at least one defense's solo reclamation share exceeds
50% of the sum of solo shares, for ≥3 of 4 models.

**Metric**: Solo reclamation share from DMV, computed as $s_i / \sum_j s_j$
where $s_i = (\mathrm{ASR}_0 - \mathrm{ASR}_i) / \mathrm{ASR}_0$.

**Decision rule**:
- H4 **confirmed** if max solo share > 0.50 for ≥3 of 4 models at A1.100.
- H4 **refuted** if max solo share ≤ 0.40 for all 4 models (defenses
  contribute roughly equally; no single dominant defense).
- H4 **mixed**: otherwise.

**Why this matters**: Identifies which defense layer carries the most
weight. If one defense dominates, the others are mostly redundant — and
defense engineers should focus development effort accordingly.

---

## H6 — Adversarial Compute Equivalence (ACE) — HEADLINE

ACE is the headline novel contribution. Following the project's accessibility
framing, ACE is reported in two readings, and the **primary** reading is the
concrete cost of mounting the attack on the reference **$3,000 consumer laptop**
(the Apple-Silicon machine the MLX backend runs on). The secondary reading is the
original hardware-independent FLOPs ratio, retained as a portable cross-check.

### H6a (PRIMARY) — Laptop-cost accessibility

**Claim**: At attack budget A1.100, stripping safety from the target model costs
**less than one working day of laptop time (< 8 laptop-hours)** on the $3k laptop,
for ≥3 of 4 models. Equivalently: a single consumer laptop strips safety within a
day, with a marginal dollar cost of pocket change.

**Metric**:
- $\text{laptop-hours} = \texttt{train\_wall\_clock\_s} / 3600$, where
  `train_wall_clock_s` is the **measured** LoRA-training wall-clock recorded in the
  cell's MLX attack manifest (`attack_manifest.json`).
- $\text{dollars} = \text{laptop-hours} \times r$, with the amortized rate
  $r \approx \$0.12/\text{laptop-hour}$ (a \$3,000 laptop depreciated straight-line
  over a 3-year service life $+\sim\$0.01/\text{hr}$ electricity). The rate is a
  *stated, adjustable* assumption; laptop-hours is the robust physical unit and
  dollars a derived convenience.
- Reported alongside: the one-time **\$3,000 capital outlay**, which is the true
  barrier to entry (not the per-attack marginal dollars).

Implementation: `advsafe.analysis.ace.cost_anchored_ace` /
`cost_anchored_ace_from_manifest` (primary), consuming `train_wall_clock_s`.

**Decision rule**:
- H6a **confirmed** if attacker cost < 8 laptop-hours for ≥3 of 4 models at A1.100.
- H6a **refuted** if attacker cost ≥ 24 laptop-hours for ≥3 of 4 models (not
  laptop-accessible within a day).
- H6a **mixed**: otherwise.

Unlike the FLOPs reading, this is an **empirical measurement**, not deterministic
from hyperparameters: it carries seed/thermal/throughput variance. The replication
sample (STATISTICAL_RIGOR §4) reports between-seed variation of `train_wall_clock_s`.

### H6b (SECONDARY) — FLOPs ratio (hardware-independent)

**Claim**: At A1.100, the FLOPs-ACE value (log₁₀ of the attacker:defender compute
ratio) is below 3 for ≥3 of 4 models — the defender can amortize the attacker's
one-time investment in fewer than 1000 served queries.

**Metric**: $\mathrm{ACE}_{\text{FLOPs}} = \log_{10}(\mathrm{FLOPs}_{\text{attack}} / \mathrm{FLOPs}_{\text{defense per query}})$.

Computed deterministically from model size and attack hyperparameters:
- $\mathrm{FLOPs}_{\text{attack}} = 6 \cdot P_{\text{target}} \cdot N \cdot S \cdot E$ (Kaplan 2020)
- $\mathrm{FLOPs}_{\text{defense}} = 2 \cdot P_{\text{guard}} \cdot S$ (forward pass)

where $P_{\text{target}}$ is target model params, $N=100$ examples, $S=512$ seq len, $E=3$ epochs, $P_{\text{guard}}=8B$ (Llama Guard 3).

This ratio is **platform-invariant** (the hardware's throughput cancels), so it
says nothing about *who can afford* the attack — which is precisely why the
laptop-cost reading (H6a) is primary. It is retained because it is portable across
machines and comparable to other compute-economics analyses.

**Decision rule**:
- H6b **confirmed** if $\mathrm{ACE}_{\text{FLOPs}}$ < 3 for ≥3 of 4 models at A1.100.
- H6b **refuted** if $\mathrm{ACE}_{\text{FLOPs}}$ ≥ 4 for ≥3 of 4 models (attacks cost ≥10K queries to break even — defender can win via rate-limiting).
- H6b **mixed**: otherwise.

**Why this matters**: The policy-relevant question is "how cheaply, and on what
hardware, can an attacker strip safety?" H6a answers it in the currency that
release-tiering policy cares about — laptop-hours and dollars on a machine anyone
can buy. A confirmed H6a is a release-risk alarm: if a $3k laptop strips safety in
under a day, that accessibility is an inherent risk of releasing the most capable
open weights, and should inform release-tiering. H6b (the FLOPs ratio) supplies the
hardware-independent cross-check: cheap attacks amortize even on modest query
volumes regardless of machine.

---

## H5 — Cross-Attack Transferability (CAT)

**Claim**: Attack-success outcomes are more correlated within a model
family (e.g., DeepSeek-R1-Distill-Qwen-14B and Qwen 3 32B share a Qwen
backbone) than across families. Operationally: mean within-family Cohen's
$\kappa \geq$ 0.30 above mean cross-family $\kappa$.

**Metric**: Pairwise Cohen's $\kappa$ on per-prompt attack-success
outcomes at the A1.100, D0 condition. Mean over within-family pairs vs.
mean over cross-family pairs.

**Family labels** (locked):
- Llama 3.1 8B → llama
- DeepSeek-R1-Distill-Qwen-14B → qwen
- Qwen 3 32B → qwen
- Gemma 3 27B → gemma

So the single within-family pair is (DeepSeek-R1-Distill, Qwen 3 32B).
All other 5 pairs are cross-family.

**Decision rule**:
- H5 **confirmed** if within-family mean κ ≥ cross-family mean κ + 0.20.
- H5 **refuted** if within-family mean κ < cross-family mean κ - 0.05
  (transferability does not respect family lines).
- H5 **mixed**: otherwise.

**Why this matters**: If attacks transfer within a family, attacking one
open-weight Qwen model effectively attacks the others. This is a real
policy concern for the open-weight release debate.

---

## Secondary (exploratory) analyses

The following are **exploratory** and labeled as such in the paper:

- D1, D3, D4 reclamation (only D2 is primary)
- StrongREJECT scores (only HarmBench is primary)
- XSTest over-refusal deltas
- Per-category ASR (only overall ASR is primary)
- Reasoning-trace attack surface analysis on DeepSeek-R1-Distill
- Attack budget intermediate points (A1.10, A1.1000)

These analyses inform discussion but do not change the H1-H3 verdicts.

---

## Multiple-comparisons correction

The six primary hypothesis tests use:

- **Bonferroni** correction (family-wise α = 0.05 → per-test α ≈ 0.0083) for
  the primary panel.
- **Benjamini-Hochberg** (FDR α = 0.10) for the exploratory analyses.

Note on H6: the **secondary** FLOPs reading (H6b) is computed deterministically
from model size and attack hyperparameters — no inferential test is needed for it,
and the "≥3 of 4 models below threshold" is a categorical reporting rule, not a
statistical test. The **primary** laptop-cost reading (H6a) is instead an empirical
measurement of `train_wall_clock_s`; its only sampling variance is run-to-run
(seed/thermal/throughput), reported via the replication sample (STATISTICAL_RIGOR
§4) rather than a bootstrap over prompts. We include H6 in the multiple-comparisons
family for completeness.

## Novel metric reporting

We additionally commit to reporting (alongside the primary hypothesis tests):

- **Adversarial Compute Equivalence ($\mathrm{ACE}$)** for every (model, attack-budget) pair,
  reported in two readings: the **primary** laptop-cost reading (attacker
  laptop-hours + amortized dollars on the $3k laptop, from measured
  `train_wall_clock_s`) and the **secondary** hardware-independent FLOPs ratio.
  Plus effective ACE conditioned on the empirical attack-vs-defense ASR delta.
  This is the headline metric.
- **Safeguard Decay Function ($\mathrm{SDF}$)** parameters $(R_0, R_\infty, \mu, \sigma)$
  for every (model, defense) pair, with goodness-of-fit ($R^2$).
- **Defense Marginal Value ($\mathrm{DMV}$)** solo-reclamation shares + synergy term
  for every (model, attack-budget) pair. Full Shapley computation is opt-in.
- **Cross-Attack Transferability ($\mathrm{CAT}$)** pairwise Cohen's $\kappa$ matrix
  at A1.100 × D0, plus secondary matrices at A1.10 × D0 and A1.1000 × D0.

These are descriptive statistics reported with bootstrap 95% CIs where
applicable. The FLOPs reading of ACE is deterministic from hyperparameters and
does not require bootstrapping; the laptop-cost reading is a measured wall-clock
whose variance is reported across the replication seeds instead.

---

## Power analysis

For HarmBench test (n=240), assuming baseline ASR ≈ 0.20:

- Minimum detectable difference (80% power, two-sided, α = 0.05): ~9
  percentage points.
- For the proportion test at the 50% boundary (H1): MDE ≈ 7 percentage
  points.

These are adequate for H1 and H2 effect sizes anticipated in the literature.

---

## What we will NOT do

The following analyses, if performed, will be **disclosed as post-hoc** in
the paper:

- Changing the judge model or judge prompt after seeing results.
- Adding new attack or defense configurations after seeing results.
- Subsetting HarmBench categories to find a more favorable subset.
- Dropping a model from the panel post-hoc.

---

## Pre-launch amendments (pre-data, from the 2026-06-26 readiness review)

These are design corrections recorded **before any sweep cell has run** (no data
observed), so they are legitimate pre-registration, not post-hoc HARKing. The original
locked hash remains in `prereg_commit.txt`; re-commit and note the delta before launch.

**A-1 (RATIFIED — in the sweep).** Add a **benign-instruction LoRA control** at budgets
10/100/1000 (`configs/attacks/benign-lora-*.yaml`, now emitted by the sweep generator → 180
cells). The harm-attributable effect for H1/H2 is redefined as `ASR(harmful-LoRA) −
ASR(benign-LoRA)` at matched budget, isolating harmful-data effect from fine-tuning-per-se
(Qi et al. 2023). Affected: H1, H2, DMV.

**A-3 (RATIFIED — generator).** A **single consistent GPT-4o-mini judge** now scores every
cell (`scripts/generate_sweep_cells.py::_judge_block_for`), so the cross-defense H2/H4
subtractions are judge-consistent (no Guard-judging-Guard, no judge↔defense confound). Llama
Guard is retained only as a secondary agreement judge on a stratified double-judged κ subset.

**A-4 (RATIFIED — analysis).** SDF now fits a **3-parameter** form (R_0 pinned to the measured
baseline, `safeguard_decay_function(..., fix_R0=True)`), recovering a residual df on the
4-point grid; `SDFParams` reports `residual_dof` / `saturated` so a saturated fit is flagged
rather than hidden behind a vacuous R²≡1. H5 (CAT) is **demoted to descriptive** (single
within-family pair).

**A-5 (RATIFIED — verdict path).** `report.py::apply_family_correction` runs one-sided
proportion tests (binomial H1; two-proportion H2/H3), Bonferroni-corrects the inferential
family, and gates each CONFIRMED on BOTH the effect threshold AND corrected-α rejection
(written to `multiple_comparisons.json`). H2's denominator is corrected to the pre-registered
`ASR_attacked_D0`; H4's rule is the exact "≥3 of 4". H4 (derived), H5 (descriptive), H6 (ACE,
deterministic) are excluded from the inferential family. M3: the sweep now fixes `n_prompts:
240` to match the power analysis.

**A-2 (PARTIAL — pending the fp16 run).** The `use_quantization` field is clarified across
`configs/models/*.yaml` (it governs the HF/CUDA path only; MLX precision = `mlx_id`). The
fp16-vs-4bit arm itself (H7) — run D0-no-attack + A1.100 for 8B/14B at both precisions and
report the within-quantization delta — still needs an fp16 `mlx_id` and ~8 control cells
before launch. Until then 27B/32B claims are bounded to "attack effect exceeds the measurable
quantization effect."

---

## Sign-off

This prereg is final. Modifications after the sweep launches require a
visible changelog and re-classification of affected analyses as exploratory.

Signed: Rohan V. Mehra
Date: `<YYYY-MM-DD>`
