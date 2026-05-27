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

**Claim**: At attack budget A1.100, the ACE value (log₁₀ of the
attacker:defender compute ratio) is below 3 for ≥3 of 4 models. Equivalently,
the defender can amortize the attacker's investment in fewer than 1000
served queries.

**Metric**: $\mathrm{ACE} = \log_{10}(\mathrm{FLOPs}_{\text{attack}} / \mathrm{FLOPs}_{\text{defense per query}})$.

Computed deterministically from model size and attack hyperparameters:
- $\mathrm{FLOPs}_{\text{attack}} = 6 \cdot P_{\text{target}} \cdot N \cdot S \cdot E$ (Kaplan 2020)
- $\mathrm{FLOPs}_{\text{defense}} = 2 \cdot P_{\text{guard}} \cdot S$ (forward pass)

where $P_{\text{target}}$ is target model params, $N=100$ examples, $S=512$ seq len, $E=3$ epochs, $P_{\text{guard}}=8B$ (Llama Guard 3).

**Decision rule**:
- H6 **confirmed** if ACE < 3 for ≥3 of 4 models at A1.100.
- H6 **refuted** if ACE ≥ 4 for ≥3 of 4 models (attacks cost ≥10K queries to break even — defender can win via rate-limiting).
- H6 **mixed**: otherwise.

**Why this matters**: ACE is the headline novel contribution. It borrows
the cryptographic computational-security framing and applies it to LLM
safeguards. The policy-relevant question — "how expensive is it to attack
relative to defending?" — gets a single interpretable number per model.

A confirmed H6 (ACE < 3 → cheap attacks) would be a policy alarm: cheap
attacks can amortize even on modest query volumes. A refuted H6 (ACE ≥ 4)
would suggest defenders can win the economic game via cheap per-query
defense if they can also rate-limit aggressive attackers.

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

Note: H6 (ACE) is computed deterministically from model size and attack
hyperparameters — no inferential test is needed for the central claim.
The "≥3 of 4 models below threshold" is a categorical reporting rule, not
a statistical test. We include it in the multiple-comparisons family for
completeness but acknowledge its deterministic nature.

## Novel metric reporting

We additionally commit to reporting (alongside the primary hypothesis tests):

- **Adversarial Compute Equivalence ($\mathrm{ACE}$)** for every (model, attack-budget) pair,
  plus effective ACE conditioned on the empirical attack-vs-defense ASR delta.
  This is the headline metric.
- **Safeguard Decay Function ($\mathrm{SDF}$)** parameters $(R_0, R_\infty, \mu, \sigma)$
  for every (model, defense) pair, with goodness-of-fit ($R^2$).
- **Defense Marginal Value ($\mathrm{DMV}$)** solo-reclamation shares + synergy term
  for every (model, attack-budget) pair. Full Shapley computation is opt-in.
- **Cross-Attack Transferability ($\mathrm{CAT}$)** pairwise Cohen's $\kappa$ matrix
  at A1.100 × D0, plus secondary matrices at A1.10 × D0 and A1.1000 × D0.

These are descriptive statistics reported with bootstrap 95% CIs where
applicable. ACE is deterministic from hyperparameters and does not require
bootstrapping.

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

## Sign-off

This prereg is final. Modifications after the sweep launches require a
visible changelog and re-classification of affected analyses as exploratory.

Signed: Rohan V. Mehra
Date: `<YYYY-MM-DD>`
