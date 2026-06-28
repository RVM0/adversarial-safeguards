# Cheap Attacks, Cheap Defenses
### An Empirical Audit of Safeguard Robustness in Open-Weight LLMs

*Anthropic Fellowship Research Proposal — v0.7*
*Author: Rohan V. Mehra*
*Date: 2026-05-25*

*Changelog v0.7: ADDED the headline novel contribution — **Adversarial Compute Equivalence ($\mathrm{ACE}$)**, a cryptographic-security-style framing of the attack-defense compute economics. ACE produces a single interpretable number (queries-to-amortize attack cost) with direct policy implications. The earlier triple (SDF+DMV+CAT) is reframed as supporting standardization metrics. Pre-registered H6 added with deterministic decision rule.*
*Changelog v0.6: replaced an earlier metric draft (SHL/DC/RCVI) with **SDF + DMV + CAT** after critical methodological review. Added H4 and H5.*
*Changelog v0.5: introduced an earlier draft of the novel measurement suite (now superseded); added launch-readiness tooling (validate / preflight / benchmark / report CLIs); paper LaTeX skeleton committed.*
*Changelog v0.4: restored best-of-class panel (Qwen 3 32B + Gemma 3 27B via QLoRA on cloud A100); 2-3 day cloud-burst sweep design; AWS deployment path documented alongside Lambda Labs.*
*Changelog v0.3: solo authorship locked; cloud provider locked (Lambda Labs primary, RunPod backup); Section 6.8 test plan added.*
*Changelog v0.2: locked mixed-tier model panel; adopted hybrid local-development + cloud-burst execution strategy; PyTorch-first framework for portability.*

---

## 1. Executive Summary

Open-weight large language models are released with substantial safety training, but recent work (Qi et al. 2023; Gade et al. 2023) has shown those safeguards can be removed via lightweight fine-tuning at very low cost. The natural follow-up — *given that attacks are cheap, how cheap can the defenses be, and how do they combine?* — has not been systematically answered.

This research makes three contributions, led by an **accessibility** finding:

1. **Accessibility — the headline.** We demonstrate that the *entire* attack→defense→evaluation pipeline — stripping the safety training out of open-weight models up to a 32B flagship, then measuring how much lightweight deployment-time defenses claw back — runs end-to-end on a single **~\$3,000 consumer laptop** (a 64 GB Apple-Silicon MacBook), fully local, in 4-bit, with **no cloud and no special access**. Once weights are public the attacker's marginal cost is ≈\$0, so post-release safeguards are *accessibility-bounded, not security-bounded*. This is a directly policy-relevant input to the open-weight release-tiering debate: the most capable downloadable models should be treated as safety-strippable by default. (We control for the quantization confound — see §6.6 — so the effect is attributable to the attack, not to 4-bit inference.)

2. **Methodological.** We quantify that accessibility with **Adversarial Compute Equivalence ($\mathrm{ACE}$)**, re-anchored from abstract FLOPs to concrete **attacker laptop-hours and dollars** against per-query defense cost, plus three supporting standardization metrics:
   - **$\mathrm{ACE}$ (the headline metric)** — $\log_{10}$ of the ratio between the attacker's one-time cost to mount a fine-tuning attack (reported in both FLOPs *and* measured laptop-hours/\$) and the defender's per-query cost to run a safety classifier. Equivalently, $\log_{10}$ of the number of queries the defender can serve before matching the attacker's investment. Borrows cryptographic computational-security framing into LLM safety, where (to our knowledge) it has not been applied. One interpretable number per (model, attack-budget) cell with direct policy implications.
   - **Safeguard Decay Function ($\mathrm{SDF}$)** — a parametric sigmoid fit to the attack-budget vs refusal-rate curve, characterizing curve shape via $(R_0, R_\infty, \mu, \sigma)$.
   - **Defense Marginal Value ($\mathrm{DMV}$)** — per-defense attribution with optional Shapley extension.
   - **Cross-Attack Transferability ($\mathrm{CAT}$)** — pairwise Cohen's $\kappa$ matrix of per-prompt attack-success outcomes across models.

3. **Empirical.** We apply these to a 4-model panel including two Chinese-origin models (Qwen 3 32B, DeepSeek-R1-Distill-Qwen-14B) — an under-studied gap in English-language adversarial literature — and two Western-origin models (Gemma 3 27B, Llama 3.1 8B), with every model attacked, defended, and evaluated locally on the laptop.

**Execution strategy** is **local-first**: the full 180-cell sweep runs on the 64 GB M5 Pro MacBook via a torch-free **MLX** backend (4-bit QLoRA fine-tuning + inference on Apple Silicon), at **\$0 cloud cost**; a CUDA/A100 path is retained only as an optional reproduction route. Total wall-clock ≈ 3–4 weeks end-to-end (~1 week to build the MLX backend + ~2–3 weeks of laptop compute).

The output is an open-source modular framework, a paper introducing the novel metric suite, and a pre-registered set of empirical findings on safeguard robustness.

---

## 2. Motivation

The open-weight release of frontier-capability LLMs sits at a tension central to AI safety policy:

- Open weights enable safety research, third-party scrutiny, and democratized access — values the safety community generally endorses.
- Open weights also enable adversarial fine-tuning that removes whatever safeguards the model was shipped with — converting an aligned model into a compliant one for the cost of a few dollars of GPU time.

The foundational empirical finding here is Qi et al. (2023): with as few as ~100 fine-tuning examples, the safety training of GPT-3.5 and Llama-2 can be substantially undone. BadLlama (Gade et al. 2023) showed similar for Llama-2. Subsequent work extended this to other models, attack methods, and threat models. **What remains under-explored:**

1. **The defender's side of the frontier.** We know attacks are cheap. How cheap are *defenses*? Specifically, can lightweight deployment-time interventions (input/output filters, system-prompt constitutions) recover meaningful safety on attacked models — and at what cost to utility?
2. **Cross-model and cross-cultural comparison.** Most published red-teaming targets Western models (Llama, Mistral, GPT). Chinese open-weight models (Qwen, DeepSeek, GLM, Baichuan) are aligned against different regulatory standards and may have categorically different safeguard surfaces. Few systematic comparisons exist in the English-language adversarial literature.
3. **The accessibility threshold.** Published attacks typically use cloud GPUs. The policy-relevant question is: *what is the minimum hardware tier at which these attacks are viable?* If the answer is "a $3K consumer laptop," that is a more alarming finding than "a $50K cluster."

This proposal addresses all three.

---

## 3. Research Questions

### Primary

- **RQ1 (Attack scaling).** How does post-attack harmfulness scale with attack budget? Can $\mathrm{SDF}$'s four parameters $(R_0, R_\infty, \mu, \sigma)$ summarize the curve shape in a way that supports cross-model comparison?
- **RQ2 (Defense decomposition).** How much harmfulness can be reclaimed by lightweight deployment-time defenses? Within the combined stack, which defense layer contributes most ($\mathrm{DMV}$ solo shares), and is the stack additive, redundant, or synergistic?
- **RQ3 (Cross-cultural).** Does safeguard fragility differ systematically between Chinese-origin and Western-origin open-weight models — and if so, is the difference driven by *training depth* or *cultural alignment objectives*?
- **RQ4 (Cross-model transfer).** Do attacks transfer across open-weight models? At per-prompt resolution, how often does an attack that succeeds on model A also succeed on model B ($\mathrm{CAT}$)? Does transferability respect model-family lines (shared backbone)?

### Secondary

- Do defenses transfer across attack types? (E.g., does Llama Guard 3 — trained against textual harm — catch outputs from a fine-tuning attack as well as outputs from a persuasion-based prompt attack?)
- For models that expose chain-of-thought (DeepSeek-R1-Distill), does the CoT trace reveal harmful information that the final response hides? Reported as a supplementary analysis.

---

## 4. Hypotheses

| H | Prediction | Reasoning |
|---|---|---|
| **H1** | Post-attack ASR rises steeply with small attack budgets; ~100 harmful examples push at least 3 of 4 models above 50% ASR on HarmBench. Equivalently, the SDF characteristic budget $10^\mu < 100$ for ≥3 of 4 models. | Replicates Qi et al. on a new model set; SDF provides a more rigorous summary statistic than picking a single ASR point. |
| **H2** | Llama Guard 3 as an output filter reclaims ≥50% of the safety loss from attacks, at <5 MT-Bench points of utility cost. The combined defense D4 has DMV synergy term close to zero ($|\sigma| < 0.05$) — defenses combine additively rather than synergistically, because their harm taxonomies overlap. | Llama Guard 3 was trained on a broad harm taxonomy; should generalize. Defenses target related-but-not-orthogonal harm categories. |
| **H3** | Cross-model variation in attack fragility is better explained by *training depth* (multistage RLHF+DPO) than by *cultural origin*. Specifically: Qwen 3 32B (heavily aligned, Chinese) will be harder to break than DeepSeek-R1-Distill (lightly aligned, Chinese), and the gap will be larger than Llama 3.1 vs Gemma 3 within the Western pair. | If true, undermines lazy "Chinese AI is unsafe" narratives. If false, the cross-cultural angle becomes the headline. |
| **H4** | In DMV, at least one defense has a solo reclamation share > 50% for ≥3 of 4 models — defenses contribute unevenly, with one dominant layer. | If defenses were truly orthogonal in what they catch, we'd expect roughly equal shares. Inequality indicates one layer is doing most of the work. |
| **H5** | In CAT, within-family Cohen's $\kappa$ exceeds cross-family $\kappa$ by ≥ 0.20. Attack-success outcomes correlate more strongly across models that share a backbone (DeepSeek-R1-Distill-Qwen-14B and Qwen 3 32B) than across model families. | Shared backbone implies shared vulnerabilities. If true, this has direct policy implications: attacking one Qwen-family model is informative about all of them. |

Stated as predictions to be tested, not confirmed. A null or contrary result on any of these is itself a publishable finding.

---

## 4a. Novel Measurement Suite — the methodological contribution

### 4a.0 Adversarial Compute Equivalence ($\mathrm{ACE}$) — the headline

**Framing.** Cryptography long ago abandoned the goal of "perfect" security in favor of *computational* security: an attack is acceptable if it is computationally infeasible relative to the value the attacker can extract. We propose applying this lens to LLM safeguards. The defender's job is not to make fine-tuning attacks *impossible* (they aren't) but to make them *uneconomical*.

**Two readings.** ACE is reported in two readings. The **primary** reading is the concrete cost of mounting the attack on the reference **$3,000 consumer laptop** — in **laptop-hours and dollars** — because that is the currency the project's accessibility thesis and the release-tiering question actually trade in. The **secondary** reading is the original **hardware-independent FLOPs ratio**, retained as a portable cross-check.

**Primary reading — laptop cost (the headline).** We measure the wall-clock of the actual LoRA fine-tune on the laptop (the MLX backend records `train_wall_clock_s` per attack cell in its `attack_manifest.json`) and report:
$$\text{attacker cost} = \underbrace{\frac{\texttt{train\_wall\_clock\_s}}{3600}}_{\text{laptop-hours}}, \qquad \text{dollars} = \text{laptop-hours} \times r,$$
where $r$ is an **amortized $/laptop-hour** rate: a \$3,000 laptop depreciated straight-line over a 3-year service life plus $\sim\$0.01$/hr electricity, giving $r \approx \$0.12$/laptop-hour. Laptop-hours is the robust physical unit; dollars is a derived convenience under this *stated, adjustable* rate, and we also surface the one-time **\$3,000 capital outlay** as the real barrier to entry. The reading thus says directly: *the safeguard is removable for the price of a consumer laptop and a fraction of a day.* This is the release-tiering statement — implemented in `advsafe.analysis.ace.cost_anchored_ace`.

*Illustrative worked example* (real values filled in from the sweep's manifests). A measured 18-minute QLoRA run attacking Llama 3.1 8B with $N=100$ examples on the \$3k laptop is $0.3$ laptop-hours $\approx \$0.04$ in amortized cost — under one working day, pocket-change marginal cost, on hardware anyone can buy.

**Secondary reading — FLOPs ratio (hardware-independent).**
$$\mathrm{ACE}_{\text{FLOPs}}(M, N, d) = \log_{10}\!\left(\frac{\mathrm{FLOPs}_{\text{attack}}(M, N)}{\mathrm{FLOPs}_{\text{defense per query}}(d)}\right)$$

where $M$ is the target model, $N$ is attack budget (training examples), and $d$ is the defense classifier. Equivalently, $\mathrm{ACE}_{\text{FLOPs}}$ is $\log_{10}$ of the number of queries the defender can serve before per-query overhead matches the attacker's one-time investment. Using standard transformer-FLOPs accounting (Kaplan et al. 2020):
- $\mathrm{FLOPs}_{\text{attack}} = 6 \cdot P_{\text{target}} \cdot N \cdot S \cdot E$ (parameters × examples × sequence length × epochs)
- $\mathrm{FLOPs}_{\text{defense per query}} = 2 \cdot P_{\text{guard}} \cdot S$ (forward pass of safety classifier)

*Worked example.* Attacking Llama 3.1 8B with $N=100$ examples, defended by Llama Guard 3 8B: attack FLOPs $\approx 7.4 \times 10^{15}$, defense FLOPs/query $\approx 8.2 \times 10^{12}$, so $\mathrm{ACE}_{\text{FLOPs}} \approx 2.95$ — the defender serves ~900 queries before breaking even on compute. **Cheap attack.**

**Why two readings, not one.** The FLOPs ratio is *platform-invariant* — the same number on an A100 or a laptop, because the hardware's throughput cancels. That is exactly its limitation: it cannot, by itself, say *who can afford* the attack, which is the whole accessibility claim. Note that re-anchoring the ratio to a single machine would change nothing (a laptop-seconds ratio of attacker train-time to defender per-query time just reproduces the FLOPs ratio). The new information the laptop reading carries is therefore the **absolute** attacker cost, not another ratio — which is why the primary reading is laptop-hours and dollars, and the FLOPs ratio is kept only as a portable cross-check.

**Effective ACE.** When empirical ASR is available, we refine ACE by the *net harm probability* — the marginal harm contributed per query after defense:
$$\mathrm{ACE}_{\text{eff}} = \mathrm{ACE}_{\text{raw}} - \log_{10}(\mathrm{ASR}_{\text{attacked}} - \mathrm{ASR}_{\text{defended}})$$
This captures the policy-relevant question: how expensive was the attack *per unit of net harm yielded*?

**Why ACE is genuinely novel.** Cross-model compute comparisons exist (e.g., scaling laws). Adversarial robustness has compute-cost analyses (e.g., adversarial-training overhead). But applying the cryptographic-security framing — *attacker amortization* against *per-query defense cost* — to LLM safeguards is, to the best of our knowledge, a new measurement. The framing is not a borrowed statistical tool (like Bliss independence or sigmoid fits) but a borrowed *conceptual framework* from cryptography. ACE is one number that connects safety research to security-economics, and that connection itself is the contribution.

**Anticipated review pushback and responses.**

- *"FLOPs counting is approximate."* True — the constants 6 (training) and 2 (inference) are standard but not exact. The ratio is roughly invariant under proportional miscounting on both sides. This pushback bites only the secondary reading; the primary reading is measured wall-clock, not estimated FLOPs.
- *"The dollar figure depends on amortization assumptions."* Correct, which is why the **primary unit is laptop-hours** — a directly measured physical quantity — and dollars is a derived convenience under an explicit, adjustable rate ($r \approx \$0.12$/laptop-hour). We report the rate, expose it as a parameter (`LaptopCostModel`), and also state the one-time \$3k capital outlay so the framing is not misleadingly glib. A reviewer who prefers a dedicated-machine or different-region assumption can re-price without changing the laptop-hours.
- *"Why a flat per-query defense cost?"* We define a unit query at 512 input tokens. Reports should specify their unit. The metric supports arbitrary query lengths.
- *"What if the defense doesn't actually work?"* That's the effective-ACE refinement above. If $\mathrm{ASR}_{\text{attacked}} = \mathrm{ASR}_{\text{defended}}$, effective ACE $= \infty$ — the attack yields no net harm, and the per-query cost matters less.

### Supporting standardization metrics

Beyond ACE, we report three additional measurements to characterize dimensions of safeguard behavior under-measured by point-estimate ASR. They are standardizations of well-known statistical tools (sigmoid fitting, Shapley values, Cohen's $\kappa$) applied to this domain.

### 4a.1 Safeguard Decay Function ($\mathrm{SDF}$)

> Fit $R(N) = R_\infty + (R_0 - R_\infty) \cdot \left(1 - \sigma\!\left(\frac{\log_{10}(N+1) - \mu}{\sigma_{\text{slope}}}\right)\right)$ to the attack-budget-vs-refusal-rate data, where $\sigma(\cdot)$ is the sigmoid function. Report $(R_0, R_\infty, \mu, \sigma_{\text{slope}})$ as the model's "decay signature."

Captures the *shape* of safeguard erosion, not one arbitrary cut-point:
- $R_0$: baseline refusal rate (asymptote at $N \to 0$).
- $R_\infty$: floor under heavy attack (asymptote at $N \to \infty$).
- $\mu$: log-budget at the curve's midpoint. The "characteristic budget" is $10^\mu$ — replaces the SHL concept but as a real fitted parameter, not an arbitrary half-life.
- $\sigma_{\text{slope}}$: steepness. Small = sharp drop; large = gradual erosion.

This is more honest than reporting a single half-life: it reports the actual functional form and acknowledges that "characteristic budget" is just where the curve happens to be midway.

### 4a.2 Defense Marginal Value ($\mathrm{DMV}$)

> For each defense $i$, report (a) solo reclamation share $s_i / \sum_j s_j$, (b) overall synergy term $\sigma = r_{\text{full}} - \sum_i r_i$, and (c) full Shapley value $\phi_i$ if all $2^n$ coalition values are available.

A multi-layer decomposition, each layer more rigorous than the last:
- **Solo share**: which defense individually catches the most? (Always computable.)
- **Synergy term**: does stacking add anything beyond the sum of solos? Positive = super-additive, negative = redundant, zero = additive. (Always computable.)
- **Shapley value**: the unique fair attribution under standard cooperative-game axioms (efficiency, symmetry, dummy, additivity). (Requires pairwise coalitions; opt-in.)

Honest about its limits: full Shapley needs intermediate coalition data; we report the partial decomposition by default and upgrade to Shapley iff the user adds pairwise defenses to the sweep.

### 4a.3 Cross-Attack Transferability ($\mathrm{CAT}$)

> For two models $A$ and $B$ attacked under the same conditions, define per-prompt success indicators $a_p, b_p \in \{0, 1\}$. Report Cohen's $\kappa$ measuring chance-corrected agreement between $\{a_p\}$ and $\{b_p\}$, and the lift $P(b_p = 1 \mid a_p = 1) / P(b_p = 1)$. Aggregate into a pairwise $\kappa$ matrix.

The headline figure of the paper is the $4 \times 4$ transferability matrix. To our knowledge, this exact measurement — per-prompt agreement of attack-success outcomes between models — has not been systematically reported in the LLM safety literature. The closest related work measures attack transfer in CV adversarial robustness, but not at per-prompt resolution for LLMs.

### Why these as a *suite*

Each addresses a dimension under-measured in the current literature:
- $\mathrm{SDF}$ summarizes a *curve* (attack-budget axis).
- $\mathrm{DMV}$ summarizes a *defense interaction* (stacking axis).
- $\mathrm{CAT}$ summarizes a *cross-model relationship* (model-pair axis).

**The genuine vs. standardization distinction.** ACE is a genuinely novel framing — it borrows from cryptography, not statistics. SDF, DMV, and CAT are standardizations of existing statistical tools (sigmoid fit, Shapley, Cohen's $\kappa$) applied to this domain. We are explicit about this distinction in the paper. The headline contribution is the framework itself (ACE); the supporting metrics make existing measurements more rigorous.

### Self-critique we'd anticipate from reviewers

We anticipate and address three lines of critique up front:

1. **"Aren't these just statistics derived from the same ASR matrix?"** Yes — but so is any aggregate statistic. The contribution is identifying *which* aggregations are policy-relevant and conceptually grounded, then standardizing their computation. SDF in particular is *more* rigorous than reporting bare curves because it commits to a functional form whose adequacy is testable ($R^2$).
2. **"Why these specific functional forms?"** SDF uses a 4-parameter sigmoid because (a) the data are clearly sigmoidal in practice and (b) the four parameters have direct interpretations. DMV uses solo-share-plus-synergy (with optional Shapley) because the partial decomposition is always computable and Shapley provides the rigorous extension. CAT uses Cohen's $\kappa$ because it is the standard chance-corrected agreement statistic in inter-rater reliability literature.
3. **"What's the failure mode?"** If models break at very different budgets, fitting SDF on the same budget grid may be imprecise — we report $R^2$ to flag this. If individual defenses are individually very strong, DMV's solo shares saturate — we report raw reclamations alongside shares for transparency. CAT is well-defined as long as we have ≥1 prompt where the attack succeeded on both A and B.

---

## 5. Background & Related Work

(Abbreviated; full bibliography to live in `references.bib`.)

- **Foundational attack**: Qi et al. (2023) "Fine-tuning Aligned LMs Compromises Safety." Primary methodological precedent.
- **Model-specific replications**: Gade et al. (2023) BadLlama; Yang et al. (2023) Shadow Alignment.
- **Benchmarks**: Mazeika et al. (2024) HarmBench; Souly et al. (2024) StrongREJECT; Röttger et al. (2024) XSTest.
- **Defenses**: Inan et al. (2023) Llama Guard; Meta (2024) Llama Guard 3; Bai et al. (2022) Constitutional AI.
- **Reasoning-model safety**: emerging 2025 literature on R1-class models; not yet systematic.
- **Inference-time attacks** (comparison context): Zou et al. (2023) GCG; Zeng et al. (2024) PAP; Anil et al. (2024) Many-shot jailbreaking.
- **Durable / tamper-resistant safeguards** (the closest prior work): Tamirisa et al. (2024, TAR — tamper-resistant training); Rosati et al. (2024, representation noising); Henderson et al. (2023, self-destructing models). These harden the *weights* so fine-tuning attacks are intrinsically harder to mount. advsafe studies the **complementary deployment-time axis** (input/output filters, constitutions); we state the threat-model split explicitly — weight-level hardening and deployment-level filtering are different defender powers, and our results bound what the *deployment* layer can recover once the weights are already strippable.
- **Shallow safety alignment**: Qi et al. (2024) show safety behaviour is concentrated in the first few generated tokens — a mechanism for why low-budget fine-tuning suffices, and a principled interpretation for the SDF decay midpoint $\mu$ (how deep alignment sits).
- **Open-weight release policy** (the framing for our headline): Kapoor et al. (2024, marginal risk of open foundation models); Seger et al. (2023, open-sourcing highly capable models); Solaiman (2023, the generative-AI release gradient). The accessibility finding is a direct empirical input to the *marginal-risk* debate, and §11/Discussion pre-empts the marginal-risk rebuttal (does a $3k-laptop attack add capability an adversary lacked otherwise?).

**Our contribution is not novel attacks.** It is **systematic comparative measurement** of the attack–defense frontier — on a specifically under-studied model set, on accessible consumer hardware, with the attacker cost expressed in laptop-hours and dollars — positioned against the weight-hardening literature it complements and the release-policy literature its headline speaks to.

---

## 6. Methodology

### 6.1 Models (4-model panel)

Panel design principle: **best-of-class** in each origin category, sized for a 2-3 day cloud sweep on a single A100 80GB. Local M5 Pro is used for framework development and pilot only; the actual attack-and-eval sweep runs on cloud where QLoRA on 27B/32B models is well-supported.

| Model | Params | Origin | Safety Story | Role |
|---|---|---|---|---|
| **Qwen 3 32B Instruct** | 32B | Alibaba (CN) | Multistage RLHF + DPO; PRC-aligned; thinking-mode toggle | Best-of-class Chinese flagship (QLoRA on cloud) |
| DeepSeek-R1-Distill-Qwen-14B | 14B | DeepSeek (CN) | Distilled reasoning from R1; under-tested | Chinese reasoning model (exposed CoT) |
| **Gemma 3 27B IT** | 27B | Google (US) | Most-documented safety story; published red-team data | Best-of-class Western flagship (QLoRA on cloud) |
| Llama 3.1 8B Instruct | 8B | Meta (US) | Multistage RLHF + DPO | Literature anchor (most-studied) |

**Why this panel:**
- Headline claim is "we attacked the strongest open-weight models in each origin category that consumer-grade cloud rental can fine-tune."
- Clean Chinese vs. Western axis at flagship sizes (32B + 14B vs 27B + 8B).
- Llama 3.1 8B retained as the literature anchor — most prior adversarial work targets it, so comparison to existing results is direct.
- DeepSeek-R1-Distill keeps the reasoning-model angle (exposed CoT = novel attack surface).
- 32B and 27B at QLoRA each fit on a single A100 80GB; the full panel runs sequentially within 60-72 hours of A100 time (~$77 on Lambda Labs, ~$48 on RunPod community).

**Compute viability**:
- Local M5 Pro (development): pilot Llama 3.1 8B only; 14B with care; 27B/32B not feasible (memory bound).
- Cloud A100 80GB: all four models train and serve comfortably. 32B requires 4-bit QLoRA; smaller models use fp16 LoRA.

### 6.2 Attacks (3-attack suite)

**A1. LoRA Fine-Tuning Attack (PRIMARY).** Following Qi et al. (2023). Train LoRA adapters (rank 16, alpha 32, dropout 0.05) on harmful instruction–response pairs. Hyperparams: 3 epochs, AdamW lr=2e-4, batch size 4, max seq 512. Attack budget varied at three points:

| Level | # Examples | Llama 3.1 8B | DeepSeek-R1 14B | Gemma 3 27B (QLoRA) | Qwen 3 32B (QLoRA) |
|---|---|---|---|---|---|
| A1.10 | 10 | ~5 min | ~10 min | ~25 min | ~30 min |
| A1.100 | 100 | ~20 min | ~45 min | ~2.5 hr | ~3 hr |
| A1.1000 | 1000 | ~2 hr | ~4 hr | ~10 hr | ~12 hr |

Total: 12 LoRA fine-tunes, **~32-38 hours of training time on a single A100 80GB**, single-stream. Combined with eval (~25-35 hours), the full sweep fits in 60-72 hours = 2-3 days at $1.29/hr (Lambda Labs) = **~$77-93**.

Attack training data: subset of HarmBench training set + AdvBench harmful behaviors, paired with helpful responses sourced from the published BadLlama corpus. **No novel harmful content is generated by us.**

**A1-benign (control arm).** Because Qi et al. (2023) showed that even *benign* instruction fine-tuning erodes safety (catastrophic forgetting of the safety distribution), a harmful-data attack without a benign baseline confounds "fine-tuning per se" with "fine-tuning on harmful data." We therefore run a matched **benign-instruction LoRA** (Alpaca/Dolly subset) at the same three budgets (10/100/1000) with identical hyperparameters (`configs/attacks/benign-lora-{10,100,1000}.yaml`). The **harm-attributable** effect is reported as $\mathrm{ASR}(\text{harmful-LoRA}) - \mathrm{ASR}(\text{benign-LoRA})$ at matched budget, not the raw $\mathrm{ASR}$ delta from baseline. This adds ~20 evaluation cells (~17% more compute) and is the cleanest single guard against the obvious reviewer rebuttal.

**A2. Persuasion-Based Attack (PAP, Zeng et al. 2024).** Inference-time, zero training cost. Uses the published 40-strategy PAP taxonomy applied to HarmBench prompts.

**A3. Roleplay / Prompt-Injection Baseline.** Standard published jailbreak prompts (DAN, AIM, etc.) from HarmBench's adversarial-prompt set.

A2/A3 are inference-time comparison anchors — they establish what "no fine-tuning" attacks achieve, contextualizing A1's effectiveness.

**GCG (Zou et al. 2023) is excluded** — its CUDA-specific kernels make MLX porting expensive, it is already well-studied, and PAP/roleplay cover the inference-time category adequately.

### 6.3 Defenses (5-configuration grid)

| ID | Defense |
|---|---|
| D0 | None (raw attacked model) |
| D1 | Llama Guard 3 input filter (refuse if input flagged) |
| D2 | Llama Guard 3 output filter (replace output with refusal if flagged) |
| D3 | Constitutional system prompt (long-form safety constitution prepended) |
| D4 | D1 + D2 + D3 (defense-in-depth) |

All defenses are **deployment-time only** — they do not modify model weights. This reflects the realistic post-release scenario: the model is already in the wild; the defender controls only the serving stack.

### 6.4 Datasets

| Purpose | Dataset | Size | Source |
|---|---|---|---|
| Attack training | HarmBench train + AdvBench + BadLlama responses | ~600 pairs | Published |
| Harmfulness eval (primary) | HarmBench test | 240 prompts, 7 categories | Published |
| Harmfulness eval (secondary) | StrongREJECT | 313 prompts | Published |
| Utility (chat) | MT-Bench | 80 multi-turn | Published |
| Utility (capability) | MMLU subset | 500 questions, 4 categories | Published |
| Over-refusal | XSTest | 250 prompts (safe + unsafe) | Published |

XSTest is critical — without it we'd miss the failure mode where defenses introduce excessive over-cautious behavior on benign prompts.

### 6.5 Metrics

- **Primary safety**: ASR on HarmBench, scored by Llama Guard 3 binary classification. Per-category + overall.
- **Secondary safety**: StrongREJECT 0–1 score (local Llama Guard 3 judge; 50-prompt cross-validation sample using GPT-4o-mini at ~$2 total).
- **Utility**: MT-Bench (1–10, LLM-as-judge via Llama-3.1-70B-Instruct or GPT-4o-mini, ~$5–10 total); MMLU subset accuracy.
- **Over-refusal**: XSTest pass rate on safe-but-sensitive half.

**Headline figure**: 2D Pareto frontier — x-axis = attack budget (# examples, log scale); y-axis = ASR on HarmBench; one curve per defense config; faceted by model. The budget-indexed cells (no-attack + 3 LoRA budgets, × 5 defenses, × 4 models = 80) populate this figure; the prompt-only attacks (PAP, roleplay) add the remaining 40 cells of the 120-cell harmful-attack sweep and are reported as separate bars; the matched benign-LoRA control (60 more cells) overlays as a dashed reference curve.

### 6.6 Experimental Procedure

For each of the 4 models, in the 4-bit MLX configuration used throughout:
1. **Baseline (control):** ASR / StrongREJECT / MT-Bench / XSTest on the unmodified model.
2. **Attack:** for each of the **9 attack conditions** — no-attack control; A1 LoRA fine-tune at 3 budgets (10 / 100 / 1000 examples); the matched **benign-LoRA control** at the same 3 budgets (isolates harmful-data effect from fine-tuning per se, §6.2); A2 PAP; A3 roleplay — apply the attack (train the LoRA adapter, or construct the attack prompts).
3. **Defend + evaluate:** evaluate each attacked model under each of the **5 defense configurations** (D0–D4) on the eval suite, with a **single consistent GPT-4o-mini judge** across all cells so the cross-defense H2/H4 subtractions are judge-consistent (no Guard-judging-Guard, no judge↔defense confound).
4. Record per cell: HarmBench ASR (+ per-category), StrongREJECT, MT-Bench, XSTest, and the MMLU subset.

**Total experimental cells**: 4 models × 9 attack conditions × 5 defenses = **180 cells** (the 120-cell harmful-attack sweep + 60 benign-control cells), each scored on the eval suite. The headline is HarmBench ASR per cell; the other evals are secondary axes. Plus **24 LoRA fine-tunes** (4 models × {3 harmful + 3 benign} budgets). Per-model and total wall-clock on the laptop is in §6.7; `advsafe-benchmark --all` replaces the estimate with measured tok/s before launch.

**Statistical verdicts are executed, not promised.** `advsafe-report` runs a one-sided proportion test per inferential hypothesis (binomial for H1; pooled two-proportion for H2/H3), Bonferroni-corrects the family, and gates each CONFIRMED on BOTH the pre-registered effect threshold AND rejection at the corrected α (written to `multiple_comparisons.json`). H4 (DMV) is derived and H5 (CAT) descriptive; H6 (ACE) is deterministic — all three excluded from the inferential family.

**Quantization control (internal validity).** Because the accessibility claim runs every model in 4-bit, the first reviewer objection is that 4-bit inference *itself* could shift refusal behaviour and confound the attack effect. We pre-register a control: for at least the 8B and 14B models (which also fit in fp16 on the laptop), we report baseline and no-attack-control refusal/ASR in **both fp16 and 4-bit**, and check that the 4-bit↔fp16 baseline gap is small relative to the attack effect. The headline result is reported as the **within-quantization delta** (4-bit attacked − 4-bit baseline), so the measured safety loss is attributable to the attack, not to quantization; the quantization deltas themselves go in an appendix.

### 6.7 Execution Strategy: Local-First on a $3k Laptop (MLX)

The pipeline runs **entirely on a 64 GB M5 Pro MacBook** via a torch-free **MLX** backend that performs 4-bit QLoRA fine-tuning and inference natively on Apple Silicon. This is not a development convenience — it *is* the experiment: the accessibility thesis requires that the demonstration hardware be the consumer laptop. A PyTorch/CUDA path (bitsandbytes QLoRA on an A100) is retained behind a `--backend hf` flag purely as an optional reproduction route for readers without a Mac.

**Why MLX (not PyTorch-MPS).** Apple-Silicon QLoRA needs CUDA-only `bitsandbytes` under PyTorch; MLX does 4-bit LoRA natively and pulls in no torch (so it even runs on Python 3.14, which has no torch wheels yet). MLX is the only path that puts the 27B/32B flagships on the laptop. The backend is selected per run with `--backend mlx`; `backend="hf"` preserves the CUDA path.

**Phase 1 — Build + pilot (~1.5 weeks):** MLX backend, smoke test, and a single-model pilot (Llama 3.1 8B) end-to-end; lock judge calibration (GPT-4o-mini cross-checks), metrics, and figure templates. Cost: $0.

**Phase 2 — Full local sweep (~2-3 weeks of laptop compute):** all 24 LoRA fine-tunes (harmful + benign control) + the 180-cell evaluation matrix, in 4-bit on the laptop. Wall-clock is dominated by the 32B model and the DeepSeek-R1 chain-of-thought; `advsafe-benchmark --all` replaces the estimate with measured tok/s before committing. Run overnight / in the background on a primary machine. Cost: $0.

**Phase 3 — Analysis (concurrent):** results land locally as cells complete; `advsafe-report` builds the metric tables and figures on the same machine. Cost: $0.

**Total: ~3-4 weeks, $0 cloud** (vs the prior hybrid plan's ~$77 + 4-5 weeks). The single external dependency is the GPT-4o-mini judge used *only* for Llama-Guard-defended cells (a few dollars of API, required to avoid Guard-judging-Guard circularity); a fully-offline local-judge variant is available at the cost of reintroducing that correlation.

### 6.8 Test Plan & Quality Assurance

The credibility of the headline result depends on whether 400 evaluation cells produce *trustworthy* numbers. The following layered safeguards are committed to in the framework design — not as nice-to-haves, but as load-bearing components without which the paper does not get submitted.

> **Honesty note.** "Zero problems" is the aspiration; in practice no ML evaluation pipeline is bug-free. The goal is: (a) catch obvious bugs before the sweep, (b) be statistically rigorous so noise doesn't masquerade as signal, (c) be transparent enough that anyone can find and report what we missed.

#### 6.8.1 Code-Level Testing

**Unit tests** (`tests/unit/`, target ≥80% coverage):
- Every attack module: loads + applies + serializes correctly on a 100M-param toy model (cheap to run repeatedly in CI).
- Every defense module: filters known-bad inputs; passes known-good inputs; preserves output formatting.
- Every eval module: produces deterministic scores on a fixed reference output.
- Tokenizer/chat-template consistency: each model's chat template applied correctly. (This is the #1 silent bug in cross-model adversarial work.)

**Integration tests** (`tests/integration/`):
- End-to-end smoke test: Llama 3.1 8B → baseline eval → A1.10 attack → eval with each defense. Runs on every PR.
- Cross-platform parity: same smoke test on MPS (Mac CI runner) and CUDA (self-hosted GPU runner or skip-with-warning).

**Continuous integration**:
- pytest + mypy + ruff on every PR.
- Pre-commit hooks for formatting + lint.
- Smoke test gates the merge.

#### 6.8.2 Methodological Controls

**Judge robustness — the most important control.**
- Llama Guard 3 (local) is the primary judge.
- For a **stratified 200-prompt sample** across HarmBench categories, we cross-validate with GPT-4o-mini and report inter-judge agreement (Cohen's κ).
- **Llama Guard 3 is never used as both defense and judge for the same cell.** When D1/D2/D4 are active, judging is done by GPT-4o-mini for the affected cells. This breaks the circular evaluation hazard.
- Cells with judge disagreement are flagged and manually spot-checked.

**Generation determinism**:
- Fixed RNG seeds for every run, recorded in output JSON.
- Temperature = 0 (greedy) for evaluation generation.
- For diversity-dependent metrics: explicit `n=5` sampling at T=0.7, results reported with bootstrap CIs.

**Eval data hygiene**:
- HarmBench has documented train/test split — we use only the test set for evaluation.
- AdvBench prompts used in attack training are explicitly held out from eval (verified by prompt-hash diff before each run).
- StrongREJECT used for cross-validation; same disjoint-set guarantee.

**Defense fairness**:
- Every defense config evaluated on the same prompt set, in the same order, with the same RNG state.
- Llama Guard 3 input filter blocks at "S1–S14" categories (per the spec); we log both the binary decision and the predicted category for failure analysis.

#### 6.8.3 Statistical Rigor

**Per-cell confidence intervals**:
- Bootstrap 1000× over eval prompts within each cell → 95% CI on ASR.
- All tables and figures report point estimate + CI, never point estimates alone.

**Multiple-comparisons correction**:
- 400 cells means many possible comparisons. We **pre-register the three primary hypothesis tests (H1, H2, H3)** in `prereg.md` *before* the Week 3 sweep runs.
- For each primary test: Bonferroni (family-wise) or Benjamini–Hochberg (FDR) correction.
- All other comparisons explicitly labeled "exploratory" in the paper.

**Effect size, not just p-values**:
- Cohen's h for ASR proportion differences.
- Every defense comparison reported as Δ-ASR with 95% CI, not "p < 0.05."

**Power check**:
- HarmBench test = 240 prompts → minimum detectable difference of ~7–10% ASR at 80% power. Documented in `power_analysis.md`.

#### 6.8.4 Reproducibility

**Repo contents** (gitignored data; tracked everything else):
- `requirements.txt` (pinned versions) + `environment.yml` (conda)
- `Dockerfile` for cloud runs
- `seeds.json` — every RNG seed used in every experiment
- `hashes.json` — SHA256 of every input dataset file
- `manifest.json` — exact model version IDs (HuggingFace revision hashes)

**Two-machine verification**:
- The Week 2 pilot is run on **both** local M5 Pro (MPS) and a 1-hr A100 (CUDA). Results must agree within ASR ±2% before we proceed to the Week 3 sweep. If they don't agree → bug; debug before sweeping.

**Cell-level results storage**:
- Every cell's output saved as JSON: prompts, generations, judge verdicts, judge confidence, defense flags, timing. ~500MB total.
- Allows re-judging or re-analysis without re-running any model.

#### 6.8.5 Failure-Mode Spot Checks

**Daily during the sweep**:
- Pull random sample of 5 prompts × 5 cells; eyeball outputs. Does the judge verdict look right? Is the defense actually triggering?
- Track ASR per HarmBench category — if any category shows suspicious values (100% or 0% across all conditions), flag for inspection.
- Auto-alert on anomalies: ASR change > 50pp between adjacent budget levels, judge disagreement > 30% in any cell, generation latency outliers.

**Pre-publication audit**:
- Pick 10 cells at random; manually review 20 prompts each (200 total).
- Rate: judge agreement, defense correctness, generation sanity.
- If <90% agreement with the automated pipeline → bug; re-investigate before submission.

**Adversarial own-review**:
- Before submission, dedicated 1-day pass actively trying to break our own results: re-run with different seeds, audit for off-by-ones, verify dataset hashes, sanity-check unusually clean curves.

#### 6.8.6 Pre-registration

The three primary hypotheses (H1, H2, H3) are pre-registered in `prereg.md` with specific success/failure criteria **before the Week 3 sweep runs**. This prevents post-hoc p-hacking. The commit hash of the prereg is included in the paper.

#### 6.8.7 What We Are NOT Promising

- **Replicability across model versions**: we pin to specific HF revision hashes. If Meta or Alibaba updates a model, our numbers may not reproduce on the new version.
- **Replicability across hardware classes**: MPS and CUDA give bit-equivalent outputs only with `deterministic=True`; we use that for evaluation but not training.
- **Bulletproof judge**: Llama Guard 3 is imperfect; we report its calibration vs GPT-4o-mini explicitly so readers can discount.
- **Generalization to closed models**: our findings apply to the four tested models. We avoid universal claims.

These are honest limitations of any work in this area; flagging them up front is better than being called out on them at review.

---

## 7. Deliverables

1. **`adversarial-safeguards/`** — open-source Python framework on GitHub:
   - `attacks/` — pluggable attack modules (LoRA, PAP, roleplay, no-attack control)
   - `defenses/` — pluggable defense layers (baseline, Llama Guard input/output, constitutional, combined)
   - `evals/` — HarmBench, StrongREJECT, MT-Bench, XSTest, MMLU wrappers
   - `analysis/` — **novel metric implementations (`novel_metrics.py`): SDF, DMV, CAT** + bootstrap statistical helpers
   - `runners/` — `advsafe-smoke`, `advsafe-validate`, `advsafe-preflight`, `advsafe-benchmark`, `advsafe-pilot`, `advsafe-sweep`, `advsafe-report` CLIs
2. **Research paper** (`paper/main.tex` LaTeX skeleton committed; ~8-10 pages workshop format). Targets: SoLaR @ NeurIPS, SafeAI @ AAAI, SaTML @ ICML workshop.
3. **Pre-registration** (`prereg.md`) — H1–H4 + decision rules locked before sweep launch.
4. **Fellowship application materials**: 1-page research summary + headline Pareto figure + novel-metric tables.
5. **Reproducibility kit**: environment manifest, dataset hashes, RNG seeds, model revision SHAs, config archive.
6. **`LAUNCH_CHECKLIST.md`** — step-by-step pre/during/post-launch procedure for the cloud sweep.
7. **Threat model document**: who would deploy this attack in the wild and what it would actually cost them.

---

## 8. Timeline (4–5 weeks, ~20–30 hrs/week)

### Week 1 — Setup & Framework Scaffolding (local)
- Set up Python environment (PyTorch + transformers + PEFT + bitsandbytes + accelerate).
- Download 4 models (~50GB total).
- Scaffold framework: attack/defense/eval base classes, YAML config system, runner CLI.
- Smoke test: one inference per model on a HarmBench prompt (MPS backend).
- Smoke test: one toy LoRA training run (10 examples, 1 epoch) on Llama 3.1 8B.
- **Milestone**: All 4 models load & generate locally; one LoRA cycle completes end-to-end on M5 Pro.

### Week 2 — Single-Model Pilot (local)
- Full pipeline on Llama 3.1 8B only: baseline → A1.100 attack → eval suite (HarmBench, StrongREJECT, MT-Bench, XSTest) → D0–D4 defense sweep.
- Establish judging pipeline (Llama Guard 3 local + GPT-4o-mini cross-check on 200-prompt sample).
- Lock metrics, output formats, figure prototypes.
- Verify framework runs identically on CUDA (rent 1× A100 for 1-2 hours, ~$3, sanity check).
- **Milestone**: One model, mini-version of the Pareto curve. Whole pipeline verified on both MPS and CUDA.

### Week 3 — Cloud Burst: Full Sweep (4× A100)
- Provision 4× A100 cluster (Lambda Labs or RunPod), ~$5-6/hr.
- Sync code + data; verify environment.
- Run all 12 LoRA fine-tunes in parallel (one per GPU): ~4-6 hours wall-clock.
- Run full evaluation matrix (400 cells): ~12-16 hours wall-clock.
- Sync results back to local; tear down cluster.
- **Wall-clock**: 1-2 days. **Cost**: $50-80.
- **Milestone**: Complete results dataframe.

### Week 4 — Analysis + Paper Draft (local)
- Generate all figures (Pareto frontier headline + supporting plots).
- Statistical analysis (bootstrap CIs over eval prompts).
- Draft paper: intro, methods, results, discussion, ethics, related work.
- **Milestone**: Paper draft + headline figure.

### Week 5 — Polish + Fellowship Materials (local)
- Code cleanup, README, reproducibility kit (env manifest, dataset hashes, seeds, configs).
- Paper revision pass(es).
- Fellowship application summary + figure pack.
- Publish framework to GitHub (attack code released; attacked weights NOT released).
- **Milestone**: Submission-ready package.

**Compression**: 3 weeks if full-time (pilot + sweep + writeup in tight succession). **Expansion**: 7-8 weeks if local-only (no cloud burst).

---

## 9. Budget

### Option A — Lambda Labs single A100 (recommended)

| Item | Cost |
|---|---|
| Local compute (M5 Pro MacBook for development, owned) | $0 |
| Week 2 CUDA sanity check (1× A100, ~1-2 hrs) | ~$3 |
| Week 3 cloud sweep (1× A100 80GB, ~60-72 hrs at $1.29/hr) | ~$77-93 |
| LLM judge API (GPT-4o-mini, ~5K calls + 200-prompt cross-validation) | ~$10-15 |
| Storage (cloud volume included; ~150GB) | $0 |
| HF Hub, GitHub | $0 |
| **Total** | **~$90-110** |

### Option B — RunPod Community (cheaper, less reliable)

Same workflow on a community A100 at ~$0.79/hr instead of $1.29/hr. Total: **~$60-75**. Risk: community providers can preempt instances.

### Option C — AWS (3-8× more expensive)

AWS only sells A100s in 8-GPU bundles (`p4de.24xlarge` at ~$40.97/hr) or A10G in `g5.48xlarge` at ~$16.29/hr. Documented for completeness in `docs/CLOUD_DEPLOYMENT.md` but **not recommended**.

| AWS instance | $/hr | 60-hr cost |
|---|---|---|
| p4de.24xlarge (8× A100 80GB) | $40.97 | ~$2,460 |
| g5.48xlarge (8× A10G 24GB) | $16.29 | ~$980 |

### Option D — 4× A100 parallel (fastest, ~$300, only if budget allows)

If $100 → $300 is fine, rent 4× A100 in parallel on Lambda Labs to compress the sweep to ~15-20 wall-clock hours. Same total compute, much faster wall-clock.

**Recommend Option A**. If RunPod community has A100s available, drop to Option B for ~$30 savings.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyTorch + MPS has rough edges on one of the 4 models | M | L | Pre-test Day 1; cloud sweep is the safety net (CUDA is the better-supported path) |
| Cloud spot instance interrupted mid-sweep | M | M | Use on-demand (not spot) for the Week 3 burst; checkpoint LoRA adapters after each fine-tune |
| LoRA fine-tunes don't converge as expected | L | M | Pilot in Week 2 verifies hyperparams on Llama 3.1 8B before sweep |
| Llama Guard 3 as judge has noisy/biased verdicts | H | M | Cross-validate on 200-prompt sample with GPT-4o-mini; report agreement; 2-judge majority on edge cases |
| Defenses produce so much over-refusal they're undeployable | M | L | This is itself a finding — report XSTest deltas as headline |
| Publication ethics concerns | L | H | Use only published datasets; do not release attacked weights; follow Qi et al. precedent; pre-submit to model developers for review |
| Headline finding is "Qi et al. replicates, again" | M | M | Defense Pareto + Chinese panel + accessibility angles remain novel; reframe around whichever shows strongest signal |
| Timeline slip (4→6 weeks) | M | L | D3/D4 and capability evals are descope-able first; cloud sweep can be re-run cheaply if needed |
| Cloud cost overrun (estimated $60-80 → actual >$100) | L | L | Hard cap the cluster session at 24 hours; tear down explicitly after sweep |

---

## 11. Ethics & Responsible Research

**Datasets.** All harmful-content data is from existing published research datasets (HarmBench, AdvBench, BadLlama corpus). **No novel harmful content is generated by us.** Same standard precedent as Qi et al., Mazeika et al., Souly et al.

**Released artifacts.** Framework code will be open-sourced. **Attack-fine-tuned model weights will NOT be released.** We release configs and dataset references such that motivated researchers can reproduce, but not ready-to-use attacked weights.

**Disclosure.** Prior to public release, findings will be shared with model developers (Meta, Mistral, Alibaba, DeepSeek) with a 30-day window for response.

**Defensive framing.** The motivation is explicitly to inform better safeguard design and policy on open-weight release. This is documented in the paper's introduction and ethics section, and is the basis for the fellowship application.

**Dual-use consideration.** Every method we use is already published. We add measurement and comparative analysis, not new attack capabilities. The benefit (clarifying the defense Pareto frontier for the field) exceeds the marginal uplift to potential attackers (who already have access to these methods).

---

## 12. Why Anthropic Fellowship

Direct relevance to Anthropic's published research priorities:

- Anthropic has published on red-teaming methodology (Ganguli et al. 2022), Constitutional AI (Bai et al. 2022), and many-shot jailbreaking (Anil et al. 2024). This work sits squarely in that lineage.
- **Open-weight model safety** is a current policy frontier where Anthropic has taken public positions; empirical work here is directly actionable for policy and product reasoning.
- The **Chinese model panel** addresses a known gap — almost no published Western safety work systematically benchmarks Qwen, DeepSeek, or GLM under adversarial fine-tuning. The fellowship will give us institutional support for a comparative analysis that is currently underdone.
- The framework is **open-source and reusable**; other safety researchers can extend it (new models, new attacks, new defenses), amplifying the contribution.
- The **consumer-hardware demonstration** has direct policy implications for the open-weight release debate that Anthropic is actively engaged in.

---

## 13. Open Decisions

### Locked (v0.2)
- ✅ **Model panel**: Qwen 3 14B Instruct, DeepSeek-R1-Distill-Qwen-14B, Gemma 3 12B IT, Llama 3.1 8B Instruct.
- ✅ **Attack budget granularity**: 3 levels (10 / 100 / 1000 examples).
- ✅ **GCG**: excluded.
- ✅ **Execution strategy**: hybrid local development + cloud burst (4× A100 in Week 3).
- ✅ **Framework**: PyTorch + HuggingFace transformers + PEFT + bitsandbytes (portable MPS/CUDA).

### Locked (v0.3)
- ✅ **Authorship**: solo for now. Implications: no advisor pipeline; will seek informal feedback on draft from safety-research practitioners pre-submission; responsible-disclosure step goes directly to model developers without institutional intermediary.
- ✅ **Cloud provider**: **Lambda Labs primary** (1× A100 80GB @ $1.29/hr, 4 instances in parallel for the sweep). **RunPod Secure Cloud backup** if Lambda capacity is unavailable in Week 3. AWS explicitly ruled out — only sells A100s in 8-GPU bundles, making it ~8× more expensive than ML-specific clouds for our workload.

### Still to decide before Week 1
1. **Workshop target.** SoLaR @ NeurIPS (best fit, safety-focused), SafeAI @ AAAI, or ICML SaTML workshop? Determines paper length and polish target.
2. **Timeline anchor.** What's the actual fellowship deadline? Sets whether we run the 5-week or compressed 3-week version.

---

## 14. Appendix: Framework Architecture (preview)

```
adversarial-safeguards/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── models/             # one YAML per model (HF id, MLX path, tokenizer)
│   ├── attacks/            # one YAML per attack config
│   ├── defenses/           # one YAML per defense config
│   ├── evals/              # one YAML per eval
│   └── experiments/        # composed: model × attack × defense × eval
├── src/
│   ├── attacks/
│   │   ├── base.py         # AttackPlugin abstract base
│   │   ├── lora_finetune.py
│   │   ├── pap.py
│   │   └── roleplay.py
│   ├── defenses/
│   │   ├── base.py         # DefensePlugin abstract base
│   │   ├── llama_guard_input.py
│   │   ├── llama_guard_output.py
│   │   └── constitutional_prompt.py
│   ├── evals/
│   │   ├── harmbench.py
│   │   ├── strongreject.py
│   │   ├── mt_bench.py
│   │   ├── mmlu.py
│   │   └── xstest.py
│   ├── runners/
│   │   ├── run_experiment.py    # main entry point
│   │   └── orchestrator.py
│   └── analysis/
│       ├── figures.py
│       └── statistics.py
├── data/                   # gitignored; pulled via scripts
├── results/                # gitignored; raw experiment outputs
└── paper/
    ├── main.tex
    ├── figures/
    └── references.bib
```

Each attack/defense/eval inherits a small abstract interface (e.g., `AttackPlugin.run(model, dataset) -> attacked_model_or_prompts`). Adding a new attack = one file + one YAML. This is the "plugin" architecture.

---

*End of v0.1 proposal. Open to substantial revision.*
