# Ethics & Responsible Research Statement

## Motivation

This project exists to inform the design of better safeguards on
open-weight language models. We do not seek to enable misuse, and we
believe the marginal uplift of this work to bad actors is small (every
attack method we use is already published in peer-reviewed venues) while
the benefit to defenders is meaningful (a clean Pareto-frontier
measurement of attack cost vs. defense effectiveness, on an under-studied
set of Chinese-origin models).

## Concrete commitments

1. **Datasets**: We use only published research datasets — HarmBench
   (Mazeika et al. 2024), AdvBench (Zou et al. 2023), StrongREJECT
   (Souly et al. 2024), MT-Bench (Zheng et al. 2023), XSTest
   (Röttger et al. 2024). For attack training pairs, we use the published
   BadLlama corpus. **We do not generate new harmful content.**

2. **Released artifacts**:
   - The framework code is open-sourced under MIT with a responsible-use
     notice in `LICENSE`.
   - **Fine-tuned attack model weights will NOT be released.**
   - Configs, dataset references, and reproduction scripts are released
     such that motivated researchers can replicate.

3. **Disclosure**: Findings will be shared with model developers (Meta,
   Mistral, Alibaba, DeepSeek, Google) with a 30-day window for response
   before public release of the paper.

4. **Defensive framing**: The paper introduction and conclusions
   explicitly frame the work as informing better safeguards. The Pareto
   frontier figure is the headline; it is constructed to be useful to
   defenders.

5. **Pre-registration**: Primary hypotheses are pre-registered in
   `prereg.md` *before* the cloud sweep runs. This prevents post-hoc
   selection of favorable findings.

## Dual-use consideration

Every attack method used (LoRA fine-tuning, persuasion-based prompts,
roleplay jailbreaks) is already published in peer-reviewed venues. We add
*measurement* and *comparative analysis* — not new offensive capabilities.

For the LoRA fine-tuning attack specifically, the dataset and methodology
are taken from Qi et al. (2023), which has been replicated extensively in
the year+ since publication. Anyone with access to a consumer GPU and a
weekend already has this capability; what is missing from the literature
is rigorous comparative measurement, which we provide.

## Limitations

- **Replicability across model versions**: We pin to specific HuggingFace
  revision SHAs. If a model is updated, our numbers may not reproduce.
- **Bulletproof judge**: Llama Guard 3 is imperfect. We report
  inter-judge agreement (κ) with GPT-4o-mini explicitly so readers can
  discount judge noise.
- **Generalization**: Findings apply to the four tested models. We avoid
  universal claims about "Chinese models" or "Western models" as a class.

## Who is this work for?

- **Safety researchers** designing defenses: the framework is reusable;
  add a new defense plugin and rerun.
- **Model developers**: the per-model breakdown identifies where each of
  the four models' safeguards is weakest under our attack suite.
- **Policy analysts**: the cost-of-attack measurement (consumer hardware,
  ~$60-100 of cloud) is a concrete data point for the open-weight
  release debate.
- **Not for** anyone seeking pre-attacked weights or a how-to guide for
  abuse. We do not publish those.

## Contact

For ethics-related questions or to report misuse of this code:
rohanvmehra4@gmail.com
