"""Adversarial Compute Equivalence (ACE) — the headline novel measurement.

Borrows the framing of cryptographic computational security and applies it
to LLM safeguards. The defender's job is not to make attacks *impossible*
(they aren't) but to make them *uneconomical* — the cost of mounting an
attack should exceed the value the attacker can extract.

We make this precise by counting FLOPs on both sides:

  - **Attacker's cost** is one-time: the FLOPs required to fine-tune a LoRA
    adapter with N harmful examples.
  - **Defender's cost** is per-query: the FLOPs to run a safety classifier
    (Llama Guard 3) on each user input/output.

The ratio between these — equivalently, the number of queries the defender
can serve before matching the attacker's one-time investment — characterizes
the attack--defense economic balance.

We define:

    ACE = log10(FLOPs_attack / FLOPs_defense_per_query)

which equals log10(queries_to_amortize). Interpretation:

  - ACE ≈ 2: defender's per-query overhead equals attacker's investment
    after ~100 queries served. Cheap attack, cheap defense; the attack
    "pays off" almost immediately.
  - ACE ≈ 5: 100,000 queries to break even. Attack pays off only at
    industrial scale.
  - Larger ACE = better for defender; smaller ACE = better for attacker.

The novelty is the framing itself: nobody in the LLM safety literature
has previously characterized the attack--defense relationship in
computational-security terms. ACE is a single number per cell that the
proposal recommends as a standard reporting unit.

### Formula references

Transformer FLOPs accounting (Kaplan et al. 2020; Hoffmann et al. 2022):
  - Training (forward + backward + activation recomputation): 6 × P × T
    where P = parameter count, T = total training tokens.
  - Inference (forward only): 2 × P × T per generated token (autoregressive)
    or 2 × P × T_in for prefill on T_in tokens.

For LoRA fine-tuning, the parameter count used for FLOPs is the FULL model
parameter count — gradients flow through all parameters even though only
the LoRA adapters are updated. So FLOPs scaling is essentially the same
as full fine-tuning, not just adapter-sized.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------


def training_flops(
    n_params: float,
    n_examples: int,
    seq_len: int,
    epochs: int = 3,
    flops_per_param_per_token: float = 6.0,
) -> float:
    """Estimate FLOPs for LoRA fine-tuning.

    Args:
        n_params: Total parameter count of the base model (not just adapters;
            gradients flow through everything).
        n_examples: Number of training examples (prompt-response pairs).
        seq_len: Per-example sequence length in tokens.
        epochs: Number of passes over the dataset.
        flops_per_param_per_token: 6 is the standard constant from
            Kaplan et al. 2020 (forward + backward + activations).

    Returns:
        Total FLOPs for the training run.

    Example:
        >>> flops = training_flops(8e9, 100, 512, epochs=3)
        >>> # 6 * 8e9 * 100 * 512 * 3 = 7.37e15 FLOPs ≈ 7.4 PFLOPs
        >>> assert 7e15 < flops < 8e15
    """
    total_tokens = n_examples * seq_len * epochs
    return flops_per_param_per_token * n_params * total_tokens


def inference_flops_per_query(
    n_params: float,
    seq_len_in: int = 512,
    seq_len_out: int = 0,
    flops_per_param_per_token: float = 2.0,
) -> float:
    """Estimate FLOPs for one inference call (prefill + optional generation).

    For a safety classifier (Llama Guard) we typically only need prefill
    (the classification head produces 'safe'/'unsafe' in one forward).
    For a target LLM responding to a user query, we need prefill + generation.

    Args:
        n_params: Parameter count of the model.
        seq_len_in: Input length (prefill tokens).
        seq_len_out: Output length (decoded tokens). 0 for classifier.
        flops_per_param_per_token: 2 (forward only) — standard constant.

    Returns:
        FLOPs for one query.
    """
    return flops_per_param_per_token * n_params * (seq_len_in + seq_len_out)


# ---------------------------------------------------------------------------
# ACE metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ACEResult:
    """Adversarial Compute Equivalence for one (attack, defense) configuration.

    Attributes:
        attack_flops:           One-time FLOPs to mount the attack.
        defense_flops_per_query: Per-query FLOPs the defender pays.
        ace:                    log10(attack_flops / defense_flops_per_query).
                                Equivalently log10(queries_to_amortize).
        queries_to_amortize:    attack_flops / defense_flops_per_query.
        interpretation:         Human-readable summary band.
        attack_params:          Params count used in attack-FLOPs estimate.
        defense_params:         Params count used in defense-FLOPs estimate.
    """

    attack_flops: float
    defense_flops_per_query: float
    ace: float
    queries_to_amortize: float
    interpretation: str
    attack_params: float
    defense_params: float
    attack_n_examples: int
    attack_seq_len: int
    attack_epochs: int


def _interpret_ace(ace: float) -> str:
    """Map ACE value to a human-readable interpretation band.

    Bands:
        ACE < 1:    extremely cheap attack (amortizes in <10 queries)
        1 ≤ ACE < 3: cheap attack (10–1000 queries)
        3 ≤ ACE < 5: moderate attack (1k–100k queries)
        5 ≤ ACE < 7: expensive attack (100k–10M queries)
        ACE ≥ 7:    very expensive attack (>10M queries)
    """
    if ace < 1:
        return "extremely cheap attack (amortizes in <10 queries)"
    if ace < 3:
        return "cheap attack (10-1k queries to amortize)"
    if ace < 5:
        return "moderate cost attack (1k-100k queries to amortize)"
    if ace < 7:
        return "expensive attack (100k-10M queries to amortize)"
    return "very expensive attack (>10M queries to amortize)"


def adversarial_compute_equivalence(
    target_model_params: float,
    n_attack_examples: int,
    attack_seq_len: int = 512,
    attack_epochs: int = 3,
    defense_classifier_params: float = 8e9,
    defense_input_tokens: int = 512,
) -> ACEResult:
    """Compute the Adversarial Compute Equivalence for an attack-defense pair.

    Args:
        target_model_params:        Parameter count of the model being attacked.
        n_attack_examples:          Number of harmful fine-tuning examples.
        attack_seq_len:             Tokens per training example.
        attack_epochs:              Training epochs.
        defense_classifier_params:  Parameter count of the defense classifier
                                    (default: Llama Guard 3 8B).
        defense_input_tokens:       Average input length the defense classifies
                                    (prompt or response).

    Returns:
        ACEResult with all intermediate quantities + the headline ACE number.

    Example:
        >>> # Attacking Llama 3.1 8B with 100 harmful examples, defended by Llama Guard 3 8B
        >>> result = adversarial_compute_equivalence(
        ...     target_model_params=8e9,
        ...     n_attack_examples=100,
        ... )
        >>> # Attack: 6 * 8e9 * 100 * 512 * 3 = 7.37e15 FLOPs
        >>> # Defense per query: 2 * 8e9 * 512 = 8.19e12 FLOPs
        >>> # ratio = 900, ACE ≈ 2.95 ("cheap attack")
        >>> assert 2.8 < result.ace < 3.1
    """
    if n_attack_examples <= 0:
        # No-attack baseline — ACE undefined
        return ACEResult(
            attack_flops=0.0,
            defense_flops_per_query=inference_flops_per_query(
                defense_classifier_params, defense_input_tokens
            ),
            ace=float("-inf"),
            queries_to_amortize=0.0,
            interpretation="no attack (ACE undefined)",
            attack_params=target_model_params,
            defense_params=defense_classifier_params,
            attack_n_examples=n_attack_examples,
            attack_seq_len=attack_seq_len,
            attack_epochs=attack_epochs,
        )

    attack_flops = training_flops(
        n_params=target_model_params,
        n_examples=n_attack_examples,
        seq_len=attack_seq_len,
        epochs=attack_epochs,
    )
    defense_flops = inference_flops_per_query(
        n_params=defense_classifier_params,
        seq_len_in=defense_input_tokens,
    )

    if defense_flops <= 0:
        ace = float("inf")
        queries = float("inf")
    else:
        ratio = attack_flops / defense_flops
        queries = ratio
        ace = math.log10(ratio) if ratio > 0 else float("-inf")

    return ACEResult(
        attack_flops=attack_flops,
        defense_flops_per_query=defense_flops,
        ace=ace,
        queries_to_amortize=queries,
        interpretation=_interpret_ace(ace),
        attack_params=target_model_params,
        defense_params=defense_classifier_params,
        attack_n_examples=n_attack_examples,
        attack_seq_len=attack_seq_len,
        attack_epochs=attack_epochs,
    )


# ---------------------------------------------------------------------------
# ACE-conditional-on-ASR (the policy-relevant version)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalACEResult:
    """ACE refined by the empirical effectiveness of the attack.

    Raw ACE doesn't know whether the attack actually worked. We refine by
    multiplying by attack effectiveness: an attack that's economically
    cheap but doesn't actually work is uninteresting.

    Attributes:
        ace_raw:           The pure FLOPs-ratio ACE (unweighted).
        attack_asr:        Measured ASR of the attack (0-1).
        defended_asr:      Measured ASR with the defense active (0-1).
        effective_ace:     ACE adjusted by net ASR gain (attack_asr - defended_asr).
                           "How many queries to amortize PER UNIT OF EFFECTIVE HARM."
        net_harm_per_query: The marginal harm probability the attack provides at
                            equilibrium (= attack_asr - defended_asr).
    """

    ace_raw: float
    attack_asr: float
    defended_asr: float
    effective_ace: float
    net_harm_per_query: float


def conditional_ace(
    ace_raw: ACEResult,
    attack_asr: float,
    defended_asr: float,
) -> ConditionalACEResult:
    """ACE conditioned on empirical attack effectiveness.

    A cheap attack that the defense fully catches is uninteresting. We
    weight ACE by the net harm probability per query (attack_asr - defended_asr).

    The effective_ace is the FLOPs ratio per unit of *net harm probability*:

        effective_ace = ace_raw - log10(net_harm_per_query)

    Equivalent to: queries-to-amortize per unit-of-harm-probability.

    Higher effective_ace = attack is expensive relative to its effective
    harm yield = defender wins economically.

    Args:
        ace_raw: The unconditional ACE result.
        attack_asr: Measured ASR of the attacked model with no defense.
        defended_asr: Measured ASR of the attacked model under the defense.
    """
    net_harm = max(0.0, attack_asr - defended_asr)
    # net_harm <= 0 → defender perfectly catches; the attack is economically worthless.
    effective_ace = float("inf") if net_harm <= 0 else ace_raw.ace - math.log10(net_harm)
    return ConditionalACEResult(
        ace_raw=ace_raw.ace,
        attack_asr=attack_asr,
        defended_asr=defended_asr,
        effective_ace=effective_ace,
        net_harm_per_query=net_harm,
    )


# ---------------------------------------------------------------------------
# Convenience: ACE matrix over the experimental grid
# ---------------------------------------------------------------------------


def ace_grid(
    model_param_counts: dict[str, float],
    attack_budgets: Iterable[int],
    defense_classifier_params: float = 8e9,
    attack_seq_len: int = 512,
    attack_epochs: int = 3,
    defense_input_tokens: int = 512,
) -> dict[tuple[str, int], ACEResult]:
    """Compute ACE over the model × attack-budget grid."""
    out = {}
    for model_name, params in model_param_counts.items():
        for budget in attack_budgets:
            out[(model_name, budget)] = adversarial_compute_equivalence(
                target_model_params=params,
                n_attack_examples=budget,
                attack_seq_len=attack_seq_len,
                attack_epochs=attack_epochs,
                defense_classifier_params=defense_classifier_params,
                defense_input_tokens=defense_input_tokens,
            )
    return out
