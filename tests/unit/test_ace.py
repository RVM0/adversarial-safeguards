"""Tests for the Adversarial Compute Equivalence (ACE) metric.

ACE is the headline novel contribution — borrows cryptographic computational-
security framing to characterize the attack-defense economic balance.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# FLOPs accounting
# ---------------------------------------------------------------------------


def test_training_flops_scales_with_params():
    from advsafe.analysis.ace import training_flops

    # Doubling parameters should double FLOPs
    base = training_flops(n_params=1e9, n_examples=100, seq_len=512)
    double = training_flops(n_params=2e9, n_examples=100, seq_len=512)
    assert abs(double - 2 * base) < 1e-3 * base


def test_training_flops_scales_with_examples():
    from advsafe.analysis.ace import training_flops

    a = training_flops(n_params=1e9, n_examples=100, seq_len=512)
    b = training_flops(n_params=1e9, n_examples=200, seq_len=512)
    assert abs(b - 2 * a) < 1e-3 * a


def test_training_flops_scales_with_epochs():
    from advsafe.analysis.ace import training_flops

    a = training_flops(n_params=1e9, n_examples=100, seq_len=512, epochs=1)
    b = training_flops(n_params=1e9, n_examples=100, seq_len=512, epochs=3)
    assert abs(b - 3 * a) < 1e-3 * a


def test_training_flops_kaplan_formula():
    """Sanity check against the published Kaplan 2020 6×P×T formula."""
    from advsafe.analysis.ace import training_flops

    n_params = 8e9
    n_examples = 100
    seq_len = 512
    epochs = 3
    expected = 6 * n_params * n_examples * seq_len * epochs
    actual = training_flops(n_params, n_examples, seq_len, epochs)
    assert abs(actual - expected) < 1e-9 * expected


def test_inference_flops_per_query_2pt_formula():
    from advsafe.analysis.ace import inference_flops_per_query

    n_params = 8e9
    seq_len = 512
    expected = 2 * n_params * seq_len
    actual = inference_flops_per_query(n_params, seq_len_in=seq_len)
    assert abs(actual - expected) < 1e-9 * expected


# ---------------------------------------------------------------------------
# ACE metric
# ---------------------------------------------------------------------------


def test_ace_basic_8b_100_examples():
    """Standard scenario: attack 8B model with 100 examples; defense by Llama Guard 8B."""
    from advsafe.analysis.ace import adversarial_compute_equivalence

    result = adversarial_compute_equivalence(
        target_model_params=8e9,
        n_attack_examples=100,
    )
    # Attack: 6 * 8e9 * 100 * 512 * 3 ≈ 7.37e15 FLOPs
    # Defense per query: 2 * 8e9 * 512 ≈ 8.19e12 FLOPs
    # Ratio ≈ 900; ACE ≈ log10(900) ≈ 2.95
    assert 7e15 < result.attack_flops < 8e15
    assert 7e12 < result.defense_flops_per_query < 9e12
    assert 2.8 < result.ace < 3.1
    assert "cheap" in result.interpretation


def test_ace_grows_with_attack_budget():
    """More examples → more attack FLOPs → higher ACE (defender wins more easily)."""
    from advsafe.analysis.ace import adversarial_compute_equivalence

    small = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=10)
    big = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=1000)
    assert big.ace > small.ace
    # Doubling examples adds log10(2) ≈ 0.30 to ACE
    a = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    b = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=200)
    assert abs((b.ace - a.ace) - math.log10(2)) < 0.01


def test_ace_grows_with_model_size():
    """Larger target → more attack FLOPs → higher ACE."""
    from advsafe.analysis.ace import adversarial_compute_equivalence

    small = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    big = adversarial_compute_equivalence(target_model_params=32e9, n_attack_examples=100)
    assert big.ace > small.ace
    # 4× params → log10(4) ≈ 0.60 added to ACE
    assert abs((big.ace - small.ace) - math.log10(4)) < 0.01


def test_ace_zero_attack_budget_returns_undefined():
    from advsafe.analysis.ace import adversarial_compute_equivalence

    result = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=0)
    assert result.ace == float("-inf")
    assert "undefined" in result.interpretation


def test_ace_interpretation_bands():
    from advsafe.analysis.ace import _interpret_ace

    assert "extremely cheap" in _interpret_ace(0.5).lower()
    assert "cheap" in _interpret_ace(2.0).lower()
    assert "moderate" in _interpret_ace(4.0).lower()
    assert "expensive" in _interpret_ace(6.0).lower()
    assert "very expensive" in _interpret_ace(8.0).lower()


def test_ace_queries_to_amortize_matches_ratio():
    from advsafe.analysis.ace import adversarial_compute_equivalence

    result = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    expected_ratio = result.attack_flops / result.defense_flops_per_query
    assert abs(result.queries_to_amortize - expected_ratio) < 1e-3 * expected_ratio
    # ACE should be log10 of this
    assert abs(result.ace - math.log10(expected_ratio)) < 1e-6


# ---------------------------------------------------------------------------
# Conditional ACE
# ---------------------------------------------------------------------------


def test_conditional_ace_perfect_defense_yields_infinity():
    """Defense fully catches the attack → effective ACE is infinite (attack worthless)."""
    from advsafe.analysis.ace import adversarial_compute_equivalence, conditional_ace

    raw = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    result = conditional_ace(raw, attack_asr=0.8, defended_asr=0.8)
    assert result.effective_ace == float("inf")
    assert result.net_harm_per_query == 0


def test_conditional_ace_no_defense_matches_raw_with_asr_correction():
    """No defense improvement → effective ACE = raw ACE - log10(attack ASR)."""
    from advsafe.analysis.ace import adversarial_compute_equivalence, conditional_ace

    raw = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    result = conditional_ace(raw, attack_asr=0.5, defended_asr=0.0)
    expected = raw.ace - math.log10(0.5)
    assert abs(result.effective_ace - expected) < 1e-6


def test_conditional_ace_partial_defense():
    from advsafe.analysis.ace import adversarial_compute_equivalence, conditional_ace

    raw = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    # Attack succeeds 70%, defense brings to 20% → net 50% harm rate
    result = conditional_ace(raw, attack_asr=0.7, defended_asr=0.2)
    assert result.net_harm_per_query == pytest.approx(0.5)
    # effective_ace = raw - log10(0.5) ≈ raw + 0.30
    assert result.effective_ace > raw.ace


# ---------------------------------------------------------------------------
# ACE grid
# ---------------------------------------------------------------------------


def test_ace_grid_covers_all_combinations():
    from advsafe.analysis.ace import ace_grid

    models = {"small-8b": 8e9, "big-32b": 32e9}
    budgets = [10, 100, 1000]
    grid = ace_grid(models, budgets)
    assert len(grid) == 6
    for model in models:
        for budget in budgets:
            assert (model, budget) in grid
            assert grid[(model, budget)].ace > 0


def test_ace_grid_monotonic_in_budget():
    from advsafe.analysis.ace import ace_grid

    grid = ace_grid({"m": 8e9}, [10, 100, 1000])
    aces = [grid[("m", b)].ace for b in [10, 100, 1000]]
    assert aces[0] < aces[1] < aces[2]
