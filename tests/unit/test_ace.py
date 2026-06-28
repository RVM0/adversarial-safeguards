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


# ---------------------------------------------------------------------------
# Cost-anchored ACE (the primary laptop reading)
# ---------------------------------------------------------------------------


def test_laptop_cost_model_default_rate():
    """Default $3k laptop, 3yr straight-line + electricity ≈ $0.12/laptop-hour."""
    from advsafe.analysis.ace import LaptopCostModel

    model = LaptopCostModel()
    # capital: 3000 / (3 * 365.25 * 24) ≈ $0.114/hr; + $0.01 electricity ≈ $0.124/hr
    assert model.amortization_hours == pytest.approx(3 * 365.25 * 24)
    assert model.usd_per_laptop_hour == pytest.approx(3000 / (3 * 365.25 * 24) + 0.01, rel=1e-9)
    assert 0.11 < model.usd_per_laptop_hour < 0.14


def test_laptop_cost_model_is_adjustable():
    """A report can re-price under its own assumptions (cheaper laptop, no power)."""
    from advsafe.analysis.ace import LaptopCostModel

    cheap = LaptopCostModel(laptop_price_usd=1500.0, electricity_usd_per_hour=0.0)
    assert cheap.usd_per_laptop_hour == pytest.approx(1500 / (3 * 365.25 * 24), rel=1e-9)
    # Half the price, no electricity → strictly cheaper than the default.
    assert cheap.usd_per_laptop_hour < LaptopCostModel().usd_per_laptop_hour


def test_laptop_cost_model_zero_amortization_raises():
    from advsafe.analysis.ace import LaptopCostModel

    with pytest.raises(ValueError, match="amortization_hours"):
        _ = LaptopCostModel(amortization_years=0.0).usd_per_laptop_hour


def test_attacker_laptop_cost_hours_and_dollars():
    from advsafe.analysis.ace import LaptopCostModel, attacker_laptop_cost

    # One full hour of training → exactly one laptop-hour at the model's rate.
    hours, usd = attacker_laptop_cost(3600.0)
    assert hours == pytest.approx(1.0)
    assert usd == pytest.approx(LaptopCostModel().usd_per_laptop_hour, rel=1e-9)


def test_attacker_laptop_cost_scales_linearly_with_time():
    from advsafe.analysis.ace import attacker_laptop_cost

    h1, u1 = attacker_laptop_cost(600.0)
    h2, u2 = attacker_laptop_cost(1200.0)
    assert h2 == pytest.approx(2 * h1)
    assert u2 == pytest.approx(2 * u1)


def test_attacker_laptop_cost_negative_raises():
    from advsafe.analysis.ace import attacker_laptop_cost

    with pytest.raises(ValueError, match="non-negative"):
        attacker_laptop_cost(-1.0)


def test_cost_anchored_ace_primary_reading():
    """An 18-minute LoRA run on the $3k laptop: sub-hour, pocket change."""
    from advsafe.analysis.ace import cost_anchored_ace

    r = cost_anchored_ace(
        train_wall_clock_s=18 * 60,
        target_model_params=8e9,
        n_attack_examples=100,
    )
    assert r.attacker_laptop_hours == pytest.approx(0.3)
    assert r.attacker_usd < 0.10  # under the default amortization
    assert r.laptop_price_usd == 3000.0  # one-time capital surfaced for context
    assert "trivial" in r.interpretation


def test_cost_anchored_ace_secondary_flops_matches_flops_ace():
    """The secondary reading must equal the standalone FLOPs-ACE for the cell."""
    from advsafe.analysis.ace import adversarial_compute_equivalence, cost_anchored_ace

    r = cost_anchored_ace(
        train_wall_clock_s=1234.0,
        target_model_params=8e9,
        n_attack_examples=100,
    )
    flops_only = adversarial_compute_equivalence(target_model_params=8e9, n_attack_examples=100)
    # The FLOPs ratio is platform-invariant: measured time must not perturb it.
    assert r.ace_flops == pytest.approx(flops_only.ace)
    assert r.flops_ace.attack_flops == pytest.approx(flops_only.attack_flops)


def test_cost_anchored_ace_time_does_not_change_flops_ace():
    """Different measured laptop times → identical secondary FLOPs-ACE."""
    from advsafe.analysis.ace import cost_anchored_ace

    fast = cost_anchored_ace(
        train_wall_clock_s=60.0, target_model_params=8e9, n_attack_examples=100
    )
    slow = cost_anchored_ace(
        train_wall_clock_s=36000.0, target_model_params=8e9, n_attack_examples=100
    )
    # Primary (laptop cost) differs; secondary (FLOPs ratio) is identical.
    assert slow.attacker_laptop_hours > fast.attacker_laptop_hours
    assert slow.attacker_usd > fast.attacker_usd
    assert fast.ace_flops == pytest.approx(slow.ace_flops)


def test_cost_anchored_ace_interpretation_bands():
    from advsafe.analysis.ace import _interpret_laptop_cost

    assert "trivial" in _interpret_laptop_cost(0.5).lower()
    assert "working day" in _interpret_laptop_cost(4.0).lower()
    assert "about a day" in _interpret_laptop_cost(12.0).lower()
    assert "multiple days" in _interpret_laptop_cost(72.0).lower()
    assert "week" in _interpret_laptop_cost(400.0).lower()


def test_cost_anchored_ace_from_manifest_reads_measured_time():
    from advsafe.analysis.ace import cost_anchored_ace, cost_anchored_ace_from_manifest

    manifest = {
        "train_wall_clock_s": 900.0,
        "n_examples_actual": 100,
        "n_examples_requested": 100,
        "epochs": 3,
    }
    r = cost_anchored_ace_from_manifest(manifest, target_model_params=8e9)
    direct = cost_anchored_ace(
        train_wall_clock_s=900.0, target_model_params=8e9, n_attack_examples=100, attack_epochs=3
    )
    assert r.attacker_laptop_hours == pytest.approx(direct.attacker_laptop_hours)
    assert r.attacker_usd == pytest.approx(direct.attacker_usd)
    assert r.ace_flops == pytest.approx(direct.ace_flops)


def test_cost_anchored_ace_from_manifest_prefers_actual_budget():
    """When actual and requested budgets differ, the actual (trained) one wins."""
    from advsafe.analysis.ace import cost_anchored_ace_from_manifest

    manifest = {
        "train_wall_clock_s": 900.0,
        "n_examples_actual": 80,  # corpus only had 80 rows
        "n_examples_requested": 100,
        "epochs": 3,
    }
    r = cost_anchored_ace_from_manifest(manifest, target_model_params=8e9)
    assert r.n_attack_examples == 80


def test_cost_anchored_ace_from_manifest_missing_timing_raises():
    from advsafe.analysis.ace import cost_anchored_ace_from_manifest

    manifest = {"n_examples_actual": 100, "epochs": 3}  # no train_wall_clock_s
    with pytest.raises(KeyError, match="train_wall_clock_s"):
        cost_anchored_ace_from_manifest(manifest, target_model_params=8e9)


def test_cost_ace_grid_covers_cells_and_uses_per_model_params():
    from advsafe.analysis.ace import cost_ace_grid

    params = {"small-8b": 8e9, "big-32b": 32e9}
    measured = {
        ("small-8b", 100): 600.0,
        ("big-32b", 100): 3600.0,
    }
    grid = cost_ace_grid(params, measured)
    assert set(grid) == set(measured)
    # Bigger model trained longer → higher laptop cost AND higher FLOPs-ACE.
    assert (
        grid[("big-32b", 100)].attacker_laptop_hours > grid[("small-8b", 100)].attacker_laptop_hours
    )
    assert grid[("big-32b", 100)].ace_flops > grid[("small-8b", 100)].ace_flops
    # Each cell's secondary FLOPs-ACE used that model's params, not a shared default.
    assert grid[("big-32b", 100)].target_model_params == 32e9


def test_cost_ace_grid_unknown_model_raises():
    from advsafe.analysis.ace import cost_ace_grid

    with pytest.raises(KeyError, match="model_param_counts"):
        cost_ace_grid({"known": 8e9}, {("unknown", 100): 600.0})
