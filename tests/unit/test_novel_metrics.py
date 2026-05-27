"""Tests for the novel metric suite: SDF, DMV, CAT.

These metrics replaced an earlier draft (SHL/DC/RCVI) on critical review:
  - SDF fits a parametric sigmoid (captures shape, not one arbitrary point)
  - DMV decomposes per-defense reclamation + synergy (Shapley-ready, honest
    about coalition data gaps)
  - CAT measures per-prompt cross-model attack transferability (genuinely
    new measurement in LLM safety literature)
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ============================================================================
# Safeguard Decay Function (SDF)
# ============================================================================


def test_sdf_recovers_known_sigmoid():
    """Generate data from a known sigmoid; fit; verify parameters recovered."""
    from advsafe.analysis.novel_metrics import safeguard_decay_function

    R_0_true, R_inf_true, mu_true, sigma_true = 0.95, 0.05, 1.5, 0.5
    budgets = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000])

    def gen(N, R0, Rinf, mu, sigma):
        x = np.log10(N + 1)
        sig = 1.0 / (1.0 + np.exp(-(x - mu) / sigma))
        return Rinf + (R0 - Rinf) * (1 - sig)

    rates = gen(budgets, R_0_true, R_inf_true, mu_true, sigma_true)
    result = safeguard_decay_function(budgets.tolist(), rates.tolist())

    assert result.converged
    assert abs(result.R_0 - R_0_true) < 0.05
    assert abs(result.R_inf - R_inf_true) < 0.05
    assert abs(result.mu - mu_true) < 0.20
    assert result.r_squared > 0.99


def test_sdf_predict_round_trip():
    from advsafe.analysis.novel_metrics import safeguard_decay_function

    budgets = [0, 10, 100, 1000, 10000]
    rates = [0.9, 0.7, 0.4, 0.15, 0.05]
    result = safeguard_decay_function(budgets, rates)
    # Prediction at midpoint budget should be roughly halfway between R_0 and R_inf
    mid = (result.R_0 + result.R_inf) / 2
    predicted_at_mid_budget = result.predict(result.characteristic_budget())
    assert abs(predicted_at_mid_budget - mid) < 0.05


def test_sdf_requires_4_points():
    from advsafe.analysis.novel_metrics import safeguard_decay_function

    with pytest.raises(ValueError):
        safeguard_decay_function([0, 100, 1000], [0.9, 0.5, 0.1])


def test_sdf_characteristic_budget_positive():
    from advsafe.analysis.novel_metrics import safeguard_decay_function

    result = safeguard_decay_function([0, 10, 100, 1000], [0.95, 0.80, 0.40, 0.05])
    assert result.characteristic_budget() > 0


# ============================================================================
# Defense Marginal Value (DMV)
# ============================================================================


def test_dmv_additive_defenses():
    """Three independent defenses, each cutting ASR by 1/3 of available range."""
    from advsafe.analysis.novel_metrics import defense_marginal_value

    # baseline 1.0; each defense individually → 0.7 (solo recl = 0.3 each)
    # additive prediction: combined = 1.0 - 0.9 = 0.1
    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.7, 0.7, 0.7],
        asr_combined=0.1,
    )
    assert all(abs(r - 1 / 3) < 0.01 for r in result.solo_shares)
    # full recl = 0.9; sum of solo recl = 0.9; synergy ≈ 0
    assert -0.05 <= result.synergy <= 0.05
    assert result.synergy_interpretation == "additive"


def test_dmv_redundant_defenses():
    """Each defense cuts to 0.7 alone but together still 0.7 — total redundancy."""
    from advsafe.analysis.novel_metrics import defense_marginal_value

    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.7, 0.7, 0.7],
        asr_combined=0.7,
    )
    assert result.synergy < -0.05
    assert "redundant" in result.synergy_interpretation


def test_dmv_synergistic_defenses():
    from advsafe.analysis.novel_metrics import defense_marginal_value

    # Each cuts to 0.9 alone; together cut to 0.2 — strongly super-additive
    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.9, 0.9, 0.9],
        asr_combined=0.2,
    )
    assert result.synergy > 0.05
    assert "synergistic" in result.synergy_interpretation


def test_dmv_solo_shares_sum_to_one():
    from advsafe.analysis.novel_metrics import defense_marginal_value

    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.5, 0.7, 0.8],
        asr_combined=0.3,
    )
    assert abs(sum(result.solo_shares) - 1.0) < 1e-9


def test_dmv_shapley_with_full_coalitions():
    """If all 2^3 coalitions provided, full Shapley is computed."""
    from advsafe.analysis.novel_metrics import defense_marginal_value

    # All three defenses identical and independent → equal Shapley
    coalitions = {
        frozenset({"A"}): 0.7,
        frozenset({"B"}): 0.7,
        frozenset({"C"}): 0.7,
        frozenset({"A", "B"}): 0.49,
        frozenset({"A", "C"}): 0.49,
        frozenset({"B", "C"}): 0.49,
        frozenset({"A", "B", "C"}): 0.343,
    }
    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.7, 0.7, 0.7],
        asr_combined=0.343,
        coalition_asrs=coalitions,
    )
    assert result.shapley_values is not None
    # By symmetry all Shapley values equal
    s = result.shapley_values
    assert abs(s[0] - s[1]) < 1e-6
    assert abs(s[1] - s[2]) < 1e-6
    # Efficiency: sum of Shapley = v(full)
    assert abs(sum(s) - result.full_reclamation) < 1e-6


def test_dmv_shapley_missing_coalitions_returns_none():
    from advsafe.analysis.novel_metrics import defense_marginal_value

    # Incomplete coalitions — missing the pairs
    coalitions = {
        frozenset({"A"}): 0.7,
        frozenset({"B"}): 0.7,
        frozenset({"C"}): 0.7,
        frozenset({"A", "B", "C"}): 0.1,
    }
    result = defense_marginal_value(
        asr_baseline=1.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.7, 0.7, 0.7],
        asr_combined=0.1,
        coalition_asrs=coalitions,
    )
    assert result.shapley_values is None
    # Partial decomposition still populated
    assert result.full_reclamation > 0


def test_dmv_baseline_zero_safe():
    from advsafe.analysis.novel_metrics import defense_marginal_value

    result = defense_marginal_value(
        asr_baseline=0.0,
        defense_names=["A", "B", "C"],
        asr_individual_defenses=[0.0, 0.0, 0.0],
        asr_combined=0.0,
    )
    assert result.synergy_interpretation.startswith("undefined")


# ============================================================================
# Cross-Attack Transferability (CAT)
# ============================================================================


def test_cat_perfect_agreement():
    """Same per-prompt outcomes on both models → kappa = 1."""
    from advsafe.analysis.novel_metrics import cross_attack_transferability

    a = [1, 1, 0, 1, 0, 0, 1, 0]
    b = [1, 1, 0, 1, 0, 0, 1, 0]
    result = cross_attack_transferability("A", "B", a, b, iterations=200)
    assert abs(result.cohens_kappa - 1.0) < 1e-6


def test_cat_perfect_disagreement():
    """Inverted outcomes → kappa = -1."""
    from advsafe.analysis.novel_metrics import cross_attack_transferability

    a = [1, 1, 0, 1, 0, 0, 1, 0]
    b = [0, 0, 1, 0, 1, 1, 0, 1]
    result = cross_attack_transferability("A", "B", a, b, iterations=200)
    assert abs(result.cohens_kappa - (-1.0)) < 1e-6


def test_cat_independent_outcomes_approximate_zero():
    """Two truly independent random samples → kappa ≈ 0."""
    from advsafe.analysis.novel_metrics import cross_attack_transferability

    rng = np.random.default_rng(0)
    n = 1000
    a = rng.binomial(1, 0.5, n).tolist()
    b = rng.binomial(1, 0.5, n).tolist()
    result = cross_attack_transferability("A", "B", a, b, iterations=200)
    assert abs(result.cohens_kappa) < 0.10  # close to zero in large sample


def test_cat_lift_greater_than_one_when_predictive():
    """When B's success is conditional on A's, lift > 1."""
    from advsafe.analysis.novel_metrics import cross_attack_transferability

    # 100 prompts: 50 where A succeeded; of those, 40 also succeed on B (high lift)
    # Of the 50 where A failed: 10 succeed on B
    a = [1] * 50 + [0] * 50
    b = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    result = cross_attack_transferability("A", "B", a, b, iterations=200)
    # p_a = 0.5, p_b = 0.5, p_joint = 0.4, P(B|A) = 0.8, lift = 0.8 / 0.5 = 1.6
    assert result.lift > 1.5
    assert result.cohens_kappa > 0.5


def test_cat_length_mismatch_raises():
    from advsafe.analysis.novel_metrics import cross_attack_transferability

    with pytest.raises(ValueError):
        cross_attack_transferability("A", "B", [1, 0], [1, 0, 1], iterations=10)


def test_transferability_matrix_diagonal_is_one():
    from advsafe.analysis.novel_metrics import transferability_matrix

    success_by_model = {
        "A": [1, 1, 0, 1, 0],
        "B": [1, 0, 0, 1, 1],
        "C": [0, 1, 1, 0, 1],
    }
    matrix = transferability_matrix(success_by_model, iterations=100)
    grid = matrix.kappa_grid()
    for i in range(3):
        assert grid[i, i] == 1.0


def test_transferability_within_vs_cross_family():
    from advsafe.analysis.novel_metrics import transferability_matrix

    # A and B agree; C disagrees with both
    success_by_model = {
        "A": [1, 1, 0, 1, 0, 0, 1, 0],
        "B": [1, 1, 0, 1, 0, 0, 1, 0],
        "C": [0, 0, 1, 0, 1, 1, 0, 1],
    }
    matrix = transferability_matrix(success_by_model, iterations=100)
    within, cross = matrix.within_vs_cross_family({"A": "x", "B": "x", "C": "y"})
    assert within > cross
    assert abs(within - 1.0) < 0.1  # A and B perfectly agree
    assert cross < 0  # C disagrees with both
