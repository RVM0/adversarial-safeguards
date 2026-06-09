"""Tests for the statistical helpers."""

from __future__ import annotations


def test_bootstrap_proportion_recovers_mean():
    from advsafe.analysis.statistics import bootstrap_proportion

    outcomes = [1] * 30 + [0] * 70
    result = bootstrap_proportion(outcomes, iterations=500, seed=42)
    assert abs(result.point - 0.30) < 1e-6
    assert result.ci_low <= result.point <= result.ci_high
    assert result.ci_low > 0.20
    assert result.ci_high < 0.40


def test_bootstrap_diff_ci_recovers_difference():
    from advsafe.analysis.statistics import bootstrap_diff_ci

    a = [1] * 60 + [0] * 40  # P(a) = 0.6
    b = [1] * 30 + [0] * 70  # P(b) = 0.3
    result = bootstrap_diff_ci(a, b, iterations=500, seed=42)
    assert abs(result.point - 0.30) < 1e-6
    assert result.ci_low > 0  # significantly different from 0


def test_cohens_h_basic():
    from advsafe.analysis.statistics import cohens_h

    # Equal proportions: h = 0
    assert abs(cohens_h(0.5, 0.5)) < 1e-9
    # Large difference: |h| > 0.5
    assert abs(cohens_h(0.1, 0.9)) > 0.5


def test_cohens_kappa_perfect_agreement():
    from advsafe.analysis.statistics import cohens_kappa

    a = [True, True, False, False, True]
    b = [True, True, False, False, True]
    assert abs(cohens_kappa(a, b) - 1.0) < 1e-9


def test_cohens_kappa_constant_rater_is_zero():
    from advsafe.analysis.statistics import cohens_kappa

    # When one rater is constant (always True), observed agreement equals the
    # agreement expected by chance, so kappa is exactly 0 — not positive.
    a = [True, False] * 50
    b = [True, True] * 50
    assert abs(cohens_kappa(a, b)) < 1e-12


def test_cohens_kappa_partial_agreement():
    from advsafe.analysis.statistics import cohens_kappa

    # Raters agree above chance but not perfectly → 0 < kappa < 1.
    # Per 5-element block: p_a=0.6, p_b=0.4, observed=0.8, expected=0.48,
    # kappa = (0.80 - 0.48) / (1 - 0.48) ≈ 0.615.
    a = [True, True, True, False, False] * 20
    b = [True, True, False, False, False] * 20
    k = cohens_kappa(a, b)
    assert 0 < k < 1
    assert abs(k - 0.615) < 0.01


def test_bonferroni():
    from advsafe.analysis.statistics import bonferroni

    pvals = [0.01, 0.02, 0.04]
    adj, rej = bonferroni(pvals, alpha=0.05)
    assert abs(adj - 0.05 / 3) < 1e-9
    assert rej == [True, False, False]


def test_benjamini_hochberg():
    from advsafe.analysis.statistics import benjamini_hochberg

    pvals = [0.001, 0.05, 0.5, 0.9]
    adj, rej = benjamini_hochberg(pvals, q=0.10)
    assert len(adj) == 4
    assert all(0 <= p <= 1 for p in adj)
    # Smallest p should be rejected
    assert rej[0] is True
