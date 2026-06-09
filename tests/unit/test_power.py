"""Unit tests for the pre-registration power analysis (advsafe.analysis.power).

Pure-Python module (no ML stack), so these run everywhere — no `ml_stack` mark.
The assertions are invariant-based (monotonicity, recorded inputs, degenerate
limits) rather than brittle exact-value checks, except for the critical-value
lookup tables, which are exact by construction.
"""

from __future__ import annotations

from advsafe.analysis.power import (
    PowerResult,
    power_for_correlation,
    power_for_proportion,
    power_for_two_proportions,
    prereg_power_table,
)


def test_z_alpha_two_sided_halves_alpha():
    from advsafe.analysis.power import _z_alpha

    # Two-sided at 0.05 uses the 0.025 tail (Z=1.960); one-sided uses 0.05 (Z=1.645).
    assert _z_alpha(0.05, two_sided=True) == 1.960
    assert _z_alpha(0.05, two_sided=False) == 1.645


def test_z_beta_known_values():
    from advsafe.analysis.power import _z_beta

    assert _z_beta(0.20) == 0.842  # 80% power
    assert _z_beta(0.50) == 0.0  # 50% power → no margin over the null


def test_power_for_proportion_mde_positive_and_shrinks_with_n():
    small = power_for_proportion(n=50)
    large = power_for_proportion(n=500)
    assert small.minimum_detectable_effect > 0
    # More data → smaller detectable effect.
    assert large.minimum_detectable_effect < small.minimum_detectable_effect


def test_power_for_proportion_records_inputs():
    r = power_for_proportion(n=240, null_value=0.5, alpha=0.0083, beta=0.20)
    assert isinstance(r, PowerResult)
    assert (r.n, r.alpha, r.beta, r.null_value) == (240, 0.0083, 0.20, 0.5)


def test_power_for_two_proportions_defaults_n2_to_n1():
    r = power_for_two_proportions(n1=240)
    # The `n` field reports the combined sample size across both groups.
    assert r.n == 480
    assert r.minimum_detectable_effect > 0


def test_power_for_two_proportions_mde_shrinks_with_n():
    small = power_for_two_proportions(n1=50)
    large = power_for_two_proportions(n1=500)
    assert large.minimum_detectable_effect < small.minimum_detectable_effect


def test_power_for_correlation_degenerate_below_four_samples():
    # n <= 3 → infinite standard error on Fisher's z → MDE saturates at 1.0.
    r = power_for_correlation(n=3)
    assert r.minimum_detectable_effect == 1.0


def test_power_for_correlation_mde_in_range_and_shrinks():
    small = power_for_correlation(n=30)
    large = power_for_correlation(n=300)
    assert 0.0 < large.minimum_detectable_effect < small.minimum_detectable_effect < 1.0


def test_prereg_power_table_covers_h1_through_h5():
    table = prereg_power_table()
    assert len(table) == 5
    assert all(isinstance(r, PowerResult) for r in table)

    labels = " ".join(r.hypothesis for r in table)
    for h in ("H1", "H2", "H3", "H4", "H5"):
        assert h in labels

    # Every row is pre-registered at the Bonferroni-adjusted alpha and 80% power.
    assert all(r.alpha == 0.0083 for r in table)
    assert all(r.beta == 0.20 for r in table)
