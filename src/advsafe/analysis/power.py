"""Power analysis for the pre-registered hypotheses.

Implements the formal power calculations referenced in
`docs/STATISTICAL_RIGOR.md` §1. Each function answers: given the sample
size we plan to collect, what effect sizes can we detect at our chosen
α and β?

These are calculations the reviewer would want to see in the paper's
methods section. The functions can also be used to plan future runs at
different sample sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerResult:
    """Result of a power analysis."""

    hypothesis: str
    n: int
    alpha: float
    beta: float
    null_value: float
    minimum_detectable_effect: float
    interpretation: str


def _z_alpha(alpha: float, two_sided: bool = True) -> float:
    """Critical Z value for a given alpha. Approximation; for α in [0.001, 0.10]."""
    one_sided = {
        0.10: 1.282,
        0.05: 1.645,
        0.025: 1.960,
        0.0125: 2.241,
        0.01: 2.326,
        0.0083: 2.394,
        0.005: 2.576,
        0.001: 3.090,
    }
    eff_alpha = alpha / 2 if two_sided else alpha
    nearest = min(one_sided.keys(), key=lambda a: abs(a - eff_alpha))
    return one_sided[nearest]


def _z_beta(beta: float) -> float:
    """Z value for power = 1 - beta. β=0.20 → Z=0.842 (80% power)."""
    table = {0.05: 1.645, 0.10: 1.282, 0.20: 0.842, 0.30: 0.524, 0.50: 0.0}
    nearest = min(table.keys(), key=lambda b: abs(b - beta))
    return table[nearest]


def power_for_proportion(
    n: int,
    null_value: float = 0.5,
    alpha: float = 0.0083,
    beta: float = 0.20,
    hypothesis_name: str = "binomial proportion",
) -> PowerResult:
    """Minimum detectable proportion above the null for a one-sample test."""
    z_a = _z_alpha(alpha, two_sided=False)
    z_b = _z_beta(beta)
    p0 = null_value
    se = math.sqrt(p0 * (1 - p0) / n)
    min_p1 = p0 + (z_a + z_b) * se
    mde = min_p1 - p0
    return PowerResult(
        hypothesis=hypothesis_name,
        n=n,
        alpha=alpha,
        beta=beta,
        null_value=p0,
        minimum_detectable_effect=mde,
        interpretation=(
            f"At n={n}, α={alpha}, power={1-beta:.0%}: detectable if true "
            f"proportion ≥ {min_p1:.3f} ({mde*100:.1f}pp above null of {p0:.0%})."
        ),
    )


def power_for_two_proportions(
    n1: int,
    n2: int | None = None,
    baseline_p: float = 0.5,
    alpha: float = 0.0083,
    beta: float = 0.20,
    hypothesis_name: str = "difference of two proportions",
) -> PowerResult:
    """Minimum detectable difference between two proportions."""
    if n2 is None:
        n2 = n1
    z_a = _z_alpha(alpha, two_sided=False)
    z_b = _z_beta(beta)
    se_null = math.sqrt(baseline_p * (1 - baseline_p) * (1 / n1 + 1 / n2))
    mde = (z_a + z_b) * se_null
    return PowerResult(
        hypothesis=hypothesis_name,
        n=n1 + n2,
        alpha=alpha,
        beta=beta,
        null_value=0.0,
        minimum_detectable_effect=mde,
        interpretation=(
            f"At n₁={n1}, n₂={n2}, α={alpha}, power={1-beta:.0%}: "
            f"detectable difference ≥ {mde*100:.1f}pp at baseline {baseline_p:.0%}."
        ),
    )


def power_for_correlation(
    n: int,
    alpha: float = 0.0083,
    beta: float = 0.20,
    hypothesis_name: str = "Cohen's kappa difference",
) -> PowerResult:
    """Minimum detectable correlation/agreement via Fisher's z-transform."""
    z_a = _z_alpha(alpha, two_sided=True)
    z_b = _z_beta(beta)
    se_z = 1 / math.sqrt(n - 3) if n > 3 else float("inf")
    mde_z = (z_a + z_b) * se_z
    mde_r = math.tanh(mde_z)
    return PowerResult(
        hypothesis=hypothesis_name,
        n=n,
        alpha=alpha,
        beta=beta,
        null_value=0.0,
        minimum_detectable_effect=mde_r,
        interpretation=(f"At n={n}, α={alpha}, power={1-beta:.0%}: detectable κ ≥ {mde_r:.3f}."),
    )


def prereg_power_table() -> list[PowerResult]:
    """Power analysis for H1-H6 at the actual sample sizes we'll use."""
    return [
        power_for_proportion(
            n=240,
            null_value=0.50,
            alpha=0.0083,
            beta=0.20,
            hypothesis_name="H1: ASR > 50% on HarmBench (per cell)",
        ),
        power_for_two_proportions(
            n1=240,
            n2=240,
            baseline_p=0.5,
            alpha=0.0083,
            beta=0.20,
            hypothesis_name="H2: defense reclamation (per model)",
        ),
        power_for_proportion(
            n=240,
            null_value=0.5,
            alpha=0.0083,
            beta=0.20,
            hypothesis_name="H3: DeepSeek-R1 fragility > others",
        ),
        power_for_proportion(
            n=240,
            null_value=0.33,
            alpha=0.0083,
            beta=0.20,
            hypothesis_name="H4: dominant defense share > 50%",
        ),
        power_for_correlation(
            n=240,
            alpha=0.0083,
            beta=0.20,
            hypothesis_name="H5: within-family κ - cross-family κ > 0.20",
        ),
    ]


def print_power_table() -> None:
    print("Power analysis for pre-registered hypotheses")
    print("=" * 70)
    for r in prereg_power_table():
        print(f"\n{r.hypothesis}")
        print(f"  n={r.n}, α={r.alpha}, power={1-r.beta:.0%}")
        print(f"  MDE: {r.minimum_detectable_effect:.3f}")
        print(f"  → {r.interpretation}")
    print(
        "\nH6 (ACE): deterministic from model size + attack hyperparameters; "
        "no inferential power needed."
    )


if __name__ == "__main__":
    print_power_table()
