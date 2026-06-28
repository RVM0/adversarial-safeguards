"""Statistical helpers — bootstrap CIs, effect sizes, multiple-comparisons."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    confidence_level: float


def bootstrap_proportion(
    binary_outcomes: Iterable[int],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for a proportion (mean of 0/1 outcomes)."""
    x = np.asarray(list(binary_outcomes), dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, confidence_level)
    samples = rng.choice(x, size=(iterations, n), replace=True).mean(axis=1)
    alpha = (1 - confidence_level) / 2
    return BootstrapResult(
        point=float(x.mean()),
        ci_low=float(np.quantile(samples, alpha)),
        ci_high=float(np.quantile(samples, 1 - alpha)),
        confidence_level=confidence_level,
    )


def bootstrap_diff_ci(
    a: Iterable[int],
    b: Iterable[int],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap CI for the difference in proportions: P(a) - P(b)."""
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iterations):
        sa = rng.choice(aa, size=len(aa), replace=True)
        sb = rng.choice(bb, size=len(bb), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    alpha = (1 - confidence_level) / 2
    return BootstrapResult(
        point=float(aa.mean() - bb.mean()),
        ci_low=float(np.quantile(diffs, alpha)),
        ci_high=float(np.quantile(diffs, 1 - alpha)),
        confidence_level=confidence_level,
    )


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
    phi2 = 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    return float(phi1 - phi2)


def cohens_kappa(labels_a: Iterable[bool], labels_b: Iterable[bool]) -> float:
    """Cohen's kappa for two raters' binary judgments."""
    a = np.asarray(list(labels_a), dtype=int)
    b = np.asarray(list(labels_b), dtype=int)
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    p_observed = float((a == b).mean())
    p_a, p_b = float(a.mean()), float(b.mean())
    p_expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_expected == 1.0:
        return float("nan")
    return (p_observed - p_expected) / (1 - p_expected)


def bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[float, list[bool]]:
    """Bonferroni-correct a list of p-values; return (alpha_adj, rejections)."""
    n = len(p_values)
    alpha_adj = alpha / n if n else alpha
    return alpha_adj, [bool(p < alpha_adj) for p in p_values]


def benjamini_hochberg(p_values: list[float], q: float = 0.10) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR correction.

    Returns (adjusted_p_values, rejections).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return [], []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    # Re-order back
    out = np.empty_like(adj)
    out[order] = adj
    out = np.minimum(out, 1.0)
    return out.tolist(), [bool(a < q) for a in out]


def binom_test_greater(k: int, n: int, p0: float) -> float:
    """One-sided exact binomial p-value for H_a: true proportion > ``p0``.

    Returns P(X >= k) under X ~ Binomial(n, p0). Used for H1 (ASR > 0.50). n<=0 → 1.0.
    """
    if n <= 0:
        return 1.0
    from scipy.stats import binom

    return float(binom.sf(k - 1, n, p0))


def two_proportion_p_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided two-proportion z-test p-value for H_a: p1 > p2 (pooled variance).

    Used for H2 (attacked ASR > defended ASR) and H3 (DeepSeek ASR > others). Degenerate
    inputs (empty group, pooled proportion 0 or 1, zero SE) return 1.0 (not significant).
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    if p_pool in (0.0, 1.0):
        return 1.0
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 1.0
    from scipy.stats import norm

    return float(norm.sf((p1 - p2) / se))
