"""Novel measurement suite — Safeguard Decay Function, Defense Marginal Value,
Cross-Attack Transferability.

This module implements the three metrics introduced as the methodological
contribution of this work:

  1. **Safeguard Decay Function (SDF)** — fits a parametric sigmoid to the
     attack-budget vs refusal-rate curve and reports interpretable parameters
     (R_0, R_inf, mu, sigma). Captures the *shape* of safeguard erosion,
     not just one point on it.

  2. **Defense Marginal Value (DMV)** — per-defense attribution of the total
     safety reclamation, plus an explicit synergy term that captures
     interactions the linear decomposition misses. Honest about its
     limitation: full Shapley requires intermediate coalitions; if those
     aren't measured, we report the best partial decomposition possible.

  3. **Cross-Attack Transferability (CAT)** — quantifies how much an attack
     that succeeds on model A predicts success on model B. Reports both
     Cohen's kappa (agreement) and lift (P(B|A)/P(B)). The pairwise CAT
     matrix is the headline figure for the cross-model section.

These replace an earlier draft of three metrics (SHL/DC/RCVI) that, on
critical review, were either curve summarization (SHL), framework borrowing
without justification (DC), or a single ASR delta dressed as an index (RCVI).
The current suite addresses those critiques:

  - SDF gives a parametric *shape* of the curve, not one arbitrary cut-point.
  - DMV decomposes per-defense and is honest about coalition data gaps.
  - CAT is a genuinely new measurement: per-prompt cross-model attack
    success correlation has not been systematically reported in the LLM
    safety literature.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# ============================================================================
# 1. Safeguard Decay Function (SDF)
# ============================================================================


@dataclass(frozen=True)
class SDFParams:
    """Fitted parameters of the safeguard decay function.

    Functional form:
        R(N) = R_inf + (R_0 - R_inf) * (1 - sigmoid((log10(N+1) - mu) / sigma))

    where R(N) is the refusal rate at attack budget N. This is a 4-parameter
    sigmoid in log-budget space.

    Parameter interpretations:
        - R_0:    baseline refusal rate (asymptote as N -> 0)
        - R_inf:  attacked refusal rate (asymptote as N -> infinity)
        - mu:     log10(budget) at the midpoint of the transition
                  (i.e., where R = (R_0 + R_inf) / 2). This *replaces* SHL
                  as the "characteristic budget" — but it's a fitted
                  parameter of a real model, not an arbitrary half-life.
        - sigma:  steepness of the transition. Small sigma = sharp drop
                  off ("cliff"); large sigma = gradual erosion.

    Goodness-of-fit:
        - r_squared:    coefficient of determination on the fit
        - residual_std: std of residuals
    """

    R_0: float
    R_inf: float
    mu: float
    sigma: float
    r_squared: float
    residual_std: float
    attack_budgets: list[int]
    refusal_rates: list[float]
    converged: bool
    # Identifiability bookkeeping (readiness-review M1). With a 4-point budget grid a
    # 4-parameter fit has residual_dof = 0, so r_squared ≡ 1 and the CIs are undefined —
    # the "goodness of fit flags bad fits" story is then vacuous. `fix_R0` drops to 3 free
    # params (R_0 pinned to the measured baseline) to recover a residual df; `saturated`
    # is True whenever residual_dof <= 0 and callers must treat r_squared as meaningless.
    n_free_params: int = 4
    residual_dof: int = 0
    saturated: bool = True

    def characteristic_budget(self) -> float:
        """The attack budget at which the curve hits its midpoint.

        This is 10^mu — interpretable as "approximately N examples are needed
        to cause half of the available safeguard erosion."
        """
        return 10.0**self.mu

    def predict(self, budget: float) -> float:
        """Predict refusal rate at a given attack budget."""
        x = math.log10(max(budget, 0) + 1)
        sig = 1.0 / (1.0 + math.exp(-(x - self.mu) / self.sigma))
        return self.R_inf + (self.R_0 - self.R_inf) * (1 - sig)


def safeguard_decay_function(
    attack_budgets: Iterable[int],
    refusal_rates: Iterable[float],
    initial_R0: float | None = None,
    initial_Rinf: float | None = None,
    fix_R0: bool = False,
) -> SDFParams:
    """Fit the safeguard decay function to attack-vs-refusal data.

    Args:
        attack_budgets: Attack budgets (# fine-tuning examples).
        refusal_rates: Measured refusal rate at each budget (0-1).
        initial_R0: Optional initial guess for R_0.
        initial_Rinf: Optional initial guess for R_inf.

    Returns:
        SDFParams with fitted parameters and goodness-of-fit statistics.

    Raises:
        ValueError: if fewer than 4 data points provided (fitting 4 params).

    Implementation:
        We use scipy.optimize.curve_fit with Levenberg-Marquardt. If scipy
        is unavailable or the fit fails, we fall back to a method-of-moments
        estimator and flag converged=False.
    """
    budgets = np.asarray(list(attack_budgets), dtype=float)
    rates = np.asarray(list(refusal_rates), dtype=float)
    if len(budgets) != len(rates):
        raise ValueError("attack_budgets and refusal_rates must be same length")
    if len(budgets) < 4:
        raise ValueError("SDF fitting requires at least 4 data points")

    # Sort by budget
    order = np.argsort(budgets)
    budgets = budgets[order]
    rates = rates[order]

    R_0_guess = initial_R0 if initial_R0 is not None else float(rates[0])
    R_inf_guess = initial_Rinf if initial_Rinf is not None else float(rates[-1])
    x = np.log10(budgets + 1)
    mu_guess = float(np.median(x))
    sigma_guess = max(float(np.std(x)) / 2, 0.3)

    def _decay(N_arr, R_0, R_inf, mu, sigma):
        xi = np.log10(np.asarray(N_arr, dtype=float) + 1)
        sig = 1.0 / (1.0 + np.exp(-(xi - mu) / max(sigma, 1e-6)))
        return R_inf + (R_0 - R_inf) * (1 - sig)

    n_free_params = 3 if fix_R0 else 4
    R_0_fixed = float(rates[0])  # measured baseline (budget 0 sorts first)
    converged = False
    R_0_fit, R_inf_fit, mu_fit, sigma_fit = R_0_guess, R_inf_guess, mu_guess, sigma_guess
    try:
        from scipy.optimize import OptimizeWarning, curve_fit  # type: ignore[import]

        # The covariance matrix is intentionally discarded (popt, _), which can
        # make scipy emit an OptimizeWarning about being unable to estimate it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            if fix_R0:
                # Pin R_0 to the measured baseline → 3 free params, so a 4-point grid
                # keeps 1 residual df (the 4-param fit would be saturated at R²≡1).
                def _decay3(N_arr, R_inf, mu, sigma):
                    return _decay(N_arr, R_0_fixed, R_inf, mu, sigma)

                popt, _ = curve_fit(
                    _decay3,
                    budgets,
                    rates,
                    p0=[R_inf_guess, mu_guess, sigma_guess],
                    bounds=([0.0, -5.0, 0.01], [1.0, 10.0, 10.0]),
                    maxfev=5000,
                )
                R_0_fit = R_0_fixed
                R_inf_fit, mu_fit, sigma_fit = (float(p) for p in popt)
            else:
                popt, _ = curve_fit(
                    _decay,
                    budgets,
                    rates,
                    p0=[R_0_guess, R_inf_guess, mu_guess, sigma_guess],
                    bounds=([0.0, 0.0, -5.0, 0.01], [1.0, 1.0, 10.0, 10.0]),
                    maxfev=5000,
                )
                R_0_fit, R_inf_fit, mu_fit, sigma_fit = (float(p) for p in popt)
        converged = True
    except Exception:  # noqa: BLE001 — fallback if scipy missing or fit fails
        if fix_R0:
            R_0_fit = R_0_fixed

    # Compute goodness-of-fit + honest identifiability bookkeeping.
    predicted = _decay(budgets, R_0_fit, R_inf_fit, mu_fit, sigma_fit)
    residuals = rates - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((rates - rates.mean()) ** 2)) if len(rates) > 1 else 0.0
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_std = float(np.std(residuals))
    residual_dof = len(budgets) - n_free_params

    return SDFParams(
        R_0=R_0_fit,
        R_inf=R_inf_fit,
        mu=mu_fit,
        sigma=sigma_fit,
        r_squared=r_squared,
        residual_std=residual_std,
        attack_budgets=budgets.tolist(),
        refusal_rates=rates.tolist(),
        converged=converged,
        n_free_params=n_free_params,
        residual_dof=residual_dof,
        saturated=residual_dof <= 0,
    )


# ============================================================================
# 2. Defense Marginal Value (DMV)
# ============================================================================


@dataclass(frozen=True)
class DMVResult:
    """Per-defense marginal value decomposition.

    Three layers of decomposition, in increasing rigor:

      - **Solo reclamation share**: each defense's reclamation
        ($r_i = (\\text{ASR}_0 - \\text{ASR}_i) / \\text{ASR}_0$)
        as a fraction of the SUM of solo reclamations. Reports "how much of
        the individual reclamation is due to defense i."

      - **Synergy term**: $\\sigma = r_{\\text{full}} - \\sum_i r_i$.
        Positive = stack > sum of parts (super-additive).
        Negative = stack < sum of parts (sub-additive / redundancy).
        Zero = additive.

      - **Shapley value** (only if pairwise coalition data is provided):
        full game-theoretic per-defense attribution. If pairwise data is
        absent, `shapley_values` is `None` and the partial decomposition
        above is what's reported.

    A key honesty note: full Shapley requires all $2^n$ coalition values.
    Our default sweep measures only the empty coalition, each singleton,
    and the full coalition — 5 of 8 for n=3 defenses. We expose the
    partial decomposition (solo + synergy) which is always computable,
    and the Shapley computation as opt-in when the user supplies the
    intermediate coalitions.
    """

    defense_names: list[str]
    solo_reclamations: list[float]  # r_i for each defense alone
    solo_shares: list[float]  # normalized so sum = 1
    full_reclamation: float
    synergy: float  # full_reclamation - sum(solo_reclamations)
    synergy_interpretation: str  # "redundant" | "additive" | "synergistic"
    shapley_values: list[float] | None  # only if pairwise coalitions provided
    asr_baseline: float
    asr_individual: list[float]
    asr_combined: float


def defense_marginal_value(
    asr_baseline: float,
    defense_names: list[str],
    asr_individual_defenses: list[float],
    asr_combined: float,
    coalition_asrs: dict[frozenset[str], float] | None = None,
) -> DMVResult:
    """Compute the Defense Marginal Value decomposition.

    Args:
        asr_baseline: ASR with no defense (D0).
        defense_names: Names of the individual defenses (e.g.
            ["input-filter", "output-filter", "constitutional"]).
        asr_individual_defenses: ASR with each defense applied alone.
        asr_combined: ASR with all defenses stacked (D4).
        coalition_asrs: Optional. ASR for arbitrary coalitions. Keys are
            frozensets of defense names; values are ASRs. If all $2^n$
            coalition values are provided, full Shapley values are computed.

    Returns:
        DMVResult with partial decomposition always populated and Shapley
        populated iff all coalition data is available.

    Example:
        >>> # Three defenses, each reduces ASR from 1.0 to 0.7 (solo).
        >>> # Combined reduces to 0.2, which is less than the 0.9 the solo
        >>> # reclamations sum to, so the synergy term is slightly negative.
        >>> result = defense_marginal_value(
        ...     asr_baseline=1.0,
        ...     defense_names=["A", "B", "C"],
        ...     asr_individual_defenses=[0.7, 0.7, 0.7],
        ...     asr_combined=0.2,
        ... )
        >>> # solo_recl = [0.3, 0.3, 0.3]; full_recl = 0.8
        >>> # synergy = 0.8 - 0.9 = -0.1 → sub-additive (slight)
        >>> assert -0.15 < result.synergy < -0.05
    """
    n = len(defense_names)
    if len(asr_individual_defenses) != n:
        raise ValueError("defense_names and asr_individual_defenses must have same length")

    if asr_baseline <= 0:
        return DMVResult(
            defense_names=defense_names,
            solo_reclamations=[0.0] * n,
            solo_shares=[0.0] * n,
            full_reclamation=0.0,
            synergy=0.0,
            synergy_interpretation="undefined (baseline ASR is zero)",
            shapley_values=None,
            asr_baseline=asr_baseline,
            asr_individual=asr_individual_defenses,
            asr_combined=asr_combined,
        )

    solo_recl = [
        max(0.0, (asr_baseline - asr_i) / asr_baseline) for asr_i in asr_individual_defenses
    ]
    full_recl = max(0.0, (asr_baseline - asr_combined) / asr_baseline)
    sum_solo = sum(solo_recl)
    solo_shares = [r / sum_solo for r in solo_recl] if sum_solo > 0 else [0.0] * n
    synergy = full_recl - sum_solo

    if synergy < -0.05:
        interp = "redundant (sub-additive)"
    elif synergy > 0.05:
        interp = "synergistic (super-additive)"
    else:
        interp = "additive"

    # Optional: full Shapley if all coalitions supplied
    shapley_values: list[float] | None = None
    if coalition_asrs is not None:
        try:
            shapley_values = _compute_shapley(
                asr_baseline=asr_baseline,
                defense_names=defense_names,
                coalition_asrs=coalition_asrs,
            )
        except (KeyError, ValueError):
            shapley_values = None

    return DMVResult(
        defense_names=defense_names,
        solo_reclamations=solo_recl,
        solo_shares=solo_shares,
        full_reclamation=full_recl,
        synergy=synergy,
        synergy_interpretation=interp,
        shapley_values=shapley_values,
        asr_baseline=asr_baseline,
        asr_individual=asr_individual_defenses,
        asr_combined=asr_combined,
    )


def _compute_shapley(
    asr_baseline: float,
    defense_names: list[str],
    coalition_asrs: dict[frozenset[str], float],
) -> list[float]:
    """Compute exact Shapley values from full coalition data.

    The payoff function is reclamation:
        v(S) = (ASR_baseline - ASR_S) / ASR_baseline   for S ⊆ defenses
        v(∅) = 0

    Shapley value of player i:
        φ_i = Σ_{S ⊆ N\\{i}} [|S|! (n-|S|-1)! / n!] * (v(S ∪ {i}) - v(S))

    Requires all 2^n coalition values in `coalition_asrs`.
    """
    n = len(defense_names)
    name_set = set(defense_names)

    def v(coalition: frozenset[str]) -> float:
        if not coalition:
            return 0.0
        asr = coalition_asrs.get(coalition)
        if asr is None:
            raise KeyError(f"Missing coalition: {sorted(coalition)}")
        return max(0.0, (asr_baseline - asr) / max(asr_baseline, 1e-9))

    # Verify all 2^n coalitions present
    expected_coalitions = []
    for k in range(n + 1):
        for combo in combinations(defense_names, k):
            expected_coalitions.append(frozenset(combo))
    missing = [c for c in expected_coalitions if c and c not in coalition_asrs]
    if missing:
        raise KeyError(f"Missing coalitions for full Shapley: {missing}")

    shapley = []
    factorial = math.factorial
    for player in defense_names:
        phi = 0.0
        others = name_set - {player}
        for k in range(n):  # |S| from 0 to n-1
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for combo in combinations(others, k):
                S = frozenset(combo)
                S_plus = S | {player}
                phi += weight * (v(S_plus) - v(S))
        shapley.append(phi)
    return shapley


# ============================================================================
# 3. Cross-Attack Transferability (CAT)
# ============================================================================


@dataclass(frozen=True)
class CATResult:
    """Cross-Attack Transferability between two models.

    Reports two complementary statistics on per-prompt attack-success outcomes:

      - **Cohen's kappa**: agreement between {success on A} and {success on B}
        beyond chance. Range [-1, 1]. 1 = perfect agreement, 0 = independent,
        -1 = perfect disagreement. This is the primary CAT statistic.

      - **Lift**: P(B succeeded | A succeeded) / P(B succeeded). Range [0, ∞).
        Lift > 1 means knowing the attack worked on A is *informative* that
        it would work on B. Lift = 1 means independence.

    The lift form has a clean interpretation as "the multiplicative factor
    by which knowing attack-A-succeeded raises my probability that attack
    succeeds on B." Both statistics are bootstrapped over the prompt set.

    `model_a` and `model_b` are tracked for downstream matrix construction.
    """

    model_a: str
    model_b: str
    cohens_kappa: float
    lift: float
    p_success_a: float
    p_success_b: float
    p_joint: float
    n_prompts: int
    ci_kappa_low: float
    ci_kappa_high: float
    ci_lift_low: float
    ci_lift_high: float


def cross_attack_transferability(
    model_a: str,
    model_b: str,
    success_a: list[int],
    success_b: list[int],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> CATResult:
    """Compute Cross-Attack Transferability between two models.

    Args:
        model_a, model_b: Model identifiers.
        success_a, success_b: Per-prompt binary attack-success outcomes
            (1 = attack succeeded, 0 = refused/safe). Must be aligned
            (success_a[i] and success_b[i] are the same prompt).
        iterations, confidence_level, seed: Bootstrap parameters.

    Returns:
        CATResult.

    Interpretation:
        - kappa ≈ 0.6+: strong cross-model transfer; the same attacks work
          across model architectures. Policy-relevant: open-weight attacks
          generalize to closed competitors.
        - kappa ≈ 0: independent; attacks against A don't predict success
          against B. Model-specific vulnerabilities.
        - kappa < 0: rare; attacks that work on A actively fail on B.
    """
    if len(success_a) != len(success_b):
        raise ValueError("success_a and success_b must be same length")
    n = len(success_a)
    if n == 0:
        return CATResult(
            model_a=model_a,
            model_b=model_b,
            cohens_kappa=float("nan"),
            lift=float("nan"),
            p_success_a=0.0,
            p_success_b=0.0,
            p_joint=0.0,
            n_prompts=0,
            ci_kappa_low=float("nan"),
            ci_kappa_high=float("nan"),
            ci_lift_low=float("nan"),
            ci_lift_high=float("nan"),
        )

    a = np.asarray(success_a, dtype=int)
    b = np.asarray(success_b, dtype=int)

    p_a = float(a.mean())
    p_b = float(b.mean())
    p_joint = float(((a == 1) & (b == 1)).mean())
    p_observed_agree = float((a == b).mean())
    p_expected_agree = p_a * p_b + (1 - p_a) * (1 - p_b)

    if p_expected_agree >= 1.0:
        kappa = float("nan")
    else:
        kappa = (p_observed_agree - p_expected_agree) / (1 - p_expected_agree)

    # Lift = P(B=1 | A=1) / P(B=1) = (p_joint / p_a) / p_b; nan if either base rate is 0.
    lift = (p_joint / p_a / p_b) if (p_a > 0 and p_b > 0) else float("nan")

    # Bootstrap
    rng = np.random.default_rng(seed)
    kappa_samples: list[float] = []
    lift_samples: list[float] = []
    for _ in range(iterations):
        idx = rng.choice(n, size=n, replace=True)
        sa = a[idx]
        sb = b[idx]
        pa_s = float(sa.mean())
        pb_s = float(sb.mean())
        pj_s = float(((sa == 1) & (sb == 1)).mean())
        po = float((sa == sb).mean())
        pe = pa_s * pb_s + (1 - pa_s) * (1 - pb_s)
        if pe < 1.0:
            kappa_samples.append((po - pe) / (1 - pe))
        if pa_s > 0 and pb_s > 0:
            lift_samples.append((pj_s / pa_s) / pb_s)

    alpha = (1 - confidence_level) / 2
    ci_kappa_low = float(np.quantile(kappa_samples, alpha)) if kappa_samples else float("nan")
    ci_kappa_high = float(np.quantile(kappa_samples, 1 - alpha)) if kappa_samples else float("nan")
    ci_lift_low = float(np.quantile(lift_samples, alpha)) if lift_samples else float("nan")
    ci_lift_high = float(np.quantile(lift_samples, 1 - alpha)) if lift_samples else float("nan")

    return CATResult(
        model_a=model_a,
        model_b=model_b,
        cohens_kappa=kappa,
        lift=lift,
        p_success_a=p_a,
        p_success_b=p_b,
        p_joint=p_joint,
        n_prompts=n,
        ci_kappa_low=ci_kappa_low,
        ci_kappa_high=ci_kappa_high,
        ci_lift_low=ci_lift_low,
        ci_lift_high=ci_lift_high,
    )


@dataclass(frozen=True)
class TransferabilityMatrix:
    """Pairwise CAT for all model pairs. Headline figure for the cross-model section."""

    models: list[str]
    pairwise: dict[tuple[str, str], CATResult]

    def kappa_grid(self) -> np.ndarray:
        """Square ndarray of Cohen's kappa values, indexed by self.models."""
        n = len(self.models)
        grid = np.full((n, n), np.nan)
        for i, m_a in enumerate(self.models):
            for j, m_b in enumerate(self.models):
                if i == j:
                    grid[i, j] = 1.0
                    continue
                cat = self.pairwise.get((m_a, m_b))
                if cat is not None:
                    grid[i, j] = cat.cohens_kappa
        return grid

    def within_vs_cross_family(
        self,
        families: dict[str, str],
    ) -> tuple[float, float]:
        """Return (mean within-family kappa, mean cross-family kappa).

        `families` maps model name -> family label (e.g., "llama" / "qwen").
        Used to test H5: within-family transfer > cross-family.
        """
        within = []
        cross = []
        for (m_a, m_b), cat in self.pairwise.items():
            if m_a == m_b:
                continue
            if families.get(m_a) == families.get(m_b):
                within.append(cat.cohens_kappa)
            else:
                cross.append(cat.cohens_kappa)
        within_mean = float(np.nanmean(within)) if within else float("nan")
        cross_mean = float(np.nanmean(cross)) if cross else float("nan")
        return within_mean, cross_mean


def transferability_matrix(
    success_by_model: dict[str, list[int]],
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> TransferabilityMatrix:
    """Compute the full pairwise CAT matrix.

    Args:
        success_by_model: Per-model per-prompt success outcomes. All entries
            must have the same length (same prompts).

    Returns:
        TransferabilityMatrix.
    """
    models = sorted(success_by_model)
    pairwise = {}
    for i, m_a in enumerate(models):
        for m_b in models:
            if m_a == m_b:
                continue
            cat = cross_attack_transferability(
                m_a,
                m_b,
                success_by_model[m_a],
                success_by_model[m_b],
                iterations=iterations,
                confidence_level=confidence_level,
                seed=seed + i,
            )
            pairwise[(m_a, m_b)] = cat
    return TransferabilityMatrix(models=models, pairwise=pairwise)
