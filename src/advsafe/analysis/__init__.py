"""Analysis: figure generation, statistical helpers, and the novel metric suite.

The headline novel contribution is **Adversarial Compute Equivalence (ACE)**,
a cryptographic-security-style framing of the attack-defense compute economics.
SDF, DMV, CAT are supporting standardization metrics.
"""

from advsafe.analysis.ace import (
    ACEResult,
    ConditionalACEResult,
    ace_grid,
    adversarial_compute_equivalence,
    conditional_ace,
    inference_flops_per_query,
    training_flops,
)
from advsafe.analysis.figures import pareto_frontier_figure
from advsafe.analysis.novel_metrics import (
    CATResult,
    DMVResult,
    SDFParams,
    TransferabilityMatrix,
    cross_attack_transferability,
    defense_marginal_value,
    safeguard_decay_function,
    transferability_matrix,
)
from advsafe.analysis.power import (
    PowerResult,
    power_for_correlation,
    power_for_proportion,
    power_for_two_proportions,
    prereg_power_table,
)
from advsafe.analysis.statistics import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_diff_ci,
    bootstrap_proportion,
    cohens_h,
    cohens_kappa,
)

__all__ = [
    # Statistics
    "bootstrap_proportion",
    "bootstrap_diff_ci",
    "cohens_h",
    "cohens_kappa",
    "bonferroni",
    "benjamini_hochberg",
    # ACE (headline)
    "adversarial_compute_equivalence",
    "ACEResult",
    "conditional_ace",
    "ConditionalACEResult",
    "ace_grid",
    "training_flops",
    "inference_flops_per_query",
    # Power analysis
    "power_for_proportion",
    "power_for_two_proportions",
    "power_for_correlation",
    "prereg_power_table",
    "PowerResult",
    # Supporting metrics
    "safeguard_decay_function",
    "SDFParams",
    "defense_marginal_value",
    "DMVResult",
    "cross_attack_transferability",
    "CATResult",
    "transferability_matrix",
    "TransferabilityMatrix",
    # Figures
    "pareto_frontier_figure",
]
