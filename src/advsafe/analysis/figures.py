"""Figure generation for the paper.

Headline figure: the attack-defense Pareto frontier.
Supporting figures: per-model breakdowns, utility vs safety trade-offs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pareto_frontier_figure(
    results_dir: Path,
    output_path: Path,
    models: list[str],
    attack_levels: list[int],  # e.g. [0, 10, 100, 1000]
    defenses: list[str],
) -> None:
    """Generate the headline 2x2 Pareto frontier figure.

    For each model, plot ASR (y) vs attack budget (x, log scale), one
    line per defense configuration. Faceted as a 2×2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(defenses)))

    for i, model in enumerate(models):
        ax = axes[i]
        for j, defense in enumerate(defenses):
            xs, ys, los, his = [], [], [], []
            for level in attack_levels:
                cell_id = f"{model}__{_attack_label(level)}__{defense}"
                score_path = results_dir / cell_id / "score.json"
                if not score_path.exists():
                    continue
                score = json.loads(score_path.read_text())
                xs.append(max(level, 1))  # avoid 0 on log scale
                ys.append(score.get("asr", 0.0))
                los.append(score.get("asr_ci_low", 0.0))
                his.append(score.get("asr_ci_high", 0.0))

            xs_arr = np.asarray(xs)
            ax.plot(xs_arr, ys, "o-", color=colors[j], label=defense, linewidth=2)
            ax.fill_between(xs_arr, los, his, color=colors[j], alpha=0.15)

        ax.set_xscale("log")
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Attack budget (LoRA training examples)")
        ax.set_ylabel("Attack Success Rate (HarmBench)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Attack–Defense Pareto Frontier (HarmBench ASR vs LoRA fine-tuning budget)",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _attack_label(n: int) -> str:
    if n == 0:
        return "no-attack"
    return f"lora-a1-{n}"
