"""A2 — Persuasion-based attack (PAP, Zeng et al. 2024).

Inference-time attack: wraps each eval prompt in one published persuasion strategy.
Does NOT modify model weights. The wrapping is applied at inference time by the runner;
this plugin just returns the transformation template.

Reference:
    Zeng et al. (2024) "How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking
    Persuasion to Challenge AI Safety by Humanizing LLMs."
"""

from __future__ import annotations

from pathlib import Path

from advsafe.attacks.base import AttackPlugin, AttackType, register_attack
from advsafe.attacks.templates import get_template, list_strategies
from advsafe.types import AttackResult, ModelHandle
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_STRATEGY = "logical_appeal"
DEFAULT_TEMPLATE_PATH = "data/pap_templates.json"


@register_attack("pap")
class PAPAttack(AttackPlugin):
    """Persuasion-based attack (inference-time prompt wrapping).

    Resolves the strategy template from the built-in published taxonomy
    (``attacks/templates.py``), or from ``config.template_path`` when that data override
    file exists. The template is validated to contain a ``{prompt}`` slot so the eval
    prompt is never silently dropped.
    """

    attack_type = AttackType.PROMPT_MOD

    def apply(self, model: ModelHandle) -> AttackResult:
        cfg = self.config
        strategy_id = cfg.template_id or DEFAULT_STRATEGY
        path = cfg.template_path or DEFAULT_TEMPLATE_PATH

        template = get_template("pap", strategy_id, path)
        source = "data-override" if Path(path).exists() else "built-in"
        logger.info(
            "PAP attack prepared",
            extra={"strategy_id": strategy_id, "source": source},
        )

        return AttackResult(
            attack_name=cfg.name,
            attack_type="PROMPT_MOD",
            attack_prompt_transform=template,
            metadata={
                "strategy_id": strategy_id,
                "template_source": source,
                "available_strategies": list_strategies("pap"),
                "n_examples": 0,
            },
        )
