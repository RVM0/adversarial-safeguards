"""A3 — Roleplay / prompt-injection baseline.

Inference-time attack using publicly catalogued persona jailbreaks (DAN, AIM, developer
mode). This is the weakest of the three attacks and serves as the comparison floor: how
much harm slips through with zero attacker effort beyond pasting a prompt?

Does NOT modify model weights; the runner applies the template at inference time.
"""

from __future__ import annotations

from pathlib import Path

from advsafe.attacks.base import AttackPlugin, AttackType, register_attack
from advsafe.attacks.templates import get_template, list_strategies
from advsafe.types import AttackResult, ModelHandle
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = "dan"
DEFAULT_TEMPLATE_PATH = "data/roleplay_templates.json"


@register_attack("roleplay")
class RoleplayAttack(AttackPlugin):
    """Roleplay / prompt-injection attack (inference-time prompt wrapping)."""

    attack_type = AttackType.PROMPT_MOD

    def apply(self, model: ModelHandle) -> AttackResult:
        cfg = self.config
        template_id = cfg.template_id or DEFAULT_TEMPLATE
        path = cfg.template_path or DEFAULT_TEMPLATE_PATH

        template = get_template("roleplay", template_id, path)
        source = "data-override" if Path(path).exists() else "built-in"
        logger.info(
            "Roleplay attack prepared",
            extra={"template_id": template_id, "source": source},
        )

        return AttackResult(
            attack_name=cfg.name,
            attack_type="PROMPT_MOD",
            attack_prompt_transform=template,
            metadata={
                "template_id": template_id,
                "template_source": source,
                "available_templates": list_strategies("roleplay"),
            },
        )
