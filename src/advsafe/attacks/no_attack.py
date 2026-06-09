"""No-attack control plugin.

Used for baseline cells where we measure the unmodified model. Cleaner than
overloading lora_finetune with n_examples=0; documents intent in the YAML.
"""

from __future__ import annotations

from advsafe.attacks.base import AttackPlugin, AttackType, register_attack
from advsafe.types import AttackResult, ModelHandle
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


@register_attack("no-attack")
class NoAttack(AttackPlugin):
    """Identity attack: returns the unmodified model.

    AttackType is technically WEIGHT_MOD (no prompt transformation), but the
    "modification" is a no-op. The runner detects the no-op via the absence of
    a checkpoint_path and skips adapter loading.
    """

    attack_type = AttackType.WEIGHT_MOD

    def apply(self, model: ModelHandle) -> AttackResult:
        logger.info("Applying NoAttack (baseline control)", extra={"model": model.config.name})
        return AttackResult(
            attack_name=self.config.name,
            attack_type="WEIGHT_MOD",
            checkpoint_path=None,  # Signals to the runner: no adapter to load
            metadata={"note": "Baseline / no-attack control"},
        )
