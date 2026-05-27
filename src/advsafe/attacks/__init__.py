"""Attack plugins.

Each attack inherits from `AttackPlugin` and lives in its own module.
Concrete attacks are registered via the `@register_attack` decorator.
"""

from advsafe.attacks.base import (
    AttackConfig,
    AttackPlugin,
    AttackType,
    get_attack,
    list_attacks,
    register_attack,
)

__all__ = [
    "AttackConfig",
    "AttackPlugin",
    "AttackType",
    "get_attack",
    "list_attacks",
    "register_attack",
]
