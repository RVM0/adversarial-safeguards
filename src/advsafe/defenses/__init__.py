"""Defense plugins.

Each defense filters input prompts and/or output responses at inference time.
Defenses do NOT modify model weights.
"""

from advsafe.defenses.base import (
    DefenseConfig,
    DefensePlugin,
    get_defense,
    list_defenses,
    register_defense,
)

__all__ = [
    "DefenseConfig",
    "DefensePlugin",
    "get_defense",
    "list_defenses",
    "register_defense",
]
