"""Abstract base class for attacks + registry.

An *attack* either:
  - modifies model weights (e.g., LoRA fine-tuning), producing a checkpoint, OR
  - modifies prompts at inference time (e.g., persuasion templates, jailbreak
    prefixes), producing an attack-prompt transformation.

Every concrete attack inherits `AttackPlugin`, registers via `@register_attack`,
and implements `apply()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from advsafe.types import AttackResult, ModelHandle


class AttackType(str, Enum):
    """How an attack manifests.

    - WEIGHT_MOD: returns a modified model checkpoint (LoRA adapter or full FT).
    - PROMPT_MOD: returns an attack-prompt transformation applied at inference.
    """

    WEIGHT_MOD = "WEIGHT_MOD"
    PROMPT_MOD = "PROMPT_MOD"


@dataclass
class AttackConfig:
    """Parsed YAML attack config (configs/attacks/*.yaml).

    `attack_type` defaults to WEIGHT_MOD; the actual attack plugin can
    override this (e.g., PAP and roleplay are PROMPT_MOD).
    """

    name: str
    attack_type: AttackType = AttackType.WEIGHT_MOD
    # For WEIGHT_MOD attacks
    dataset_path: str | None = None
    n_examples: int | None = None
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    max_seq_len: int = 512
    # For PROMPT_MOD attacks
    template_path: str | None = None
    template_id: str | None = None
    # Generic
    seed: int = 0
    output_dir: str = "checkpoints"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttackConfig:
        """Build an AttackConfig from a dict, routing unknown keys to `extra`.

        This makes YAML configs forgiving: adding new keys for new attack
        plugins doesn't break the loader.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        extras = {k: v for k, v in data.items() if k not in known_fields}
        if "extra" in kwargs and extras:
            kwargs["extra"] = {**kwargs["extra"], **extras}
        elif extras:
            kwargs["extra"] = extras
        # attack_type from string
        if "attack_type" in kwargs and isinstance(kwargs["attack_type"], str):
            kwargs["attack_type"] = AttackType(kwargs["attack_type"])
        return cls(**kwargs)


class AttackPlugin(ABC):
    """Abstract base for attacks.

    Subclasses override `name`, `attack_type`, and `apply`.
    Use `@register_attack` to make the class discoverable.
    """

    name: str = ""
    attack_type: AttackType = AttackType.WEIGHT_MOD

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, model: ModelHandle) -> AttackResult:
        """Apply the attack.

        For WEIGHT_MOD: should fine-tune the model and return an `AttackResult`
        with `checkpoint_path` set to the saved adapter.

        For PROMPT_MOD: should NOT modify the model in-place; should return an
        `AttackResult` with `attack_prompt_transform` set.

        Args:
            model: The clean model to attack.

        Returns:
            AttackResult describing the attack output.
        """

    def output_path(self) -> Path:
        """Standard location for this attack's saved artifacts.

        Uses `output_dir` directly (does NOT append the attack name) — the
        config's `output_dir` is already specific to the attack instance.
        """
        return Path(self.config.output_dir)


# ----- Registry -------------------------------------------------------------

_ATTACK_REGISTRY: dict[str, type[AttackPlugin]] = {}


def register_attack(name: str):
    """Decorator: register an attack class under a canonical name."""

    def _decorator(cls: type[AttackPlugin]) -> type[AttackPlugin]:
        if name in _ATTACK_REGISTRY:
            raise ValueError(f"Attack '{name}' is already registered")
        cls.name = name
        _ATTACK_REGISTRY[name] = cls
        return cls

    return _decorator


def get_attack(name: str) -> type[AttackPlugin]:
    """Look up an attack class by name. Raises if not registered."""
    if name not in _ATTACK_REGISTRY:
        available = ", ".join(sorted(_ATTACK_REGISTRY))
        raise KeyError(f"Attack '{name}' not registered. Available: {available}")
    return _ATTACK_REGISTRY[name]


def list_attacks() -> list[str]:
    """Return all registered attack names."""
    return sorted(_ATTACK_REGISTRY)


# Import concrete attacks so they self-register on package import.
# (Each module imports its decorator and calls it at module-load time.)
def _autoload_attacks() -> None:
    # Lazy imports to avoid circular dependency.
    from advsafe.attacks import lora_finetune, no_attack, pap, roleplay  # noqa: F401


# Defer autoload to avoid surprises during testing.
def autoload() -> None:
    _autoload_attacks()
