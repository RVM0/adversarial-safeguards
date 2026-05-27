"""Abstract base class for defenses + registry.

A *defense* is a deployment-time intervention that:
  - filters the prompt before it reaches the model (input filter), OR
  - filters or rewrites the response before it reaches the user (output filter), OR
  - modifies the system prompt the model sees (constitutional prompt).

Defenses do NOT change model weights; they wrap the inference pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from advsafe.types import DefenseDecision


@dataclass
class DefenseConfig:
    """Parsed YAML defense config (configs/defenses/*.yaml)."""

    name: str
    description: str = ""
    # For Llama Guard-based defenses
    guard_model_id: str | None = None
    guard_revision: str | None = None
    block_categories: list[str] = field(default_factory=list)
    # For constitutional defenses
    system_prompt_path: str | None = None
    system_prompt_inline: str | None = None
    # Generic
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefenseConfig":
        """Build a DefenseConfig from a dict, routing unknown keys to `extra`."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        extras = {k: v for k, v in data.items() if k not in known_fields}
        if "extra" in kwargs and extras:
            kwargs["extra"] = {**kwargs["extra"], **extras}
        elif extras:
            kwargs["extra"] = extras
        return cls(**kwargs)


class DefensePlugin(ABC):
    """Abstract base for defenses.

    A defense may implement any subset of:
      - `filter_input(prompt) -> DefenseDecision`
      - `filter_output(prompt, response) -> DefenseDecision`
      - `wrap_system_prompt(system_prompt) -> str`

    The base class provides identity implementations; subclasses override.
    """

    name: str = ""

    def __init__(self, config: DefenseConfig) -> None:
        self.config = config

    def filter_input(self, prompt: str) -> DefenseDecision:
        """Default: pass through unmodified."""
        return DefenseDecision(allowed=True, reason="no-op input filter")

    def filter_output(self, prompt: str, response: str) -> DefenseDecision:
        """Default: pass through unmodified."""
        return DefenseDecision(allowed=True, reason="no-op output filter")

    def wrap_system_prompt(self, system_prompt: str) -> str:
        """Default: pass through unmodified."""
        return system_prompt

    def setup(self) -> None:
        """One-time initialization hook (e.g., load Llama Guard 3 weights).

        Called by the runner before any filter_* calls. Default is no-op.
        """
        return None

    def teardown(self) -> None:
        """One-time cleanup hook (e.g., free GPU memory). Default is no-op."""
        return None


# ----- Registry -------------------------------------------------------------

_DEFENSE_REGISTRY: dict[str, type[DefensePlugin]] = {}


def register_defense(name: str):
    """Decorator: register a defense class under a canonical name."""

    def _decorator(cls: type[DefensePlugin]) -> type[DefensePlugin]:
        if name in _DEFENSE_REGISTRY:
            raise ValueError(f"Defense '{name}' is already registered")
        cls.name = name
        _DEFENSE_REGISTRY[name] = cls
        return cls

    return _decorator


def get_defense(name: str) -> type[DefensePlugin]:
    if name not in _DEFENSE_REGISTRY:
        available = ", ".join(sorted(_DEFENSE_REGISTRY))
        raise KeyError(f"Defense '{name}' not registered. Available: {available}")
    return _DEFENSE_REGISTRY[name]


def list_defenses() -> list[str]:
    return sorted(_DEFENSE_REGISTRY)


def autoload() -> None:
    """Import concrete defenses so they self-register."""
    from advsafe.defenses import (  # noqa: F401
        baseline,
        combined,
        constitutional_prompt,
        llama_guard_input,
        llama_guard_output,
    )
