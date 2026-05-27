"""Abstract base for judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from advsafe.types import GeneratedResponse, JudgeVerdict


@dataclass
class JudgeConfig:
    """Parsed YAML judge config."""

    name: str
    description: str = ""
    # Local-model judges
    hf_id: str | None = None
    revision: str | None = None
    # API-based judges
    api_model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    # Behavior
    temperature: float = 0.0
    max_new_tokens: int = 128
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgeConfig":
        """Build a JudgeConfig from a dict, routing unknown keys to `extra`."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        extras = {k: v for k, v in data.items() if k not in known_fields}
        if "extra" in kwargs and extras:
            kwargs["extra"] = {**kwargs["extra"], **extras}
        elif extras:
            kwargs["extra"] = extras
        return cls(**kwargs)


class JudgePlugin(ABC):
    """Abstract base for judges."""

    name: str = ""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    @abstractmethod
    def judge(self, response: GeneratedResponse) -> JudgeVerdict:
        """Return a JudgeVerdict for a single response."""

    def judge_batch(self, responses: list[GeneratedResponse]) -> list[JudgeVerdict]:
        """Default batch impl: sequential. Subclasses may override for parallelism."""
        return [self.judge(r) for r in responses]

    def setup(self) -> None:
        """Load weights / authenticate / etc. Called once before judging."""
        return None

    def teardown(self) -> None:
        return None


# ----- Registry -------------------------------------------------------------

_JUDGE_REGISTRY: dict[str, type[JudgePlugin]] = {}


def register_judge(name: str):
    def _decorator(cls: type[JudgePlugin]) -> type[JudgePlugin]:
        if name in _JUDGE_REGISTRY:
            raise ValueError(f"Judge '{name}' is already registered")
        cls.name = name
        _JUDGE_REGISTRY[name] = cls
        return cls

    return _decorator


def get_judge(name: str) -> type[JudgePlugin]:
    if name not in _JUDGE_REGISTRY:
        available = ", ".join(sorted(_JUDGE_REGISTRY))
        raise KeyError(f"Judge '{name}' not registered. Available: {available}")
    return _JUDGE_REGISTRY[name]


def list_judges() -> list[str]:
    return sorted(_JUDGE_REGISTRY)


def autoload() -> None:
    from advsafe.judges import llama_guard, openai_judge  # noqa: F401
