"""Abstract base for evaluations.

An *eval* knows:
  - How to load its benchmark dataset (`load_prompts`).
  - How to score model responses against a judge (`score`).

Concrete evals: HarmBench, StrongREJECT, MT-Bench, XSTest, MMLU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse


@dataclass
class EvalConfig:
    """Parsed YAML eval config (configs/evals/*.yaml)."""

    name: str
    dataset_path: str | None = None  # local path or HF dataset name
    split: str = "test"
    n_prompts: int | None = None  # if set, subsample; else use full set
    judge_name: str = "llama-guard-3"
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        """Build an EvalConfig from a dict, routing unknown keys to `extra`."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        extras = {k: v for k, v in data.items() if k not in known_fields}
        if "extra" in kwargs and extras:
            kwargs["extra"] = {**kwargs["extra"], **extras}
        elif extras:
            kwargs["extra"] = extras
        return cls(**kwargs)


class EvalPlugin(ABC):
    """Abstract base for evaluations."""

    name: str = ""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    @abstractmethod
    def load_prompts(self) -> list[EvalPrompt]:
        """Load the benchmark prompts. Should be deterministic given config.seed."""

    @abstractmethod
    def score(
        self,
        responses: list[GeneratedResponse],
        judge: JudgePlugin,
    ) -> tuple[EvalScore, list]:
        """Aggregate per-prompt judgements into an EvalScore with bootstrap CIs.

        Returns a 2-tuple of ``(EvalScore, verdicts)`` where ``verdicts`` is the
        list of per-prompt judge verdicts (empty for evals that don't use a
        judge, e.g. the heuristic utility/over-refusal scorers). The runner
        unpacks both — see ``run_experiment.run_cell``.
        """

    def setup(self) -> None:
        """Optional setup hook called once before scoring."""
        return None


# ----- Registry -------------------------------------------------------------

_EVAL_REGISTRY: dict[str, type[EvalPlugin]] = {}


def register_eval(name: str):
    def _decorator(cls: type[EvalPlugin]) -> type[EvalPlugin]:
        if name in _EVAL_REGISTRY:
            raise ValueError(f"Eval '{name}' is already registered")
        cls.name = name
        _EVAL_REGISTRY[name] = cls
        return cls

    return _decorator


def get_eval(name: str) -> type[EvalPlugin]:
    if name not in _EVAL_REGISTRY:
        available = ", ".join(sorted(_EVAL_REGISTRY))
        raise KeyError(f"Eval '{name}' not registered. Available: {available}")
    return _EVAL_REGISTRY[name]


def list_evals() -> list[str]:
    return sorted(_EVAL_REGISTRY)


def autoload() -> None:
    from advsafe.evals import (  # noqa: F401
        harmbench,
        mmlu,
        mt_bench,
        strongreject,
        xstest,
    )
