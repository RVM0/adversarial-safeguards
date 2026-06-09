"""Shared pytest fixtures."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Ensure we never accidentally try to authenticate to HF Hub during tests
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked `ml_stack` when torch is not importable.

    The pure-Python core (analysis, configs, registries, types) runs anywhere.
    Tests that exercise the heavy ML plugins (LoRA fine-tune, Llama Guard, device
    helpers) need torch/transformers/peft; on a box without them (e.g. a laptop
    that can't install the pinned CUDA stack) we skip rather than fail. On CI and
    the cloud GPU box, torch is present, so these run normally.
    """
    if _HAS_TORCH:
        return
    skip_ml = pytest.mark.skip(reason="requires the ML stack (torch not importable)")
    for item in items:
        if "ml_stack" in item.keywords:
            item.add_marker(skip_ml)


@pytest.fixture
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test scratch directory."""
    return tmp_path / "out"
