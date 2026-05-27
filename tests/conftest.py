"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure we never accidentally try to authenticate to HF Hub during tests
os.environ.setdefault("HF_HUB_OFFLINE", "1")


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
