"""Model configuration registry.

Models are declared in `configs/models/*.yaml`. This module provides loading
and lookup helpers. The YAML is the source of truth; this is just glue.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from advsafe.types import ModelConfig
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_DIR = Path("configs/models")


def load_model_config(path: str | Path) -> ModelConfig:
    """Parse a YAML model config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def get_model_config(name: str, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> ModelConfig:
    """Look up a model by short name (e.g. 'llama-3.1-8b')."""
    config_dir = Path(config_dir)
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        available = list_models(config_dir)
        raise KeyError(f"No model config '{name}' in {config_dir}. Available: {available}")
    return load_model_config(path)


def list_models(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> list[str]:
    """List all model config short names available in `config_dir`."""
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []
    return sorted(p.stem for p in config_dir.glob("*.yaml"))
