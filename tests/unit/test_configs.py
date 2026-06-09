"""Validate that every YAML config in configs/ is parseable and well-formed.

This is one of the most valuable tests: it catches typos, missing fields,
and accidentally-changed plugin names before the cloud sweep wastes hours
on a config error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _yaml_files(configs_dir: Path, sub: str) -> list[Path]:
    return sorted((configs_dir / sub).glob("*.yaml"))


@pytest.mark.parametrize(
    "subdir,required_keys",
    [
        ("models", ["name", "hf_id", "family"]),
        ("attacks", ["plugin"]),
        ("defenses", ["plugin"]),
        ("evals", ["plugin"]),
    ],
)
def test_config_required_keys(configs_dir: Path, subdir: str, required_keys: list[str]):
    files = _yaml_files(configs_dir, subdir)
    assert files, f"No YAMLs in configs/{subdir}/"
    for path in files:
        with path.open() as f:
            data = yaml.safe_load(f)
        for key in required_keys:
            assert key in data, f"{path.name} missing '{key}'"


def test_experiment_configs_parseable(configs_dir: Path):
    """Experiment configs are composite — should at least parse."""
    for path in _yaml_files(configs_dir, "experiments"):
        with path.open() as f:
            data = yaml.safe_load(f)
        assert "cells" in data, f"{path.name} missing 'cells'"
        assert isinstance(data["cells"], list), f"{path.name} 'cells' must be a list"
        assert data["cells"], f"{path.name} has empty 'cells' list"
        for i, cell in enumerate(data["cells"]):
            # Each cell must have a model (or inherit it from common)
            common = data.get("common", {})
            model = cell.get("model", common.get("model"))
            assert model, f"{path.name} cell {i} has no model"
            assert "attack" in cell, f"{path.name} cell {i} missing attack"
            assert "defense" in cell, f"{path.name} cell {i} missing defense"


def test_model_configs_reference_real_families(configs_dir: Path):
    """family must be one of our supported chat-template families."""
    supported_families = {"llama", "qwen", "gemma", "deepseek", "mistral", "phi"}
    for path in _yaml_files(configs_dir, "models"):
        with path.open() as f:
            data = yaml.safe_load(f)
        assert (
            data["family"].lower() in supported_families
        ), f"{path.name} family='{data['family']}' not in {supported_families}"


@pytest.mark.ml_stack
def test_attack_plugin_names_match_registry(configs_dir: Path):
    """Every attack YAML's plugin field must exist in the attack registry."""
    from advsafe.attacks.base import autoload, list_attacks

    autoload()
    valid = set(list_attacks())
    for path in _yaml_files(configs_dir, "attacks"):
        with path.open() as f:
            data = yaml.safe_load(f)
        plugin = data["plugin"]
        assert plugin in valid, f"{path.name} references unknown attack '{plugin}'; valid: {valid}"


@pytest.mark.ml_stack
def test_defense_plugin_names_match_registry(configs_dir: Path):
    from advsafe.defenses.base import autoload, list_defenses

    autoload()
    valid = set(list_defenses())
    for path in _yaml_files(configs_dir, "defenses"):
        with path.open() as f:
            data = yaml.safe_load(f)
        plugin = data["plugin"]
        assert plugin in valid, f"{path.name} references unknown defense '{plugin}'; valid: {valid}"
