"""Config validator — `advsafe-validate`.

Checks every YAML config against the framework's expectations *without*
loading models or running anything. Catches typos, missing fields, plugin
name mismatches, and dangling dataset paths before you spend $77 on a
cloud sweep.

Usage:
    advsafe-validate                       # validates everything in configs/
    advsafe-validate --strict              # also requires dataset files to exist
    advsafe-validate --experiment <path>   # validate a single experiment
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from advsafe.attacks.base import autoload as autoload_attacks, list_attacks
from advsafe.defenses.base import autoload as autoload_defenses, list_defenses
from advsafe.evals.base import autoload as autoload_evals, list_evals
from advsafe.judges.base import autoload as autoload_judges, list_judges

console = Console()


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    path: str
    field: str
    message: str


def _check_required(data: dict, required: list[str], path: str) -> list[ValidationIssue]:
    return [
        ValidationIssue("error", path, key, f"missing required field '{key}'")
        for key in required
        if key not in data
    ]


def _check_plugin(
    plugin_name: str, valid_names: set[str], path: str, field: str = "plugin"
) -> list[ValidationIssue]:
    if plugin_name not in valid_names:
        return [
            ValidationIssue(
                "error",
                path,
                field,
                f"unknown plugin '{plugin_name}'. Valid: {sorted(valid_names)}",
            )
        ]
    return []


def _check_file_exists(
    file_path: str | None, ref_path: str, field: str
) -> list[ValidationIssue]:
    if file_path and not Path(file_path).exists():
        return [
            ValidationIssue(
                "warning",
                ref_path,
                field,
                f"path '{file_path}' does not exist (run download_datasets.sh)",
            )
        ]
    return []


def validate_model_config(path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    with path.open() as f:
        data = yaml.safe_load(f)
    issues += _check_required(data, ["name", "hf_id", "family"], str(path))
    if "family" in data:
        if data["family"].lower() not in {
            "llama", "qwen", "gemma", "deepseek", "mistral", "phi"
        }:
            issues.append(
                ValidationIssue(
                    "warning",
                    str(path),
                    "family",
                    f"unfamiliar family '{data['family']}'; chat template may not work",
                )
            )
    if data.get("use_quantization") and data.get("params_billion", 0) < 5:
        issues.append(
            ValidationIssue(
                "warning",
                str(path),
                "use_quantization",
                "quantization unnecessary for <5B models — wastes compute",
            )
        )
    return issues


def validate_attack_config(
    path: Path, valid_attacks: set[str], strict: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    with path.open() as f:
        data = yaml.safe_load(f)
    issues += _check_required(data, ["plugin"], str(path))
    if "plugin" in data:
        issues += _check_plugin(data["plugin"], valid_attacks, str(path))
    if data.get("plugin") == "lora-finetune":
        if data.get("n_examples") and data["n_examples"] > 0:
            if strict and "dataset_path" in data:
                issues += _check_file_exists(data["dataset_path"], str(path), "dataset_path")
            if not 0 < data.get("learning_rate", 2e-4) < 1:
                issues.append(
                    ValidationIssue("error", str(path), "learning_rate", "must be 0 < lr < 1")
                )
            if data.get("lora_rank", 16) < 1:
                issues.append(
                    ValidationIssue("error", str(path), "lora_rank", "must be >= 1")
                )
    return issues


def validate_defense_config(
    path: Path, valid_defenses: set[str], strict: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    with path.open() as f:
        data = yaml.safe_load(f)
    issues += _check_required(data, ["plugin"], str(path))
    if "plugin" in data:
        issues += _check_plugin(data["plugin"], valid_defenses, str(path))
    if strict and data.get("system_prompt_path"):
        issues += _check_file_exists(
            data["system_prompt_path"], str(path), "system_prompt_path"
        )
    return issues


def validate_eval_config(
    path: Path, valid_evals: set[str], strict: bool
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    with path.open() as f:
        data = yaml.safe_load(f)
    issues += _check_required(data, ["plugin"], str(path))
    if "plugin" in data:
        issues += _check_plugin(data["plugin"], valid_evals, str(path))
    if strict and data.get("dataset_path"):
        issues += _check_file_exists(data["dataset_path"], str(path), "dataset_path")
    return issues


def validate_experiment_config(
    path: Path,
    valid_attacks: set[str],
    valid_defenses: set[str],
    valid_evals: set[str],
    valid_judges: set[str],
    valid_models: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    with path.open() as f:
        data = yaml.safe_load(f)
    if "cells" not in data:
        issues.append(ValidationIssue("error", str(path), "cells", "missing"))
        return issues
    if not data["cells"]:
        issues.append(ValidationIssue("error", str(path), "cells", "empty"))
        return issues
    common = data.get("common", {})
    for i, cell in enumerate(data["cells"]):
        merged = {**common, **cell}
        ref = f"{path}:cell[{i}]"

        model = merged.get("model")
        if not model:
            issues.append(ValidationIssue("error", ref, "model", "missing"))
        elif model not in valid_models:
            issues.append(
                ValidationIssue(
                    "error", ref, "model", f"unknown model '{model}'; available: {sorted(valid_models)}"
                )
            )

        for sub, valid_set, label in [
            ("attack", valid_attacks, "attack"),
            ("defense", valid_defenses, "defense"),
            ("eval", valid_evals, "eval"),
            ("judge", valid_judges, "judge"),
        ]:
            block = merged.get(sub)
            if not block:
                issues.append(ValidationIssue("error", ref, sub, "missing block"))
                continue
            plug = block.get("plugin")
            if not plug:
                issues.append(ValidationIssue("error", f"{ref}.{sub}", "plugin", "missing"))
                continue
            issues += _check_plugin(plug, valid_set, f"{ref}.{sub}")
    return issues


@click.command()
@click.option(
    "--configs-dir",
    default="configs",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--strict/--no-strict", default=False, show_default=True,
              help="Also require dataset files to exist")
@click.option("--experiment", "experiment_path", default=None,
              type=click.Path(), help="Validate a single experiment file")
def cli(configs_dir: str, strict: bool, experiment_path: str | None) -> None:
    """Validate all YAML configs against the framework's expectations."""
    # Load all plugin registries
    autoload_attacks()
    autoload_defenses()
    autoload_evals()
    autoload_judges()
    valid_attacks = set(list_attacks())
    valid_defenses = set(list_defenses())
    valid_evals = set(list_evals())
    valid_judges = set(list_judges())
    valid_models = {p.stem for p in Path(configs_dir, "models").glob("*.yaml")}

    issues: list[ValidationIssue] = []

    if experiment_path:
        issues += validate_experiment_config(
            Path(experiment_path),
            valid_attacks,
            valid_defenses,
            valid_evals,
            valid_judges,
            valid_models,
        )
    else:
        for p in Path(configs_dir, "models").glob("*.yaml"):
            issues += validate_model_config(p)
        for p in Path(configs_dir, "attacks").glob("*.yaml"):
            issues += validate_attack_config(p, valid_attacks, strict)
        for p in Path(configs_dir, "defenses").glob("*.yaml"):
            issues += validate_defense_config(p, valid_defenses, strict)
        for p in Path(configs_dir, "evals").glob("*.yaml"):
            issues += validate_eval_config(p, valid_evals, strict)
        for p in Path(configs_dir, "experiments").glob("*.yaml"):
            issues += validate_experiment_config(
                p, valid_attacks, valid_defenses, valid_evals, valid_judges, valid_models
            )

    if not issues:
        console.print("[green]✓ All configs valid[/green]")
        sys.exit(0)

    table = Table(title="Validation issues")
    table.add_column("severity")
    table.add_column("path")
    table.add_column("field")
    table.add_column("message")
    n_errors = 0
    for issue in issues:
        color = "red" if issue.severity == "error" else "yellow"
        table.add_row(
            f"[{color}]{issue.severity}[/{color}]",
            issue.path,
            issue.field,
            issue.message,
        )
        if issue.severity == "error":
            n_errors += 1
    console.print(table)
    console.print(f"\n[bold]{n_errors} errors, {len(issues) - n_errors} warnings[/bold]")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    cli()
