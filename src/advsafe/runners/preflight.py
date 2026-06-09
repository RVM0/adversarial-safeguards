"""Preflight check — `advsafe-preflight`.

Single-command "are we ready to launch the cloud sweep" check. Verifies
environment, dependencies, configs, datasets, model access, HF token,
disk space, and runs `advsafe-smoke` end-to-end on the smallest model.

Exits 0 if launch-ready, non-zero (with a checklist of failures) otherwise.

Usage:
    advsafe-preflight                          # full check
    advsafe-preflight --skip-smoke             # skip the model-load smoke test
    advsafe-preflight --skip-models            # skip HF model access checks
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class Check:
    name: str
    status: str  # "pass" | "fail" | "warn" | "skip"
    detail: str = ""


def check_python_version() -> Check:
    # Runtime guard: the package can be run from source on an unsupported
    # interpreter even though requires-python is >=3.10. ruff's UP036 judges this
    # statically against the configured target version, so silence it here.
    if sys.version_info < (3, 10):  # noqa: UP036
        return Check("Python ≥ 3.10", "fail", f"found {sys.version.split()[0]}")
    return Check("Python ≥ 3.10", "pass", sys.version.split()[0])


def check_imports() -> list[Check]:
    checks = []
    required = [
        ("torch", "2.4"),
        ("transformers", "4.40"),
        ("peft", "0.10"),
        ("accelerate", "1.0"),
        ("datasets", "3.0"),
        ("yaml", None),
        ("click", None),
        ("rich", None),
        ("numpy", "1.26"),
        ("scipy", None),
    ]
    for pkg, _min_ver in required:
        try:
            mod = importlib.import_module(pkg.replace("-", "_"))
            version = getattr(mod, "__version__", "?")
            checks.append(Check(f"import {pkg}", "pass", version))
        except ImportError as e:
            checks.append(Check(f"import {pkg}", "fail", str(e)))
    return checks


def check_device() -> Check:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return Check("GPU available", "pass", f"{n}× {name} ({mem:.0f} GB)")
        if torch.backends.mps.is_available():
            return Check("GPU available", "pass", "Apple MPS (unified memory)")
        return Check("GPU available", "warn", "CPU only — pilot will work but sweep needs CUDA")
    except Exception as e:  # noqa: BLE001
        return Check("GPU available", "fail", str(e))


def check_hf_token() -> Check:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return Check(
            "HF_TOKEN set", "fail", "required for Llama/Gemma; export HF_TOKEN=<your-token>"
        )
    return Check("HF_TOKEN set", "pass", f"token len={len(token)}")


def check_openai_token() -> Check:
    if os.environ.get("OPENAI_API_KEY"):
        return Check("OPENAI_API_KEY set", "pass", "")
    return Check(
        "OPENAI_API_KEY set", "warn", "optional — used for the judge cross-validation step only"
    )


def check_disk_space(min_gb: float = 200) -> Check:
    total, used, free = shutil.disk_usage(Path.cwd())
    free_gb = free / 1e9
    if free_gb < min_gb:
        return Check(
            f"Disk free ≥ {min_gb} GB",
            "fail",
            f"only {free_gb:.0f} GB free; need ~{min_gb} GB for all 5 model weights",
        )
    return Check(f"Disk free ≥ {min_gb} GB", "pass", f"{free_gb:.0f} GB free")


def check_configs_validate() -> Check:
    """Run the validator as a subroutine."""
    from advsafe.attacks.base import autoload as autoload_attacks
    from advsafe.attacks.base import list_attacks
    from advsafe.defenses.base import autoload as autoload_defenses
    from advsafe.defenses.base import list_defenses
    from advsafe.evals.base import autoload as autoload_evals
    from advsafe.evals.base import list_evals
    from advsafe.judges.base import autoload as autoload_judges
    from advsafe.judges.base import list_judges
    from advsafe.runners.validate import (
        validate_attack_config,
        validate_defense_config,
        validate_eval_config,
        validate_experiment_config,
        validate_model_config,
    )

    autoload_attacks()
    autoload_defenses()
    autoload_evals()
    autoload_judges()
    va, vd, ve, vj = (
        set(list_attacks()),
        set(list_defenses()),
        set(list_evals()),
        set(list_judges()),
    )
    vm = {p.stem for p in Path("configs/models").glob("*.yaml")}

    issues = []
    for p in Path("configs/models").glob("*.yaml"):
        issues += validate_model_config(p)
    for p in Path("configs/attacks").glob("*.yaml"):
        issues += validate_attack_config(p, va, strict=False)
    for p in Path("configs/defenses").glob("*.yaml"):
        issues += validate_defense_config(p, vd, strict=False)
    for p in Path("configs/evals").glob("*.yaml"):
        issues += validate_eval_config(p, ve, strict=False)
    for p in Path("configs/experiments").glob("*.yaml"):
        issues += validate_experiment_config(p, va, vd, ve, vj, vm)

    n_errors = sum(1 for i in issues if i.severity == "error")
    if n_errors:
        return Check("Configs validate", "fail", f"{n_errors} errors; run advsafe-validate")
    if issues:
        return Check("Configs validate", "warn", f"{len(issues)} warnings; run advsafe-validate")
    return Check("Configs validate", "pass", "")


def check_datasets() -> list[Check]:
    expected = [
        ("HarmBench", "data/harmbench/harmbench_test.csv"),
        ("StrongREJECT", "data/strongreject/strongreject_dataset.csv"),
        ("MT-Bench", "data/mt_bench/question.jsonl"),
        ("XSTest", "data/xstest/xstest_v2_prompts.csv"),
        ("AdvBench", "data/advbench/harmful_behaviors.csv"),
        ("Attack train data", "data/attacks/harmful_train.jsonl"),
    ]
    checks = []
    for name, path in expected:
        p = Path(path)
        if not p.exists():
            checks.append(
                Check(
                    f"Dataset: {name}", "fail", f"{path} missing — run scripts/download_datasets.sh"
                )
            )
        else:
            size_mb = p.stat().st_size / 1e6
            checks.append(Check(f"Dataset: {name}", "pass", f"{size_mb:.1f} MB"))
    return checks


def check_smoke_test() -> Check:
    """Loads the smallest panel model and runs one generation."""
    try:
        from advsafe.models import generate, get_model_config, load_model
        from advsafe.types import GenerationConfig
    except ImportError as e:
        return Check("Smoke test", "fail", f"import error: {e}")

    try:
        cfg = get_model_config("llama-3.1-8b", config_dir="configs/models")
        handle = load_model(cfg)
        gen = GenerationConfig(max_new_tokens=8, do_sample=False)
        r = generate(handle, prompt="Hello", gen_config=gen)
        if r.n_output_tokens > 0:
            return Check("Smoke test", "pass", f"generated {r.n_output_tokens} tokens")
        return Check("Smoke test", "fail", "generated 0 tokens")
    except Exception as e:  # noqa: BLE001
        return Check("Smoke test", "fail", str(e)[:200])


@click.command()
@click.option(
    "--skip-smoke/--no-skip-smoke",
    default=False,
    show_default=True,
    help="Skip the model-load smoke test (faster)",
)
@click.option(
    "--skip-models/--no-skip-models",
    default=True,
    show_default=True,
    help="Skip HF model access checks (default: skip; rely on smoke test)",
)
@click.option("--min-disk-gb", default=200, show_default=True)
def cli(skip_smoke: bool, skip_models: bool, min_disk_gb: float) -> None:
    """Run all preflight checks."""
    checks: list[Check] = []

    console.print("[bold]advsafe preflight check[/bold]\n")

    checks.append(check_python_version())
    checks.extend(check_imports())
    checks.append(check_device())
    checks.append(check_hf_token())
    checks.append(check_openai_token())
    checks.append(check_disk_space(min_disk_gb))
    checks.append(check_configs_validate())
    checks.extend(check_datasets())
    if not skip_smoke:
        checks.append(check_smoke_test())
    else:
        checks.append(Check("Smoke test", "skip", "(skipped by flag)"))

    table = Table(title="Preflight results")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for c in checks:
        color = {"pass": "green", "fail": "red", "warn": "yellow", "skip": "dim"}[c.status]
        table.add_row(c.name, f"[{color}]{c.status}[/{color}]", c.detail)
    console.print(table)

    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")
    if n_fail:
        console.print(f"\n[red]NOT READY: {n_fail} failures, {n_warn} warnings.[/red]")
        sys.exit(1)
    if n_warn:
        console.print(
            f"\n[yellow]READY WITH CAVEATS: {n_warn} warnings — review before launching.[/yellow]"
        )
        sys.exit(0)
    console.print("\n[green]READY TO LAUNCH.[/green]")


if __name__ == "__main__":
    cli()
