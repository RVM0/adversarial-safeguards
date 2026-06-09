"""Pilot runner — Week 2.

Runs the full pipeline on Llama 3.1 8B only:
    baseline → A1.100 attack → eval suite (HarmBench) × all 5 defenses

Used to validate the pipeline end-to-end before the Week 3 cloud sweep.

Usage:
    advsafe-pilot --config configs/experiments/pilot.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from advsafe.runners.run_experiment import run_cell
from advsafe.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--config", "config_path", default="configs/experiments/pilot.yaml", show_default=True
)
@click.option("--output", "output_dir", default="results/pilot", show_default=True)
def cli(config_path: str, output_dir: str) -> None:
    """Run the Week 2 pilot."""
    setup_logging("INFO")
    with Path(config_path).open() as f:
        pilot_config = yaml.safe_load(f)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # The pilot config is a list of cells (one per attack × defense combo).
    cells = pilot_config["cells"]
    common = pilot_config.get("common", {})

    results = []
    for cell_spec in cells:
        # Merge common + per-cell overrides
        cell = {**common, **cell_spec}
        cell_id = cell.get(
            "id", f"{cell['model']}__{cell['attack']['plugin']}__{cell['defense']['plugin']}"
        )
        cell["id"] = cell_id
        out = output_root / cell_id

        console.print(f"\n[bold cyan]Running cell: {cell_id}[/bold cyan]")
        manifest = run_cell(cell, out)
        results.append(manifest)

    # Summary table
    table = Table(title="Pilot summary")
    table.add_column("cell")
    table.add_column("attack")
    table.add_column("defense")
    table.add_column("ASR")
    table.add_column("CI")
    table.add_column("n")
    for r in results:
        asr = r["score"].get("asr")
        lo = r["score"].get("asr_ci_low")
        hi = r["score"].get("asr_ci_high")
        table.add_row(
            r["experiment_id"],
            r["attack"]["name"],
            r["defense"]["name"],
            f"{asr:.3f}" if asr is not None else "n/a",
            f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "n/a",
            str(r["eval"]["n_prompts"]),
        )
    console.print(table)

    (output_root / "pilot_summary.json").write_text(json.dumps(results, default=str, indent=2))
    console.print(f"\n[green]Pilot complete.[/green] Results: {output_root}")


if __name__ == "__main__":
    cli()
