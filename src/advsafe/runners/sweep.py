"""Sweep runner — Week 3 cloud burst.

Iterates over all experiment cells in `configs/experiments/sweep.yaml`,
running each one and collecting results into a single dataframe-ready output.

This is the script that runs on the 4× A100 (or single A100) cloud instance
for ~2-3 days. It is designed to be:
  - Resumable: cells already in `results/sweep/` are skipped on rerun
  - Robust to single-cell failures: errors are logged + the run continues
  - Checkpointable: each cell saves before moving to the next

Usage:
    advsafe-sweep --config configs/experiments/sweep.yaml
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import click
import yaml
from rich.console import Console

from advsafe.runners.run_experiment import run_cell
from advsafe.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option("--config", "config_path", default="configs/experiments/sweep.yaml", show_default=True)
@click.option("--output", "output_dir", default="results/sweep", show_default=True)
@click.option("--skip-existing/--no-skip-existing", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="Print the cells that would run and exit")
def cli(
    config_path: str,
    output_dir: str,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    setup_logging("INFO")
    with Path(config_path).open() as f:
        sweep_config = yaml.safe_load(f)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cells = sweep_config["cells"]
    common = sweep_config.get("common", {})

    if dry_run:
        for cell_spec in cells:
            cell = {**common, **cell_spec}
            cell_id = cell.get("id", f"{cell['model']}__{cell['attack']['plugin']}__{cell['defense']['plugin']}")
            console.print(f"would run: {cell_id}")
        console.print(f"\nTotal cells: {len(cells)}")
        return

    n_done = 0
    n_skipped = 0
    n_failed = 0
    summary: list[dict] = []
    sweep_start = time.perf_counter()

    for i, cell_spec in enumerate(cells, start=1):
        cell = {**common, **cell_spec}
        cell_id = cell.get(
            "id", f"{cell['model']}__{cell['attack']['plugin']}__{cell['defense']['plugin']}"
        )
        cell["id"] = cell_id
        out = output_root / cell_id

        if skip_existing and (out / "manifest.json").exists():
            console.print(f"[yellow]Skipping (already done): {cell_id}[/yellow]")
            n_skipped += 1
            summary.append({"cell_id": cell_id, "status": "skipped"})
            continue

        console.print(f"\n[bold cyan]({i}/{len(cells)}) {cell_id}[/bold cyan]")
        t0 = time.perf_counter()
        try:
            manifest = run_cell(cell, out)
            summary.append(
                {
                    "cell_id": cell_id,
                    "status": "ok",
                    "asr": manifest["score"].get("asr"),
                    "elapsed_s": time.perf_counter() - t0,
                }
            )
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            tb = traceback.format_exc()
            logger.error("Cell failed", extra={"cell_id": cell_id, "error": str(exc)})
            (out / "ERROR.txt").parent.mkdir(parents=True, exist_ok=True)
            (out / "ERROR.txt").write_text(tb)
            summary.append({"cell_id": cell_id, "status": "failed", "error": str(exc)})

        # Save running summary after every cell (so partial sweep state is recoverable)
        (output_root / "sweep_summary.json").write_text(
            json.dumps(
                {
                    "n_done": n_done,
                    "n_skipped": n_skipped,
                    "n_failed": n_failed,
                    "elapsed_total_s": time.perf_counter() - sweep_start,
                    "cells": summary,
                },
                default=str,
                indent=2,
            )
        )

    console.print(
        f"\n[green]Sweep complete.[/green] "
        f"done={n_done} skipped={n_skipped} failed={n_failed}"
    )


if __name__ == "__main__":
    cli()
