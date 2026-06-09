"""Runtime benchmark — `advsafe-benchmark`.

Empirically measures tokens/sec for inference on the current device for
each panel model, then projects total sweep cost and wall-clock from the
experiment matrix. Useful for catching surprises (e.g., MPS is 4× slower
than projected) before commit.

Usage:
    advsafe-benchmark --model llama-3.1-8b
    advsafe-benchmark --all                       # all panel models (slow)
    advsafe-benchmark --all --estimate-sweep      # also project full sweep
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    inference_tokens_per_sec: float
    training_examples_per_sec: float | None
    avg_inference_latency_s: float
    notes: str


def _benchmark_inference(
    model_name: str, n_prompts: int = 5, max_tokens: int = 64
) -> BenchmarkResult:
    from advsafe.models import generate, get_model_config, load_model
    from advsafe.types import GenerationConfig
    from advsafe.utils.device import get_device

    cfg = get_model_config(model_name, config_dir="configs/models")
    handle = load_model(cfg)

    gen = GenerationConfig(max_new_tokens=max_tokens, do_sample=False)
    prompts = [
        "Explain what an LLM is in 50 words.",
        "Translate to Spanish: 'Good morning, how are you?'",
        "Write a one-sentence summary of the French Revolution.",
        "List three reasons clouds form.",
        "What is the boiling point of water at sea level?",
    ][:n_prompts]

    # Warmup
    generate(handle, prompts[0], gen_config=gen)

    t0 = time.perf_counter()
    total_out = 0
    for p in prompts:
        r = generate(handle, p, gen_config=gen)
        total_out += r.n_output_tokens
    elapsed = time.perf_counter() - t0

    tps = total_out / elapsed if elapsed > 0 else 0.0
    avg_latency = elapsed / n_prompts

    return BenchmarkResult(
        model_name=model_name,
        device=str(get_device()),
        inference_tokens_per_sec=tps,
        training_examples_per_sec=None,  # not benchmarked here
        avg_inference_latency_s=avg_latency,
        notes=f"n_prompts={n_prompts}, max_tokens={max_tokens}",
    )


def _project_sweep_cost(
    results: dict[str, BenchmarkResult],
    sweep_path: Path,
    a100_hourly: float = 1.29,
) -> dict:
    """Project sweep wall-clock + cost from per-model benchmarks."""
    with sweep_path.open() as f:
        sweep = yaml.safe_load(f)

    # Heuristic: average 200 tokens out per eval prompt; 880 prompts per cell
    # (HarmBench + StrongREJECT + MT-Bench + XSTest combined).
    AVG_TOKENS_PER_PROMPT = 200
    PROMPTS_PER_CELL = 240  # if running HarmBench only; sweep config has n_prompts=None=full

    total_inference_sec = 0.0
    total_training_sec = 0.0
    per_model_breakdown = {}

    for cell in sweep["cells"]:
        model_name = cell["model"]
        attack = cell["attack"]
        n_examples = attack.get("n_examples", 0)

        # Training cost (only for non-zero LoRA attacks)
        if attack.get("plugin") == "lora-finetune" and n_examples and n_examples > 0:
            # Heuristic: ~5 examples/sec on A100 for 8-14B fp16,
            # ~2 ex/sec for 27-32B QLoRA
            params_b = _params_for(model_name)
            examples_per_sec = 5.0 if params_b <= 14 else 2.0
            epochs = attack.get("epochs", 3)
            training_time = (n_examples * epochs) / examples_per_sec
            total_training_sec += training_time
            per_model_breakdown.setdefault(model_name, {"train": 0, "infer": 0})
            per_model_breakdown[model_name]["train"] += training_time

        # Inference cost (default 30 tok/s if this model wasn't benchmarked).
        tps = results[model_name].inference_tokens_per_sec if model_name in results else 30.0
        inference_time = (PROMPTS_PER_CELL * AVG_TOKENS_PER_PROMPT) / max(tps, 1)
        total_inference_sec += inference_time
        per_model_breakdown.setdefault(model_name, {"train": 0, "infer": 0})
        per_model_breakdown[model_name]["infer"] += inference_time

    total_hours = (total_training_sec + total_inference_sec) / 3600
    cost = total_hours * a100_hourly

    return {
        "total_hours": total_hours,
        "total_cost_usd": cost,
        "training_hours": total_training_sec / 3600,
        "inference_hours": total_inference_sec / 3600,
        "per_model": {
            k: {"train_hr": v["train"] / 3600, "infer_hr": v["infer"] / 3600}
            for k, v in per_model_breakdown.items()
        },
    }


def _params_for(model_name: str) -> float:
    """Best-effort params (in billions) from the model config."""
    try:
        with Path(f"configs/models/{model_name}.yaml").open() as f:
            return yaml.safe_load(f).get("params_billion", 0)
    except Exception:  # noqa: BLE001
        return 0


@click.command()
@click.option("--model", "model_name", default=None)
@click.option("--all", "all_models", is_flag=True)
@click.option("--n-prompts", default=5, show_default=True)
@click.option("--max-tokens", default=64, show_default=True)
@click.option("--estimate-sweep/--no-estimate-sweep", default=False, show_default=True)
@click.option("--sweep-config", default="configs/experiments/sweep.yaml", show_default=True)
@click.option(
    "--a100-hourly",
    default=1.29,
    show_default=True,
    help="$/hr for projection (Lambda Labs A100 80GB default)",
)
def cli(
    model_name: str | None,
    all_models: bool,
    n_prompts: int,
    max_tokens: int,
    estimate_sweep: bool,
    sweep_config: str,
    a100_hourly: float,
) -> None:
    """Benchmark inference throughput; optionally project sweep cost."""
    if not all_models and not model_name:
        console.print("[red]Specify --model <name> or --all[/red]")
        raise SystemExit(1)

    models = [p.stem for p in Path("configs/models").glob("*.yaml")] if all_models else [model_name]

    results: dict[str, BenchmarkResult] = {}
    for name in models:
        console.print(f"\n[bold]Benchmarking {name}...[/bold]")
        try:
            r = _benchmark_inference(name, n_prompts=n_prompts, max_tokens=max_tokens)
            results[name] = r
            console.print(
                f"  {r.inference_tokens_per_sec:.1f} tok/s · "
                f"latency {r.avg_inference_latency_s:.2f}s per gen on {r.device}"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]FAILED: {e}[/red]")
            continue

    if results:
        table = Table(title="Inference benchmark")
        table.add_column("model")
        table.add_column("device")
        table.add_column("tokens/sec")
        table.add_column("avg latency (s)")
        for r in results.values():
            table.add_row(
                r.model_name,
                r.device,
                f"{r.inference_tokens_per_sec:.1f}",
                f"{r.avg_inference_latency_s:.2f}",
            )
        console.print(table)

    if estimate_sweep:
        sweep_path = Path(sweep_config)
        if not sweep_path.exists():
            console.print(f"[red]Sweep config not found: {sweep_path}[/red]")
            raise SystemExit(1)
        projection = _project_sweep_cost(results, sweep_path, a100_hourly=a100_hourly)
        console.print("\n[bold]Sweep projection[/bold]")
        console.print(
            f"  Training: {projection['training_hours']:.1f} hr · "
            f"Inference: {projection['inference_hours']:.1f} hr"
        )
        console.print(
            f"  [bold]Total: {projection['total_hours']:.1f} hr ≈ "
            f"${projection['total_cost_usd']:.2f} @ ${a100_hourly}/hr[/bold]"
        )
        table = Table(title="Per-model projection")
        table.add_column("model")
        table.add_column("train hr")
        table.add_column("infer hr")
        for model, breakdown in projection["per_model"].items():
            table.add_row(
                model,
                f"{breakdown['train_hr']:.1f}",
                f"{breakdown['infer_hr']:.1f}",
            )
        console.print(table)


if __name__ == "__main__":
    cli()
