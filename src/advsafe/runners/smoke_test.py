"""Smoke test: load a model, generate one response, exit.

Verifies that the environment is set up correctly and the model can be
loaded on the available device. Run this first when bootstrapping a new
machine.

Usage:
    advsafe-smoke --model llama-3.1-8b --prompt "Hello, are you online?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from advsafe.models import generate, get_model_config, list_models, load_model
from advsafe.types import GenerationConfig
from advsafe.utils.device import describe_device, get_device
from advsafe.utils.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


@click.command()
@click.option(
    "--model",
    "model_name",
    default="llama-3.1-8b",
    show_default=True,
    help="Model short name (must have config in configs/models/)",
)
@click.option(
    "--prompt",
    default="Hello! In one sentence, what is the capital of France?",
    show_default=True,
    help="User prompt to send to the model",
)
@click.option("--max-tokens", default=64, show_default=True, help="Max new tokens")
@click.option("--temperature", default=0.0, show_default=True, help="Generation temperature")
@click.option("--config-dir", default="configs/models", show_default=True)
def cli(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    config_dir: str,
) -> None:
    """Run the smoke test."""
    setup_logging("INFO")
    dev = get_device()
    dev_info = describe_device(dev)

    mem_str = (
        f"{dev_info.available_memory_gb:.1f} GB"
        if dev_info.available_memory_gb is not None
        else "(unknown)"
    )
    panel_text = (
        f"[bold]Smoke test[/bold]\n"
        f"Device: {dev_info.name} ({dev_info.backend_version})\n"
        f"Available memory: {mem_str}\n"
        f"Model: {model_name}"
    )
    console.print(Panel.fit(panel_text, title="advsafe"))

    if model_name not in list_models(Path(config_dir)):
        console.print(f"[red]Model '{model_name}' not found in {config_dir}[/red]")
        console.print(f"Available: {', '.join(list_models(Path(config_dir)))}")
        sys.exit(1)

    config = get_model_config(model_name, config_dir=Path(config_dir))
    handle = load_model(config)

    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    response = generate(handle, prompt=prompt, gen_config=gen_config)

    console.print(Panel(response.response, title="Response", border_style="green"))
    console.print(
        f"[dim]elapsed={response.elapsed_seconds:.2f}s · "
        f"in_tokens={response.n_input_tokens} · out_tokens={response.n_output_tokens}[/dim]"
    )


if __name__ == "__main__":
    cli()
