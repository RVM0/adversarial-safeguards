"""Unit tests for the sweep cost/wall-clock projection (G8).

The $-figure is an A100 heuristic and must be suppressed on the local MLX path,
where the laptop is owned (not rented) and the A100 training-throughput estimate
does not apply. Measured inference hours are still reported. Torch-free.
"""

from __future__ import annotations

import yaml

from advsafe.runners.benchmark import BenchmarkResult, _is_mlx_device, _project_sweep_cost


def _result(model: str, device: str, tps: float = 20.0) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=model,
        device=device,
        inference_tokens_per_sec=tps,
        training_examples_per_sec=None,
        avg_inference_latency_s=1.0,
        notes="",
    )


def _write_sweep(path) -> None:
    sweep = {
        "cells": [
            {"model": "m1", "attack": {"plugin": "lora-finetune", "n_examples": 10, "epochs": 1}},
            {"model": "m1", "attack": {"plugin": "lora-finetune", "n_examples": 0}},
        ]
    }
    path.write_text(yaml.safe_dump(sweep))


def test_is_mlx_device() -> None:
    assert _is_mlx_device("mlx")
    assert _is_mlx_device("MLX")
    assert not _is_mlx_device("cuda:0")
    assert not _is_mlx_device("mps")


def test_projection_suppresses_dollar_cost_on_mlx(tmp_path) -> None:
    sweep = tmp_path / "sweep.yaml"
    _write_sweep(sweep)

    proj = _project_sweep_cost({"m1": _result("m1", "mlx")}, sweep)

    assert proj["is_local"] is True
    assert proj["total_cost_usd"] is None  # $-figure gated off on the local path
    assert "local MLX" in proj["cost_basis"]
    assert proj["inference_hours"] > 0  # measured wall-clock still reported


def test_projection_reports_dollars_on_cuda(tmp_path) -> None:
    sweep = tmp_path / "sweep.yaml"
    _write_sweep(sweep)

    proj = _project_sweep_cost({"m1": _result("m1", "cuda:0")}, sweep, a100_hourly=2.0)

    assert proj["is_local"] is False
    assert isinstance(proj["total_cost_usd"], float)
    assert proj["total_cost_usd"] == proj["total_hours"] * 2.0
    assert "A100" in proj["cost_basis"]


def test_projection_empty_results_defaults_to_cloud(tmp_path) -> None:
    """No benchmarked models → can't prove local; keep the A100 $-projection."""
    sweep = tmp_path / "sweep.yaml"
    _write_sweep(sweep)

    proj = _project_sweep_cost({}, sweep)

    assert proj["is_local"] is False
    assert proj["total_cost_usd"] is not None
