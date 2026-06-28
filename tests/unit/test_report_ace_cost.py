"""Tests for the cost-anchored ACE wiring in the report runner.

These exercise `compute_ace_grid` (attaching laptop-hours + dollars from the
measured `train_wall_clock_s`) and `evaluate_h6` (primary laptop-cost reading,
secondary FLOPs reading) without needing any real sweep artifacts on disk.
"""

from __future__ import annotations

import pytest

from advsafe.runners.report import compute_ace_grid, evaluate_h6

MODELS = [
    "llama-3.1-8b",
    "deepseek-r1-distill-qwen-14b",
    "gemma-3-27b",
    "qwen-3-32b",
]


def _cell(asr: float, train_s: float | None = None) -> dict:
    """A minimal cell manifest: ASR score + (optional) measured attack time."""
    manifest: dict = {
        "score": {"asr": asr},
        "attack": {"name": "lora-finetune", "metadata": {}},
    }
    if train_s is not None:
        manifest["attack"]["metadata"]["train_wall_clock_s"] = train_s
    return manifest


def _index(train_seconds: dict[str, float | None], attack_id: str = "lora-a1-100") -> dict:
    """Build a (model, attack, defense) index over the 4 models for one attack."""
    index: dict = {}
    for model in MODELS:
        index[(model, attack_id, "baseline")] = _cell(0.7, train_seconds.get(model))
        index[(model, attack_id, "output-filter")] = _cell(0.2)
    return index


def test_compute_ace_grid_attaches_laptop_cost():
    index = _index({"llama-3.1-8b": 18 * 60}, attack_id="lora-a1-100")
    grid = compute_ace_grid(index, ["llama-3.1-8b"], ["lora-a1-100"])
    entry = grid[("llama-3.1-8b", "lora-a1-100")]
    # Primary (laptop) reading present and correct.
    assert entry["train_wall_clock_s"] == 18 * 60
    assert entry["attacker_laptop_hours"] == pytest.approx(0.3)
    assert entry["attacker_usd"] < 0.10
    assert entry["laptop_price_usd"] == 3000.0
    # Secondary (FLOPs) reading still present.
    assert 2.8 < entry["raw_ace"] < 3.1


def test_compute_ace_grid_without_measured_time_omits_cost():
    index = _index({"llama-3.1-8b": None}, attack_id="lora-a1-100")
    grid = compute_ace_grid(index, ["llama-3.1-8b"], ["lora-a1-100"])
    entry = grid[("llama-3.1-8b", "lora-a1-100")]
    assert "attacker_laptop_hours" not in entry  # no measured time → no cost reading
    assert "raw_ace" in entry  # FLOPs reading still computed


def test_evaluate_h6_primary_confirmed_when_all_accessible():
    # Every model strips safety in well under a working day.
    index = _index(dict.fromkeys(MODELS, 30 * 60))  # 0.5 laptop-hours each
    grid = compute_ace_grid(index, MODELS, ["lora-a1-100"])
    h6 = evaluate_h6(grid)
    assert h6["primary_reading"] == "laptop_cost"
    assert h6["verdict"] == "CONFIRMED"
    assert h6["n_laptop_accessible"] == 4
    assert h6["n_with_measured_time"] == 4


def test_evaluate_h6_primary_refuted_when_all_slow():
    # Every model needs more than a day of laptop time.
    index = _index(dict.fromkeys(MODELS, 30 * 3600))  # 30 laptop-hours each
    grid = compute_ace_grid(index, MODELS, ["lora-a1-100"])
    h6 = evaluate_h6(grid)
    assert h6["primary_reading"] == "laptop_cost"
    assert h6["verdict"] == "REFUTED"
    assert h6["n_laptop_accessible"] == 0


def test_evaluate_h6_falls_back_to_flops_without_measured_time():
    index = _index(dict.fromkeys(MODELS, None))  # no measured times anywhere
    grid = compute_ace_grid(index, MODELS, ["lora-a1-100"])
    h6 = evaluate_h6(grid)
    assert h6["primary_reading"] == "flops"
    assert h6["laptop_verdict"] == "NOT_TESTABLE"
    # FLOPs reading still yields a verdict (these 8-32B cells are all "cheap").
    assert h6["verdict"] == h6["flops_verdict"]
    assert h6["flops_verdict"] in {"CONFIRMED", "REFUTED", "MIXED"}


def test_evaluate_h6_reports_both_readings_in_per_model():
    index = _index(dict.fromkeys(MODELS, 60 * 60))  # 1 laptop-hour each
    grid = compute_ace_grid(index, MODELS, ["lora-a1-100"])
    h6 = evaluate_h6(grid)
    for model in MODELS:
        pm = h6["per_model"][model]
        assert pm["attacker_laptop_hours"] == pytest.approx(1.0)
        assert pm["laptop_accessible"] is True
        assert "ace" in pm  # FLOPs reading retained per model
