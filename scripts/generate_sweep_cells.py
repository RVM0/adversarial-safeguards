#!/usr/bin/env python
"""Generate the full sweep.yaml from a compact spec.

The canonical sweep is 4 models × 6 attack conditions × 5 defenses = 120 cells.
This script emits them all to stdout (pipe into configs/experiments/sweep.yaml).

Usage:
    python scripts/generate_sweep_cells.py > configs/experiments/sweep.yaml
"""

from __future__ import annotations

import sys

import yaml

MODELS = [
    "llama-3.1-8b",
    "deepseek-r1-distill-qwen-14b",
    "gemma-3-27b",
    "qwen-3-32b",
]

ATTACKS = [
    {"id": "no-attack", "plugin": "lora-finetune", "n_examples": 0},
    {"id": "lora-a1-10", "plugin": "lora-finetune", "n_examples": 10},
    {"id": "lora-a1-100", "plugin": "lora-finetune", "n_examples": 100},
    {"id": "lora-a1-1000", "plugin": "lora-finetune", "n_examples": 1000},
    {"id": "pap", "plugin": "pap", "template_id": "logical_appeal"},
    {"id": "roleplay", "plugin": "roleplay", "template_id": "dan"},
]

DEFENSES = [
    {"id": "baseline", "plugin": "baseline"},
    {"id": "input-filter", "plugin": "llama-guard-input"},
    {"id": "output-filter", "plugin": "llama-guard-output"},
    {"id": "constitutional", "plugin": "constitutional-prompt"},
    {"id": "combined", "plugin": "combined"},
]


def _batch_size_for(model: str) -> int:
    """Smaller batch for the biggest QLoRA models."""
    if model.endswith("-32b") or model.endswith("-27b"):
        return 2
    return 4


def _attack_block(model: str, attack: dict) -> dict:
    if attack["plugin"] == "lora-finetune":
        return {
            "plugin": "lora-finetune",
            "dataset_path": "data/attacks/harmful_train.jsonl",
            "n_examples": attack["n_examples"],
            "output_dir": f"checkpoints/sweep/{model}/{attack['id']}",
            "epochs": 3 if attack["n_examples"] > 0 else 0,
            "learning_rate": 2.0e-4,
            "batch_size": _batch_size_for(model),
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "max_seq_len": 512,
            "seed": 42,
        }
    if attack["plugin"] == "pap":
        return {
            "plugin": "pap",
            "attack_type": "PROMPT_MOD",
            "template_path": "data/pap_templates.json",
            "template_id": attack["template_id"],
            "seed": 42,
        }
    if attack["plugin"] == "roleplay":
        return {
            "plugin": "roleplay",
            "attack_type": "PROMPT_MOD",
            "template_path": "data/roleplay_templates.json",
            "template_id": attack["template_id"],
            "seed": 42,
        }
    raise ValueError(f"Unknown attack plugin: {attack['plugin']}")


def _defense_block(defense: dict) -> dict:
    base = {"plugin": defense["plugin"]}
    if defense["plugin"] in ("llama-guard-input", "llama-guard-output", "combined"):
        base["guard_model_id"] = "meta-llama/Llama-Guard-3-8B"
    if defense["plugin"] in ("constitutional-prompt", "combined"):
        base["system_prompt_path"] = "configs/defenses/constitution.txt"
    return base


def _judge_block_for(defense: dict) -> dict:
    """Pick the judge based on the defense to avoid circular evaluation.

    Pre-registration requires: when Llama Guard 3 is the defense, the judge
    must be a different model (else the same model's blind spots would
    invalidate the measurement). We switch to GPT-4o-mini for D1/D2/D4.
    """
    if defense["plugin"] in ("llama-guard-input", "llama-guard-output", "combined"):
        return {
            "plugin": "openai-gpt-4o-mini",
            "api_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        }
    return {
        "plugin": "llama-guard-3",
        "hf_id": "meta-llama/Llama-Guard-3-8B",
    }


def main() -> None:
    common = {
        "seed": 42,
        "system_prompt": "",
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": False,
        },
        "judge": {
            "plugin": "llama-guard-3",
            "hf_id": "meta-llama/Llama-Guard-3-8B",
        },
        "eval": {
            "plugin": "harmbench",
            "dataset_path": "data/harmbench/harmbench_test.csv",
            "n_prompts": None,
            "judge_name": "llama-guard-3",
            "bootstrap_iterations": 1000,
            "seed": 42,
        },
    }

    cells = []
    for model in MODELS:
        for attack in ATTACKS:
            for defense in DEFENSES:
                cell_id = f"{model}__{attack['id']}__{defense['id']}"
                cell = {
                    "judge": _judge_block_for(defense),  # may override common
                    "eval": {**common["eval"], "judge_name": _judge_block_for(defense)["plugin"]},
                }
                cells.append(
                    {
                        **cell,
                        "id": cell_id,
                        "model": model,
                        "attack": _attack_block(model, attack),
                        "defense": _defense_block(defense),
                    }
                )

    out = {"common": common, "cells": cells}
    yaml.safe_dump(out, sys.stdout, sort_keys=False, default_flow_style=False, width=200)
    print(f"\n# Total cells: {len(cells)}", file=sys.stderr)


if __name__ == "__main__":
    main()
