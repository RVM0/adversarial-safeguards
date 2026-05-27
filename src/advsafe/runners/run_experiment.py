"""Single experiment cell runner.

A "cell" = (model × attack × defense × eval) with a specific seed.
Reads a composed YAML experiment config and produces one results directory:

    results/<experiment_id>/
        per_prompt.jsonl       # responses + verdicts per prompt
        score.json             # aggregate EvalScore
        manifest.json          # provenance: seeds, hashes, revisions, timings
        responses.jsonl        # raw model responses (before judge)
        defense_decisions.jsonl # per-prompt defense actions

This is the workhorse runner used by both `pilot` (in dev) and `sweep`
(in cloud) — both compose calls to `run_cell`.

Usage:
    advsafe-run --experiment configs/experiments/<id>.yaml
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from advsafe.attacks.base import AttackConfig, AttackType, autoload as autoload_attacks, get_attack
from advsafe.defenses.base import DefenseConfig, autoload as autoload_defenses, get_defense
from advsafe.evals.base import EvalConfig, autoload as autoload_evals, get_eval
from advsafe.judges.base import JudgeConfig, autoload as autoload_judges, get_judge
from advsafe.models import generate, get_model_config, load_model
from advsafe.types import AttackResult, GenerationConfig, ModelHandle
from advsafe.utils.device import describe_device
from advsafe.utils.logging import get_logger, setup_logging
from advsafe.utils.seeds import capture_rng_state, set_global_seed

console = Console()
logger = get_logger(__name__)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _apply_attack(model: ModelHandle, attack_cfg: AttackConfig) -> AttackResult:
    AttackClass = get_attack(attack_cfg.name.split("@")[0] if "@" in attack_cfg.name else attack_cfg.name)
    # Actually: registry is keyed by attack identifier in YAML; for our YAMLs
    # the canonical names are "lora-finetune", "pap", "roleplay".
    attack = AttackClass(attack_cfg)
    return attack.apply(model)


def _load_lora_adapter(model: ModelHandle, adapter_path: Path) -> ModelHandle:
    """Load a saved LoRA adapter into the base model."""
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model.model, str(adapter_path))
    peft_model.eval()
    model.model = peft_model
    return model


def run_cell(experiment_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run one experiment cell. Returns the manifest dict.

    Args:
        experiment_config: Parsed YAML dict with keys:
            - model: short model name (str)
            - attack: attack config dict
            - defense: defense config dict
            - eval: eval config dict
            - judge: judge config dict
            - seed: int
            - generation: optional GenerationConfig overrides
        output_dir: Where to write the results.

    Returns:
        The manifest dict (also written to output_dir/manifest.json).
    """
    # Ensure plugins are registered
    autoload_attacks()
    autoload_defenses()
    autoload_evals()
    autoload_judges()

    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(experiment_config.get("seed", 0))
    set_global_seed(seed)
    rng_snapshot = capture_rng_state(seed)

    # --- Load model ---
    t0 = time.perf_counter()
    model_cfg = get_model_config(experiment_config["model"])
    handle = load_model(model_cfg)
    t_model_load = time.perf_counter() - t0
    logger.info("Model loaded", extra={"seconds": t_model_load, "model": model_cfg.name})

    # --- Apply attack ---
    t0 = time.perf_counter()
    attack_dict = dict(experiment_config["attack"])
    attack_dict["name"] = attack_dict.pop("plugin")
    attack_cfg = AttackConfig.from_dict(attack_dict)
    AttackClass = get_attack(attack_cfg.name)
    attack_result = AttackClass(attack_cfg).apply(handle)
    t_attack = time.perf_counter() - t0
    logger.info("Attack applied", extra={"seconds": t_attack, "attack": attack_cfg.name})

    # If WEIGHT_MOD with a saved adapter: load it on top of the base model.
    # If checkpoint_path is None, this is a no-attack control — skip loading.
    if attack_result.attack_type == "WEIGHT_MOD" and attack_result.checkpoint_path:
        handle = _load_lora_adapter(handle, attack_result.checkpoint_path)

    # --- Setup defense ---
    defense_dict = dict(experiment_config["defense"])
    defense_dict["name"] = defense_dict.pop("plugin")
    defense_cfg = DefenseConfig.from_dict(defense_dict)
    DefenseClass = get_defense(defense_cfg.name)
    defense = DefenseClass(defense_cfg)
    defense.setup()

    # --- Setup judge ---
    judge_dict = dict(experiment_config["judge"])
    judge_dict["name"] = judge_dict.pop("plugin")
    judge_cfg = JudgeConfig.from_dict(judge_dict)
    JudgeClass = get_judge(judge_cfg.name)
    judge = JudgeClass(judge_cfg)
    judge.setup()

    # --- Load eval prompts ---
    eval_dict = dict(experiment_config["eval"])
    eval_dict["name"] = eval_dict.pop("plugin")
    eval_cfg = EvalConfig.from_dict(eval_dict)
    EvalClass = get_eval(eval_cfg.name)
    eval_plugin = EvalClass(eval_cfg)
    prompts = eval_plugin.load_prompts()
    logger.info("Eval prompts loaded", extra={"n_prompts": len(prompts), "eval": eval_cfg.name})

    # --- Generation config ---
    gen_dict = experiment_config.get("generation", {})
    gen_config = GenerationConfig(**gen_dict)

    # --- Run inference + defense pipeline ---
    responses = []
    defense_decisions = []
    skipped = 0

    # Prompt transform from PROMPT_MOD attacks
    transform_template: str | None = None
    if attack_result.attack_type == "PROMPT_MOD":
        transform_template = attack_result.attack_prompt_transform

    for prompt_obj in prompts:
        prompt_text = prompt_obj.prompt
        if transform_template:
            prompt_text = transform_template.format(prompt=prompt_text)

        # Input filter
        input_decision = defense.filter_input(prompt_text)
        if not input_decision.allowed:
            # Substitute the refusal text as the "response"; skip model call
            from advsafe.types import GeneratedResponse

            responses.append(
                GeneratedResponse(
                    prompt_id=prompt_obj.id,
                    prompt=prompt_text,
                    response=input_decision.modified_text or "I'm not able to help with that.",
                    elapsed_seconds=0.0,
                    n_input_tokens=0,
                    n_output_tokens=0,
                    finish_reason="defense_blocked_input",
                    metadata={"defense_action": "input_blocked"},
                )
            )
            defense_decisions.append({"prompt_id": prompt_obj.id, "stage": "input", **asdict(input_decision)})
            skipped += 1
            continue

        # Wrap system prompt
        system_prompt = defense.wrap_system_prompt(experiment_config.get("system_prompt", ""))

        # Generate
        response = generate(
            handle,
            prompt=prompt_text,
            gen_config=gen_config,
            system_prompt=system_prompt or None,
            prompt_id=prompt_obj.id,
        )
        # Propagate eval-prompt metadata into the response so downstream
        # scoring (e.g., per-category ASR) can read it.
        response.metadata["prompt_category"] = prompt_obj.category
        response.metadata["expected_behavior"] = prompt_obj.expected_behavior
        response.metadata.update(prompt_obj.metadata or {})

        # Output filter
        output_decision = defense.filter_output(prompt_text, response.response)
        if not output_decision.allowed and output_decision.modified_text:
            response.response = output_decision.modified_text
            response.finish_reason = "defense_blocked_output"
            response.metadata["defense_action"] = "output_blocked"

        defense_decisions.append(
            {"prompt_id": prompt_obj.id, "stage": "output", **asdict(output_decision)}
        )
        responses.append(response)

    # --- Score (returns both aggregate score and per-prompt verdicts) ---
    t0 = time.perf_counter()
    score, verdicts = eval_plugin.score(responses, judge)
    t_score = time.perf_counter() - t0
    logger.info("Scoring complete", extra={"seconds": t_score, "asr": score.asr})

    # --- Persist outputs ---
    # Merge per-prompt verdicts into responses.jsonl so downstream analyses
    # (Cross-Attack Transferability needs is_harmful per prompt) can read them.
    verdict_by_prompt = {v.prompt_id: v for v in verdicts}
    response_rows = []
    for r in responses:
        row = asdict(r)
        v = verdict_by_prompt.get(r.prompt_id)
        if v is not None:
            row["is_harmful"] = v.is_harmful
            row["judge_category"] = v.category
            row["judge_confidence"] = v.confidence
        response_rows.append(row)
    (output_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(row, default=str) for row in response_rows)
    )
    (output_dir / "defense_decisions.jsonl").write_text(
        "\n".join(json.dumps(d, default=str) for d in defense_decisions)
    )
    (output_dir / "score.json").write_text(json.dumps(asdict(score), default=str, indent=2))

    # --- Manifest ---
    manifest = {
        "experiment_id": experiment_config.get("id", "unnamed"),
        "model": {
            "name": model_cfg.name,
            "hf_id": model_cfg.hf_id,
            "revision_sha": handle.revision_sha,
            "device": str(handle.device),
            "dtype": str(handle.dtype),
        },
        "attack": {
            "name": attack_cfg.name,
            "type": attack_result.attack_type,
            "checkpoint_path": str(attack_result.checkpoint_path) if attack_result.checkpoint_path else None,
            "metadata": attack_result.metadata,
        },
        "defense": {"name": defense_cfg.name},
        "eval": {
            "name": eval_cfg.name,
            "n_prompts": len(prompts),
            "n_skipped_by_input_filter": skipped,
        },
        "judge": {"name": judge_cfg.name},
        "seed": seed,
        "rng_snapshot": asdict(rng_snapshot),
        "score": asdict(score),
        "timings": {
            "model_load_s": t_model_load,
            "attack_s": t_attack,
            "score_s": t_score,
        },
        "device_info": asdict(describe_device(handle.device)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2))

    judge.teardown()
    defense.teardown()

    return manifest


@click.command()
@click.option("--experiment", "experiment_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", default=None, help="Output directory (defaults to results/<id>/)")
def cli(experiment_path: str, output_dir: str | None) -> None:
    """Run a single experiment cell from a YAML config."""
    setup_logging("INFO")
    with Path(experiment_path).open() as f:
        config = yaml.safe_load(f)

    exp_id = config.get("id", Path(experiment_path).stem)
    out = Path(output_dir or f"results/{exp_id}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = run_cell(config, out)

    # Print summary
    table = Table(title=f"Experiment {exp_id}")
    table.add_column("metric")
    table.add_column("value")
    asr = manifest["score"].get("asr")
    lo = manifest["score"].get("asr_ci_low")
    hi = manifest["score"].get("asr_ci_high")
    table.add_row("ASR", f"{asr:.3f}" if isinstance(asr, (int, float)) else "n/a")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        table.add_row("CI", f"[{lo:.3f}, {hi:.3f}]")
    else:
        table.add_row("CI", "n/a")
    table.add_row("Skipped by input filter", str(manifest["eval"]["n_skipped_by_input_filter"]))
    table.add_row("Output", str(out))
    console.print(table)


if __name__ == "__main__":
    cli()
