"""Tests for the shared dataclass types."""

from __future__ import annotations

from advsafe.types import (
    AttackResult,
    DefenseDecision,
    EvalPrompt,
    EvalScore,
    GeneratedResponse,
    GenerationConfig,
    JudgeVerdict,
    ModelConfig,
)


def test_model_config_minimal():
    cfg = ModelConfig(name="x", hf_id="x/y")
    assert cfg.name == "x"
    assert cfg.family == "llama"  # default


def test_generation_config_defaults_are_greedy():
    cfg = GenerationConfig()
    assert cfg.temperature == 0.0
    assert cfg.do_sample is False


def test_eval_prompt_round_trips():
    p = EvalPrompt(id="1", prompt="hello", category="benign")
    assert p.id == "1"
    assert p.expected_behavior == "refuse"  # default


def test_generated_response_minimal():
    r = GeneratedResponse(
        prompt_id="1",
        prompt="x",
        response="y",
        elapsed_seconds=0.1,
        n_input_tokens=1,
        n_output_tokens=2,
    )
    assert r.finish_reason == "stop"


def test_defense_decision_default_allowed():
    d = DefenseDecision(allowed=True)
    assert d.allowed is True
    assert d.modified_text is None


def test_judge_verdict_minimal():
    v = JudgeVerdict(prompt_id="1", is_harmful=False)
    assert not v.is_harmful


def test_attack_result_weight_mod():
    r = AttackResult(attack_name="x", attack_type="WEIGHT_MOD")
    assert r.checkpoint_path is None
    assert r.attack_prompt_transform is None


def test_eval_score_minimal():
    s = EvalScore(eval_name="harmbench", n_prompts=100, asr=0.3)
    assert s.asr == 0.3
    assert s.per_category_asr == {}
