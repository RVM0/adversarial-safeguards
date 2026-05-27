"""Tests for the AttackConfig.from_dict / DefenseConfig.from_dict / EvalConfig.from_dict
unknown-fields-to-extra routing.

These methods make YAML configs forgiving: adding new keys for new plugins
doesn't break the loader.
"""

from __future__ import annotations


def test_attack_config_from_dict_known_keys():
    from advsafe.attacks.base import AttackConfig, AttackType

    data = {
        "name": "lora-finetune",
        "attack_type": "WEIGHT_MOD",
        "n_examples": 100,
        "epochs": 3,
    }
    cfg = AttackConfig.from_dict(data)
    assert cfg.name == "lora-finetune"
    assert cfg.attack_type == AttackType.WEIGHT_MOD
    assert cfg.n_examples == 100
    assert cfg.extra == {}


def test_attack_config_from_dict_unknown_keys_go_to_extra():
    from advsafe.attacks.base import AttackConfig, AttackType

    data = {
        "name": "lora-finetune",
        "attack_type": "WEIGHT_MOD",
        "n_examples": 100,
        "future_field": "future_value",
        "another_unknown": 42,
    }
    cfg = AttackConfig.from_dict(data)
    assert cfg.n_examples == 100
    assert cfg.extra == {"future_field": "future_value", "another_unknown": 42}


def test_defense_config_from_dict():
    from advsafe.defenses.base import DefenseConfig

    data = {"name": "baseline", "novel_param": True}
    cfg = DefenseConfig.from_dict(data)
    assert cfg.name == "baseline"
    assert cfg.extra == {"novel_param": True}


def test_eval_config_from_dict():
    from advsafe.evals.base import EvalConfig

    data = {"name": "harmbench", "n_prompts": 50, "custom_judging_mode": "strict"}
    cfg = EvalConfig.from_dict(data)
    assert cfg.name == "harmbench"
    assert cfg.n_prompts == 50
    assert cfg.extra == {"custom_judging_mode": "strict"}


def test_judge_config_from_dict():
    from advsafe.judges.base import JudgeConfig

    data = {"name": "llama-guard-3", "max_new_tokens": 64, "future_knob": "x"}
    cfg = JudgeConfig.from_dict(data)
    assert cfg.max_new_tokens == 64
    assert cfg.extra == {"future_knob": "x"}
