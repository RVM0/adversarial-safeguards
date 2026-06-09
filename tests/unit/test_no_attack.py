"""Tests for the NoAttack plugin and the n_examples=0 short-circuit in LoRAFineTuneAttack."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_no_attack_returns_no_checkpoint():
    from advsafe.attacks.base import AttackConfig, AttackType
    from advsafe.attacks.no_attack import NoAttack

    cfg = AttackConfig(name="no-attack", attack_type=AttackType.WEIGHT_MOD)
    attack = NoAttack(cfg)

    fake_model = MagicMock()
    fake_model.config.name = "fake-model"
    result = attack.apply(fake_model)

    assert result.attack_type == "WEIGHT_MOD"
    assert result.checkpoint_path is None  # signals no adapter to load
    assert result.attack_name == "no-attack"


@pytest.mark.ml_stack
def test_lora_with_zero_examples_short_circuits():
    """LoRAFineTuneAttack with n_examples=0 should return no checkpoint, not crash."""
    from advsafe.attacks.base import AttackConfig, AttackType
    from advsafe.attacks.lora_finetune import LoRAFineTuneAttack

    cfg = AttackConfig(
        name="lora-finetune",
        attack_type=AttackType.WEIGHT_MOD,
        n_examples=0,
        dataset_path="/does/not/exist.jsonl",
    )
    attack = LoRAFineTuneAttack(cfg)

    fake_model = MagicMock()
    fake_model.config.name = "fake-model"
    result = attack.apply(fake_model)

    assert result.checkpoint_path is None
    assert "no-attack" in result.metadata.get("note", "").lower()


@pytest.mark.ml_stack
def test_no_attack_registered():
    from advsafe.attacks.base import autoload, list_attacks

    autoload()
    assert "no-attack" in list_attacks()
