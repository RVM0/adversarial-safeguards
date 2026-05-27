"""Verify that all plugins self-register correctly on import.

Catches: typos in @register_* decorators, duplicate names, missing autoload
calls in __init__.py.
"""

from __future__ import annotations

import pytest


def test_attack_registry():
    from advsafe.attacks.base import autoload, list_attacks

    autoload()
    names = list_attacks()
    assert "lora-finetune" in names
    assert "pap" in names
    assert "roleplay" in names


def test_defense_registry():
    from advsafe.defenses.base import autoload, list_defenses

    autoload()
    names = list_defenses()
    for expected in (
        "baseline",
        "llama-guard-input",
        "llama-guard-output",
        "constitutional-prompt",
        "combined",
    ):
        assert expected in names, f"defense '{expected}' missing from registry"


def test_eval_registry():
    from advsafe.evals.base import autoload, list_evals

    autoload()
    names = list_evals()
    for expected in ("harmbench", "strongreject", "mt-bench", "xstest", "mmlu"):
        assert expected in names, f"eval '{expected}' missing from registry"


def test_judge_registry():
    from advsafe.judges.base import autoload, list_judges

    autoload()
    names = list_judges()
    assert "llama-guard-3" in names
    assert "openai-gpt-4o-mini" in names


def test_duplicate_registration_fails():
    """Double-registering a name should raise."""
    from advsafe.attacks.base import AttackPlugin, register_attack

    @register_attack("test_duplicate_xyz")
    class TestAttack1(AttackPlugin):
        def apply(self, model):
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):

        @register_attack("test_duplicate_xyz")
        class TestAttack2(AttackPlugin):
            def apply(self, model):
                raise NotImplementedError
