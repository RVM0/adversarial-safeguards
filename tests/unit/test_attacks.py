"""Tests for the inference-time attack plugins (A2 PAP, A3 roleplay) + the template module.

These are torch-free: PROMPT_MOD attacks never load a model, they only return a prompt
transformation. We verify the transform is well-formed (carries the {prompt} slot, applies
cleanly the way the runner does), that validation catches bad ids, and that the built-in
published set is used when no data/ override exists.
"""

from __future__ import annotations

import json

import pytest

from advsafe.attacks.base import AttackConfig
from advsafe.attacks.pap import PAPAttack
from advsafe.attacks.roleplay import RoleplayAttack
from advsafe.attacks.templates import (
    BUILTIN,
    PAP_TEMPLATES,
    PLACEHOLDER,
    ROLEPLAY_TEMPLATES,
    get_template,
    list_strategies,
    load_templates,
)

# ----- template module ------------------------------------------------------


def test_all_builtin_templates_carry_the_prompt_slot() -> None:
    for kind, mapping in BUILTIN.items():
        for tid, tmpl in mapping.items():
            assert PLACEHOLDER in tmpl, f"{kind}/{tid} is missing {PLACEHOLDER}"


def test_builtin_templates_format_cleanly() -> None:
    # The runner applies the transform via str.format(prompt=...); a stray brace would
    # raise. Every built-in must substitute cleanly and contain the substituted text.
    for mapping in BUILTIN.values():
        for tmpl in mapping.values():
            out = tmpl.format(prompt="BENCHMARK_PROMPT")
            assert "BENCHMARK_PROMPT" in out


def test_pap_has_expanded_strategy_set() -> None:
    assert len(PAP_TEMPLATES) >= 10  # expanded from the original 3
    assert "logical_appeal" in PAP_TEMPLATES


def test_get_template_unknown_id_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_template("pap", "no_such_strategy", path="data/does-not-exist.json")


def test_load_templates_unknown_kind_raises_valueerror() -> None:
    with pytest.raises(ValueError, match="Unknown template kind"):
        load_templates("bogus")


def test_get_template_missing_placeholder_raises(tmp_path) -> None:
    override = tmp_path / "pap.json"
    override.write_text(json.dumps({"broken": "no placeholder here"}))
    with pytest.raises(ValueError, match="placeholder"):
        get_template("pap", "broken", path=override)


def test_data_override_takes_precedence_else_builtin(tmp_path) -> None:
    override = tmp_path / "pap.json"
    override.write_text(json.dumps({"custom": "wrap {prompt}"}))
    assert load_templates("pap", override) == {"custom": "wrap {prompt}"}
    # Missing override → built-in published set.
    assert load_templates("pap", tmp_path / "missing.json") == PAP_TEMPLATES
    assert load_templates("roleplay", None) == ROLEPLAY_TEMPLATES


def test_list_strategies() -> None:
    assert list_strategies("pap") == sorted(PAP_TEMPLATES)
    assert "dan" in list_strategies("roleplay")


# ----- PAP plugin -----------------------------------------------------------


def _nonexistent(tmp_path) -> str:
    return str(tmp_path / "nope.json")  # forces the built-in path


def test_pap_attack_returns_applicable_transform(tmp_path) -> None:
    cfg = AttackConfig(
        name="pap", template_id="authority_endorsement", template_path=_nonexistent(tmp_path)
    )
    result = PAPAttack(cfg).apply(None)  # PROMPT_MOD: model is unused
    assert result.attack_type == "PROMPT_MOD"
    assert result.checkpoint_path is None
    assert PLACEHOLDER in result.attack_prompt_transform
    applied = result.attack_prompt_transform.format(prompt="X_BENCH")
    assert "X_BENCH" in applied
    assert result.metadata["template_source"] == "built-in"
    assert "authority_endorsement" in result.metadata["available_strategies"]


def test_pap_defaults_to_logical_appeal(tmp_path) -> None:
    cfg = AttackConfig(name="pap", template_path=_nonexistent(tmp_path))  # no template_id
    result = PAPAttack(cfg).apply(None)
    assert result.metadata["strategy_id"] == "logical_appeal"


def test_pap_unknown_strategy_raises(tmp_path) -> None:
    cfg = AttackConfig(
        name="pap", template_id="does_not_exist", template_path=_nonexistent(tmp_path)
    )
    with pytest.raises(KeyError):
        PAPAttack(cfg).apply(None)


# ----- roleplay plugin ------------------------------------------------------


def test_roleplay_attack_returns_applicable_transform(tmp_path) -> None:
    cfg = AttackConfig(name="roleplay", template_id="dan", template_path=_nonexistent(tmp_path))
    result = RoleplayAttack(cfg).apply(None)
    assert result.attack_type == "PROMPT_MOD"
    applied = result.attack_prompt_transform.format(prompt="X_BENCH")
    assert "X_BENCH" in applied
    assert result.metadata["template_id"] == "dan"


def test_roleplay_defaults_to_dan(tmp_path) -> None:
    cfg = AttackConfig(name="roleplay", template_path=_nonexistent(tmp_path))
    result = RoleplayAttack(cfg).apply(None)
    assert result.metadata["template_id"] == "dan"
