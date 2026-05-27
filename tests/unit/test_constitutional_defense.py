"""Unit test for the constitutional-prompt defense.

This defense is pure-Python (no model load), so we can test it directly.
"""

from __future__ import annotations

from advsafe.defenses.base import DefenseConfig
from advsafe.defenses.constitutional_prompt import ConstitutionalPromptDefense


def test_constitutional_prepends_default_when_no_system_prompt():
    defense = ConstitutionalPromptDefense(DefenseConfig(name="d"))
    out = defense.wrap_system_prompt("")
    assert "harmless" in out.lower()
    assert "weapons" in out.lower()


def test_constitutional_prepends_to_caller_system_prompt():
    defense = ConstitutionalPromptDefense(DefenseConfig(name="d"))
    out = defense.wrap_system_prompt("You are a poetry assistant.")
    # Constitution comes first, then separator, then caller's content.
    assert out.index("harmless") < out.index("poetry assistant")


def test_constitutional_uses_inline_text_if_provided():
    cfg = DefenseConfig(name="d", system_prompt_inline="Be safe.")
    defense = ConstitutionalPromptDefense(cfg)
    out = defense.wrap_system_prompt("")
    assert out.strip() == "Be safe."


def test_filter_input_passes_through():
    defense = ConstitutionalPromptDefense(DefenseConfig(name="d"))
    decision = defense.filter_input("hello")
    assert decision.allowed is True


def test_filter_output_passes_through():
    defense = ConstitutionalPromptDefense(DefenseConfig(name="d"))
    decision = defense.filter_output("hello", "world")
    assert decision.allowed is True
