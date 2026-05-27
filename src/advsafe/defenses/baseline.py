"""D0 — Baseline (no defense). Identity pass-through."""

from __future__ import annotations

from advsafe.defenses.base import DefenseConfig, DefensePlugin, register_defense


@register_defense("baseline")
class BaselineDefense(DefensePlugin):
    """Identity defense — no filtering, no system prompt modification.

    This is the D0 condition: the raw attacked model output. Always present in
    the panel as the control.
    """

    # Inherits no-op implementations from DefensePlugin.
    pass
