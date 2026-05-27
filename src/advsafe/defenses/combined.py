"""D4 — Combined defense (D1 + D2 + D3).

Defense-in-depth: applies the input filter, output filter, and constitutional
system prompt simultaneously. Measures whether stacking defenses materially
improves over any single defense.
"""

from __future__ import annotations

from advsafe.defenses.base import DefenseConfig, DefensePlugin, register_defense
from advsafe.defenses.constitutional_prompt import ConstitutionalPromptDefense
from advsafe.defenses.llama_guard_input import LlamaGuardInputDefense
from advsafe.defenses.llama_guard_output import LlamaGuardOutputDefense
from advsafe.types import DefenseDecision


@register_defense("combined")
class CombinedDefense(DefensePlugin):
    """Defense-in-depth: D1 (input filter) + D2 (output filter) + D3 (constitution).

    The Llama Guard classifier is shared between input and output to avoid
    loading the model twice.
    """

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        # Build sub-defenses with the same config; they each pick up the guard model
        self._input = LlamaGuardInputDefense(config)
        self._output = LlamaGuardOutputDefense(config)
        self._constitution = ConstitutionalPromptDefense(config)

    def setup(self) -> None:
        # Shared classifier: setup once, attach to both
        self._input.setup()
        # Reuse the loaded classifier for output (avoid double-loading 8B weights)
        self._output._classifier = self._input._classifier  # noqa: SLF001
        self._constitution.setup()

    def filter_input(self, prompt: str) -> DefenseDecision:
        return self._input.filter_input(prompt)

    def filter_output(self, prompt: str, response: str) -> DefenseDecision:
        return self._output.filter_output(prompt, response)

    def wrap_system_prompt(self, system_prompt: str) -> str:
        return self._constitution.wrap_system_prompt(system_prompt)

    def teardown(self) -> None:
        # Only tear down once; sub-defenses share the same classifier
        self._input.teardown()
        # _output shares the classifier; just clear its reference
        self._output._classifier = None  # noqa: SLF001
