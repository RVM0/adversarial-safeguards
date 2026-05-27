"""D2 — Llama Guard 3 output-side filter.

Lets the target model respond, then classifies the *response*. If unsafe,
substitutes a refusal. This catches cases where the prompt looked innocent
but the model produced harmful content (the LoRA-attack failure mode).
"""

from __future__ import annotations

from advsafe.defenses.base import DefenseConfig, DefensePlugin, register_defense
from advsafe.judges.llama_guard import LlamaGuardClassifier
from advsafe.types import DefenseDecision
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


@register_defense("llama-guard-output")
class LlamaGuardOutputDefense(DefensePlugin):
    """Llama Guard 3 output-side classifier."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._classifier: LlamaGuardClassifier | None = None
        self._block_categories: set[str] = set(config.block_categories or [
            "S1", "S2", "S3", "S4", "S5", "S6", "S7",
            "S8", "S9", "S10", "S11", "S12", "S13", "S14",
        ])

    def setup(self) -> None:
        if self._classifier is None:
            self._classifier = LlamaGuardClassifier(
                hf_id=self.config.guard_model_id or "meta-llama/Llama-Guard-3-8B",
                revision=self.config.guard_revision,
            )
            self._classifier.setup()

    def filter_output(self, prompt: str, response: str) -> DefenseDecision:
        if self._classifier is None:
            self.setup()
        assert self._classifier is not None

        # Llama Guard 3 supports two-turn classification (prompt + assistant response)
        verdict = self._classifier.classify_conversation(
            user_message=prompt, assistant_message=response
        )
        if not verdict.is_unsafe:
            return DefenseDecision(
                allowed=True,
                reason="llama-guard-3: response safe",
                category=None,
                confidence=verdict.confidence,
            )
        if verdict.category and verdict.category in self._block_categories:
            return DefenseDecision(
                allowed=False,
                reason=f"llama-guard-3: response blocked ({verdict.category})",
                category=verdict.category,
                confidence=verdict.confidence,
                modified_text="I'm not able to help with that request.",
            )
        return DefenseDecision(
            allowed=True,
            reason=f"llama-guard-3: response flagged but category {verdict.category} not blocked",
            category=verdict.category,
            confidence=verdict.confidence,
        )

    def teardown(self) -> None:
        if self._classifier is not None:
            self._classifier.teardown()
