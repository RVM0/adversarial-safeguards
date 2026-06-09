"""D1 — Llama Guard 3 input-side filter.

Wraps the input prompt with Llama Guard 3's classifier. If the prompt is
flagged as unsafe under any of the configured categories (S1–S14 by default),
the defense refuses without ever calling the target model.

Reference:
    Meta (2024) "Llama Guard 3 Model Card."
"""

from __future__ import annotations

from advsafe.defenses.base import DefenseConfig, DefensePlugin, register_defense
from advsafe.judges.llama_guard import LlamaGuardClassifier
from advsafe.types import DefenseDecision
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


@register_defense("llama-guard-input")
class LlamaGuardInputDefense(DefensePlugin):
    """Llama Guard 3 input-side classifier."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self._classifier: LlamaGuardClassifier | None = None
        self._block_categories: set[str] = set(
            config.block_categories
            or [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8",
                "S9",
                "S10",
                "S11",
                "S12",
                "S13",
                "S14",
            ]
        )

    def setup(self) -> None:
        if self._classifier is None:
            self._classifier = LlamaGuardClassifier(
                hf_id=self.config.guard_model_id or "meta-llama/Llama-Guard-3-8B",
                revision=self.config.guard_revision,
            )
            self._classifier.setup()

    def filter_input(self, prompt: str) -> DefenseDecision:
        if self._classifier is None:
            self.setup()
        assert self._classifier is not None

        verdict = self._classifier.classify(role="user", content=prompt)
        if not verdict.is_unsafe:
            return DefenseDecision(
                allowed=True,
                reason="llama-guard-3: safe",
                category=None,
                confidence=verdict.confidence,
            )
        if verdict.category and verdict.category in self._block_categories:
            return DefenseDecision(
                allowed=False,
                reason=f"llama-guard-3: blocked ({verdict.category})",
                category=verdict.category,
                confidence=verdict.confidence,
                modified_text="I'm not able to help with that request.",
            )
        # Flagged but not in our block list — let it through
        return DefenseDecision(
            allowed=True,
            reason=f"llama-guard-3: flagged but category {verdict.category} not in block list",
            category=verdict.category,
            confidence=verdict.confidence,
        )

    def teardown(self) -> None:
        if self._classifier is not None:
            self._classifier.teardown()
