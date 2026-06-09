"""MMLU subset eval (Hendrycks et al. 2021).

Capability check: ensures defenses don't tank general knowledge accuracy.
We use a 500-question stratified subset across 4 categories for speed.

Reference:
    Hendrycks et al. (2021) "Measuring Massive Multitask Language Understanding."

Status: scaffolded. Full MMLU implementation deferred to follow-up work.
"""

from __future__ import annotations

from advsafe.evals.base import EvalPlugin, register_eval
from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse


@register_eval("mmlu")
class MMLUEval(EvalPlugin):
    """MMLU capability subset (TODO: full implementation)."""

    def load_prompts(self) -> list[EvalPrompt]:
        # Placeholder: the real implementation will pull from HF datasets
        # `cais/mmlu` and format multiple-choice prompts.
        return []

    def score(
        self, responses: list[GeneratedResponse], judge: JudgePlugin
    ) -> tuple[EvalScore, list]:
        return (
            EvalScore(
                eval_name=self.name,
                n_prompts=0,
                score=None,
                metadata={"note": "MMLU implementation deferred; use lm-eval-harness for now"},
            ),
            [],
        )
