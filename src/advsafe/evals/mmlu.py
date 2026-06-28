"""MMLU subset eval (Hendrycks et al. 2021).

Capability check: ensures defenses don't tank general knowledge accuracy.
We use a 500-question stratified subset across 4 categories for speed.

Reference:
    Hendrycks et al. (2021) "Measuring Massive Multitask Language Understanding."

Status: scaffolded, NOT implemented. Registered so configs validate and
``list_evals()`` is complete, but both hooks fail loudly rather than silently
returning 0 prompts / a null score — which would let a sweep cell "succeed"
while measuring nothing. No experiment config selects ``mmlu`` today.
"""

from __future__ import annotations

from advsafe.evals.base import EvalPlugin, register_eval
from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse

_NOT_IMPLEMENTED = (
    "The 'mmlu' eval is a scaffold only: no prompts or scoring are implemented yet, and "
    "no experiment config selects it. Measure MMLU with lm-eval-harness, or choose "
    "harmbench/strongreject/mt-bench/xstest. To implement it here, load `cais/mmlu`, "
    "format multiple-choice prompts, and score via generate()/mlx_lm."
)


@register_eval("mmlu")
class MMLUEval(EvalPlugin):
    """MMLU capability subset — registered but not yet implemented (errors clearly)."""

    def load_prompts(self) -> list[EvalPrompt]:
        # Guard: fail fast instead of returning [] (which would yield a null-score run).
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def score(
        self, responses: list[GeneratedResponse], judge: JudgePlugin
    ) -> tuple[EvalScore, list]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
