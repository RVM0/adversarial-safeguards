"""MT-Bench eval (Zheng et al. 2023).

Utility / chat capability benchmark — 80 multi-turn prompts, LLM-as-judge
scoring 1-10. Used to verify that defenses don't tank utility on benign prompts.

Reference:
    Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."

Status: scaffolded; full multi-turn handling + 1-10 rubric scoring is a TODO.
For the pilot, we run only the first turn and score binary (acceptable / not).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from advsafe.evals.base import EvalConfig, EvalPlugin, register_eval
from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse


@register_eval("mt-bench")
class MTBenchEval(EvalPlugin):
    """MT-Bench utility eval. First-turn only for now."""

    def load_prompts(self) -> list[EvalPrompt]:
        path = Path(self.config.dataset_path or "data/mt_bench/question.jsonl")
        if not path.exists():
            raise FileNotFoundError(
                f"MT-Bench data not found at {path}. "
                "Run `scripts/download_datasets.sh` to fetch it."
            )
        prompts: list[EvalPrompt] = []
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                turns = row.get("turns", [])
                if not turns:
                    continue
                prompts.append(
                    EvalPrompt(
                        id=str(row["question_id"]),
                        prompt=turns[0],
                        category=row.get("category", "unspecified"),
                        expected_behavior="comply",
                        metadata={"source": "mt-bench", "all_turns": turns},
                    )
                )
        if self.config.n_prompts and self.config.n_prompts < len(prompts):
            prompts = prompts[: self.config.n_prompts]
        return prompts

    def score(
        self, responses: list[GeneratedResponse], judge: JudgePlugin
    ) -> tuple[EvalScore, list]:
        # For now: a heuristic acceptable-rate based on response length + non-refusal.
        # A real implementation would use a 1-10 LLM-as-judge rubric.
        accepted = 0
        for r in responses:
            if len(r.response.split()) >= 20 and "I cannot" not in r.response[:50]:
                accepted += 1
        n = len(responses)
        rate = accepted / n if n else 0.0
        # No judge verdicts for utility eval; return empty list
        return (
            EvalScore(
                eval_name=self.name,
                n_prompts=n,
                score=rate,
                metadata={
                    "note": "First-turn-only heuristic; full MT-Bench rubric is a TODO",
                    "judge_name": judge.name,
                },
            ),
            [],
        )
