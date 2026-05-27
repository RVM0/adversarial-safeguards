"""StrongREJECT eval (Souly et al. 2024).

Secondary harmfulness benchmark with rubric-based scoring. Designed to be
harder to game than HarmBench's binary classification.

Reference:
    Souly et al. (2024) "A StrongREJECT for Empty Jailbreaks."
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from advsafe.evals.base import EvalConfig, EvalPlugin, register_eval
from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


@register_eval("strongreject")
class StrongREJECTEval(EvalPlugin):
    """StrongREJECT eval — rubric-based harmfulness score.

    Status: scaffolded. Full rubric judging is delegated to the judge plugin
    (which should implement the StrongREJECT rubric — currently we fall back
    to binary judgment, marked as a known limitation in the manifest).
    """

    def load_prompts(self) -> list[EvalPrompt]:
        path = Path(self.config.dataset_path or "data/strongreject/strongreject_dataset.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"StrongREJECT data not found at {path}. "
                "Run `scripts/download_datasets.sh` to fetch it."
            )
        import csv

        prompts: list[EvalPrompt] = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("id") or row.get("BehaviorID") or str(len(prompts))
                text = row.get("forbidden_prompt") or row.get("prompt")
                category = row.get("category") or "unspecified"
                if not text:
                    continue
                prompts.append(
                    EvalPrompt(
                        id=str(pid),
                        prompt=str(text),
                        category=str(category),
                        expected_behavior="refuse",
                        metadata={"source": "strongreject"},
                    )
                )

        if self.config.n_prompts and self.config.n_prompts < len(prompts):
            rng = np.random.default_rng(self.config.seed)
            indices = rng.choice(len(prompts), size=self.config.n_prompts, replace=False)
            prompts = [prompts[i] for i in indices]

        return prompts

    def score(
        self, responses: list[GeneratedResponse], judge: JudgePlugin
    ) -> tuple[EvalScore, list]:
        verdicts = judge.judge_batch(responses)
        harmful = np.array([1 if v.is_harmful else 0 for v in verdicts])
        n = len(harmful)
        asr = float(harmful.mean()) if n else 0.0

        # Bootstrap CI
        rng = np.random.default_rng(self.config.seed)
        boots = np.array([
            rng.choice(harmful, size=n, replace=True).mean()
            for _ in range(self.config.bootstrap_iterations)
        ])
        alpha = (1 - self.config.confidence_level) / 2

        return (
            EvalScore(
                eval_name=self.name,
                n_prompts=n,
                asr=asr,
                asr_ci_low=float(np.quantile(boots, alpha)),
                asr_ci_high=float(np.quantile(boots, 1 - alpha)),
                metadata={
                    "judge_name": judge.name,
                    "note": "Binary judgment fallback; full StrongREJECT rubric is a TODO",
                },
            ),
            verdicts,
        )
