"""HarmBench eval (Mazeika et al. 2024).

Primary harmfulness benchmark in the panel. Loads the standard HarmBench test
set, runs the model on each prompt, judges with the configured judge, and
reports Attack Success Rate (ASR) with bootstrap 95% CIs.

Reference:
    Mazeika et al. (2024) "HarmBench: A Standardized Evaluation Framework for
    Automated Red Teaming and Robust Refusal."
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from advsafe.evals.base import EvalPlugin, register_eval
from advsafe.judges.base import JudgePlugin
from advsafe.types import EvalPrompt, EvalScore, GeneratedResponse
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


@register_eval("harmbench")
class HarmBenchEval(EvalPlugin):
    """HarmBench harmfulness eval."""

    def load_prompts(self) -> list[EvalPrompt]:
        path = Path(self.config.dataset_path or "data/harmbench/harmbench_test.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"HarmBench data not found at {path}. "
                "Run `scripts/download_datasets.sh` to fetch it."
            )

        prompts: list[EvalPrompt] = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("BehaviorID") or row.get("behavior_id") or row.get("id")
                text = row.get("Behavior") or row.get("behavior") or row.get("prompt")
                # The CAIS HarmBench CSV uses SemanticCategory/FunctionalCategory (no
                # "Category" column), so without these fallbacks every prompt collapses
                # to "unspecified" and per-category ASR is degenerate.
                category = (
                    row.get("Category")
                    or row.get("category")
                    or row.get("SemanticCategory")
                    or row.get("FunctionalCategory")
                    or "unspecified"
                )
                if not pid or not text:
                    continue
                prompts.append(
                    EvalPrompt(
                        id=str(pid),
                        prompt=str(text),
                        category=str(category),
                        expected_behavior="refuse",
                        metadata={"source": "harmbench"},
                    )
                )

        if self.config.n_prompts and self.config.n_prompts < len(prompts):
            rng = np.random.default_rng(self.config.seed)
            indices = rng.choice(len(prompts), size=self.config.n_prompts, replace=False)
            prompts = [prompts[i] for i in indices]

        logger.info("Loaded HarmBench prompts", extra={"n": len(prompts), "path": str(path)})
        return prompts

    def score(
        self, responses: list[GeneratedResponse], judge: JudgePlugin
    ) -> tuple[EvalScore, list]:
        verdicts = judge.judge_batch(responses)
        harmful = np.array([1 if v.is_harmful else 0 for v in verdicts])
        n = len(harmful)
        asr = float(harmful.mean()) if n else 0.0

        # Bootstrap 95% CI
        rng = np.random.default_rng(self.config.seed)
        boots = []
        for _ in range(self.config.bootstrap_iterations):
            sample = rng.choice(harmful, size=n, replace=True)
            boots.append(sample.mean())
        boots = np.array(boots)
        alpha = (1 - self.config.confidence_level) / 2
        ci_low = float(np.quantile(boots, alpha))
        ci_high = float(np.quantile(boots, 1 - alpha))

        # Per-category ASR (categories are propagated into response.metadata
        # by the runner; falls back to "unspecified" if missing).
        per_cat: dict[str, list[int]] = {}
        for r, v in zip(responses, verdicts, strict=False):
            cat = r.metadata.get("prompt_category", "unspecified")
            per_cat.setdefault(str(cat), []).append(1 if v.is_harmful else 0)
        per_cat_asr = {k: float(np.mean(v)) for k, v in per_cat.items() if v}

        return (
            EvalScore(
                eval_name=self.name,
                n_prompts=n,
                asr=asr,
                asr_ci_low=ci_low,
                asr_ci_high=ci_high,
                per_category_asr=per_cat_asr,
                metadata={
                    "judge_name": judge.name,
                    "n_harmful": int(harmful.sum()),
                    "bootstrap_iterations": self.config.bootstrap_iterations,
                },
            ),
            verdicts,
        )

    def save_results(
        self,
        responses: list[GeneratedResponse],
        verdicts: list,
        score: EvalScore,
        output_dir: str | Path,
    ) -> None:
        """Persist per-prompt outputs + aggregate score to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cell-level outputs (one row per prompt)
        records = []
        for r, v in zip(responses, verdicts, strict=False):
            records.append(
                {
                    "prompt_id": r.prompt_id,
                    "prompt": r.prompt,
                    "response": r.response,
                    "is_harmful": v.is_harmful,
                    "judge_category": v.category,
                    "judge_raw": v.raw_output,
                    "elapsed_s": r.elapsed_seconds,
                    "n_input_tokens": r.n_input_tokens,
                    "n_output_tokens": r.n_output_tokens,
                }
            )
        (output_dir / "per_prompt.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        (output_dir / "score.json").write_text(json.dumps(score.__dict__, default=str, indent=2))
