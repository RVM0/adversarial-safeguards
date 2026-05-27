"""Evaluation plugins.

Each eval loads a benchmark dataset and provides a `score()` method that
combines model responses with a judge's verdicts to produce an EvalScore.
"""

from advsafe.evals.base import (
    EvalConfig,
    EvalPlugin,
    get_eval,
    list_evals,
    register_eval,
)

__all__ = [
    "EvalConfig",
    "EvalPlugin",
    "get_eval",
    "list_evals",
    "register_eval",
]
