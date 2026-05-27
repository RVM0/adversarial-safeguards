"""Judge plugins.

A judge consumes (prompt, response) pairs and emits a JudgeVerdict
(binary harmful / not, plus optional category, confidence, raw output).
"""

from advsafe.judges.base import JudgeConfig, JudgePlugin, get_judge, list_judges, register_judge

__all__ = [
    "JudgeConfig",
    "JudgePlugin",
    "get_judge",
    "list_judges",
    "register_judge",
]
