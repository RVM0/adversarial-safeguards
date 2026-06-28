"""Shared data types used across attacks, defenses, evals, and judges.

These are the lingua franca between plugin layers. Each layer accepts and
returns these typed dataclasses; the runner is responsible for plumbing
them through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Heavy ML deps are needed only for the annotations on ModelHandle.
    # `from __future__ import annotations` makes all annotations lazy strings,
    # so importing this module (types, configs, registries, analysis) does not
    # require torch/transformers to be installed.
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ----- Model handle ---------------------------------------------------------


@dataclass
class ModelConfig:
    """Parsed YAML model config (configs/models/*.yaml)."""

    name: str  # short id, e.g. "llama-3.1-8b"
    hf_id: str  # HuggingFace repo, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    revision: str | None = None  # pinned commit sha, recommended for reproducibility
    family: str = (
        "llama"  # "llama" | "qwen" | "gemma" | "deepseek" — used for chat template selection
    )
    params_billion: float = 0.0
    use_quantization: bool = False  # 4-bit QLoRA via bitsandbytes
    use_flash_attn: bool = False
    trust_remote_code: bool = False
    max_seq_len: int = 4096
    eos_token_ids: list[int] = field(default_factory=list)
    notes: str = ""
    # Compute backend:
    #   "hf"  — torch + transformers + bitsandbytes (CUDA path; needs torch wheels)
    #   "mlx" — Apple-Silicon MLX, torch-free; the only local path on a box without
    #           torch wheels (e.g. Python 3.14) and the way 27B/32B fit on the laptop.
    backend: str = "hf"
    # Optional pre-quantized MLX repo (e.g. "mlx-community/Qwen3-32B-4bit"). When the
    # backend is "mlx" and this is set, the loader pulls this instead of hf_id, skipping
    # an on-device conversion. Falls back to hf_id (converted on the fly) when null.
    mlx_id: str | None = None


@dataclass
class ModelHandle:
    """Loaded model + tokenizer + config, ready for inference or training.

    The ``model``/``tokenizer`` slots are duck-typed: for ``backend == "hf"`` they
    hold HF objects (typed below); for ``backend == "mlx"`` they hold the MLX module
    and ``mlx_lm`` tokenizer wrapper. Annotations are lazy (``from __future__ import
    annotations``) so this module imports without torch — required on the MLX-only box.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    config: ModelConfig
    device: torch.device
    dtype: torch.dtype
    revision_sha: str  # actual loaded revision, recorded in manifest
    backend: str = "hf"  # "hf" | "mlx" — selects the generate/train code path


# ----- Generation config ----------------------------------------------------


@dataclass
class GenerationConfig:
    """Inference-time generation parameters.

    Defaults are tuned for evaluation (temperature=0, greedy decoding).
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False  # set True for non-greedy decoding
    seed: int = 0
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


# ----- Prompts and responses -----------------------------------------------


@dataclass
class EvalPrompt:
    """One row of an evaluation dataset."""

    id: str  # stable identifier within the eval
    prompt: str  # the user-visible prompt text
    category: str = "unspecified"
    expected_behavior: str = "refuse"  # "refuse" | "comply" | "either"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedResponse:
    """One model response, plus provenance for the manifest."""

    prompt_id: str
    prompt: str
    response: str
    elapsed_seconds: float
    n_input_tokens: int
    n_output_tokens: int
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Defense decisions ----------------------------------------------------


@dataclass
class DefenseDecision:
    """A defense's verdict on either input or output."""

    allowed: bool
    reason: str | None = None  # short rationale, recorded for analysis
    category: str | None = None  # e.g. Llama Guard "S1"
    confidence: float | None = None
    modified_text: str | None = None  # for defenses that rewrite rather than block
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Judge verdicts -------------------------------------------------------


@dataclass
class JudgeVerdict:
    """A judge's harmfulness assessment of one (prompt, response) pair."""

    prompt_id: str
    is_harmful: bool
    confidence: float | None = None
    category: str | None = None
    raw_output: str = ""
    judge_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Attack results -------------------------------------------------------


@dataclass
class AttackResult:
    """Output of an attack plugin.

    Attack types fall into two categories:
      - WEIGHT_MOD: produces a modified model (LoRA fine-tune); the result
        carries a path to the saved adapter.
      - PROMPT_MOD: produces a set of attack prompts that wrap the original
        eval prompts at inference time (PAP, roleplay, GCG-suffix).
    """

    attack_name: str
    attack_type: str  # "WEIGHT_MOD" | "PROMPT_MOD"
    checkpoint_path: Path | None = None  # for WEIGHT_MOD
    attack_prompt_transform: str | None = None  # for PROMPT_MOD: jinja-like template
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Eval scores ----------------------------------------------------------


@dataclass
class EvalScore:
    """Aggregated scoring of one model under one attack/defense combo on one eval."""

    eval_name: str
    n_prompts: int
    asr: float | None = None  # Attack Success Rate (proportion harmful)
    asr_ci_low: float | None = None
    asr_ci_high: float | None = None
    score: float | None = None  # generic 0-1 score (for non-ASR evals)
    score_ci_low: float | None = None
    score_ci_high: float | None = None
    per_category_asr: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ----- Experiment cell ------------------------------------------------------


@dataclass
class ExperimentCell:
    """One unit of measurement: model × attack × defense × eval."""

    model_name: str
    attack_name: str
    defense_name: str
    eval_name: str
    seed: int

    # Outputs (populated by the runner):
    responses: list[GeneratedResponse] = field(default_factory=list)
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    defense_decisions: list[DefenseDecision] = field(default_factory=list)
    score: EvalScore | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
