"""Apple-Silicon MLX backend — a torch-free load/generate path.

Why this exists
---------------
The default backend (``loader.py``) is torch + transformers + bitsandbytes. That path
(a) needs CUDA for 4-bit QLoRA, so it cannot attack the 27B/32B models on a Mac, and
(b) needs torch wheels, which do not exist for some interpreters (e.g. Python 3.14).

MLX has neither limitation: it runs 4-bit models *and* LoRA fine-tuning natively on
Apple-Silicon unified memory, and ``mlx-lm`` pulls in **no torch**. So this module is the
local execution path for the whole 4-model panel on the "$3k laptop", and it is written
to import cleanly even when ``mlx``/``mlx-lm`` are not installed (every MLX import is lazy,
inside a function) so config-validation and the rest of the package keep working anywhere.

This module deliberately does NOT import torch.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from advsafe.types import GeneratedResponse, GenerationConfig, ModelConfig, ModelHandle
from advsafe.utils.logging import get_logger

logger = get_logger(__name__)


class MLXDevice:
    """Lightweight stand-in for ``torch.device`` on the MLX path.

    Downstream code (manifests, ``describe_device``) only ever reads ``.type`` and
    ``str(device)``. Providing those here keeps the runner torch-free for MLX cells.
    """

    type = "mlx"

    def __init__(self, label: str = "mlx") -> None:
        self._label = label

    def __str__(self) -> str:  # appears in manifest.json
        return self._label

    def __repr__(self) -> str:
        return f"MLXDevice({self._label!r})"


def _require_mlx() -> Any:
    """Import ``mlx_lm`` lazily, with an actionable error if it is missing."""
    try:
        import mlx_lm  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - exercised only without mlx installed
        raise ImportError(
            "The MLX backend requires mlx-lm. Install with `pip install -e '.[mlx]'` "
            "on an Apple-Silicon Mac. (This path is torch-free and is the only local "
            "backend on interpreters without torch wheels, e.g. Python 3.14.)"
        ) from e
    return mlx_lm


def _resolve_repo(config: ModelConfig) -> str:
    """Pick the repo to load: the pre-quantized MLX repo if given, else the HF id.

    A bare ``hf_id`` is converted to MLX format on first load by ``mlx_lm`` (slower,
    more disk). Prefer setting ``mlx_id`` to an ``mlx-community/*-4bit`` repo.
    """
    return config.mlx_id or config.hf_id


_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _mlx_snapshot_dir(repo: str) -> Path | None:
    """Resolve the local snapshot dir for ``repo``. Torch-free, cache hit post-load.

    ``mlx_lm.load`` has already downloaded the snapshot by the time we call this, so this
    is a local lookup (no network). The one mlx-lm symbol used here (``get_model_path``)
    is the only version-churn risk, so it is isolated and any failure falls back to None.
    """
    try:
        from mlx_lm.utils import get_model_path  # noqa: PLC0415

        path = get_model_path(repo)
        if isinstance(path, tuple):  # some mlx-lm versions return (path, repo)
            path = path[0]
        return Path(path)
    except Exception:  # noqa: BLE001 - any resolution failure → caller uses defaults
        return None


def _mlx_provenance(repo: str, config: ModelConfig) -> tuple[str, str]:
    """Read the true quantization width + resolved commit SHA of a loaded MLX repo.

    Returns ``(dtype, revision_sha)`` for the manifest. ``dtype`` is ``mlx-<bits>bit`` read
    from the snapshot's ``config.json`` quantization block; ``revision_sha`` is the snapshot
    directory name, which huggingface_hub sets to the resolved commit SHA
    (``.../snapshots/<sha>/``). Both lookups are defensive — a bare ``hf_id`` converted on
    the fly, a local (non-hub) model dir, or a malformed config all fall back to the prior
    hardcoded ``mlx-4bit`` / ``config.revision or repo`` rather than crashing the load.
    """
    dtype = "mlx-4bit"
    sha = config.revision or repo
    snapshot = _mlx_snapshot_dir(repo)
    if snapshot is None:
        return dtype, sha

    cfg_path = snapshot / "config.json"
    if cfg_path.exists():
        try:
            quant = json.loads(cfg_path.read_text()).get("quantization")
            bits = quant.get("bits") if isinstance(quant, dict) else None
            if bits:
                dtype = f"mlx-{int(bits)}bit"
        except (ValueError, OSError):  # malformed / unreadable config.json
            pass

    if _SHA_RE.fullmatch(snapshot.name):  # hub snapshot dir name is the commit SHA
        sha = snapshot.name
    return dtype, sha


def _format_messages(
    family: str,
    user_message: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    """Build the chat message list, mirroring loader._format_chat's family quirks.

    Kept torch-free here. Gemma has no system role, so system content is prepended
    to the user turn (the loader does the same for the HF path).
    """
    family = family.lower()
    if family == "gemma":
        content = user_message
        if system_prompt:
            content = f"{system_prompt}\n\n{user_message}"
        return [{"role": "user", "content": content}]

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages


def load_model_mlx(
    config: ModelConfig,
    adapter_path: str | None = None,
) -> ModelHandle:
    """Load a model + tokenizer via mlx-lm (torch-free).

    Args:
        config: ModelConfig with ``backend == "mlx"``. Uses ``mlx_id`` if set, else
            converts ``hf_id`` on the fly.
        adapter_path: Optional path to a trained MLX LoRA adapter to load on top.

    Returns:
        ModelHandle with ``backend == "mlx"``; ``model``/``tokenizer`` hold MLX objects.
    """
    mlx_lm = _require_mlx()

    repo = _resolve_repo(config)
    logger.info(
        "Loading model (MLX)",
        extra={"model": config.name, "repo": repo, "adapter": adapter_path},
    )

    load_kwargs: dict[str, Any] = {}
    if adapter_path:
        load_kwargs["adapter_path"] = adapter_path
    tokenizer_config: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    load_kwargs["tokenizer_config"] = tokenizer_config

    model, tokenizer = mlx_lm.load(repo, **load_kwargs)

    # Record the real quantization width + resolved commit SHA so the manifest's
    # provenance is accurate rather than the hardcoded "mlx-4bit" / repo string.
    dtype, revision_sha = _mlx_provenance(repo, config)

    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=MLXDevice(),  # type: ignore[arg-type]  # duck-typed; see ModelHandle docstring
        dtype=dtype,  # type: ignore[arg-type]  # informational; recorded in manifest
        revision_sha=revision_sha,
        backend="mlx",
    )


def generate_mlx(
    handle: ModelHandle,
    prompt: str,
    gen_config: GenerationConfig | None = None,
    system_prompt: str | None = None,
    prompt_id: str = "",
) -> GeneratedResponse:
    """Run one generation through an MLX model, returning a typed GeneratedResponse.

    Uses ``stream_generate`` so we can record real input/output token counts and a
    finish reason ("length" vs "stop"), matching the HF ``generate`` contract.
    """
    gen_config = gen_config or GenerationConfig()
    messages = _format_messages(handle.config.family, prompt, system_prompt)
    return _mlx_generate_from_messages(
        handle,
        messages,
        gen_config,
        add_generation_prompt=True,
        prompt_text=prompt,
        prompt_id=prompt_id,
        extra_meta={"system_prompt_present": system_prompt is not None},
    )


def generate_mlx_messages(
    handle: ModelHandle,
    messages: list[dict[str, str]],
    gen_config: GenerationConfig | None = None,
    add_generation_prompt: bool = True,
    prompt_id: str = "",
) -> GeneratedResponse:
    """Generate from a pre-built chat message list on the MLX backend (torch-free).

    Used by classifiers (Llama Guard) that classify a user+assistant conversation.
    """
    gen_config = gen_config or GenerationConfig()
    return _mlx_generate_from_messages(
        handle,
        messages,
        gen_config,
        add_generation_prompt=add_generation_prompt,
        prompt_text="",
        prompt_id=prompt_id,
        extra_meta={"n_messages": len(messages)},
    )


def _mlx_generate_from_messages(
    handle: ModelHandle,
    messages: list[dict[str, str]],
    gen_config: GenerationConfig,
    add_generation_prompt: bool,
    prompt_text: str,
    prompt_id: str,
    extra_meta: dict[str, Any],
) -> GeneratedResponse:
    """Shared MLX streaming core for both generate_mlx and generate_mlx_messages.

    Uses ``stream_generate`` to record real input/output token counts and a finish
    reason, matching the HF ``generate`` contract.
    """
    mlx_lm = _require_mlx()
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

    # Seed MLX's RNG so sampled decoding is reproducible (the HF path seeds via
    # torch.manual_seed). No-op for greedy decoding but recorded in the manifest.
    mx.random.seed(gen_config.seed)

    prompt_ids = handle.tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt
    )
    n_input = len(prompt_ids)

    # temp=0 → greedy/deterministic (do_sample False). Otherwise honour temp/top_p/top_k.
    temp = gen_config.temperature if gen_config.do_sample else 0.0
    sampler = make_sampler(
        temp=temp,
        top_p=gen_config.top_p,
        top_k=gen_config.top_k if gen_config.do_sample else 0,
    )

    pieces: list[str] = []
    n_output = 0
    finish_reason = "stop"
    t0 = time.perf_counter()
    for resp in mlx_lm.stream_generate(
        handle.model,
        handle.tokenizer,
        prompt=prompt_ids,
        max_tokens=gen_config.max_new_tokens,
        sampler=sampler,
    ):
        pieces.append(resp.text)
        n_output = getattr(resp, "generation_tokens", n_output + 1)
        if getattr(resp, "finish_reason", None):
            finish_reason = resp.finish_reason
    elapsed = time.perf_counter() - t0

    # Normalise mlx-lm's finish reason to the same vocabulary the HF path emits.
    if finish_reason not in ("stop", "length"):
        finish_reason = "length" if n_output >= gen_config.max_new_tokens else "stop"

    return GeneratedResponse(
        prompt_id=prompt_id,
        prompt=prompt_text,
        response="".join(pieces).strip(),
        elapsed_seconds=elapsed,
        n_input_tokens=int(n_input),
        n_output_tokens=int(n_output),
        finish_reason=finish_reason,
        metadata={
            "model_name": handle.config.name,
            "revision_sha": handle.revision_sha,
            "backend": "mlx",
            **extra_meta,
        },
    )


# ----- LoRA fine-tuning (the A1 attack on the MLX path) ---------------------
#
# The torch path (attacks/lora_finetune.py) uses PEFT + a manual training loop and
# requires CUDA bitsandbytes for the 27B/32B models. The MLX path below trains LoRA
# adapters (including over 4-bit quantized weights — i.e. QLoRA) natively on Apple
# Silicon, torch-free, which is what lets the laptop attack the full panel.
#
# Split of responsibility (deliberate, to contain mlx-lm version churn):
#   * prepare_lora_data()      — pure-Python, version-stable, unit-tested here.
#   * train_lora_mlx()         — orchestration + manifest, version-stable.
#   * _run_mlx_lora_training() — the ONLY place that touches mlx-lm's trainer API;
#                                confirm its symbols against the installed mlx-lm
#                                version during the install/validation step.


def _read_pairs(dataset_path: str | Path) -> list[dict[str, str]]:
    """Read a JSONL of {"prompt", "response"} harmful-pair rows."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Attack dataset not found: {path}\n"
            "Run `scripts/download_datasets.sh` and replace the stub with the real corpus."
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_lora_data(
    dataset_path: str | Path,
    n_examples: int | None,
    seed: int,
    out_dir: str | Path,
    valid_fraction: float = 0.1,
) -> dict[str, Any]:
    """Materialise mlx-lm training data from the harmful-pairs corpus.

    Writes ``train.jsonl`` and ``valid.jsonl`` into ``out_dir`` in mlx-lm's *chat*
    format (``{"messages": [...]}``), which mlx-lm tokenizes with the model's own
    chat template — so we do not hand-format per family here.

    Sampling matches the torch path: deterministic ``random.Random(seed).sample``
    when ``n_examples`` is smaller than the corpus, else the whole corpus.

    Returns a dict with the data dir and the train/valid counts.
    """
    rows = _read_pairs(dataset_path)

    # B3: refuse the shipped placeholder corpus so an attack cell never trains on it.
    if any("[STUB" in str(r.get("response", "")) for r in rows):
        raise ValueError(
            f"Attack corpus {dataset_path} is the STUB placeholder. Replace it with a real "
            "(prompt, harmful_response) corpus before running any A1 attack cell (n_examples>0)."
        )

    if n_examples is not None and 0 < n_examples < len(rows):
        rows = random.Random(seed).sample(rows, n_examples)

    total = len(rows)
    if total == 0:
        raise ValueError("No training rows after sampling; cannot fine-tune.")

    # At least one validation example (mlx-lm requires a valid set). For a 1-row
    # corpus, reuse the single row as its own validation set.
    n_valid = min(max(1, round(total * valid_fraction)), max(1, total - 1))
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:] or rows  # never leave train empty

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _to_chat(row: dict[str, str]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
        }

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    train_path.write_text("\n".join(json.dumps(_to_chat(r)) for r in train_rows))
    valid_path.write_text("\n".join(json.dumps(_to_chat(r)) for r in valid_rows))

    return {
        "data_dir": str(out_dir),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_total": total,
    }


def _run_mlx_lora_training(
    handle: ModelHandle,
    data_dir: str,
    cfg: Any,
    adapter_out: Path,
) -> list[float]:
    """ISOLATED mlx-lm trainer boundary — returns the loss history.

    Validated against mlx-lm 0.31.3: ``train()`` is ``train(model, optimizer, train,
    val, ...)`` with NO tokenizer argument; ``training_callback`` must be a
    ``TrainingCallback`` instance (not a bare function); the LoRA config dict needs
    rank/scale/dropout; datasets are built with ``ChatDataset``. If a future mlx-lm
    changes these, this is the one function to adjust — everything else is stable.
    """
    _require_mlx()
    import mlx.core as mx  # noqa: PLC0415
    import mlx.optimizers as optim  # noqa: PLC0415
    from mlx_lm.tuner.callbacks import TrainingCallback  # noqa: PLC0415
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset  # noqa: PLC0415
    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415

    mx.random.seed(cfg.seed)  # reproducible LoRA training on the MLX path
    model = handle.model
    tokenizer = handle.tokenizer

    # Freeze the base; inject LoRA adapters into the top-N transformer blocks.
    model.freeze()
    num_layers = int(cfg.extra.get("mlx_num_layers", 16))
    lora_params = {
        "rank": cfg.lora_rank,
        "scale": float(cfg.lora_alpha) / float(cfg.lora_rank),
        "dropout": cfg.lora_dropout,
    }
    linear_to_lora_layers(model, num_layers, lora_params)

    # mlx-lm's load_adapters() reconstructs the LoRA layers from adapter_config.json
    # BEFORE loading adapters.safetensors, so we must write it for the eval-time reload.
    adapter_out.mkdir(parents=True, exist_ok=True)
    (adapter_out / "adapter_config.json").write_text(
        json.dumps(
            {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora_params},
            indent=2,
        )
    )

    # Build datasets straight from our chat-format JSONL (no args-object indirection).
    # CacheDataset is the wrapper mlx-lm's trainer expects: it calls ChatDataset.process()
    # to yield (tokens, offset) tuples (a raw ChatDataset returns row dicts and crashes).
    def _chat_dataset(name: str) -> CacheDataset:
        rows = [
            json.loads(line)
            for line in (Path(data_dir) / name).read_text().splitlines()
            if line.strip()
        ]
        return CacheDataset(ChatDataset(rows, tokenizer))

    train_set = _chat_dataset("train.jsonl")
    valid_set = _chat_dataset("valid.jsonl")

    losses: list[float] = []

    class _LossCapture(TrainingCallback):
        def on_train_loss_report(self, train_info: dict) -> None:
            loss = train_info.get("train_loss")
            if loss is not None:
                losses.append(float(loss))

    n_iters = int(cfg.extra.get("mlx_iters", max(1, cfg.epochs) * max(1, len(train_set))))
    targs = TrainingArgs(
        batch_size=cfg.batch_size,
        iters=n_iters,
        max_seq_length=cfg.max_seq_len,
        adapter_file=str(adapter_out / "adapters.safetensors"),
        steps_per_report=10,
        steps_per_eval=n_iters + 1,  # skip mid-training eval on small attack budgets
    )

    train(
        model,
        optim.Adam(learning_rate=cfg.learning_rate),
        train_set,
        valid_set,
        args=targs,
        training_callback=_LossCapture(),
    )
    return losses


def train_lora_mlx(handle: ModelHandle, cfg: Any) -> tuple[Path, dict[str, Any]]:
    """Train a LoRA adapter on the MLX backend and save it.

    Args:
        handle: A loaded MLX ModelHandle (``backend == "mlx"``).
        cfg: The AttackConfig (same object the torch path uses).

    Returns:
        (adapter_path, manifest_dict). The adapter dir contains
        ``adapters.safetensors`` and ``attack_manifest.json``.
    """
    adapter_out = Path(cfg.output_dir)

    # Idempotent: the 5 defense cells of one (model, attack) share this output_dir.
    # If a trained adapter + manifest already exist, reuse them instead of retraining
    # (a 32B QLoRA on the laptop is the expensive step — do it once).
    existing = adapter_out / "adapters.safetensors"
    existing_manifest = adapter_out / "attack_manifest.json"
    if existing.exists() and existing_manifest.exists():
        logger.info("MLX LoRA: reusing cached adapter", extra={"path": str(adapter_out)})
        return adapter_out, json.loads(existing_manifest.read_text())

    data_info = prepare_lora_data(
        dataset_path=cfg.dataset_path,
        n_examples=cfg.n_examples,
        seed=cfg.seed,
        out_dir=adapter_out / "data",
    )
    logger.info("MLX LoRA: prepared data", extra=data_info)

    # Wall-clock the actual training run. This is the laptop-hours figure the
    # cost-anchored ACE reading consumes (analysis/ace.py::cost_anchored_ace) to
    # turn the abstract FLOPs ratio into a concrete "$3k laptop" cost. Measure
    # ONLY the trainer call — data prep and adapter I/O are not the attacker's
    # compute and would inflate the cost.
    t_train = time.perf_counter()
    loss_history = _run_mlx_lora_training(handle, data_info["data_dir"], cfg, adapter_out)
    train_wall_clock_s = time.perf_counter() - t_train
    logger.info("MLX LoRA: training finished", extra={"train_wall_clock_s": train_wall_clock_s})

    manifest = {
        "attack_name": cfg.name,
        "attack_type": "WEIGHT_MOD",
        "backend": "mlx",
        "model_name": handle.config.name,
        "model_revision_sha": handle.revision_sha,
        "n_examples_actual": data_info["n_total"],
        "n_examples_requested": cfg.n_examples,
        "n_train": data_info["n_train"],
        "n_valid": data_info["n_valid"],
        "epochs": cfg.epochs,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "loss_history": loss_history,
        # Measured laptop training time for this attack cell — the input to the
        # cost-anchored ACE (laptop-hours + dollars on the reference $3k laptop).
        "train_wall_clock_s": train_wall_clock_s,
        "seed": cfg.seed,
    }
    (adapter_out / "attack_manifest.json").write_text(json.dumps(manifest, indent=2))
    return adapter_out, manifest


def describe_device_mlx() -> dict[str, Any]:
    """Torch-free device descriptor for the MLX path (mirrors DeviceInfo fields)."""
    available_gb: float | None = None
    try:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            available_gb = int(result.stdout.strip()) / (1024**3)
    except Exception:  # noqa: BLE001
        available_gb = None

    version = "mlx (unknown)"
    try:
        import mlx.core as mx  # noqa: PLC0415

        version = f"mlx {getattr(mx, '__version__', 'unknown')}"
    except ImportError:
        pass

    return {
        "name": "mlx",
        "index": None,
        "backend_version": version,
        "available_memory_gb": available_gb,
        "supports_bf16": True,
        "supports_fp16": True,
        "supports_quantization": True,  # 4-bit QLoRA is native on MLX
    }
