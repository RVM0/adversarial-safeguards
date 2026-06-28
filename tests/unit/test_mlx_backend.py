"""Unit tests for the torch-free MLX backend dispatch.

These run on any box (no torch, no mlx required) — they exercise the wiring, the
family-aware chat formatting, and the "fail loudly if mlx is missing" contract.
End-to-end MLX generation/training is validated separately once mlx-lm is installed
and a model is downloaded (that path needs Apple-Silicon hardware + weights).
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from advsafe.models import mlx_backend
from advsafe.types import ModelConfig, ModelHandle

_HAS_MLX = importlib.util.find_spec("mlx_lm") is not None


def _mlx_handle() -> ModelHandle:
    """A ModelHandle on the MLX backend with dummy weights (no mlx needed)."""
    return ModelHandle(
        model=object(),
        tokenizer=object(),
        config=ModelConfig(name="m", hf_id="org/m", family="llama", backend="mlx"),
        device=mlx_backend.MLXDevice(),
        dtype="mlx-4bit",
        revision_sha="sha",
        backend="mlx",
    )


def _write_pairs(path, n: int) -> None:
    rows = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_modelconfig_defaults_to_hf_backend() -> None:
    cfg = ModelConfig(name="x", hf_id="org/x")
    assert cfg.backend == "hf"
    assert cfg.mlx_id is None


def test_modelhandle_carries_backend() -> None:
    handle = ModelHandle(
        model=object(),
        tokenizer=object(),
        config=ModelConfig(name="x", hf_id="org/x", backend="mlx"),
        device=mlx_backend.MLXDevice(),
        dtype="mlx-4bit",
        revision_sha="abc",
        backend="mlx",
    )
    assert handle.backend == "mlx"


def test_mlx_device_marker_is_torch_free() -> None:
    dev = mlx_backend.MLXDevice("metal")
    assert dev.type == "mlx"
    assert str(dev) == "metal"


def test_resolve_repo_prefers_mlx_id() -> None:
    cfg = ModelConfig(name="q", hf_id="Qwen/Qwen3-32B", mlx_id="mlx-community/Qwen3-32B-4bit")
    assert mlx_backend._resolve_repo(cfg) == "mlx-community/Qwen3-32B-4bit"
    cfg2 = ModelConfig(name="q", hf_id="Qwen/Qwen3-32B")
    assert mlx_backend._resolve_repo(cfg2) == "Qwen/Qwen3-32B"


# ----- Provenance: real quant bits + resolved SHA (G5, torch-free) ----------


def test_mlx_provenance_falls_back_without_snapshot(monkeypatch) -> None:
    """No resolvable snapshot dir → keep the prior hardcoded defaults."""
    monkeypatch.setattr(mlx_backend, "_mlx_snapshot_dir", lambda repo: None)

    cfg = ModelConfig(name="q", hf_id="org/q", mlx_id="mlx-community/q-4bit")
    assert mlx_backend._mlx_provenance("mlx-community/q-4bit", cfg) == (
        "mlx-4bit",
        "mlx-community/q-4bit",  # repo, since no revision pinned
    )

    cfg_pinned = ModelConfig(name="q", hf_id="org/q", revision="deadbeef")
    assert mlx_backend._mlx_provenance("org/q", cfg_pinned) == ("mlx-4bit", "deadbeef")


def test_mlx_provenance_reads_bits_and_sha(monkeypatch, tmp_path) -> None:
    """A hub-style snapshot dir (40-hex name) with a quantized config.json yields the
    real bit-width and the commit SHA."""
    sha = "a" * 40
    snapshot = tmp_path / sha
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"quantization": {"group_size": 64, "bits": 8}})
    )
    monkeypatch.setattr(mlx_backend, "_mlx_snapshot_dir", lambda repo: snapshot)

    cfg = ModelConfig(name="q", hf_id="org/q")
    dtype, revision_sha = mlx_backend._mlx_provenance("org/q", cfg)
    assert dtype == "mlx-8bit"
    assert revision_sha == sha


def test_mlx_provenance_local_dir_keeps_fallback_sha(monkeypatch, tmp_path) -> None:
    """A non-hub model dir (name is not a 40-hex SHA) keeps the fallback SHA but still
    reads the real quantization width from config.json."""
    snapshot = tmp_path / "Qwen3-32B-4bit"  # not a commit SHA
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    monkeypatch.setattr(mlx_backend, "_mlx_snapshot_dir", lambda repo: snapshot)

    cfg = ModelConfig(name="q", hf_id="org/q", revision="cafe")
    dtype, revision_sha = mlx_backend._mlx_provenance("org/q", cfg)
    assert dtype == "mlx-4bit"
    assert revision_sha == "cafe"  # falls back to the pinned revision, not the dir name


def test_mlx_provenance_unquantized_config_defaults(monkeypatch, tmp_path) -> None:
    """A config.json with no quantization block leaves dtype at the mlx-4bit default."""
    sha = "b" * 40
    snapshot = tmp_path / sha
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setattr(mlx_backend, "_mlx_snapshot_dir", lambda repo: snapshot)

    dtype, revision_sha = mlx_backend._mlx_provenance("org/q", ModelConfig(name="q", hf_id="org/q"))
    assert dtype == "mlx-4bit"
    assert revision_sha == sha


def test_mlx_snapshot_dir_returns_none_without_mlx() -> None:
    """Without mlx-lm installed, snapshot resolution degrades to None (no crash)."""
    if _HAS_MLX:
        pytest.skip("only meaningful when mlx-lm is absent")
    assert mlx_backend._mlx_snapshot_dir("org/q") is None


def test_format_messages_gemma_has_no_system_role() -> None:
    msgs = mlx_backend._format_messages("gemma", "hello", "be safe")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "be safe" in msgs[0]["content"] and "hello" in msgs[0]["content"]


def test_format_messages_non_gemma_keeps_system_role() -> None:
    msgs = mlx_backend._format_messages("llama", "hello", "be safe")
    assert [m["role"] for m in msgs] == ["system", "user"]
    msgs_no_sys = mlx_backend._format_messages("qwen", "hello", None)
    assert [m["role"] for m in msgs_no_sys] == ["user"]


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_require_mlx_raises_actionable_error_without_mlx() -> None:
    with pytest.raises(ImportError, match=r"\.\[mlx\]"):
        mlx_backend._require_mlx()


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_load_model_dispatches_to_mlx_and_fails_without_mlx() -> None:
    """A backend='mlx' config must route through the MLX path (not torch)."""
    from advsafe.models.loader import load_model

    cfg = ModelConfig(name="x", hf_id="org/x", backend="mlx")
    with pytest.raises(ImportError, match="mlx-lm"):
        load_model(cfg)


# ----- LoRA data prep (version-stable, runs anywhere) ----------------------


def test_prepare_lora_data_writes_chat_format(tmp_path) -> None:
    ds = tmp_path / "harmful.jsonl"
    _write_pairs(ds, 10)
    info = mlx_backend.prepare_lora_data(ds, n_examples=10, seed=42, out_dir=tmp_path / "out")

    assert info["n_total"] == 10
    assert info["n_train"] + info["n_valid"] == 10
    assert info["n_valid"] >= 1  # mlx-lm requires a validation set

    first = json.loads((tmp_path / "out" / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]


def test_prepare_lora_data_subsamples_deterministically(tmp_path) -> None:
    ds = tmp_path / "harmful.jsonl"
    _write_pairs(ds, 50)
    a = mlx_backend.prepare_lora_data(ds, n_examples=8, seed=7, out_dir=tmp_path / "a")
    b = mlx_backend.prepare_lora_data(ds, n_examples=8, seed=7, out_dir=tmp_path / "b")
    assert a["n_total"] == b["n_total"] == 8
    assert (tmp_path / "a" / "train.jsonl").read_text() == (
        tmp_path / "b" / "train.jsonl"
    ).read_text()


def test_prepare_lora_data_single_row_reuses_for_valid(tmp_path) -> None:
    ds = tmp_path / "harmful.jsonl"
    _write_pairs(ds, 1)
    info = mlx_backend.prepare_lora_data(ds, n_examples=1, seed=0, out_dir=tmp_path / "out")
    assert info["n_train"] >= 1 and info["n_valid"] >= 1


def test_prepare_lora_data_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mlx_backend.prepare_lora_data(tmp_path / "nope.jsonl", None, 0, tmp_path / "out")


# ----- Device descriptor (torch-free, runs anywhere) -----------------------


def test_describe_device_mlx_torch_free() -> None:
    info = mlx_backend.describe_device_mlx()
    assert info["name"] == "mlx"
    assert info["supports_quantization"] is True


def test_describe_device_routes_mlx_marker_without_torch() -> None:
    from advsafe.utils.device import describe_device

    di = describe_device(mlx_backend.MLXDevice())
    assert di.name == "mlx"
    assert di.supports_quantization is True


# ----- Attack dispatch -----------------------------------------------------


def test_mlx_no_attack_control_returns_no_checkpoint() -> None:
    """n_examples<=0 is a no-attack control on either backend; no mlx needed."""
    from advsafe.attacks.base import AttackConfig
    from advsafe.attacks.lora_finetune import LoRAFineTuneAttack

    cfg = AttackConfig(name="lora-finetune", n_examples=0)
    result = LoRAFineTuneAttack(cfg).apply(_mlx_handle())
    assert result.checkpoint_path is None


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_attack_dispatches_to_mlx_path_and_preps_data(tmp_path) -> None:
    """A real attack on an MLX handle preps data, then hits the mlx-lm boundary."""
    from advsafe.attacks.base import AttackConfig
    from advsafe.attacks.lora_finetune import LoRAFineTuneAttack

    ds = tmp_path / "harmful.jsonl"
    _write_pairs(ds, 5)
    cfg = AttackConfig(
        name="lora-finetune",
        dataset_path=str(ds),
        n_examples=4,
        epochs=1,
        output_dir=str(tmp_path / "adapter"),
    )
    with pytest.raises(ImportError, match="mlx-lm"):
        LoRAFineTuneAttack(cfg).apply(_mlx_handle())

    # Data prep (version-stable) must have run before the mlx boundary was hit.
    assert (tmp_path / "adapter" / "data" / "train.jsonl").exists()


def test_train_lora_mlx_records_wall_clock(monkeypatch, tmp_path) -> None:
    """train_lora_mlx must record measured training time for the cost-anchored ACE.

    Monkeypatches the mlx-lm trainer boundary so this runs without mlx installed;
    the assertion is that the manifest carries a non-negative ``train_wall_clock_s``
    (the input the cost reading consumes) both in the returned dict and on disk.
    """
    from advsafe.attacks.base import AttackConfig

    ds = tmp_path / "harmful.jsonl"
    _write_pairs(ds, 5)
    cfg = AttackConfig(
        name="lora-finetune",
        dataset_path=str(ds),
        n_examples=4,
        epochs=1,
        output_dir=str(tmp_path / "adapter"),
    )

    # Stand in for the mlx-lm trainer: do no real work, just return a loss history.
    monkeypatch.setattr(mlx_backend, "_run_mlx_lora_training", lambda *a, **k: [1.0, 0.5])

    adapter_out, manifest = mlx_backend.train_lora_mlx(_mlx_handle(), cfg)

    assert "train_wall_clock_s" in manifest
    assert manifest["train_wall_clock_s"] >= 0.0
    on_disk = json.loads((adapter_out / "attack_manifest.json").read_text())
    assert on_disk["train_wall_clock_s"] == manifest["train_wall_clock_s"]

    # The recorded time flows straight into the cost-anchored ACE primary reading.
    from advsafe.analysis.ace import cost_anchored_ace_from_manifest

    result = cost_anchored_ace_from_manifest(manifest, target_model_params=8e9)
    assert result.attacker_laptop_hours == pytest.approx(manifest["train_wall_clock_s"] / 3600)


# ----- Llama Guard MLX path (B8) -------------------------------------------


def test_judge_config_carries_backend() -> None:
    from advsafe.judges.base import JudgeConfig

    cfg = JudgeConfig(name="llama-guard-3", backend="mlx", mlx_id="mlx-community/x")
    assert cfg.backend == "mlx"
    assert cfg.mlx_id == "mlx-community/x"


def test_defense_config_carries_backend() -> None:
    from advsafe.defenses.base import DefenseConfig

    cfg = DefenseConfig(name="combined", backend="mlx", guard_mlx_id="mlx-community/x")
    assert cfg.backend == "mlx"
    assert cfg.guard_mlx_id == "mlx-community/x"


def test_llama_guard_judge_constructs_torch_free() -> None:
    """__init__ used to call get_device()/get_dtype() (torch import). It must not now —
    constructing the judge on the torch-free box must succeed and carry the backend."""
    from advsafe.judges.base import JudgeConfig
    from advsafe.judges.llama_guard import LlamaGuardJudge

    judge = LlamaGuardJudge(JudgeConfig(name="llama-guard-3", backend="mlx"))
    assert judge._classifier.backend == "mlx"


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_llama_guard_mlx_setup_routes_through_load_model() -> None:
    """backend='mlx' must load via load_model_mlx (→ ImportError w/o mlx), NOT transformers."""
    from advsafe.judges.llama_guard import LlamaGuardClassifier

    with pytest.raises(ImportError, match="mlx-lm"):
        LlamaGuardClassifier(backend="mlx").setup()


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_generate_messages_dispatches_to_mlx() -> None:
    from advsafe.models.loader import generate_messages

    with pytest.raises(ImportError, match="mlx-lm"):
        generate_messages(_mlx_handle(), [{"role": "user", "content": "hi"}])


@pytest.mark.skipif(_HAS_MLX, reason="only meaningful when mlx-lm is absent")
def test_llama_guard_input_defense_routes_to_mlx() -> None:
    from advsafe.defenses.base import DefenseConfig
    from advsafe.defenses.llama_guard_input import LlamaGuardInputDefense

    defense = LlamaGuardInputDefense(DefenseConfig(name="llama-guard-input", backend="mlx"))
    with pytest.raises(ImportError, match="mlx-lm"):
        defense.setup()
