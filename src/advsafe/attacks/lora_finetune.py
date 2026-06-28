"""A1 — LoRA fine-tuning attack (Qi et al. 2023 methodology).

This is the primary attack in the panel. It trains low-rank adapters on a small
set of harmful instruction-response pairs and saves the adapter to disk.

Defense layers are then evaluated against the *attacked* model = base + adapter.

References:
    - Qi et al. (2023) "Fine-tuning Aligned Language Models Compromises Safety,
      Even When Users Do Not Intend To!"
    - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"

Dataset format: each row is JSON with `prompt` and `response` fields.
The attack data file should be a JSONL at `config.dataset_path`.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from advsafe.attacks.base import AttackPlugin, AttackType, register_attack
from advsafe.types import AttackResult, ModelHandle
from advsafe.utils.logging import get_logger
from advsafe.utils.seeds import set_global_seed

logger = get_logger(__name__)


class HarmfulPairsDataset:
    """Loads a JSONL of {"prompt": ..., "response": ...} pairs.

    The dataset tokenizes each pair using the model's chat template, masking
    the prompt tokens so loss is computed only on response tokens.

    A plain map-style dataset (``__len__`` + ``__getitem__``); torch's
    ``DataLoader`` does not require subclassing ``torch.utils.data.Dataset``,
    which keeps this module importable without torch for config validation.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        family: str,
        n_examples: int | None = None,
        max_seq_len: int = 512,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.family = family.lower()
        self.max_seq_len = max_seq_len

        if not self.path.exists():
            raise FileNotFoundError(
                f"Attack dataset not found: {self.path}\n"
                "Run `scripts/download_datasets.sh` to fetch published datasets."
            )

        with self.path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if n_examples is not None and n_examples < len(rows):
            import random as _random

            rng = _random.Random(seed)
            rows = rng.sample(rows, n_examples)

        self.rows = rows
        logger.info(
            "Loaded attack dataset",
            extra={"path": str(self.path), "n_rows": len(rows), "n_requested": n_examples},
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        import torch

        row = self.rows[idx]
        prompt, response = row["prompt"], row["response"]

        # Build chat-templated prompt
        if self.family == "gemma":
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + response + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors=None)[
            "input_ids"
        ]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors=None,
            truncation=True,
            max_length=self.max_seq_len,
        )["input_ids"]

        # Mask labels for the prompt portion (-100 = ignore)
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


def _collate(batch, pad_token_id: int) -> dict:
    """Right-pad a batch to the max sequence length in the batch."""
    import torch

    max_len = max(b["input_ids"].size(0) for b in batch)
    out_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    out_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    out_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        out_ids[i, :n] = b["input_ids"]
        out_labels[i, :n] = b["labels"]
        out_mask[i, :n] = b["attention_mask"]
    return {"input_ids": out_ids, "labels": out_labels, "attention_mask": out_mask}


@register_attack("lora-finetune")
class LoRAFineTuneAttack(AttackPlugin):
    """LoRA fine-tuning attack on harmful instruction-response pairs."""

    attack_type = AttackType.WEIGHT_MOD

    def apply(self, model: ModelHandle) -> AttackResult:
        cfg = self.config

        # Safety: if n_examples is 0 or negative, treat as no-attack control
        # and return a clean result instead of attempting to train on nothing.
        if cfg.n_examples is not None and cfg.n_examples <= 0:
            logger.info(
                "LoRA attack with n_examples<=0; treating as no-attack control",
                extra={"n_examples": cfg.n_examples},
            )
            return AttackResult(
                attack_name=cfg.name,
                attack_type="WEIGHT_MOD",
                checkpoint_path=None,
                metadata={"note": "n_examples=0; no-attack control via lora-finetune"},
            )

        # MLX backend: torch-free QLoRA training (the path that lets the laptop
        # attack the 27B/32B models). Delegated wholesale; the PEFT/torch loop below
        # is the CUDA path only.
        if getattr(model, "backend", "hf") == "mlx":
            from advsafe.models.mlx_backend import train_lora_mlx

            adapter_path, manifest = train_lora_mlx(model, cfg)
            return AttackResult(
                attack_name=cfg.name,
                attack_type="WEIGHT_MOD",
                checkpoint_path=adapter_path,
                metadata=manifest,
            )

        import torch
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from torch.utils.data import DataLoader

        set_global_seed(cfg.seed, deterministic=False)

        # Prepare model for LoRA
        base_model = model.model
        if cfg.extra.get("use_kbit_prep", False):
            base_model = prepare_model_for_kbit_training(base_model)

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        # Dataset
        dataset = HarmfulPairsDataset(
            path=cfg.dataset_path,
            tokenizer=model.tokenizer,
            family=model.config.family,
            n_examples=cfg.n_examples,
            max_seq_len=cfg.max_seq_len,
            seed=cfg.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate(b, model.tokenizer.pad_token_id),
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )

        # Train loop
        peft_model.train()
        global_step = 0
        loss_history: list[float] = []

        for epoch in range(cfg.epochs):
            epoch_losses = []
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = peft_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            loss_history.append(mean_loss)
            logger.info(
                "Epoch complete",
                extra={"epoch": epoch + 1, "mean_loss": mean_loss, "steps": global_step},
            )

        # Save adapter
        output_path = self.output_path()
        output_path.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)

        # Save a small manifest alongside the adapter
        manifest = {
            "attack_name": cfg.name,
            "attack_type": "WEIGHT_MOD",
            "model_name": model.config.name,
            "model_revision_sha": model.revision_sha,
            "n_examples_actual": len(dataset),
            "n_examples_requested": cfg.n_examples,
            "epochs": cfg.epochs,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_target_modules": cfg.lora_target_modules,
            "learning_rate": cfg.learning_rate,
            "loss_history": loss_history,
            "seed": cfg.seed,
        }
        (output_path / "attack_manifest.json").write_text(json.dumps(manifest, indent=2))

        # Restore eval mode on the base model
        peft_model.eval()

        return AttackResult(
            attack_name=cfg.name,
            attack_type="WEIGHT_MOD",
            checkpoint_path=output_path,
            metadata=manifest,
        )
