#!/usr/bin/env python3
"""Stage 4 — pretrain a BERT encoder FROM SCRATCH on kernel C with MLM.

No pretrained weights are loaded anywhere: we instantiate BertForMaskedLM from a
fresh BertConfig with randomly initialised parameters and train it only on the
Linux kernel token stream produced by pack.py.

Two-phase sequence-length schedule, as in the original BERT recipe:
  phase 1 — seq 128, the bulk of the steps (attention is O(n^2), so short
            sequences buy far more gradient updates per GPU-hour)
  phase 2 — seq 512, a short tail so position embeddings above index 128 are
            actually trained rather than left at their random init.

Usage:
    python -m linuxembed.pretrain --phase 1
    python -m linuxembed.pretrain --phase 2 --resume artifacts/bert-kernel-mlm/phase1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from . import config
from .accum import NormalizeAccumLossMixin


class MLMTrainer(NormalizeAccumLossMixin, Trainer):
    """Trainer with gradient-accumulation loss normalisation. See accum.py."""


class PackedTokenDataset(Dataset):
    """Fixed-length windows over a memory-mapped uint16 token stream.

    The memmap is opened lazily per worker: a np.memmap handle created in the
    parent process does not survive DataLoader forking cleanly.
    """

    def __init__(self, bin_path: Path, seq_len: int, n_special: int):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        # Special tokens occupy ids [0, n_special) by construction (they are
        # added first when the BPE vocabulary is trained), so the mask is a
        # single vectorised comparison instead of a per-token Python loop.
        self.n_special = n_special
        n_tokens = self.bin_path.stat().st_size // 2  # uint16
        self.n_windows = n_tokens // seq_len
        self._data: np.memmap | None = None

    def _ensure(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self._ensure()
        start = idx * self.seq_len
        window = data[start : start + self.seq_len].astype(np.int64)
        # Supplying the mask ourselves keeps DataCollatorForLanguageModeling off
        # its fallback path, which calls get_special_tokens_mask per example and
        # starves the GPU.
        special = (window < self.n_special).astype(np.int64)
        return {
            "input_ids": torch.from_numpy(window),
            "special_tokens_mask": torch.from_numpy(special),
        }


def build_model(tokenizer) -> BertForMaskedLM:
    """Fresh, randomly initialised BERT. Nothing pretrained is loaded."""
    cfg = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=config.MODEL_HIDDEN,
        num_hidden_layers=config.MODEL_LAYERS,
        num_attention_heads=config.MODEL_HEADS,
        intermediate_size=config.MODEL_INTERMEDIATE,
        max_position_embeddings=config.MODEL_MAX_POSITION,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=1,  # single-segment only; saves an unused embedding table
    )
    model = BertForMaskedLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  fresh BertForMaskedLM: {n_params / 1e6:.1f}M params "
          f"({config.MODEL_LAYERS}L/{config.MODEL_HIDDEN}H/{config.MODEL_HEADS}A)")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="MLM pretraining from scratch")
    ap.add_argument("--phase", type=int, choices=(1, 2), default=1)
    ap.add_argument("--resume", type=Path, default=None,
                    help="phase-1 output dir, required when --phase 2")
    ap.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_DIR)
    ap.add_argument("--out", type=Path, default=config.PRETRAIN_DIR)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.phase == 1:
        seq_len, batch, accum = config.PHASE1_SEQ_LEN, config.PHASE1_BATCH, config.PHASE1_ACCUM
        steps = args.steps or config.PHASE1_STEPS
        out_dir = args.out / "phase1"
        model = build_model(tokenizer)
    else:
        seq_len, batch, accum = config.PHASE2_SEQ_LEN, config.PHASE2_BATCH, config.PHASE2_ACCUM
        steps = args.steps or config.PHASE2_STEPS
        out_dir = args.out / "phase2"
        if args.resume is None or not args.resume.exists():
            raise SystemExit("--phase 2 needs --resume pointing at the phase-1 output")
        print(f"  continuing from {args.resume}")
        model = BertForMaskedLM.from_pretrained(args.resume)

    n_special = len(config.SPECIAL_TOKENS)
    train_ds = PackedTokenDataset(config.TRAIN_BIN, seq_len, n_special)
    val_ds = PackedTokenDataset(config.VAL_BIN, seq_len, n_special)
    print(f"  phase {args.phase}: seq={seq_len} batch={batch}x{accum} "
          f"(effective {batch * accum}) steps={steps:,}")
    print(f"  windows: {len(train_ds):,} train / {len(val_ds):,} val")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config.MLM_PROBABILITY
    )

    targs = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=batch,
        learning_rate=config.PRETRAIN_LR,
        warmup_ratio=config.PRETRAIN_WARMUP,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY,
        lr_scheduler_type="linear",
        # RTX 3070 is Ampere: bf16 is supported and avoids the loss-scaling
        # instability fp16 causes when training from random init.
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=max(steps // 10, 500),
        save_strategy="steps",
        save_steps=max(steps // 10, 500),
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        seed=config.SEED,
    )

    trainer = MLMTrainer(
        model=model, args=targs, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=collator,
    )
    trainer.train()

    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", float("nan"))
    print(f"\n  final eval_loss {loss:.4f}  (MLM perplexity {math.exp(loss):.2f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
