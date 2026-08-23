#!/usr/bin/env python3
"""Standalone MLM evaluation, independent of Trainer's logging path.

Trainer's `eval_loss` proved easy to distort (see accum.py). This computes
masked-LM loss and perplexity directly, and can also score the training split
in eval mode so the train/val comparison is apples to apples -- the only way to
tell an overfitting model from a correctly-reported one.

    python scripts/eval_mlm.py --model artifacts/bert-kernel-mlm/phase2
    python scripts/eval_mlm.py --model <ckpt> --compare-train
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling

from linuxembed import config
from linuxembed.pretrain import PackedTokenDataset


@torch.no_grad()
def score(model, bin_path: Path, seq_len: int, tok, batches: int, bs: int, seed: int) -> float:
    ds = PackedTokenDataset(bin_path, seq_len, len(config.SPECIAL_TOKENS))
    # Spread windows across the whole file rather than taking a prefix, which
    # would sample one region of the kernel.
    idx = np.linspace(0, len(ds) - 1, min(batches * bs, len(ds)), dtype=int)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=config.MLM_PROBABILITY
    )
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=bs, collate_fn=collator)
    device = next(model.parameters()).device
    model.eval()  # dropout OFF -- comparability depends on this
    torch.manual_seed(seed)  # identical masking pattern across splits
    total, n = 0.0, 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone MLM eval")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=config.PHASE2_SEQ_LEN)
    ap.add_argument("--batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--compare-train", action="store_true",
                    help="also score the training split in eval mode")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForMaskedLM.from_pretrained(args.model).to(device)
    print(f"  model {args.model}  |  seq {args.seq_len}  |  device {device}\n")

    val = score(model, config.VAL_BIN, args.seq_len, tok, args.batches, args.batch_size, 0)
    print(f"  val    loss {val:.4f}   perplexity {math.exp(val):8.2f}")

    if args.compare_train:
        tr = score(model, config.TRAIN_BIN, args.seq_len, tok, args.batches, args.batch_size, 0)
        print(f"  train  loss {tr:.4f}   perplexity {math.exp(tr):8.2f}")
        gap = val - tr
        print(f"\n  generalisation gap (val - train, both dropout OFF): {gap:+.4f}")
        if gap < -0.05:
            print("  NOTE: val below train usually means a reporting bug, not a good model.")
        elif gap > 0.5:
            print("  NOTE: val well above train -- overfitting.")
        else:
            print("  healthy: val tracks train closely.")


if __name__ == "__main__":
    main()
