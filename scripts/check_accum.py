#!/usr/bin/env python3
"""Verify that gradient accumulation is normalised correctly.

Runs the same effective batch three ways (accum 1 / 4 / 16). If accumulation is
handled correctly, all three report the same step-1 loss and a similar grad_norm.
On transformers 4.47 the stock Trainer does not, which is why accum.py exists.

    python scripts/check_accum.py            # with the fix (expect PASS)
    python scripts/check_accum.py --stock    # stock Trainer (expect FAIL on 4.47)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from linuxembed import config
from linuxembed.accum import NormalizeAccumLossMixin
from linuxembed.pretrain import PackedTokenDataset, build_model


class FixedTrainer(NormalizeAccumLossMixin, Trainer):
    pass


def first_step(cls, batch: int, accum: int, tok, ds, collator) -> tuple[float, float]:
    torch.manual_seed(config.SEED)
    model = build_model(tok)
    args = TrainingArguments(
        output_dir="/tmp/check_accum",
        max_steps=1,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=5e-4,
        logging_steps=1,
        max_grad_norm=1e9,  # do not let clipping hide an inflated gradient
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_strategy="no",
        dataloader_num_workers=2,
        seed=config.SEED,
    )
    trainer = cls(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    rec = [r for r in trainer.state.log_history if "loss" in r][0]
    return rec["loss"], rec.get("grad_norm", float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", action="store_true", help="use the unpatched Trainer")
    args = ap.parse_args()
    cls = Trainer if args.stock else FixedTrainer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=config.MLM_PROBABILITY
    )
    ds = PackedTokenDataset(config.TRAIN_BIN, 128, len(config.SPECIAL_TOKENS))

    print(f"  trainer: {cls.__name__}   effective batch 64 in every configuration\n")
    results = {}
    for batch, accum in ((64, 1), (16, 4), (4, 16)):
        loss, gnorm = first_step(cls, batch, accum, tok, ds, collator)
        results[accum] = (loss, gnorm)
        print(f"  accum {accum:<3d} batch {batch:<3d}  loss {loss:8.4f}  grad_norm {gnorm:8.3f}")

    # Evaluation must NOT be normalised: prediction_step also routes through
    # compute_loss, and dividing there makes eval_loss look artificially good.
    print("\n  eval loss must match across accum (evaluation does no accumulation):")
    eval_losses = {}
    for batch, accum in ((64, 1), (16, 4)):
        torch.manual_seed(config.SEED)
        model = build_model(tok)
        args_e = TrainingArguments(
            output_dir="/tmp/check_accum_eval", per_device_eval_batch_size=32,
            gradient_accumulation_steps=accum, bf16=torch.cuda.is_bf16_supported(),
            report_to="none", save_strategy="no", seed=config.SEED,
        )
        small = torch.utils.data.Subset(ds, list(range(256)))
        t = cls(model=model, args=args_e, eval_dataset=small, data_collator=collator)
        eval_losses[accum] = t.evaluate()["eval_loss"]
        print(f"    accum {accum:<3d} eval_loss {eval_losses[accum]:8.4f}")
    eval_gap = abs(eval_losses[4] - eval_losses[1])

    ref = results[1][0]
    worst = max(abs(results[a][0] - ref) for a in results)
    print()
    if eval_gap > 0.01:
        print(f"  FAIL — eval_loss differs by {eval_gap:.4f} across accum settings;")
        print("         the normalisation is leaking into the evaluation path.")
        sys.exit(1)
    if worst < 0.1:
        print(f"  PASS — train losses within {worst:.4f} of the accum=1 reference,")
        print(f"         eval losses within {eval_gap:.4f}")
    else:
        print(f"  FAIL — losses differ by up to {worst:.4f}; accumulation is not normalised")
        print("         gradients are inflated by the accumulation factor, so the")
        print("         effective learning rate is accum x the configured value.")
        sys.exit(1)


if __name__ == "__main__":
    main()
