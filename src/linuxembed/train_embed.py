#!/usr/bin/env python3
"""Stage 6 — contrastive training of the embedding model.

Turns the from-scratch MLM encoder into a retrieval model, in two stages:

  stage 1  loss = InfoNCE only.
           The MLM checkpoint has never done retrieval and its mean-pooled
           output is not yet a metric space, so a guide-based mask would be
           filtering against noise. Plain InfoNCE establishes the geometry.

  stage 2  loss = 0.6 * GISTEmbed + 0.4 * InfoNCE, guided by the FROZEN stage-1
           checkpoint. Because this repo uses no pretrained weights, the guide
           is our own stage-1 model — self-guided GIST. It shares the tokenizer
           with the student by construction, so the same tokenized batch feeds
           both with no re-encoding.

Usage:
    python -m linuxembed.train_embed --stage 1
    python -m linuxembed.train_embed --stage 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import SequentialSampler
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

from . import config
from .accum import NormalizeAccumLossMixin
from .losses import GISTInfoNCELoss


class EmbedTrainer(NormalizeAccumLossMixin, SentenceTransformerTrainer):
    """SentenceTransformerTrainer with the same accumulation fix. See accum.py."""


class SequentialBatchTrainer(EmbedTrainer):
    """Feeds the dataset in the order given, without reshuffling.

    Used with subsystem-grouped ordering so each micro-batch is drawn from one
    kernel subsystem. GIST can only mask a candidate the guide ranks above the
    true positive, which effectively never happens when in-batch candidates are
    unrelated functions. Measured mask rate on the held-out set:

        random batches + sibling negatives   0.027% of candidates, 1.2% of rows
        same-subsystem batches + siblings    0.372% of candidates, 8.4% of rows

    Grouping is what gives GIST anything to act on.
    """

    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


def subsystem(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def group_by_subsystem(rows: list[dict], batch: int, seed: int) -> list[dict]:
    """Order rows so consecutive `batch`-sized runs share a subsystem.

    Rows are shuffled within each subsystem and the subsystem order is shuffled,
    so grouping does not also impose a fixed curriculum. Subsystems with fewer
    than `batch` rows are pooled into a remainder so nothing is discarded.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[subsystem(row["path"])].append(row)

    blocks: list[list[dict]] = []
    remainder: list[dict] = []
    for group in buckets.values():
        rng.shuffle(group)
        for i in range(0, len(group), batch):
            chunk = group[i : i + batch]
            if len(chunk) == batch:
                blocks.append(chunk)
            else:
                remainder.extend(chunk)

    # Tail chunks from every subsystem are pooled and re-batched, so no row is
    # dropped; these mixed batches are the price of keeping the full dataset.
    rng.shuffle(remainder)
    for i in range(0, len(remainder), batch):
        blocks.append(remainder[i : i + batch])

    rng.shuffle(blocks)
    return [row for block in blocks for row in block]


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_sentence_transformer(encoder_dir: Path) -> SentenceTransformer:
    """Wrap the from-scratch MLM encoder as a retrieval model.

    Mean pooling, not CLS: the [CLS] token only carries sentence-level meaning
    if it was trained with a sentence-level objective (NSP), and our MLM-only
    pretraining never gave it one. Mean pooling over token states is the right
    readout for an encoder pretrained with MLM alone.
    """
    word = models.Transformer(str(encoder_dir), max_seq_length=config.EMBED_MAX_SEQ)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    norm = models.Normalize()
    return SentenceTransformer(modules=[word, pool, norm])


def build_evaluator(val_rows: list[dict], name: str) -> InformationRetrievalEvaluator:
    """IR eval: every held-out positive AND negative is in the corpus as a distractor."""
    queries, corpus, relevant = {}, {}, {}
    for i, r in enumerate(val_rows):
        qid, pid, nid = f"q{i}", f"d{i}p", f"d{i}n"
        queries[qid] = r["anchor"]
        corpus[pid] = r["positive"]
        corpus[nid] = r["negative"]
        relevant[qid] = {pid}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5],
        precision_recall_at_k=[10],
        batch_size=64,
        show_progress_bar=True,
        write_csv=False,
        name=name,
    )


def report(scores: dict[str, float], label: str) -> None:
    print(f"\n{'─' * 62}\n  {label}\n{'─' * 62}")
    for k in sorted(scores):
        if any(m in k.lower() for m in ("accuracy", "mrr", "ndcg", "recall")):
            print(f"  {k:<48s} {scores[k]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrastive training")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--pairs", type=Path, default=config.PAIRS_JSONL)
    ap.add_argument("--encoder", type=Path, default=config.PRETRAIN_DIR / "phase2")
    ap.add_argument("--homogeneous-batches", action="store_true",
                    help="draw each micro-batch from one subsystem, so in-batch "
                         "candidates are confusable and GIST has something to mask")
    ap.add_argument("--w-gist", type=float, default=None, help="override W_GIST")
    ap.add_argument("--w-infonce", type=float, default=None, help="override W_INFONCE")
    ap.add_argument("--out", type=Path, default=None, help="override output dir")
    args = ap.parse_args()

    torch.manual_seed(config.SEED)

    rows = load_pairs(args.pairs)
    val_rows = rows[: config.PAIR_VAL_SIZE]
    train_rows = rows[config.PAIR_VAL_SIZE :]
    print(f"  pairs: {len(train_rows):,} train  /  {len(val_rows):,} held-out")

    if args.homogeneous_batches:
        train_rows = group_by_subsystem(train_rows, config.EMBED_BATCH, config.SEED)
        print(f"  batching: subsystem-grouped (micro-batch {config.EMBED_BATCH})")

    train_ds = Dataset.from_list([
        {"anchor": r["anchor"], "positive": r["positive"], "negative": r["negative"]}
        for r in train_rows
    ])

    if args.stage == 1:
        model = build_sentence_transformer(args.encoder)
        guide = None
        out_dir = config.STAGE1_DIR
        epochs, lr = config.EMBED_EPOCHS_STAGE1, config.EMBED_LR_STAGE1
        print("  stage 1 — InfoNCE only (no guide yet)")
    else:
        if not config.STAGE1_DIR.exists():
            raise SystemExit("stage 2 needs stage 1 to have been run first")
        model = SentenceTransformer(str(config.STAGE1_DIR))
        # A separate frozen snapshot of the same checkpoint: the student moves,
        # the guide must not.
        guide = SentenceTransformer(str(config.STAGE1_DIR))
        out_dir = config.STAGE2_DIR
        epochs, lr = config.EMBED_EPOCHS_STAGE2, config.EMBED_LR_STAGE2

    w_gist = config.W_GIST if args.w_gist is None else args.w_gist
    w_infonce = config.W_INFONCE if args.w_infonce is None else args.w_infonce
    if args.out is not None:
        out_dir = args.out
    if args.stage == 2:
        # w_gist=0 turns stage 2 into an InfoNCE-only control, which is the only
        # way to attribute a metric change to GIST rather than to the extra epochs.
        kind = "InfoNCE only (GIST disabled)" if w_gist == 0 else \
               f"{w_gist} * GIST + {w_infonce} * InfoNCE (self-guided by stage 1)"
        print(f"  stage 2 — {kind}")
        if w_gist == 0:
            guide = None

    model.max_seq_length = config.EMBED_MAX_SEQ
    evaluator = build_evaluator(val_rows, name=f"kernel-stage{args.stage}")

    before = evaluator(model)
    report(before, f"BEFORE stage {args.stage}")

    loss = GISTInfoNCELoss(
        model=model,
        guide=guide,
        scale=config.EMBED_SCALE,
        w_gist=w_gist,
        w_infonce=w_infonce,
    )

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.EMBED_BATCH,
        gradient_accumulation_steps=config.EMBED_ACCUM,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        # NO_DUPLICATES keeps the same text from appearing twice in a batch,
        # which would otherwise be a guaranteed false negative.
        # Subsystem grouping must survive to the dataloader, so it cannot also
        # be reshuffled by NO_DUPLICATES.
        batch_sampler=BatchSamplers.BATCH_SAMPLER if args.homogeneous_batches
        else BatchSamplers.NO_DUPLICATES,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        seed=config.SEED,
    )

    trainer_cls = SequentialBatchTrainer if args.homogeneous_batches else EmbedTrainer
    trainer = trainer_cls(
        model=model, args=targs, train_dataset=train_ds, loss=loss
    )
    trainer.train()

    after = evaluator(model)
    report(after, f"AFTER stage {args.stage}")

    print(f"\n{'═' * 62}\n  Δ (after − before)\n{'═' * 62}")
    for k in sorted(before):
        if any(m in k.lower() for m in ("accuracy", "mrr", "ndcg", "recall")):
            d = after.get(k, 0.0) - before[k]
            print(f"  {k:<48s} {before[k]:.4f} → {after.get(k, 0.0):.4f}  "
                  f"({'▲' if d >= 0 else '▼'}{abs(d):.4f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
