#!/usr/bin/env python3
"""Stage 2 — train a byte-level BPE tokenizer on the kernel corpus.

We train our own rather than reuse BERT's vocabulary. Two reasons:

1. Provenance — this repo uses no pretrained artifacts of any kind.
2. Quality — BERT's uncased WordPiece is actively bad for C. It lowercases
   (so `KMALLOC` and `kmalloc` collapse, and every macro/constant convention in
   the kernel is destroyed) and splits identifiers into English word pieces.
   A byte-level BPE fit on kernel C learns `struct`, `->`, `EXPORT_SYMBOL`,
   `spin_lock_irqsave` as units, and can never emit [UNK].

Vocab is 32768 so packed token ids fit in uint16 (2 bytes/token on disk).

Usage:
    python -m linuxembed.tokenizer_train --corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from . import config


def corpus_iterator(path: Path, batch_size: int = 1000):
    """Stream texts from corpus.jsonl in batches (never loads the file at once)."""
    batch: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            batch.append(json.loads(line)["text"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    # ByteLevel with add_prefix_space=False: source code is not prose, and a
    # forced leading space would make `int` and ` int` different tokens.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Train kernel BPE tokenizer")
    ap.add_argument("--corpus", type=Path, default=config.CORPUS_JSONL)
    ap.add_argument("--out", type=Path, default=config.TOKENIZER_DIR)
    ap.add_argument("--vocab-size", type=int, default=config.VOCAB_SIZE)
    args = ap.parse_args()

    print(f"Training byte-level BPE (vocab={args.vocab_size:,}) on {args.corpus} …")

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=config.TOKENIZER_MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(corpus_iterator(args.corpus), trainer=trainer)

    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    # BERT-style single/pair templates so the model sees [CLS] x [SEP].
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=config.MODEL_MAX_POSITION,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.out)
    print(f"  saved -> {args.out}  (vocab={fast.vocab_size:,})")

    sample = "static inline void spin_lock_irqsave(spinlock_t *lock, unsigned long flags)"
    ids = fast(sample)["input_ids"]
    print(f"\n  sanity: {len(ids)} tokens for {len(sample)} chars")
    print(f"  {fast.convert_ids_to_tokens(ids)[:24]}")


if __name__ == "__main__":
    main()
