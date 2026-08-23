#!/usr/bin/env python3
"""Stage 3 — tokenize the corpus into a flat uint16 token stream on disk.

Instead of a HuggingFace Dataset of padded examples, we pack every file's tokens
end-to-end into one contiguous .bin (nanoGPT style) and slice random windows at
train time. This matters on an 8 GB card: there is zero padding waste, no
per-example Python overhead in the dataloader, and the whole ~230M-token corpus
is ~460 MB on disk and memory-mapped rather than resident.

Files are separated by [SEP] so the model still sees document boundaries.

Usage:
    python -m linuxembed.pack --corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from . import config

_TOK = None
_SEP_ID = None


def _init_worker(tokenizer_dir: str) -> None:
    global _TOK, _SEP_ID
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir)
    _SEP_ID = _TOK.sep_token_id


def _encode(text: str) -> np.ndarray:
    # add_special_tokens=False: we are packing a continuous stream, so we append
    # a single [SEP] as a document separator rather than wrapping every file.
    ids = _TOK(text, add_special_tokens=False, truncation=False)["input_ids"]
    ids.append(_SEP_ID)
    return np.asarray(ids, dtype=np.uint16)


def read_texts(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack corpus into uint16 token bins")
    ap.add_argument("--corpus", type=Path, default=config.CORPUS_JSONL)
    ap.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_DIR)
    ap.add_argument("--train-out", type=Path, default=config.TRAIN_BIN)
    ap.add_argument("--val-out", type=Path, default=config.VAL_BIN)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    assert config.VOCAB_SIZE <= 65536, "uint16 packing requires vocab <= 65536"
    args.train_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Tokenizing {args.corpus} with {args.workers} workers …")
    chunks: list[np.ndarray] = []
    total = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(str(args.tokenizer),)) as pool:
        for i, arr in enumerate(pool.imap(_encode, read_texts(args.corpus), chunksize=64)):
            chunks.append(arr)
            total += arr.size
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,} files … {total / 1e6:.1f}M tokens", flush=True)

    stream = np.concatenate(chunks)
    del chunks
    n_val = int(len(stream) * config.VAL_FRACTION)
    # Take validation from the tail so it is a contiguous held-out region rather
    # than tokens interleaved with training windows.
    train, val = stream[:-n_val], stream[-n_val:]

    train.tofile(args.train_out)
    val.tofile(args.val_out)

    print(f"\n  total  {len(stream) / 1e6:.1f}M tokens")
    print(f"  train  {len(train) / 1e6:.1f}M -> {args.train_out} "
          f"({args.train_out.stat().st_size / 1048576:.0f} MB)")
    print(f"  val    {len(val) / 1e6:.1f}M -> {args.val_out}")


if __name__ == "__main__":
    main()
