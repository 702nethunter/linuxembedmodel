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
import hashlib
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


def read_docs(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                yield doc["path"], doc["text"]


def is_val_file(rel_path: str, fraction: float) -> bool:
    """Assign whole files to validation by a hash of their path.

    An earlier version took validation as the contiguous tail of the token
    stream. That looked like a clean held-out region but was not: corpus.jsonl
    is in directory-traversal order, so the tail was entirely mm/kasan,
    mm/kmsan and mm/damon/tests -- one subsystem of highly formulaic sanitizer
    and kunit boilerplate, with its sibling files all in training. The result
    was an eval loss 3x *lower* than train loss (0.42 vs 1.34), which is
    backwards and would have hidden overfitting for the whole run.

    Hashing the path instead gives a random sample spread across every
    subsystem, and holding out whole files means no window of a validation file
    is adjacent to a training window of the same file.
    """
    digest = hashlib.blake2b(rel_path.encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) % 1_000_000) < fraction * 1_000_000


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
    docs = list(read_docs(args.corpus))
    paths = [p for p, _ in docs]
    train_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []
    total = 0

    with Pool(args.workers, initializer=_init_worker, initargs=(str(args.tokenizer),)) as pool:
        # imap preserves input order, so arr[i] corresponds to paths[i].
        stream = pool.imap(_encode, (t for _, t in docs), chunksize=64)
        for i, arr in enumerate(stream):
            if is_val_file(paths[i], config.VAL_FRACTION):
                val_chunks.append(arr)
            else:
                train_chunks.append(arr)
            total += arr.size
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,} files … {total / 1e6:.1f}M tokens", flush=True)

    train = np.concatenate(train_chunks)
    val = np.concatenate(val_chunks)
    del train_chunks, val_chunks

    train.tofile(args.train_out)
    val.tofile(args.val_out)

    print(f"\n  total  {total / 1e6:.1f}M tokens across {len(docs):,} files")
    print(f"  train  {len(train) / 1e6:.1f}M tokens / "
          f"{len(docs) - sum(is_val_file(p, config.VAL_FRACTION) for p in paths):,} files "
          f"-> {args.train_out} ({args.train_out.stat().st_size / 1048576:.0f} MB)")
    print(f"  val    {len(val) / 1e6:.1f}M tokens / "
          f"{sum(is_val_file(p, config.VAL_FRACTION) for p in paths):,} files "
          f"-> {args.val_out}")

    val_paths = [p for p in paths if is_val_file(p, config.VAL_FRACTION)]
    subsys = sorted({p.split("/")[0] for p in val_paths})
    print(f"  val spans {len(subsys)} top-level dirs: {', '.join(subsys[:12])}"
          f"{' …' if len(subsys) > 12 else ''}")


if __name__ == "__main__":
    main()
