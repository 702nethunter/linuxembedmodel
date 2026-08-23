#!/usr/bin/env python3
"""Stage 7 — retrieval evaluation on held-out kernel-doc pairs.

The held-out slice is the first PAIR_VAL_SIZE rows of pairs.jsonl, which
train_embed.py excludes from training. The eval corpus contains every held-out
positive *and* its hard negative, so the model must separate a function from its
own sibling rather than from unrelated code.

Usage:
    python -m linuxembed.evaluate --model artifacts/embed-stage2-gist
    python -m linuxembed.evaluate --compare artifacts/embed-stage1-infonce \
                                            artifacts/embed-stage2-gist
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from . import config
from .train_embed import build_evaluator, build_sentence_transformer, load_pairs

METRIC_KEYS = ("accuracy@1", "accuracy@5", "mrr@10", "ndcg@10", "recall@10")


def load_any(path: Path) -> SentenceTransformer:
    """Load either a saved SentenceTransformer or a bare HF encoder dir."""
    if (path / "modules.json").exists():
        return SentenceTransformer(str(path))
    # A raw MLM checkpoint — wrap it so an untrained baseline can be scored too.
    return build_sentence_transformer(path)


def short(key: str) -> str:
    return key.split("_")[-1].lower()


def pick(scores: dict[str, float]) -> dict[str, float]:
    out = {}
    for k, v in scores.items():
        s = short(k)
        if s in METRIC_KEYS:
            out[s] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality")
    ap.add_argument("--model", type=Path, default=config.STAGE2_DIR)
    ap.add_argument("--compare", type=Path, nargs="+", default=None)
    ap.add_argument("--pairs", type=Path, default=config.PAIRS_JSONL)
    args = ap.parse_args()

    rows = load_pairs(args.pairs)
    val_rows = rows[: config.PAIR_VAL_SIZE]
    print(f"  evaluating on {len(val_rows):,} held-out pairs "
          f"({len(val_rows) * 2:,} corpus docs)\n")
    evaluator = build_evaluator(val_rows, name="kernel-eval")

    targets = args.compare or [args.model]
    results: dict[str, dict[str, float]] = {}
    for path in targets:
        if not path.exists():
            print(f"  ! skipping missing {path}")
            continue
        print(f"  scoring {path} …")
        model = load_any(path)
        model.max_seq_length = config.EMBED_MAX_SEQ
        results[path.name] = pick(evaluator(model))
        del model
        torch.cuda.empty_cache()

    if not results:
        raise SystemExit("nothing to evaluate")

    names = list(results)
    width = max(len(n) for n in names) + 2
    header = "  " + "metric".ljust(14) + "".join(n.ljust(width) for n in names)
    print(f"\n{'═' * len(header)}\n{header}\n{'═' * len(header)}")
    for m in METRIC_KEYS:
        if not any(m in results[n] for n in names):
            continue
        row = "  " + m.ljust(14)
        for n in names:
            row += f"{results[n].get(m, float('nan')):.4f}".ljust(width)
        print(row)
    print()


if __name__ == "__main__":
    main()
