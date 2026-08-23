#!/usr/bin/env python3
"""Search the Linux kernel by natural language with the trained model.

Indexes every mined kernel function and lets you query it in plain English.
Dense retrieval by default; `--hybrid` fuses the dense ranking with BM25 via
reciprocal rank fusion, which is the standard fix when a dense model and a
lexical model fail on different queries.

The index is a plain float32 numpy array. At ~49k documents a brute-force
matmul is well under a millisecond and exact, so an ANN index would add a
dependency and approximation error to solve a problem this corpus does not
have.

    python -m linuxembed.search build
    python -m linuxembed.search query "how does the buddy allocator split pages"
    python -m linuxembed.search query "acquire a spinlock without disabling irqs" --hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import config

INDEX_NPY = config.OUT_DIR / "search_index.npy"
INDEX_META = config.OUT_DIR / "search_meta.jsonl"


def load_documents(pairs_path: Path) -> list[dict]:
    """One entry per mined definition, deduplicated by (path, symbol)."""
    seen: set[tuple[str, str]] = set()
    docs: list[dict] = []
    with open(pairs_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["path"], row["name"])
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "path": row["path"],
                "name": row["name"],
                "kind": row["kind"],
                "code": row["positive"],
            })
    return docs


def cmd_build(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    docs = load_documents(args.pairs)
    print(f"  indexing {len(docs):,} kernel definitions with {args.model}")

    model = SentenceTransformer(str(args.model))
    model.max_seq_length = config.EMBED_MAX_SEQ
    vecs = model.encode(
        [d["code"] for d in docs],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    INDEX_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_NPY, vecs)
    with open(INDEX_META, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    size = INDEX_NPY.stat().st_size / 1048576
    print(f"  saved {vecs.shape} -> {INDEX_NPY} ({size:.0f} MB)")
    print(f"  saved metadata -> {INDEX_META}")


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion.

    Scores from a cosine-similarity model and from BM25 are on different
    scales and cannot be added directly. RRF combines the *ranks* instead, so
    no calibration between the two is required.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def cmd_query(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    if not INDEX_NPY.exists():
        sys.exit(f"no index at {INDEX_NPY} — run `search build` first")

    vecs = np.load(INDEX_NPY)
    docs = [json.loads(l) for l in open(INDEX_META, encoding="utf-8") if l.strip()]

    model = SentenceTransformer(str(args.model))
    model.max_seq_length = config.EMBED_MAX_SEQ
    q = model.encode([args.text], normalize_embeddings=True, convert_to_numpy=True)
    dense_scores = (vecs @ q[0]).astype(np.float32)
    dense_order = np.argsort(-dense_scores)

    if args.hybrid:
        from .bm25 import BM25

        pool = args.k * 20  # fuse over a deeper pool than we display
        bm = BM25([d["code"] for d in docs])
        lexical = [i for i, _ in bm.top_k(args.text, pool)]
        fused = rrf_fuse([list(dense_order[:pool]), lexical])
        order = sorted(fused, key=lambda d: -fused[d])[: args.k]
        label = "hybrid (dense + BM25, RRF)"
    else:
        order = list(dense_order[: args.k])
        label = "dense"

    print(f"\n  query: {args.text!r}   [{label}]\n")
    for rank, idx in enumerate(order, start=1):
        d = docs[idx]
        first = d["code"].split("\n")[0][:96]
        print(f"  {rank:2d}. {d['name']}  ({d['kind']})   score {dense_scores[idx]:.4f}")
        print(f"      {d['path']}")
        print(f"      {first}")
        if args.show_code:
            body = "\n".join("        " + l for l in d["code"].split("\n")[:20])
            print(body)
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic search over the kernel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="embed every mined definition")
    b.add_argument("--model", type=Path, default=config.STAGE1_DIR)
    b.add_argument("--pairs", type=Path, default=config.PAIRS_JSONL)
    b.add_argument("--batch-size", type=int, default=64)
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="search the index")
    q.add_argument("text")
    q.add_argument("--model", type=Path, default=config.STAGE1_DIR)
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--hybrid", action="store_true", help="fuse dense with BM25 via RRF")
    q.add_argument("--show-code", action="store_true")
    q.set_defaults(func=cmd_query)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
