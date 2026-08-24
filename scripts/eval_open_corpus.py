#!/usr/bin/env python3
"""Open-corpus retrieval evaluation — the number that actually matters for RAG.

evaluate.py scores 2,000 held-out queries against 4,000 candidates with the
answer guaranteed present. That measures the contrastive objective, but it is
not the retrieval task: RAG searches the whole tree, where the answer competes
with ~900k chunks, most of them undocumented code the model never saw as a
positive.

This scores held-out kernel-doc anchors against the FULL index, matching a
query to its function by (path, name).

Recall@50 is the number to read: a cross-encoder reranker can only fix ordering
inside the pool it is given, so if recall@50 is low, reranking cannot help and
the fix has to be recall.

    python scripts/eval_open_corpus.py --n 500
    python scripts/eval_open_corpus.py --n 500 --hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from linuxembed import config
from linuxembed.bm25 import BM25
from linuxembed.search import INDEX_META, INDEX_NPY, dense_similarity, rrf_fuse
from linuxembed.train_embed import load_pairs

KS = (1, 5, 10, 50, 100, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="held-out queries to score")
    ap.add_argument("--encoder", type=Path, default=config.STAGE1_DIR)
    ap.add_argument("--hybrid", action="store_true", help="BM25 rerank of dense top-200")
    ap.add_argument("--pool", type=int, default=200)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    docs = [json.loads(l) for l in open(INDEX_META, encoding="utf-8") if l.strip()]
    vecs = np.asarray(np.load(INDEX_NPY, mmap_mode="r"))
    print(f"  index: {len(docs):,} chunks")

    # (path, name) -> chunk id. A definition can appear once per file only.
    location = {}
    for i, d in enumerate(docs):
        location.setdefault((d["path"], d["name"]), i)

    pairs = load_pairs(config.PAIRS_JSONL)[: config.PAIR_VAL_SIZE]
    scored = [p for p in pairs if (p["path"], p["name"]) in location][: args.n]
    print(f"  queries: {len(scored)} held-out anchors resolvable to a chunk\n")

    model = SentenceTransformer(str(args.encoder))
    model.max_seq_length = config.EMBED_MAX_SEQ

    qv = model.encode([p["anchor"] for p in scored], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=64, show_progress_bar=True)

    hits = {k: 0 for k in KS}
    rr = 0.0
    ranks: list[int] = []
    for j, p in enumerate(scored):
        gold = location[(p["path"], p["name"])]
        sims = dense_similarity(vecs, qv[j])
        order = np.argsort(-sims)[: max(KS)]

        if args.hybrid:
            pool = list(order[: args.pool])
            bm = BM25([docs[i]["code"] for i in pool])
            lex = [pool[i] for i, _ in bm.top_k(p["anchor"], len(pool))]
            fused = rrf_fuse([pool, lex])
            order = np.array(sorted(fused, key=lambda d: -fused[d]))

        where = np.where(order == gold)[0]
        rank = int(where[0]) + 1 if len(where) else None
        ranks.append(rank if rank else 10**9)
        if rank:
            rr += 1.0 / rank
            for k in KS:
                if rank <= k:
                    hits[k] += 1

    n = len(scored)
    print(f"\n  {'hybrid' if args.hybrid else 'dense'} over the full index")
    for k in KS:
        print(f"    recall@{k:<4d} {hits[k] / n:.4f}")
    print(f"    MRR        {rr / n:.4f}")
    finite = [r for r in ranks if r < 10**9]
    if finite:
        print(f"    median rank (found only) {int(np.median(finite))}")


if __name__ == "__main__":
    main()
