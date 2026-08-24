#!/usr/bin/env python3
"""Code RAG over the Linux kernel, built on the from-scratch embedding model.

Retrieval uses our own encoder; generation is delegated to a local Ollama model.
Nothing here calls a hosted API.

    python -m linuxembed.rag ask "how does the buddy allocator split pages"
    python -m linuxembed.rag ask "what happens on mutex contention" --hybrid
    python -m linuxembed.rag ask "how is a page freed" --model qwen2.5-coder:14b

The retriever is 43M parameters (~170 MB in fp32), which is small enough to sit
on the same 8 GB card as a 7B generator. A larger encoder forces you to unload
one to run the other.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import config
from .search import INDEX_META, INDEX_NPY, dense_similarity, rrf_fuse

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# qwen2.5-coder:14b over deepseek-coder:6.7b. On the same retrieved context the
# 6.7b model ignored the citation instruction and described page *splitting* as
# merging -- it answered from general knowledge despite the prompt forbidding it.
# The 14b cited its sources and named the actual functions in the excerpts
# (find_buddy_page_pfn, __del_page_from_free_list, set_buddy_order). Grounding is
# what a RAG is for, so the larger model is worth the VRAM here.
DEFAULT_LLM = "qwen2.5-coder:14b"

SYSTEM_PROMPT = """You are a Linux kernel expert answering questions about kernel source code.

Rules:
- Answer ONLY from the numbered code excerpts provided. They are the ground truth.
- Cite the excerpts you used as [1], [2], … inline.
- If the excerpts do not contain the answer, say so plainly. Do not guess, and do
  not fall back on general knowledge about older kernel versions.
- Be concrete: name the functions, structs and flags involved.
"""


def build_context(hits: list[dict], budget_chars: int) -> tuple[str, list[dict]]:
    """Format retrieved chunks as a numbered context block within a char budget."""
    parts: list[str] = []
    used: list[dict] = []
    total = 0
    for i, d in enumerate(hits, start=1):
        block = (
            f"[{i}] {d['path']}:{d.get('line', 0)}  ({d['kind']} {d['name']})\n"
            f"```c\n{d['code']}\n```\n"
        )
        if total + len(block) > budget_chars and used:
            break
        parts.append(block)
        used.append(d)
        total += len(block)
    return "\n".join(parts), used


def call_ollama(model: str, prompt: str, timeout: int, num_ctx: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Ollama silently truncates to the model default (often 2048) otherwise,
        # which would drop most of the retrieved context and produce an answer
        # that looks grounded but is not.
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"could not reach Ollama at {OLLAMA_URL}: {exc}\n"
            "This must run on the machine hosting Ollama (it binds localhost)."
        )


def retrieve_redis(text: str, k: int, hybrid: bool, encoder: Path) -> list[dict]:
    """Retrieve from the Redis vector index.

    Redis does the KNN, so no 900 MB array is loaded per process — which is what
    makes this usable from a long-lived server rather than only a CLI.
    """
    from sentence_transformers import SentenceTransformer

    from .vectorstore import connect, redis_knn

    client = connect()
    model = SentenceTransformer(str(encoder))
    model.max_seq_length = config.EMBED_MAX_SEQ
    q = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Over-fetch when re-ranking so BM25 has a pool to work with.
    hits = redis_knn(client, q, k * 20 if hybrid else k)
    if not hybrid:
        return hits[:k]

    from .bm25 import BM25

    bm = BM25([h["code"] for h in hits])
    lexical = [i for i, _ in bm.top_k(text, len(hits))]
    fused = rrf_fuse([list(range(len(hits))), lexical])
    order = sorted(fused, key=lambda d: -fused[d])[:k]
    return [hits[i] for i in order]


def retrieve(text: str, k: int, hybrid: bool, encoder: Path) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    if not INDEX_NPY.exists():
        sys.exit(f"no index at {INDEX_NPY} — run `python -m linuxembed.search build` first")

    vecs = np.load(INDEX_NPY)
    docs = [json.loads(l) for l in open(INDEX_META, encoding="utf-8") if l.strip()]

    model = SentenceTransformer(str(encoder))
    model.max_seq_length = config.EMBED_MAX_SEQ
    q = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    scores = dense_similarity(vecs, q[0])
    dense_order = np.argsort(-scores)

    if hybrid:
        from .bm25 import BM25

        pool = min(k * 20, len(docs))
        # BM25 over the whole tree would be rebuilt per query, so restrict the
        # lexical pass to the dense candidate pool. This is re-ranking, not a
        # full independent lexical search.
        cand = list(dense_order[:pool])
        bm = BM25([docs[i]["code"] for i in cand])
        lexical = [cand[i] for i, _ in bm.top_k(text, pool)]
        fused = rrf_fuse([cand, lexical])
        order = sorted(fused, key=lambda d: -fused[d])[:k]
    else:
        order = list(dense_order[:k])

    out = []
    for i in order:
        d = dict(docs[i])
        d["score"] = float(scores[i])
        out.append(d)
    return out


def cmd_ask(args: argparse.Namespace) -> None:
    fetch = retrieve_redis if args.store == "redis" else retrieve
    hits = fetch(args.question, args.k, args.hybrid, args.encoder)
    context, used = build_context(hits, args.context_chars)

    print(f"\n  retrieved {len(used)} excerpts "
          f"({'hybrid' if args.hybrid else 'dense'}, store={args.store}):")
    for i, d in enumerate(used, start=1):
        print(f"    [{i}] {d['path']}:{d.get('line', 0)}  {d['name']}  ({d['score']:.3f})")

    if args.retrieve_only:
        return

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Code excerpts from the Linux kernel:\n\n{context}\n"
        f"Question: {args.question}\n\nAnswer:"
    )
    print(f"\n  asking {args.llm} ({len(prompt):,} chars of prompt) …\n")
    answer = call_ollama(args.llm, prompt, args.timeout, args.num_ctx)
    print(answer)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Code RAG over the Linux kernel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="answer a question from retrieved kernel code")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=8, help="excerpts to retrieve")
    # Hybrid is on by default: measured over 400 held-out queries against the
    # full 914k index it lifts recall@1 from 0.8125 to 0.9050 and MRR from
    # 0.8682 to 0.9374. Two hand-picked queries had suggested it was a wash;
    # they were wrong.
    a.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                   help="dense only, skipping the BM25 rerank (default: hybrid)")
    a.set_defaults(hybrid=True)
    a.add_argument("--store", choices=("redis", "npy"), default="redis",
                   help="redis: server-side KNN, nothing loaded per process; "
                        "npy: local array, no Redis needed")
    a.add_argument("--encoder", type=Path, default=config.STAGE1_DIR)
    a.add_argument("--llm", default=DEFAULT_LLM)
    a.add_argument("--num-ctx", type=int, default=8192)
    a.add_argument("--context-chars", type=int, default=12000)
    a.add_argument("--timeout", type=int, default=300)
    a.add_argument("--retrieve-only", action="store_true",
                   help="show retrieved excerpts without calling the LLM")
    a.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
