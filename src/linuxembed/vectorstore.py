#!/usr/bin/env python3
"""Persist the kernel index to Redis (RediSearch vector index) and to FAISS.

Two stores, deliberately:

* **Redis** is the live store the RAG queries. It survives process restarts,
  can be shared by several clients (an MCP server, a CLI, a notebook) without
  each loading a 900 MB array, and keeps the chunk text next to its vector so
  retrieval is a single round trip.
* **FAISS** is a portable on-disk artifact — one file you can copy to another
  machine, diff against a rebuild, or load without a Redis running.

The Redis index is FLAT with COSINE distance, which is the exact analogue of
`faiss.IndexFlatIP` over L2-normalised vectors: both compute a true inner
product against every vector, with no approximation. HNSW is available via
`--hnsw` and is far faster per query, but it is approximate — recall depends on
EF_RUNTIME, so the two stores would no longer agree. For 865k vectors FLAT is
comfortably fast enough for RAG.

The index name is namespaced (`idx:lem:*`) so it cannot collide with an
existing code-rag index on the same Redis.

    python -m linuxembed.vectorstore build
    python -m linuxembed.vectorstore query "how are free pages coalesced"
    python -m linuxembed.vectorstore info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import config
from .search import INDEX_META, INDEX_NPY

REDIS_URL = os.environ.get("LEM_REDIS_URL", "redis://127.0.0.1:6379")
INDEX_NAME = os.environ.get("LEM_REDIS_INDEX", "idx:lem:linux")
KEY_PREFIX = os.environ.get("LEM_REDIS_PREFIX", "lem:chunk:")
FAISS_PATH = config.OUT_DIR / "kernel.faiss"
FAISS_IDS = config.OUT_DIR / "kernel.faiss.ids.jsonl"


def connect():
    import redis

    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    client.ping()
    return client


def drop_index(client, delete_docs: bool) -> None:
    from redis.exceptions import ResponseError

    try:
        # DD deletes the documents too; without it the hashes survive and a
        # rebuild would double-write into a stale keyspace.
        client.execute_command("FT.DROPINDEX", INDEX_NAME, *(["DD"] if delete_docs else []))
        print(f"  dropped existing index {INDEX_NAME}"
              f"{' and its documents' if delete_docs else ''}")
    except ResponseError as exc:
        if "unknown index" not in str(exc).lower():
            raise


def create_index(client, dim: int, hnsw: bool) -> None:
    algo = "HNSW" if hnsw else "FLAT"
    vec_args = [
        "TYPE", "FLOAT32",
        "DIM", str(dim),
        "DISTANCE_METRIC", "COSINE",
    ]
    if hnsw:
        vec_args += ["M", "16", "EF_CONSTRUCTION", "200"]

    # `code` is deliberately NOT in the schema. It is still written to the hash
    # and still returnable by FT.SEARCH RETURN -- only the inverted index is
    # skipped. Full-text indexing ~700 MB of C would cost a large amount of
    # memory and ingest time to support a lexical path that hybrid retrieval
    # already covers in Python over the small candidate pool.
    client.execute_command(
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH",
        "PREFIX", "1", KEY_PREFIX,
        "SCHEMA",
        "path", "TEXT",
        "name", "TEXT", "NOSTEM", "SORTABLE",
        "kind", "TAG",
        "line", "NUMERIC",
        "embedding", "VECTOR", algo, str(len(vec_args)), *vec_args,
    )
    print(f"  created {INDEX_NAME}  ({algo}, COSINE, dim={dim})")


def cmd_build(args: argparse.Namespace) -> None:
    if not INDEX_NPY.exists():
        sys.exit(f"no embeddings at {INDEX_NPY} — run `linuxembed.search build` first")

    vecs = np.load(INDEX_NPY)
    docs = [json.loads(l) for l in open(INDEX_META, encoding="utf-8") if l.strip()]
    if len(docs) != vecs.shape[0]:
        sys.exit(f"metadata/vector mismatch: {len(docs)} docs vs {vecs.shape[0]} vectors")
    dim = int(vecs.shape[1])
    print(f"  {len(docs):,} chunks, dim {dim}")

    if not args.faiss_only:
        client = connect()
        drop_index(client, delete_docs=True)
        create_index(client, dim, args.hnsw)

        pipe = client.pipeline(transaction=False)
        for i, doc in enumerate(docs):
            # Redis stores FLOAT32; the .npy is float16 to keep the file small.
            blob = vecs[i].astype(np.float32).tobytes()
            pipe.hset(f"{KEY_PREFIX}{i}", mapping={
                b"path": doc["path"].encode(),
                b"name": doc["name"].encode(),
                b"kind": doc["kind"].encode(),
                b"line": str(doc.get("line", 0)).encode(),
                b"code": doc["code"].encode(),
                b"embedding": blob,
            })
            if (i + 1) % args.batch == 0:
                pipe.execute()
                pipe = client.pipeline(transaction=False)
                print(f"    ingested {i + 1:,}/{len(docs):,}", flush=True)
        pipe.execute()

        info = client.execute_command("FT.INFO", INDEX_NAME)
        info = {info[j].decode(): info[j + 1] for j in range(0, len(info) - 1, 2)}
        print(f"  redis: {info.get('num_docs')} docs indexed")

    if not args.redis_only:
        import faiss

        FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # IndexFlatIP over L2-normalised vectors == cosine similarity, matching
        # the Redis FLAT/COSINE index exactly.
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        block = 100_000
        for start in range(0, vecs.shape[0], block):
            stop = min(start + block, vecs.shape[0])
            index.add_with_ids(
                np.ascontiguousarray(vecs[start:stop].astype(np.float32)),
                np.arange(start, stop, dtype=np.int64),
            )
        faiss.write_index(index, str(FAISS_PATH))
        with open(FAISS_IDS, "w", encoding="utf-8") as fh:
            for i, doc in enumerate(docs):
                fh.write(json.dumps({
                    "id": i, "path": doc["path"], "name": doc["name"],
                    "kind": doc["kind"], "line": doc.get("line", 0),
                }, ensure_ascii=False) + "\n")
        mb = FAISS_PATH.stat().st_size / 1048576
        print(f"  faiss: {index.ntotal:,} vectors -> {FAISS_PATH} ({mb:.0f} MB)")
        print(f"  faiss ids -> {FAISS_IDS}")


def redis_knn(client, query_vec: np.ndarray, k: int) -> list[dict]:
    blob = query_vec.astype(np.float32).tobytes()
    # KNN syntax: filter => [KNN k @field $vec]. "*" means no pre-filter.
    res = client.execute_command(
        "FT.SEARCH", INDEX_NAME,
        f"*=>[KNN {k} @embedding $vec AS score]",
        "PARAMS", "2", "vec", blob,
        "SORTBY", "score",
        "RETURN", "6", "path", "name", "kind", "line", "code", "score",
        "DIALECT", "2",
    )
    def as_str(value) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    out = []
    # res is [total, key1, [field, value, ...], key2, [...], ...]
    for i in range(1, len(res), 2):
        fields = res[i + 1]
        if not fields:
            continue
        # Keys and values both come back as bytes; normalise keys to str once
        # and read them as str everywhere (mixing the two silently yields empty
        # fields rather than an error).
        d = {as_str(fields[j]): fields[j + 1] for j in range(0, len(fields) - 1, 2)}
        out.append({
            "path": as_str(d.get("path", "")),
            "name": as_str(d.get("name", "")),
            "kind": as_str(d.get("kind", "")),
            "line": int(as_str(d.get("line", "0")) or 0),
            "code": as_str(d.get("code", "")),
            # RediSearch returns COSINE *distance*; similarity is 1 - distance.
            "score": 1.0 - float(as_str(d.get("score", "1"))),
        })
    return out


def cmd_query(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    client = connect()
    model = SentenceTransformer(str(args.encoder))
    model.max_seq_length = config.EMBED_MAX_SEQ
    q = model.encode([args.text], normalize_embeddings=True, convert_to_numpy=True)[0]

    hits = redis_knn(client, q, args.k)
    print(f"\n  query: {args.text!r}   [redis {INDEX_NAME}]\n")
    for rank, h in enumerate(hits, start=1):
        first = h["code"].split("\n")[0][:96]
        print(f"  {rank:2d}. {h['name']}  ({h['kind']})   sim {h['score']:.4f}")
        print(f"      {h['path']}:{h['line']}")
        print(f"      {first}\n")


def cmd_info(args: argparse.Namespace) -> None:
    client = connect()
    try:
        raw = client.execute_command("FT.INFO", INDEX_NAME)
    except Exception as exc:
        sys.exit(f"index {INDEX_NAME} not found: {exc}")
    info = {raw[i].decode(): raw[i + 1] for i in range(0, len(raw) - 1, 2)}
    for key in ("num_docs", "num_records", "inverted_sz_mb", "vector_index_sz_mb",
                "total_indexing_time", "indexing"):
        if key in info:
            val = info[key]
            print(f"  {key:<22s} {val.decode() if isinstance(val, bytes) else val}")
    print(f"  faiss file            {FAISS_PATH} "
          f"({'present' if FAISS_PATH.exists() else 'missing'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Redis + FAISS vector store")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ingest embeddings into Redis and write a FAISS index")
    b.add_argument("--batch", type=int, default=2000, help="pipeline flush size")
    b.add_argument("--hnsw", action="store_true",
                   help="approximate HNSW index instead of exact FLAT")
    b.add_argument("--redis-only", action="store_true")
    b.add_argument("--faiss-only", action="store_true")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="KNN search against the Redis index")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--encoder", type=Path, default=config.STAGE1_DIR)
    q.set_defaults(func=cmd_query)

    i = sub.add_parser("info", help="show index statistics")
    i.set_defaults(func=cmd_info)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
