"""BM25 lexical baseline for the retrieval evaluation.

A retrieval score in isolation means nothing. "NDCG@10 = 0.72" is only
interpretable against a reference, and the reference that matters for code
search is lexical matching: if a dense model cannot beat BM25 on this task, it
is not earning its training cost.

BM25 is also a *fair* baseline here, unlike a pretrained encoder — it is a
classical algorithm with no learned weights, so it does not violate this repo's
no-pretrained-artifacts constraint.

It is a genuinely strong baseline on this dataset, and the anchor construction
in mine_pairs.py is what keeps it honest. The symbol name is stripped from the
anchor, so BM25 cannot simply match `kmalloc_node` in the query against
`kmalloc_node` in the signature. What is left is prose-versus-code overlap:
identifier fragments, parameter names and type names that survive in both.

Pure Python, no dependencies — the eval corpus is a few thousand documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

# Split on non-alphanumerics, then split snake_case and camelCase, so
# `spin_lock_irqsave` contributes `spin`, `lock`, `irqsave` and
# `kmemLeakScan` contributes `kmem`, `leak`, `scan`.
_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SPLIT_RE.split(text):
        if not piece:
            continue
        for sub in _CAMEL_RE.split(piece):
            if sub:
                out.append(sub.lower())
    return out


class BM25:
    """Okapi BM25 over a fixed document collection."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n_docs = len(docs)
        self.avg_len = sum(self.doc_len) / max(self.n_docs, 1)

        # postings: term -> list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

        # Standard BM25 idf with the +0.5 smoothing; floored at a small positive
        # value so terms appearing in most documents cannot contribute negative
        # score and rank a document below one that matched nothing.
        self.idf: dict[str, float] = {}
        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = max(
                math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0), 1e-6
            )

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_i, tf in plist:
                norm = 1.0 - self.b + self.b * self.doc_len[doc_i] / max(self.avg_len, 1e-9)
                scores[doc_i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scored = list(enumerate(self.scores(query)))
        scored.sort(key=lambda x: (-x[1], x[0]))  # stable: ties broken by index
        return scored[:k]


def evaluate_bm25(val_rows: list[dict], k_values=(1, 5, 10)) -> dict[str, float]:
    """Score BM25 on the same held-out pairs the dense evaluator uses.

    Corpus layout matches build_evaluator(): for row i, document 2i is the
    positive and 2i+1 is its hard negative, so the correct answer for query i
    is always document 2i.
    """
    corpus: list[str] = []
    for row in val_rows:
        corpus.append(row["positive"])
        corpus.append(row["negative"])

    bm25 = BM25(corpus)
    max_k = max(max(k_values), 10)

    hits = {k: 0 for k in k_values}
    mrr = 0.0
    ndcg = 0.0
    recall10 = 0

    for i, row in enumerate(val_rows):
        gold = 2 * i
        ranked = [doc for doc, _ in bm25.top_k(row["anchor"], max_k)]
        rank = ranked.index(gold) + 1 if gold in ranked else None
        for k in k_values:
            if rank is not None and rank <= k:
                hits[k] += 1
        if rank is not None and rank <= 10:
            mrr += 1.0 / rank
            # Single relevant document, so IDCG is 1 and NDCG reduces to this.
            ndcg += 1.0 / math.log2(rank + 1)
            recall10 += 1

    n = max(len(val_rows), 1)
    out = {f"accuracy@{k}": hits[k] / n for k in k_values if k in (1, 5)}
    out["mrr@10"] = mrr / n
    out["ndcg@10"] = ndcg / n
    out["recall@10"] = recall10 / n
    return out
