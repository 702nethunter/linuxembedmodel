---
license: gpl-2.0
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - code-retrieval
  - code-search
  - linux-kernel
  - c
language:
  - code
---

# linuxembed — a C code embedding model trained from scratch on the Linux kernel

A 43M-parameter BERT encoder for retrieving Linux kernel C code from natural
language. **No pretrained weights were used at any stage** — the tokenizer, the
encoder, and the retrieval objective were all trained from random
initialisation on kernel source alone, on a single 8 GB RTX 3070.

Ask it a question in English; it returns the kernel function that answers it.

Code, training pipeline and evaluation: **https://github.com/702nethunter/linuxembedmodel**

## Results

Retrieval over the **whole kernel — 914,554 chunks** — using 400 held-out
kernel-doc anchors as queries:

| metric | dense | hybrid (dense + BM25 RRF) |
|---|---|---|
| recall@1 | 0.8125 | **0.9050** |
| recall@5 | 0.9375 | **0.9725** |
| recall@10 | 0.9575 | **0.9775** |
| recall@50 | 0.9850 | **0.9950** |
| MRR | 0.8682 | **0.9374** |
| median rank | 1 | 1 |

The correct function ranks first out of 914,554 candidates 90% of the time.

On the closed evaluation (2,000 held-out pairs, 4,000 candidates, hard negatives
are sibling functions from the same file):

| metric | BM25 | MLM only | **this model** |
|---|---|---|---|
| accuracy@1 | 0.7115 | 0.2535 | **0.9225** |
| NDCG@10 | 0.8297 | 0.3856 | **0.9659** |

The MLM-only column is the same encoder before contrastive training. It shows
that MLM pretraining teaches the model C but almost nothing about retrieval —
the contrastive stage supplies nearly all of the retrieval ability.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("702nethunter/linuxembed")

query = "how are free pages coalesced into larger blocks"
code = """
static inline void __free_one_page(struct page *page, unsigned long pfn,
                                   struct zone *zone, unsigned int order,
                                   int migratetype, fpi_t fpi_flags)
{
        ...
}
"""

emb = model.encode([query, code], normalize_embeddings=True)
print(emb @ emb.T)
```

Queries and code go through the **same** encoder with no prefix or instruction —
it is a symmetric bi-encoder.

## Architecture

| | |
|---|---|
| parameters | 42.6M |
| layers / hidden / heads | 8 / 512 / 8 |
| intermediate | 2048 |
| vocabulary | 32,768 byte-level BPE, trained on kernel C |
| max sequence length | **320** |
| pooling | mean |
| output | 512-dim, L2-normalised |
| size on disk | 162 MB |

Deliberately not BERT-base: against a 342M-token corpus on one 3070, a 110M
model would overfit and take days.

**Do not raise `max_seq_length` above 320.** MLM pretraining saw 512, but the
contrastive stage trained at 320, and feeding longer inputs measurably *hurts*
retrieval (a spot check moved a correct answer from rank 12 to rank 40).

## Training

Trained only on the Linux kernel (v7.1-rc5). No CodeSearchNet, no CoRNStack, no
external corpus, no pretrained checkpoint.

**Tokenizer** — byte-level BPE fit on kernel C. BERT's uncased WordPiece is a
poor fit for this domain: it lowercases, so `KMALLOC` and `kmalloc` collapse and
every kernel macro convention is destroyed, and it shatters identifiers into
English word pieces.

**Pretraining** — masked language modelling from random init on 342.6M tokens
from 62,595 `.c`/`.h` files. Two-phase sequence length (128, then 512). Final
MLM loss 0.5314, perplexity 1.70, with train and eval tracking within 0.003.

Only 872 MB of the kernel's 1340 MB of `.c`/`.h` is usable. 468 MB of it — 35%,
concentrated in 476 files — is auto-generated hardware register headers that are
up to 89% bare `#define FOO_MASK 0x1L` lines. Left in, a third of every batch
teaches hex-constant formatting rather than C.

**Contrastive** — 48,777 (anchor, positive, negative) triples mined from the
kernel's ~64,000 hand-written kernel-doc comments. Loss is InfoNCE
(MultipleNegativesRankingLoss).

Two choices in the pair construction do most of the work:

- **The symbol name is stripped from the anchor.** Left in, `kmalloc_node`
  appears in both the query and the positive's signature, and the model learns
  to match an identifier — scoring well on the eval while being useless in
  practice.
- **The kernel-doc block is excluded from the positive**, or the positive
  literally contains the query text.

Hard negatives are **sibling functions from the same file**: same subsystem,
same types, same idioms, wrong function.

### A note on GISTEmbed

The pipeline implements a combined `0.6·GISTEmbed + 0.4·InfoNCE` objective, and
it is **not** what this model uses. Measured against an InfoNCE-only control
under identical batching, learning rate and epochs, GIST was marginally worse
(0.9200 vs 0.9220 accuracy@1) — even with batches grouped by subsystem to make
its masking fire 14× more often.

GIST masks in-batch candidates a frozen guide ranks above the true positive.
That needs a guide confused where the student is confident, and a negative
sampler that actually produces false negatives. Curated sibling negatives and a
self-guide give it neither. This is a result about this dataset, not about
GISTEmbed.

## Limitations

- **Kernel C only.** Trained on one corpus and specialised to it. Do not expect
  it to work on Python, Rust, or non-kernel C.
- **Evaluated on documented functions.** The reported numbers use held-out
  kernel-doc anchors as queries. Roughly 95% of kernel definitions carry no
  kernel-doc, and retrieval over undocumented code is untested.
- **320-token ceiling**, which truncates long functions.
- **Symmetric bi-encoder**, no reranker. `recall@50 = 0.995` means a
  cross-encoder reranker has room to improve `recall@1` substantially.

## License and provenance

**GPL-2.0**, because the training data is.

This model was trained exclusively on Linux kernel source, which is GPL-2.0.
Whether model weights trained on GPL source constitute a derivative work is
genuinely unsettled law. Treat these weights as GPL-2.0-encumbered and get real
legal advice before shipping them in a product.

Note that building from scratch does **not** clean up that provenance. Common
alternatives are permissively licensed — `bert-base-uncased` is Apache-2.0,
`microsoft/codebert-base` is MIT — so avoiding a pretrained base removes very
little legal exposure. The training data is the encumbered part either way.
From-scratch training is well justified here on tokenizer quality and provenance
clarity, not on licensing.

## Citation

```bibtex
@software{linuxembed,
  title  = {linuxembed: a C code embedding model trained from scratch on the Linux kernel},
  author = {702nethunter},
  year   = {2026},
  url    = {https://github.com/702nethunter/linuxembedmodel}
}
```
