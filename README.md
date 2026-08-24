# linuxembedmodel

A code embedding model for C, **built from scratch on the Linux kernel** — own
tokenizer, own BERT encoder, own contrastive objective. No pretrained weights
are downloaded or loaded at any stage.

Trained on a single 8 GB RTX 3070.

```
kernel source ──▶ corpus ──▶ BPE tokenizer ──▶ packed tokens
                                                    │
                                              MLM pretrain (from random init)
                                                    │
kernel-doc ──▶ mined NL→C pairs ──▶ stage 1: InfoNCE
                                                    │
                                    stage 2: 0.6·GISTEmbed + 0.4·InfoNCE
                                             (self-guided by stage 1)
                                                    │
                                              embedding model
```

## The model

Weights are on the Hugging Face Hub — 162 MB, loads directly with
`sentence-transformers`:

**https://huggingface.co/HF_REPO_ID**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("HF_REPO_ID")
emb = model.encode(["how are free pages coalesced into larger blocks"],
                   normalize_embeddings=True)
```

**0.9050 recall@1 over 914,554 kernel chunks.** Queries and code share one
encoder, with no prefix or instruction. Do not raise `max_seq_length` above 320
— see [Open-corpus retrieval](#open-corpus-retrieval--the-number-that-matters).

To publish a rebuild: `HF_REPO=<user>/linuxembed bash scripts/publish_hf.sh`
(run it on the training box, where the weights already are).

## Why from scratch

The usual move is to fine-tune `bert-base-uncased` or `microsoft/codebert-base`.
This project does not, for provenance reasons — see [Licensing](#licensing),
which is more nuanced than "avoid pretrained weights."

There is also a technical argument. BERT's uncased WordPiece vocabulary is
genuinely poor for C:

| | `bert-base-uncased` | this repo |
|---|---|---|
| casing | lowercased — `KMALLOC` ≡ `kmalloc`, so every kernel macro convention is destroyed | preserved |
| `spin_lock_irqsave` | shattered into English word-pieces | learned as few units |
| unknown bytes | `[UNK]` | impossible (byte-level) |
| trained on | Wikipedia + BooksCorpus | 872 MB of kernel C |

## The corpus is not just "the kernel"

The kernel tree holds **1340 MB** of `.c`/`.h`. Only **845 MB** of it survives
filtering — 62,595 files, **342.6M tokens**.

The other **468 MB — 35% of the corpus, in just 476 files** — is auto-generated
hardware register headers, overwhelmingly under
`drivers/gpu/drm/amd/include/asic_reg`. They look like this, for 150,000 lines
at a stretch:

```c
#define DCN_REG_FIELD_MASK   0x00000001L
#define DCN_REG_FIELD_SHIFT  0x0
```

One such file measures **89% bare `#define` lines**. Left in, more than a third
of every MLM batch would teach the model hex-constant formatting rather than C.

`corpus.py` drops them with a *structural* test — ≥500 lines and ≥60% `#define`
density — rather than a hardcoded path, so equivalent generated headers
elsewhere in the tree are caught too. On v7.1-rc5 that removes 702 files by
density plus 37 more too large to be hand-written.

## Where supervision comes from

The kernel ships **~64,000 hand-written kernel-doc blocks**. That is a free,
human-authored natural-language → C dataset, and it is the reason this project
needs no synthetic query generation.

```c
/**
 * kmalloc_node - allocate memory from a specific node
 * @size: how many bytes of memory are required
 * @flags: describe the allocation context
 *
 * Return: pointer to the allocated memory, or NULL.
 */
void *kmalloc_node(size_t size, gfp_t flags) { ... }
```

Two details decide whether the model learns semantics or a shortcut:

- **The symbol name is stripped from the anchor.** Left in, `kmalloc_node`
  appears in both the anchor and the positive's signature, and the model learns
  to string-match an identifier — scoring well on the eval while being useless
  on real queries.
- **The kernel-doc block is excluded from the positive.** Left in, the positive
  literally contains the anchor text and the task is trivial.

Hard negatives are **sibling functions from the same file** — same subsystem,
same types, same idioms, wrong function. Records with no sibling are dropped
rather than given a random negative, which would be trivially easy and
contribute no gradient signal.

Yield on v7.1-rc5: 50,881 parsed → 48,865 after anchor dedup → **48,777 pairs**
with hard negatives (37,367 function, 11,410 struct).

## The loss

```
total = 0.6 · GISTEmbed + 0.4 · InfoNCE
```

**InfoNCE** treats every other item in the batch as a negative. That assumption
is wrong often enough to hurt: the kernel is full of near-equivalent helpers, so
a batch routinely contains a "negative" that answers the anchor perfectly well.
Pushing it away is a wrong gradient.

**GISTEmbed** ([Solatorio, 2024](https://arxiv.org/abs/2402.16829)) fixes that
with a frozen *guide* encoder. For anchor *i*, the guide scores its true
positive; any in-batch candidate the guide rates **higher** than that true
positive is presumed a false negative and is masked out of the softmax.

Both terms are kept. GIST alone can over-filter — early on the guide's judgement
is noisy, and aggressive masking starves the model of negatives — so the plain
InfoNCE term stays as a floor.

### Self-guided GIST

GIST needs a guide model, and this repo permits no external weights. The guide
is therefore **our own stage-1 checkpoint, frozen**:

| stage | loss | guide |
|---|---|---|
| 1 | InfoNCE only | — |
| 2 | 0.6·GIST + 0.4·InfoNCE | frozen stage-1 checkpoint |

Stage 1 exists because the MLM checkpoint has never done retrieval — its
mean-pooled output is not yet a metric space, so guide-based masking would be
filtering against noise. Stage 1 establishes the geometry; stage 2 refines it.

A convenient consequence: guide and student share a tokenizer by construction,
so one tokenized batch feeds both with no re-encoding.

Both terms are computed from **one forward pass** (`losses.py`). Composing two
stock sentence-transformers losses would encode every text twice, which does not
fit in 8 GB at a useful batch size.

## Model

~43M parameters — deliberately not BERT-base. Against a ~230M-token corpus and
one 3070, a 110M-parameter model would overfit and take days.

| | |
|---|---|
| layers / hidden / heads | 8 / 512 / 8 |
| intermediate | 2048 |
| vocab | 32,768 byte-level BPE (fits `uint16`) |
| max position | 512 |
| pooling | mean |

Mean pooling, not CLS: `[CLS]` only carries sentence-level meaning if trained
with a sentence-level objective (NSP), and MLM-only pretraining never gives it
one.

## Fitting an 8 GB card

- **Packed `uint16` token stream.** The corpus is tokenized once into a flat
  memory-mapped `.bin` (nanoGPT style) and sliced into windows at train time —
  zero padding waste, no per-example Python overhead, ~460 MB on disk, never
  resident.
- **Two-phase sequence length.** Attention is O(n²), so most steps run at
  seq 128; a short tail at seq 512 trains the position embeddings above index
  128 that would otherwise stay at random init.
- **bf16, not fp16.** The 3070 is Ampere. bf16 avoids the loss-scaling
  instability fp16 causes when training from random init.
- **Single-forward-pass loss**, gradient checkpointing, and gradient
  accumulation to an effective batch of 256 (pretrain) / 96 (contrastive).

Measured on the 3070: **~69k tokens/sec at both seq 128 and seq 512** (5.31 GB
and 4.77 GB peak). Throughput is flat across sequence length, so at this model
size the run is compute-bound rather than attention-bound. seq 512 at batch 32
OOMs; batch 16 leaves comfortable headroom.

## A bug worth knowing about

Gradient accumulation is **silently broken** in transformers 4.47. At the same
effective batch of 64:

| | logged loss | grad_norm |
|---|---|---|
| `accum=1` | 10.478 | 5.43 |
| `accum=4` | 41.997 | 23.75 |

`grad_norm` scales with the accumulation factor, so those are genuinely inflated
gradients, not just a mis-printed number — the optimizer sees **`accum` × the
intended learning rate**. Pretrain phase 2 uses `accum=16`, which would have run
at an effective 8e-3 against an intended 5e-4, and would not have trained.

The cause is the `model_accepts_loss_kwargs` path: `BertForMaskedLM.forward`
accepts `**loss_kwargs`, so `Trainer` passes `num_items_in_batch` and then skips
its own division on the assumption the model normalised the loss itself — which
this model does not. Setting `trainer.model_accepts_loss_kwargs = False` after
construction does **not** help (verified: byte-identical losses).

`accum.py` divides the loss before `backward()`, fixing reporting and gradients
together. After the fix, `accum` 1 / 4 / 16 agree: 10.478 / 10.499 / 10.494.

```bash
python scripts/check_accum.py          # expect PASS
python scripts/check_accum.py --stock  # expect FAIL on 4.47
```

If you upgrade transformers, re-run that check — the mixin should be removed
once it is no longer needed.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export PYTHONPATH=src KERNEL_ROOT=~/linux

bash scripts/run_all.sh          # or run stages individually:

python -m linuxembed.corpus              # kernel tree  -> data/corpus.jsonl
python -m linuxembed.tokenizer_train     # corpus       -> artifacts/tokenizer
python -m linuxembed.pack                # corpus       -> data/{train,val}.bin
python -m linuxembed.pretrain --phase 1  # random init  -> MLM, seq 128
python -m linuxembed.pretrain --phase 2 --resume artifacts/bert-kernel-mlm/phase1
python -m linuxembed.mine_pairs          # kernel-doc   -> data/pairs.jsonl
python -m linuxembed.train_embed --stage 1   # InfoNCE
python -m linuxembed.train_embed --stage 2   # GIST + InfoNCE
python -m linuxembed.evaluate            # held-out retrieval metrics
```

Every stage is resumable and writes to `data/` or `artifacts/`, both gitignored
— they are large and fully reproducible from this code plus a kernel tree.

## Results

2,000 held-out kernel-doc pairs. The corpus contains every held-out positive
**and** its hard negative, so the model has to separate a function from its own
sibling, not from unrelated code.

| metric | BM25 | MLM only | **stage 1 (InfoNCE)** | stage 2 (GIST+InfoNCE) |
|---|---|---|---|---|
| accuracy@1 | 0.7115 | 0.2535 | **0.9225** | 0.9210 |
| accuracy@5 | 0.9035 | 0.4525 | **0.9930** | 0.9930 |
| MRR@10 | 0.7948 | 0.3371 | **0.9558** | 0.9542 |
| NDCG@10 | 0.8297 | 0.3856 | **0.9659** | 0.9644 |
| recall@10 | 0.9365 | 0.5415 | **0.9950** | 0.9940 |

A 43M-parameter encoder trained from scratch on one 8 GB card beats BM25
decisively: accuracy@1 0.71 → 0.92, NDCG@10 0.83 → 0.97.

The MLM column matters as much as the BM25 one. Pretraining alone scores 0.386
NDCG — the encoder learns C, but nothing about retrieval. Contrastive training
supplies almost all of the retrieval ability.

MLM pretraining itself: final loss 0.5314, perplexity 1.70, with train and eval
tracking within 0.003.

```bash
python -m linuxembed.evaluate --compare \
    artifacts/bert-kernel-mlm/phase2 \
    artifacts/embed-stage1-infonce \
    artifacts/embed-stage2-gist
```

### GIST did not earn its place

Stage 2 is a small regression on every metric (−0.0015 NDCG). That is not GIST
hurting — it is GIST doing **nothing**, and two extra epochs adding noise.

GIST masks **8 of 33,840 candidates: 0.024%**, touching 1.2% of rows.

The mechanism only fires when the guide ranks a distractor *above* the true
positive. Our guide is the stage-1 model, already 92% accurate, so it almost
never does. And the hard negatives are curated sibling functions — genuine
negatives, not the false negatives GIST exists to remove. Self-guided GIST also
cannot contribute information the student lacks, since guide and student start
identical.

Larger batches are the obvious suspect and are **not** the answer — measured
across batch sizes, GIST still fires on only 3.6% of rows at batch 1024:

| batch | mask rate | rows hit |
|---|---|---|
| 24 | 0.018% | 0.8% |
| 256 | 0.004% | 1.8% |
| 1024 | 0.003% | 3.6% |

What *does* change it is making in-batch candidates confusable. Drawing each
micro-batch from a single kernel subsystem raises the firing rate 14×:

| regime | mask rate | rows hit |
|---|---|---|
| random batches + sibling negatives | 0.027% | 1.2% |
| random batches, in-batch negatives only | 0.000% | 0.0% |
| **same-subsystem batches + siblings** | **0.372%** | **8.4%** |

`--homogeneous-batches` implements that grouping, and `--w-gist 0` gives an
InfoNCE-only control under identical batching — the only way to attribute a
metric change to GIST rather than to the extra epochs.

Run that way, with GIST firing 14× more often, it still loses to its own control:

| arm (both from stage 1, grouped batches, same LR and epochs) | accuracy@1 | NDCG@10 |
|---|---|---|
| A — InfoNCE only (control) | 0.9220 | 0.9655 |
| B — 0.6·GIST + 0.4·InfoNCE | 0.9200 | 0.9646 |

So the verdict is not "GIST needed better conditions." Given conditions
engineered in its favour, it is still marginally worse than not using it. The
stage-1 InfoNCE checkpoint is the model this repo ships.

That is a negative result about *this dataset*, not about GISTEmbed. Its
mechanism needs a guide that is confused where the student is confident, and a
negative-sampling scheme that actually produces false negatives. Curated sibling
negatives and a self-guide give it neither.

## Open-corpus retrieval — the number that matters

The table above scores 2,000 queries against 4,000 candidates with the answer
guaranteed present. That measures the contrastive objective, **not** retrieval.
RAG searches the whole tree, where the answer competes with ~914k chunks, most
of them undocumented code the model never saw as a positive.

400 held-out kernel-doc anchors, scored against the **full 914,554-chunk index**:

| metric | dense | **hybrid (dense + BM25 RRF)** |
|---|---|---|
| recall@1 | 0.8125 | **0.9050** |
| recall@5 | 0.9375 | **0.9725** |
| recall@10 | 0.9575 | **0.9775** |
| recall@50 | 0.9850 | **0.9950** |
| MRR | 0.8682 | **0.9374** |
| median rank | 1 | 1 |

The correct function is rank 1 out of 914,554 candidates 90% of the time. Hybrid
is the default because of the +9.3 point recall@1 it buys.

```bash
python scripts/eval_open_corpus.py --n 400 --hybrid
```

**Do not judge this from spot checks.** Picking a symbol and asking where it
ranks measures the wrong thing: for "acquire a mutex and sleep if it is
contended" the true `mutex_lock` sits at rank 68, but the chunks above it are
`mutex_lock_interruptible`, `__mutex_lock_slowpath` and friends — all good
answers. Anecdotes said retrieval was weak and hybrid was a wash. Both were
wrong, and only the 400-query measurement showed it.

recall@50 = 0.995 also means a cross-encoder reranker is worth building: the
answer is essentially always inside the pool a reranker would see.

## Searching the kernel

```bash
python -m linuxembed.search build
python -m linuxembed.search query "how does the buddy allocator split pages"
python -m linuxembed.search query "acquire a spinlock without disabling irqs" --hybrid
```

`--hybrid` fuses the dense ranking with BM25 by reciprocal rank fusion. Cosine
scores and BM25 scores are on different scales and cannot be added, so RRF
combines ranks instead and needs no calibration.

Whether hybrid actually beats dense alone here is measured, not assumed — see
the complementarity numbers below.

## Licensing

Stated plainly, because the motivation for building from scratch was legal
caution and the picture is not what it first looks like:

- **`bert-base-uncased` is Apache-2.0; `microsoft/codebert-base` is MIT.** Both
  are permissive. Avoiding them removes very little legal exposure.
- **The Linux kernel is GPL-2.0** — the most restrictive input in this pipeline.
  Whether model weights trained on GPL source are a derivative work is
  genuinely unsettled law, and dropping the pretrained base does not affect it.

So: building from scratch is well justified on *tokenizer quality* and
*provenance clarity*, but it does **not** make the output legally unencumbered.
The training data is GPL-2.0 either way. Treat the resulting weights as
GPL-2.0-encumbered and get real legal advice before shipping them in a product.

The code in this repository is GPL-2.0, matching its data source.
