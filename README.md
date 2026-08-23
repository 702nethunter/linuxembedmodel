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

## Evaluation

`evaluate.py` reports Accuracy@1/@5, MRR@10 and NDCG@10 on held-out kernel-doc
pairs, against a corpus that includes every held-out positive **and** its hard
negative as a distractor. `--compare` scores several checkpoints side by side so
the contribution of each stage is visible:

```bash
python -m linuxembed.evaluate --compare \
    artifacts/bert-kernel-mlm/phase2 \
    artifacts/embed-stage1-infonce \
    artifacts/embed-stage2-gist
```

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
