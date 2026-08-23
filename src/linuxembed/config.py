"""Central configuration for the linuxembedmodel pipeline.

Every stage reads its knobs from here so the pipeline stays reproducible.
Sizes are tuned for a single 8 GB RTX 3070.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
KERNEL_ROOT = Path(os.environ.get("KERNEL_ROOT", Path.home() / "linux"))
DATA_DIR = Path(os.environ.get("LEM_DATA", "data"))
OUT_DIR = Path(os.environ.get("LEM_OUT", "artifacts"))

CORPUS_JSONL = DATA_DIR / "corpus.jsonl"
TOKENIZER_DIR = OUT_DIR / "tokenizer"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"
PAIRS_JSONL = DATA_DIR / "pairs.jsonl"

PRETRAIN_DIR = OUT_DIR / "bert-kernel-mlm"
STAGE1_DIR = OUT_DIR / "embed-stage1-infonce"
STAGE2_DIR = OUT_DIR / "embed-stage2-gist"

# ── Corpus extraction ──────────────────────────────────────────────────────────
# The kernel ships ~470 MB of auto-generated register headers (mostly
# drivers/gpu/drm/amd/include/asic_reg) that are ~89% bare "#define FOO_MASK 0x1L"
# lines. Left in, they are ~35% of the corpus and dominate MLM pretraining with
# a pattern that teaches the model nothing about C. We detect them structurally
# rather than by path, so equivalent generated headers elsewhere are caught too.
GENERATED_DEFINE_RATIO = 0.60  # >60% of non-blank lines are #define
GENERATED_MIN_LINES = 500  # ...and the file is long enough for that to be meaningful
MAX_FILE_BYTES = 4 * 1024 * 1024  # skip anything pathologically large

# Fraction of FILES (not tokens) held out for MLM eval. Whole files are held
# out, chosen by hashing the path, so validation spans every subsystem and no
# validation window sits adjacent to a training window of the same file.
VAL_FRACTION = 0.01

# ── Tokenizer ──────────────────────────────────────────────────────────────────
# Byte-level BPE: never emits [UNK] on arbitrary bytes, and keeps C identifiers
# like `kmalloc_node` far more intact than BERT's uncased WordPiece would.
VOCAB_SIZE = 32768  # fits in uint16, so the packed corpus is 2 bytes/token
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOKENIZER_MIN_FREQUENCY = 2

# ── Model (from scratch — no pretrained weights anywhere in this repo) ─────────
# ~43M params. Sized against a ~230M-token corpus: BERT-base (110M) would badly
# overfit this much data on this much compute; this config trains in ~a night.
MODEL_HIDDEN = 512
MODEL_LAYERS = 8
MODEL_HEADS = 8
MODEL_INTERMEDIATE = 2048
MODEL_MAX_POSITION = 512

# ── MLM pretraining ────────────────────────────────────────────────────────────
MLM_PROBABILITY = 0.15
# Two-phase schedule (the original BERT recipe): most steps at short sequence
# length because attention is quadratic, then a short phase at full length so
# the position embeddings past 128 actually get trained.
PHASE1_SEQ_LEN = 128
PHASE1_BATCH = 64
PHASE1_ACCUM = 4  # effective 256
PHASE1_STEPS = 40_000

PHASE2_SEQ_LEN = 512
PHASE2_BATCH = 16
PHASE2_ACCUM = 16  # effective 256
PHASE2_STEPS = 6_000

PRETRAIN_LR = 5e-4  # from-scratch, so much higher than a fine-tune LR
PRETRAIN_WARMUP = 0.02
PRETRAIN_WEIGHT_DECAY = 0.01

# ── Pair mining ────────────────────────────────────────────────────────────────
MIN_ANCHOR_CHARS = 40  # drop stub kernel-doc with no real prose
MAX_POSITIVE_LINES = 120  # cap giant function bodies
MAX_POSITIVE_CHARS = 6000
PAIR_VAL_SIZE = 2000  # held-out pairs for IR evaluation

# ── Contrastive training ───────────────────────────────────────────────────────
EMBED_MAX_SEQ = 320
EMBED_BATCH = 24
EMBED_ACCUM = 4  # effective 96
EMBED_EPOCHS_STAGE1 = 2
EMBED_EPOCHS_STAGE2 = 2
EMBED_LR_STAGE1 = 3e-5
EMBED_LR_STAGE2 = 1e-5
EMBED_SCALE = 20.0  # 1/temperature for the cosine-similarity softmax

# Loss mix for stage 2: total = W_GIST * GIST + W_INFONCE * InfoNCE
W_GIST = 0.6
W_INFONCE = 0.4

SEED = 42
