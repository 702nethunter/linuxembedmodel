#!/usr/bin/env bash
# Full pipeline, from a kernel tree to an evaluated embedding model.
# Every stage is skipped if its output already exists, so this is re-runnable.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"
export KERNEL_ROOT="${KERNEL_ROOT:-$HOME/linux}"
# The 3070 fragments easily at 8 GB; expandable segments avoids OOM from
# fragmentation rather than from genuine capacity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Without this, Trainer's loss lines sit in a block-buffered stdout for hours
# while only tqdm (stderr) flows, so a diverging run looks identical to a
# healthy one until the buffer happens to flush.
export PYTHONUNBUFFERED=1
# Silences the fork warning once dataloader workers start.
export TOKENIZERS_PARALLELISM=false

PY="${PY:-python}"
log() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

log "1/8 corpus"
[ -f data/corpus.jsonl ] || $PY -m linuxembed.corpus

log "2/8 tokenizer"
[ -f artifacts/tokenizer/tokenizer.json ] || $PY -m linuxembed.tokenizer_train

log "3/8 pack"
[ -f data/train.bin ] || $PY -m linuxembed.pack

log "4/8 MLM pretrain — phase 1 (seq 128)"
[ -f artifacts/bert-kernel-mlm/phase1/model.safetensors ] || \
    $PY -m linuxembed.pretrain --phase 1

log "5/8 MLM pretrain — phase 2 (seq 512)"
[ -f artifacts/bert-kernel-mlm/phase2/model.safetensors ] || \
    $PY -m linuxembed.pretrain --phase 2 --resume artifacts/bert-kernel-mlm/phase1

log "6/8 mine kernel-doc pairs"
[ -f data/pairs.jsonl ] || $PY -m linuxembed.mine_pairs

log "7/8 contrastive stage 1 — InfoNCE"
[ -f artifacts/embed-stage1-infonce/modules.json ] || \
    $PY -m linuxembed.train_embed --stage 1

log "8/8 contrastive stage 2 — GIST + InfoNCE"
[ -f artifacts/embed-stage2-gist/modules.json ] || \
    $PY -m linuxembed.train_embed --stage 2

log "evaluation"
$PY -m linuxembed.evaluate --compare \
    artifacts/bert-kernel-mlm/phase2 \
    artifacts/embed-stage1-infonce \
    artifacts/embed-stage2-gist
