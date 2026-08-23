#!/usr/bin/env python3
"""CPU smoke test for the contrastive stage.

Exercises the real code path -- MLM checkpoint -> SentenceTransformer -> both
training stages -> evaluator -- with a tiny model and a handful of pairs, so
wiring bugs surface in seconds instead of after a night of pretraining.

    CUDA_VISIBLE_DEVICES= python scripts/smoke_embed.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM

from linuxembed import config


def make_tiny_encoder(dest: Path, tokenizer_dir: Path) -> None:
    """A 2-layer BERT saved exactly the way pretrain.py saves the real one."""
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    cfg = BertConfig(
        vocab_size=len(tok), hidden_size=64, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        type_vocab_size=1,
    )
    BertForMaskedLM(cfg).save_pretrained(dest)
    tok.save_pretrained(dest)


def main() -> None:
    if not config.PAIRS_JSONL.exists():
        sys.exit(f"need {config.PAIRS_JSONL}; run mine_pairs first")

    work = Path(tempfile.mkdtemp(prefix="smoke_embed_"))
    print(f"  workdir {work}")
    try:
        encoder = work / "encoder"
        make_tiny_encoder(encoder, config.TOKENIZER_DIR)

        # A small slice of the real mined pairs, in the real format.
        pairs = work / "pairs.jsonl"
        with open(config.PAIRS_JSONL) as src, open(pairs, "w") as dst:
            for i, line in enumerate(src):
                if i >= 240:
                    break
                dst.write(line)

        # Point the pipeline at the scratch dirs without touching real artifacts.
        config.STAGE1_DIR = work / "stage1"
        config.STAGE2_DIR = work / "stage2"
        config.PAIR_VAL_SIZE = 40
        config.EMBED_BATCH = 8
        config.EMBED_ACCUM = 2
        config.EMBED_EPOCHS_STAGE1 = 1
        config.EMBED_EPOCHS_STAGE2 = 1
        config.EMBED_MAX_SEQ = 64

        from linuxembed import train_embed

        for stage in (1, 2):
            print(f"\n  ---- stage {stage} ----")
            sys.argv = ["train_embed", "--stage", str(stage),
                        "--pairs", str(pairs), "--encoder", str(encoder)]
            train_embed.main()

        assert (config.STAGE1_DIR / "modules.json").exists(), "stage 1 did not save"
        assert (config.STAGE2_DIR / "modules.json").exists(), "stage 2 did not save"

        # The saved model must actually embed text.
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(config.STAGE2_DIR))
        v = m.encode(["allocate memory from a specific node"])
        assert v.shape[1] == 64, v.shape
        norm = float((v[0] ** 2).sum() ** 0.5)
        assert abs(norm - 1.0) < 1e-3, f"embeddings should be L2-normalised, got {norm}"
        print(f"\n  embedding dim {v.shape[1]}, L2 norm {norm:.4f}")
        print("  SMOKE TEST PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
