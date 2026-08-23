#!/usr/bin/env python3
"""Stage 1 — extract a clean C/H text corpus from the Linux kernel tree.

The kernel is not a ready-made pretraining corpus. ~35% of its .c/.h bytes are
auto-generated hardware register headers whose lines look like

    #define DCN_REG_FIELD_MASK  0x00000001L

Those files are ~89% bare #define lines. Leaving them in means a third of every
MLM batch teaches the model hex-constant formatting instead of C. We drop them
with a structural heuristic (define density) so generated headers are caught
wherever they live, not just under the known amd/asic_reg path.

Usage:
    python -m linuxembed.corpus --kernel ~/linux --out data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def iter_source_files(root: Path):
    """Yield every .c/.h file in the kernel tree, skipping VCS and build dirs."""
    skip_dirs = {".git", ".github", "Documentation/output"}
    for path in root.rglob("*"):
        if path.suffix not in (".c", ".h"):
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        yield path


def define_ratio(text: str) -> tuple[float, int]:
    """Return (fraction of non-blank lines that are #define, non-blank line count)."""
    non_blank = 0
    defines = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        non_blank += 1
        if stripped.startswith("#define"):
            defines += 1
    if non_blank == 0:
        return 0.0, 0
    return defines / non_blank, non_blank


def is_generated_noise(text: str) -> bool:
    """True for auto-generated register-definition headers."""
    ratio, lines = define_ratio(text)
    return lines >= config.GENERATED_MIN_LINES and ratio >= config.GENERATED_DEFINE_RATIO


def read_source(path: Path) -> str | None:
    """Read a kernel source file. Kernel sources are mostly UTF-8 but not all."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > config.MAX_FILE_BYTES:
        return None
    if b"\x00" in raw[:8192]:  # binary masquerading as .h
        return None
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract kernel C corpus")
    ap.add_argument("--kernel", type=Path, default=config.KERNEL_ROOT)
    ap.add_argument("--out", type=Path, default=config.CORPUS_JSONL)
    args = ap.parse_args()

    if not args.kernel.exists():
        sys.exit(f"kernel tree not found: {args.kernel}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped_generated = skipped_read = 0
    kept_bytes = dropped_bytes = 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for i, path in enumerate(iter_source_files(args.kernel)):
            text = read_source(path)
            if text is None:
                skipped_read += 1
                continue
            if is_generated_noise(text):
                skipped_generated += 1
                dropped_bytes += len(text)
                continue
            rel = path.relative_to(args.kernel).as_posix()
            fh.write(json.dumps({"path": rel, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            kept_bytes += len(text)

            if (i + 1) % 5000 == 0:
                print(f"  scanned {i + 1:,} files … kept {kept:,}", flush=True)

    print(f"\n  kept              {kept:,} files  ({kept_bytes / 1048576:.1f} MB)")
    print(f"  dropped generated {skipped_generated:,} files  ({dropped_bytes / 1048576:.1f} MB)")
    print(f"  dropped unreadable{skipped_read:,} files")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
