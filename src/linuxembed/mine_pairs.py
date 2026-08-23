#!/usr/bin/env python3
"""Stage 5 — mine (anchor, positive, negative) triples from kernel-doc comments.

The kernel carries ~64k hand-written kernel-doc blocks:

    /**
     * kmalloc_node - allocate memory from a specific node
     * @size: how many bytes of memory are required
     * @flags: describe the allocation context
     *
     * Return: pointer to the allocated memory, or NULL.
     */
    void *kmalloc_node(size_t size, gfp_t flags) { ... }

That is a free, human-authored, natural-language -> C supervision set. Two
details decide whether the resulting model learns semantics or a shortcut:

* the leading "symbol_name - " is STRIPPED from the anchor. Left in, the model
  learns to string-match the identifier that also appears in the positive's
  signature, and scores well on the eval while being useless on real queries.
* the kernel-doc block itself is EXCLUDED from the positive. Left in, the
  positive literally contains the anchor text and the task is trivial.

Hard negatives are sibling functions from the same file: same subsystem, same
idioms, same types, wrong function. Those are far harder than random negatives
and are what make the contrastive signal informative.

Usage:
    python -m linuxembed.mine_pairs --corpus data/corpus.jsonl --out data/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from . import config

# A kernel-doc block: /** on its own line, through the closing */
KDOC_RE = re.compile(r"^[ \t]*/\*\*[ \t]*\n(.*?)^[ \t]*\*/[ \t]*\n", re.S | re.M)
# First kernel-doc line: " * name - short description"
HEADER_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-–]\s*(.+)$")
STRUCT_HEADER_RE = re.compile(r"^\s*(?:struct|union|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[-–]\s*(.+)$")
PARAM_RE = re.compile(r"^\s*@([A-Za-z0-9_.]+):\s*(.*)$")


def strip_comment_prefix(block: str) -> list[str]:
    """Turn raw kernel-doc body lines into plain text lines."""
    out = []
    for line in block.split("\n"):
        line = re.sub(r"^[ \t]*\*[ \t]?", "", line)
        out.append(line.rstrip())
    return out


def parse_kdoc(block: str) -> dict | None:
    """Parse a kernel-doc body into {name, kind, anchor}."""
    lines = strip_comment_prefix(block)
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None

    header = lines[0]
    m = STRUCT_HEADER_RE.match(header)
    kind = "struct"
    if not m:
        m = HEADER_RE.match(header)
        kind = "function"
    if not m:
        return None
    name, short_desc = m.group(1), m.group(2).strip()

    # Everything after the header: @param descriptions plus free prose.
    prose: list[str] = []
    params: list[str] = []
    for line in lines[1:]:
        pm = PARAM_RE.match(line)
        if pm:
            field, desc = pm.group(1), pm.group(2).strip()
            if desc:
                params.append(f"{field}: {desc}")
        elif line.strip():
            prose.append(line.strip())

    # The anchor deliberately omits `name` so the model cannot lexically cheat.
    parts = [short_desc]
    if prose:
        parts.append(" ".join(prose))
    if params:
        parts.append("Parameters: " + "; ".join(params))
    anchor = " ".join(parts)
    anchor = re.sub(r"\s+", " ", anchor).strip()

    return {"name": name, "kind": kind, "anchor": anchor}


def match_braces(text: str, open_idx: int) -> int | None:
    """Return index just past the '}' matching text[open_idx] == '{'.

    Skips over string literals, char literals, and comments so that a '{' inside
    a string (common in kernel format strings) does not throw off the depth.
    """
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if c in "\"'":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def extract_definition(text: str, search_from: int, name: str) -> str | None:
    """Grab the definition that follows a kernel-doc block.

    Looks only in the window immediately after the comment: kernel-doc sits
    directly above what it documents, so anything further away is a different
    symbol and would be a mislabelled positive.
    """
    window = text[search_from : search_from + 4000]
    m = re.search(r"\b" + re.escape(name) + r"\b", window)
    if m is None:
        return None
    # The signature starts at the beginning of the line the name appears on.
    line_start = window.rfind("\n", 0, m.start()) + 1
    brace = window.find("{", m.end())
    if brace == -1:
        # Bodyless: a struct/enum declaration or prototype ending in ';'
        semi = window.find(";", m.end())
        if semi == -1 or semi - m.end() > 400:
            return None
        return window[line_start : semi + 1].strip()

    # Reject a '{' that is really the start of an unrelated later block.
    if window.count("\n", m.end(), brace) > 6:
        return None
    end = match_braces(window, brace)
    if end is None:
        return None
    return window[line_start:end].strip()


def truncate_code(code: str) -> str:
    lines = code.split("\n")
    if len(lines) > config.MAX_POSITIVE_LINES:
        lines = lines[: config.MAX_POSITIVE_LINES] + ["\t/* ... truncated ... */", "}"]
    out = "\n".join(lines)
    return out[: config.MAX_POSITIVE_CHARS]


def mine_file(path: str, text: str) -> list[dict]:
    """Extract every well-formed (anchor, positive) pair from one source file."""
    found = []
    for m in KDOC_RE.finditer(text):
        parsed = parse_kdoc(m.group(1))
        if parsed is None:
            continue
        if len(parsed["anchor"]) < config.MIN_ANCHOR_CHARS:
            continue
        code = extract_definition(text, m.end(), parsed["name"])
        if code is None or len(code) < 40:
            continue
        found.append({
            "path": path,
            "name": parsed["name"],
            "kind": parsed["kind"],
            "anchor": parsed["anchor"],
            "positive": truncate_code(code),
        })
    return found


def subsystem(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def attach_negatives(records: list[dict], rng: random.Random) -> list[dict]:
    """Pair each record with a hard negative: a sibling definition."""
    by_file: dict[str, list[int]] = defaultdict(list)
    by_subsys: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_file[r["path"]].append(i)
        by_subsys[subsystem(r["path"])].append(i)

    out = []
    for i, r in enumerate(records):
        # Prefer a sibling in the same file (hardest), then same subsystem.
        for pool in (by_file[r["path"]], by_subsys[subsystem(r["path"])]):
            cands = [j for j in pool if j != i and records[j]["name"] != r["name"]]
            if cands:
                r = {**r, "negative": records[rng.choice(cands)]["positive"]}
                out.append(r)
                break
        # No sibling anywhere: drop it rather than fall back to a random
        # negative, which would be trivially easy and add no signal.
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine kernel-doc training pairs")
    ap.add_argument("--corpus", type=Path, default=config.CORPUS_JSONL)
    ap.add_argument("--out", type=Path, default=config.PAIRS_JSONL)
    args = ap.parse_args()

    rng = random.Random(config.SEED)
    records: list[dict] = []
    n_files = 0
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            n_files += 1
            records.extend(mine_file(doc["path"], doc["text"]))
            if n_files % 5000 == 0:
                print(f"  {n_files:,} files … {len(records):,} pairs", flush=True)

    print(f"\n  parsed {len(records):,} kernel-doc pairs from {n_files:,} files")

    # Drop near-duplicate anchors (boilerplate docs repeated across drivers),
    # which would otherwise show up as false negatives inside a batch.
    seen: set[str] = set()
    deduped = []
    for r in records:
        key = r["anchor"][:200].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  after anchor dedup: {len(deduped):,}")

    final = attach_negatives(deduped, rng)
    print(f"  with hard negatives: {len(final):,}")

    rng.shuffle(final)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds: dict[str, int] = defaultdict(int)
    for r in final:
        kinds[r["kind"]] += 1
    print(f"  kinds: {dict(kinds)}")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
