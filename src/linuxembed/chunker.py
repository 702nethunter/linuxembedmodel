#!/usr/bin/env python3
"""Chunk the whole kernel into retrievable units for RAG.

This is deliberately not the same extraction as mine_pairs.py. That one takes
only kernel-doc'd definitions (~49k) and strips the doc comment out of the
positive, because leaving it in would let the model match the anchor text
against itself.

For RAG both choices invert:

* Coverage must be total. A question about an undocumented static helper is
  still a fair question, and only ~8% of kernel definitions carry kernel-doc.
* The leading comment is KEPT and prepended to the chunk. Nothing is being
  trained here, so there is no leakage to avoid, and the comment is exactly the
  natural-language text a query is most likely to match -- plus it is useful
  context for the model that has to answer from the chunk.

Anything a definition scan cannot reach (headers that are mostly macros,
Kconfig-heavy files, assembly stubs) falls back to overlapping line windows so
no part of the tree is unreachable.

    python -m linuxembed.chunker --out data/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config
from .mine_pairs import match_braces

# A top-level definition starts in column 0 -- inside a function everything is
# indented -- contains a parameter list, and opens a brace.
#
# The return-type prefix is OPTIONAL. Kernel style routinely breaks a long
# signature across lines:
#
#     struct task_struct *
#     pick_next_task_fair(struct rq *rq, ...)
#
# which leaves the name alone in column 0. Requiring a same-line return type
# silently dropped every such function -- pick_next_task_fair among them.
FUNC_RE = re.compile(
    r"^(?![\s#/])"                                       # column 0, not a directive
    r"(?:[A-Za-z_][A-Za-z0-9_ \t\*\(\),\[\]]*?\b)?"      # optional return type
    r"([A-Za-z_][A-Za-z0-9_]*)\s*"                       # symbol name
    r"\(",                                                # start of parameter list
    re.M,
)
# A line holding only the return type of the definition below it.
RETURN_TYPE_LINE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \t\*]*\**\s*$")
RECORD_RE = re.compile(
    r"^(?:typedef\s+)?(struct|union|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
    re.M,
)
MAX_COMMENT_LINES = 40

MAX_CHUNK_LINES = 160
WINDOW_LINES = 60
WINDOW_STRIDE = 45


def leading_comment(text: str, start: int) -> str:
    """Return the comment block directly above `start`, if any.

    Scanned backwards line by line rather than by regex. The obvious pattern,
    `(?:^[ \\t]*(?:/\\*.*?\\*/|//[^\\n]*)\\s*)+\\Z` with re.S, is wrong: when the
    trailing `\\s*\\Z` fails, `.*?` backtracks to a *later* `*/` and the match
    swallows every line of code in between. In practice that attached
    __mutex_handoff's comment -- plus its entire body -- to mutex_lock.

    A line scan cannot span code, because the first non-comment line stops it.
    """
    lines = text[:start].split("\n")
    if lines and not lines[-1].strip():
        lines.pop()  # empty element left by the newline before the definition
    # Kernel style puts the comment flush against the definition; tolerate one
    # blank line, but no more, so a distant comment cannot be dragged in.
    if lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    collected: list[str] = []
    if lines and lines[-1].rstrip().endswith("*/"):
        # Walk back to the opening /* of this block.
        while lines and len(collected) < MAX_COMMENT_LINES:
            line = lines.pop()
            collected.append(line)
            if line.lstrip().startswith("/*"):
                break
        else:
            return ""  # never found an opening delimiter
    elif lines and lines[-1].lstrip().startswith("//"):
        while lines and lines[-1].lstrip().startswith("//") \
                and len(collected) < MAX_COMMENT_LINES:
            collected.append(lines.pop())
    else:
        return ""

    return "\n".join(reversed(collected)).strip()


def match_parens(text: str, open_idx: int) -> int | None:
    """Index just past the ')' matching text[open_idx] == '('.

    A plain find(')') is wrong for function-pointer parameters such as
    `int foo(void (*cb)(int), int x)`, where the first ')' closes the callback's
    own list and everything after it gets mis-sliced.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        elif c in ";{}" and depth == 0:
            return None
    return None


def widen_to_return_type(text: str, start: int) -> int:
    """Move `start` back over a return type that sits on its own line."""
    line_start = text.rfind("\n", 0, start) + 1
    if line_start == 0:
        return start
    prev_start = text.rfind("\n", 0, line_start - 1) + 1
    prev = text[prev_start : line_start - 1]
    if RETURN_TYPE_LINE_RE.match(prev) and not prev.strip().endswith((";", "{", "}", ":", ",")):
        return prev_start
    return start


def truncate(code: str) -> str:
    lines = code.split("\n")
    if len(lines) > MAX_CHUNK_LINES:
        lines = lines[:MAX_CHUNK_LINES] + ["\t/* ... truncated ... */", "}"]
    return "\n".join(lines)


def extract_definitions(path: str, text: str) -> list[dict]:
    """Every top-level function and record definition in one file."""
    out: list[dict] = []
    covered: list[tuple[int, int]] = []

    for m in FUNC_RE.finditer(text):
        name = m.group(1)
        if name in ("if", "for", "while", "switch", "return", "sizeof", "defined",
                    "do", "else", "case", "goto"):
            continue
        close = match_parens(text, m.end() - 1)
        if close is None:
            continue
        # Between ')' and '{' only attributes may appear (__must_hold, __acquires,
        # __init and friends); a ';' means it is a prototype, not a definition.
        gap = text[close : close + 300]
        brace_off = gap.find("{")
        if brace_off == -1 or ";" in gap[:brace_off]:
            continue
        brace = close + brace_off
        end = match_braces(text, brace)
        if end is None:
            continue
        start = widen_to_return_type(text, m.start())
        comment = leading_comment(text, start)
        body = text[start:end]
        out.append({
            "path": path, "name": name, "kind": "function",
            "code": truncate((comment + "\n" + body) if comment else body),
            "line": text.count("\n", 0, m.start()) + 1,
        })
        covered.append((m.start(), end))

    for m in RECORD_RE.finditer(text):
        brace = text.find("{", m.start())
        end = match_braces(text, brace) if brace != -1 else None
        if end is None:
            continue
        if any(s <= m.start() < e for s, e in covered):
            continue  # a record defined inside a function we already captured
        comment = leading_comment(text, m.start())
        body = text[m.start() : end]
        out.append({
            "path": path, "name": m.group(2), "kind": m.group(1),
            "code": truncate((comment + "\n" + body) if comment else body),
            "line": text.count("\n", 0, m.start()) + 1,
        })
        covered.append((m.start(), end))

    return out


def window_chunks(path: str, text: str) -> list[dict]:
    """Overlapping line windows, for files no definition scan reaches."""
    lines = text.split("\n")
    out = []
    for start in range(0, max(len(lines) - 1, 1), WINDOW_STRIDE):
        block = lines[start : start + WINDOW_LINES]
        if not any(l.strip() for l in block):
            continue
        out.append({
            "path": path, "name": f"{Path(path).name}:{start + 1}", "kind": "window",
            "code": "\n".join(block), "line": start + 1,
        })
        if start + WINDOW_LINES >= len(lines):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk the kernel for RAG")
    ap.add_argument("--corpus", type=Path, default=config.CORPUS_JSONL)
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "chunks.jsonl")
    ap.add_argument("--min-chars", type=int, default=60)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_files = n_chunks = n_fallback = 0
    kinds: dict[str, int] = {}

    with open(args.corpus, encoding="utf-8") as src, \
         open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            doc = json.loads(line)
            n_files += 1
            chunks = extract_definitions(doc["path"], doc["text"])
            if not chunks:
                chunks = window_chunks(doc["path"], doc["text"])
                n_fallback += 1
            for c in chunks:
                if len(c["code"]) < args.min_chars:
                    continue
                kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
                dst.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += 1
            if n_files % 5000 == 0:
                print(f"  {n_files:,} files … {n_chunks:,} chunks", flush=True)

    print(f"\n  files    {n_files:,}  ({n_fallback:,} fell back to line windows)")
    print(f"  chunks   {n_chunks:,}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {k:<10s} {v:,}")
    print(f"  wrote -> {args.out}")


if __name__ == "__main__":
    main()
