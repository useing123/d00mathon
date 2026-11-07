#!/usr/bin/env python3
"""Doc-consistency guard: fails on stale fine-tune epoch claims or hardcoded test counts.

Usage: python scripts/check_docs.py   (exit 1 on any finding)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_GLOBS = ["README.md", "EXPERIMENTS.md", "*.md", "docs/**/*.md", "docs/*.html", "models/**/README.md"]

# The fine-tune ran 16 steps ≈ 0.32 epoch; "1 эпоха" / "one epoch" claims are wrong.
EPOCH_RE = re.compile(r"(?<![\d.])(1|одн[аоуи]\w*)\s+эпох|(?<![\d.])(1|one)\s+epoch\b(?!s)", re.I)
# Test counts go stale on every commit; say "full test suite" instead.
TESTS_RE = re.compile(r"\b\d+\+?\s*(авто)?тест|\b\d+\+?\s+(unit\s+)?tests\b", re.I)


def doc_files() -> list[Path]:
    seen: set[Path] = set()
    for g in DOC_GLOBS:
        for p in ROOT.glob(g):
            if p.is_file() and ".venv" not in p.parts and "node_modules" not in p.parts:
                seen.add(p)
    return sorted(seen)


def check_text(text: str) -> list[str]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if EPOCH_RE.search(line):
            out.append(f"{i}: fine-tune epoch claim (use '16 шагов (≈ 0.32 эпохи)'): {line.strip()[:120]}")
        if TESTS_RE.search(line):
            out.append(f"{i}: hardcoded test count: {line.strip()[:120]}")
    return out


def main() -> int:
    bad = 0
    for p in doc_files():
        for msg in check_text(p.read_text(encoding="utf-8", errors="replace")):
            print(f"{p.relative_to(ROOT)}:{msg}")
            bad += 1
    if bad:
        print(f"{bad} doc issue(s)")
        return 1
    print("docs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
