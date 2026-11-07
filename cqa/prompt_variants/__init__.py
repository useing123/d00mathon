"""Pluggable prompt variants (selected by CQA_PROMPT_VARIANT or a preset's "prompt" key).

The base prompts live in cqa/prompts.py and are NOT edited by variants; a variant only overrides
``SYSTEM`` and/or wraps ``criterion_block`` at import time via :func:`apply`. Checklists, the tool schema,
evidence validation and scoring are identical for every variant.

* ``v0`` — base prompts (no change).
* ``v1`` — few-shot: 1–2 short SYNTHETIC examples per criterion appended to the criterion block. Written from the
  checklist rules only; never copied from examples/gold_holdout or DEV dialogs.
* ``v2`` — checklist-first: an explicit step-by-step decision procedure replaces the closing instruction.
* ``v3`` — compact: a much shorter system prompt (same rubric and rules in the criterion block).
"""
from .variants import VARIANTS


def apply(name, ns):
    """Patch a prompts module namespace ``ns`` (its globals()) in place."""
    if name in ("", "v0", None):
        return
    if name not in VARIANTS:
        raise ValueError(f"unknown prompt variant {name!r}; known: v0, {', '.join(sorted(VARIANTS))}")
    VARIANTS[name](ns)
    ns["PROMPT_VERSION"] = f"{ns['PROMPT_VERSION']}+{name}"
