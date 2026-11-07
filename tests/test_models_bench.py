"""Presets, prompt variants, per-model routing and price table (no network)."""
import importlib
import os

import pytest

from cqa import llm, presets, pricing


def test_cloud_route_by_model_id():
    assert llm.cloud_route("claude-opus-5-5") == "anthropic"
    assert llm.cloud_route("gpt-5.3-codex") == "openai_cloud"
    assert llm.cloud_route("gpt-6-luna") == "openai_cloud"
    assert llm.cloud_route("gemini-4-argon") == "gemini"


def test_pricing_table_and_usage_by_model():
    assert pricing.price("claude-haiku-5-5")[:2] == (0.10, 0.50)
    assert pricing.usd("gpt-5.3-codex", input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(15.75)
    before = dict(llm.USAGE_BY_MODEL.get("gpt-6-luna", {"usd": 0.0, "calls": 0}))
    llm._add_usage({"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 1000}, "gpt-6-luna")
    after = llm.USAGE_BY_MODEL["gpt-6-luna"]
    assert after["calls"] == before["calls"] + 1
    assert after["usd"] - before["usd"] == pytest.approx((1000 * 0.10 + 100 * 0.50 + 1000 * 0.01) / 1e6)


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
def test_prompt_variants_change_only_prompt_text(monkeypatch, variant):
    from cqa import checklist as ckl, prompts
    cl = ckl.load("freedom_chat_v1")
    cr = cl["criteria"][0]
    monkeypatch.setenv("CQA_PROMPT_VARIANT", "v0")
    base = importlib.reload(prompts)
    b_sys, b_blk, b_tool = base.SYSTEM, base.criterion_block(cl, cr), base.TOOL
    monkeypatch.setenv("CQA_PROMPT_VARIANT", variant)
    v = importlib.reload(prompts)
    try:
        assert v.PROMPT_VERSION.endswith("+" + variant)
        assert (v.SYSTEM, v.criterion_block(cl, cr)) != (b_sys, b_blk)
        assert v.TOOL == b_tool                      # same tool schema / output contract
        assert cr["rules"].strip()[:80] in v.criterion_block(cl, cr)   # rubric + rules kept verbatim
    finally:
        monkeypatch.setenv("CQA_PROMPT_VARIANT", "v0")
        importlib.reload(prompts)


def test_fewshot_examples_are_not_copied_from_holdout():
    import json
    from pathlib import Path

    from cqa.prompt_variants.variants import FEWSHOT
    root = Path(__file__).resolve().parent.parent
    texts = [t["text"] for f in (root / "examples" / "gold_holdout").glob("*.json")
             for t in json.loads(f.read_text(encoding="utf-8"))["dialog"].get("transcript", []) if len(t.get("text", "")) > 25]
    shots = " ".join(s for ex in FEWSHOT.values() for s, _ in ex)
    assert not [t for t in texts if t in shots]


def test_preset_apply(monkeypatch):
    ns = {}
    monkeypatch.setenv("CQA_PROMPT_VARIANT", "v0")
    p = presets.apply("max", ns)
    assert ns["ENSEMBLE"] is False and ns["MODEL"] == p["model"] and ns["PRESET"] == "max"
    presets.apply("jury", ns)
    assert ns["ENSEMBLE"] is True and ns["JUDGE"] == "claude-opus-5-5"
    with pytest.raises(ValueError):
        presets.apply("nope", ns)
    os.environ["CQA_PROMPT_VARIANT"] = "v0"
    from cqa import prompts
    importlib.reload(prompts)
