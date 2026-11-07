"""Cascade router: pure routing rules, logprob → option probabilities, tier flow with fake models."""
import math

from cqa import config, router
from cqa.gold import load_gold


def S(**kw):
    base = dict(t1=0.9, hard="04,05,06", hard_conf=0.99, t2=0.9, haiku_low=0.6, judge="claude-sonnet-5-5", first="local")
    base.update(kw)
    return router.settings(**base)


def test_tier1_reasons():
    s = S()
    assert router.tier1_reasons({"id": "01"}, {"value": "YES", "confidence": 0.95}, s) == []
    assert router.tier1_reasons({"id": "01"}, {"error": "x"}, s) == ["local_error"]
    assert "low_conf" in router.tier1_reasons({"id": "01"}, {"value": "YES", "confidence": 0.5}, s)
    assert router.tier1_reasons({"id": "06"}, {"value": 10, "confidence": 0.95}, s) == ["hard_criterion"]
    assert router.tier1_reasons({"id": "06"}, {"value": 10, "confidence": 0.995}, s) == []
    assert "evidence_failed" in router.tier1_reasons({"id": "01"}, {"value": "YES", "confidence": 1,
                                                                     "rejected_quotes": [{"turn": 1}]}, s)
    assert "gross_yes" in router.tier1_reasons({"id": "08", "gross": True}, {"value": "YES", "confidence": 1}, s)
    assert "decision_disagree" in router.tier1_reasons({"id": "01"}, {"value": "NO", "confidence": 1,
                                                                       "decision_disagree": True}, s)


def test_tier2_reasons():
    s = S()
    assert router.tier2_reasons({"value": "YES"}, {"value": "YES", "confidence": 0.95}, s) == []
    assert router.tier2_reasons({"value": "YES"}, {"value": "NO", "confidence": 0.8}, s) == ["disagree_low_conf"]
    assert router.tier2_reasons({"value": "YES"}, {"value": "NO", "confidence": 0.95}, s) == []
    assert router.tier2_reasons({"value": "YES"}, {"value": "YES", "confidence": 0.4}, s) == ["haiku_low_conf"]
    assert router.tier2_reasons({"value": "YES"}, {"error": "boom", "value": None}, s) == ["haiku_error"]


def _tok(t, lp, top):
    return {"token": t, "logprob": lp, "top_logprobs": [{"token": k, "logprob": v} for k, v in top.items()]}


def test_option_probs_multitoken_scale():
    ln = math.log
    toks = [_tok('{"value": "', 0, {}), _tok("1", ln(0.8), {"1": ln(0.8), "2": ln(0.15), "N": ln(0.05)}),
            _tok("5", ln(0.75), {"5": ln(0.75), "0": ln(0.25)}), _tok('"}', 0, {})]
    P = router.option_probs(toks, ["0", "5", "10", "15", "20", "NA"])
    assert max(P, key=P.get) == "15"
    assert abs(P["15"] - 0.6) < 1e-3 and abs(P["10"] - 0.2) < 1e-3 and abs(P["20"] - 0.15) < 1e-3


def test_value_prob():
    content = '{"reasoning": "x", "value": "10", "j": 1}'
    toks = [{"token": '{"reasoning": "x", "value": "', "logprob": 0.0}, {"token": "1", "logprob": math.log(0.5)},
            {"token": "0", "logprob": 0.0}, {"token": '", "j": 1}', "logprob": 0.0}]
    assert abs(router.value_prob(content, {"content": toks}) - 0.5) < 1e-3


class Fake:
    def __init__(self, model, value, conf):
        self.provider, self.model, self.value, self.conf, self.calls = "fake", model, value, conf, 0

    def call(self, system, cached_user, user, tool, temperature=0.0):
        from cqa.llm import _TURN_RE
        self.calls += 1
        ops = [(int(i), tx) for i, sp, tx in _TURN_RE.findall(cached_user) if sp == "OPERATOR"]
        ev = [{"turn": ops[0][0], "quote": ops[0][1][:40]}] if ops else []
        return {"value": self.value, "confidence": self.conf, "evidence": ev, "justification": self.model}


def _run(monkeypatch, tmp_path, local, haiku, sonnet, cid="06"):
    monkeypatch.setattr(config, "PROVIDER", "mock")
    monkeypatch.setattr(router, "CACHE", tmp_path)
    ev = router.CascadeEvaluator(use_cache=False, **dict(t1=0.9, hard="04,05,06", hard_conf=0.99, t2=0.9,
                                                         haiku_low=0.6, judge="claude-sonnet-5-5", logprobs=False))
    ev.local, ev.esc, ev.judge_llm, ev.tier1_name = local, haiku, sonnet, local.model
    g = load_gold()[0]
    from cqa import checklist as ckl, data
    rec = data.normalize(g["dialog"], 0)
    cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
    cr = next(c for c in cl["criteria"] if c["id"] == cid)
    return ev.evaluate_criterion(rec, cl, cr)


def test_flow_tiers(monkeypatch, tmp_path):
    # 06 hard criterion, local 0.95 < hard_conf → Haiku; Haiku agrees confidently → tier 2, Sonnet not called
    loc, hai, son = Fake("loc", "10", 0.95), Fake("haiku", "10", 0.95), Fake("sonnet", "7", 0.9)
    r = _run(monkeypatch, tmp_path, loc, hai, son)
    assert r["routing"]["tier"] == 2 and son.calls == 0 and r["value"] == 10
    # Haiku disagrees with low confidence → Sonnet decides
    loc, hai, son = Fake("loc", "10", 0.95), Fake("haiku", "7", 0.7), Fake("sonnet", "4", 0.9)
    r = _run(monkeypatch, tmp_path, loc, hai, son)
    assert r["routing"]["tier"] == 3 and r["value"] == 4 and r["decided_by"].endswith("sonnet")
    # confident local on an easy criterion stays local
    loc, hai, son = Fake("loc", "YES", 1.0), Fake("haiku", "NO", 1.0), Fake("sonnet", "NO", 1.0)
    r = _run(monkeypatch, tmp_path, loc, hai, son, cid="01")
    assert r["routing"]["tier"] == 1, r["routing"]["reasons"]
    assert hai.calls == 0 and r["value"] == "YES"


def test_lora_args_and_preset(monkeypatch):
    assert router.vllm_lora_args("") == []
    assert "--lora-modules cqa-adapter=org/ad" in router.vllm_lora_args("org/ad")
    from cqa import presets
    assert presets.PRESETS["cascade"]["router"] is True
