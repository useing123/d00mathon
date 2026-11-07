"""Evidence step: cite-only call after non-escalated tier-1 verdicts (mock provider, no network)."""
from cqa import evidence_step as es

CR_BIN = {"id": "01", "type": "binary", "name": "x"}
CR_SCALE = {"id": "04", "type": "scale", "scale": [0, 10, 20], "name": "x"}
CR_08 = {"id": "08", "type": "binary", "name": "x", "gross": True}
REC = {"hash": "h1", "conversation_id": "c1", "transcript": [
    {"index": 0, "speaker": "client", "text": "Здравствуйте, у меня вопрос"},
    {"index": 1, "speaker": "operator", "text": "Добрый день. Чем помочь?"}]}


class FakeLLM:
    provider, model = "fake", "claude-haiku-5-5"

    def __init__(self, out=None, exc=None):
        self.out, self.exc, self.calls = out, exc, []

    def call(self, system, cached_user, user, tool, temperature=0.0):
        self.calls.append(user)
        if self.exc:
            raise self.exc
        return self.out


def _ck(monkeypatch):
    monkeypatch.setattr(es.prompts, "criterion_block", lambda ck, cr: "КРИТЕРИЙ\n\nОцени ТОЛЬКО это")
    monkeypatch.setattr(es.ckl, "criterion_hash", lambda ck, cr: "ch")


def test_is_disputed(monkeypatch):
    assert es.is_disputed(CR_BIN, "NO") and es.is_disputed(CR_BIN, "NA") and not es.is_disputed(CR_BIN, "YES")
    assert es.is_disputed(CR_SCALE, "10") and not es.is_disputed(CR_SCALE, "20")
    assert es.is_disputed(CR_08, "YES")
    monkeypatch.setenv("CQA_EVIDENCE_08_YES", "0")
    assert not es.is_disputed(CR_08, "YES")


def test_mode(monkeypatch):
    monkeypatch.delenv("CQA_EVIDENCE_STEP", raising=False)
    assert es.mode() == "disputed"
    monkeypatch.setenv("CQA_EVIDENCE_STEP", "ALL")
    assert es.mode() == "all"
    monkeypatch.setenv("CQA_EVIDENCE_STEP", "bogus")
    assert es.mode() == "disputed"


def test_verified_quote_kept_value_unchanged(monkeypatch):
    _ck(monkeypatch)
    llm = FakeLLM({"evidence": [{"turn": 1, "quote": "Добрый день."}, {"turn": 1, "quote": "выдумка"}],
                   "rationale": "Нет имени"})
    res = {"value": "NO", "evidence": [], "justification": "jev p=1"}
    assert es.apply(res, llm, REC, "tx", {}, CR_BIN, m="disputed", use_cache=False) is False
    assert res["value"] == "NO" and res["evidence"][0]["turn"] == 1 and len(res["evidence"]) == 1
    assert res["routing"]["evidence_step"]["status"] == "ok" and res["routing"]["evidence_step"]["rejected"] == 1
    assert "РЕШЕНИЕ УЖЕ ПРИНЯТО: value = NO" in llm.calls[0]


def test_skips(monkeypatch):
    _ck(monkeypatch)
    llm = FakeLLM({"evidence": [], "rationale": ""})
    for m, v in (("off", "NO"), ("disputed", "YES")):
        res = {"value": v, "evidence": []}
        es.apply(res, llm, REC, "tx", {}, CR_BIN, m=m, use_cache=False)
        assert res["routing"]["evidence_step"]["status"] == "skipped"
    assert not llm.calls
    es.apply({"value": "YES", "evidence": []}, llm, REC, "tx", {}, CR_BIN, m="all", use_cache=False)
    assert len(llm.calls) == 1


def test_missing_and_escalate(monkeypatch):
    _ck(monkeypatch)
    for llm in (FakeLLM({"evidence": [{"turn": 0, "quote": "нет такого"}], "rationale": "r"}),
                FakeLLM(exc=RuntimeError("boom"))):
        res = {"value": "NA", "evidence": []}
        monkeypatch.delenv("CQA_EVIDENCE_ESCALATE", raising=False)
        assert es.apply(res, llm, REC, "tx", {}, CR_BIN, m="disputed", use_cache=False) is False
        assert res["evidence_missing"] and res["value"] == "NA" and not res["evidence"]
        monkeypatch.setenv("CQA_EVIDENCE_ESCALATE", "1")
        assert es.apply({"value": "NA", "evidence": []}, llm, REC, "tx", {}, CR_BIN, m="disputed", use_cache=False)


def test_cascade_hook_mock(monkeypatch, tmp_path):
    """End-to-end through CascadeEvaluator with the mock provider: tier-1 verdicts never change."""
    from cqa import config, data, router
    from cqa.gold import load_gold
    monkeypatch.setattr(config, "PROVIDER", "mock")
    monkeypatch.setattr(router, "CACHE", tmp_path)
    monkeypatch.setattr(es, "CACHE", tmp_path)
    gold = load_gold(config.ROOT / "examples" / "gold")[:2]
    recs = [data.normalize(g["dialog"], i) for i, g in enumerate(gold)]
    vals = {}
    for m in ("off", "all"):
        monkeypatch.setenv("CQA_EVIDENCE_STEP", m)
        ev = router.CascadeEvaluator(use_cache=False, t1=0.0, hard="", hard_conf=0.0, first="local", logprobs=False,
                                     judge="")
        res, _ = ev.evaluate_many(recs, concurrency=1)
        vals[m] = [(c["criterion_id"], c["value"]) for d in res for c in d["criteria"]]
        steps = {c["routing"].get("evidence_step", {}).get("status") for d in res for c in d["criteria"]}
        assert "skipped" in steps and steps <= ({None, "skipped"} if m == "off" else {None, "skipped", "ok", "evidence_missing"})
    assert vals["off"] == vals["all"]
