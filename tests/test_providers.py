"""Provider response paths with mocked SDK/HTTP: tool_use parsing, missing tool_use, malformed JSON,
retry on 5xx, no retry on 4xx, usage accounting."""
import json
from types import SimpleNamespace

import httpx
import pytest

from cqa import config, llm, prompts


def _anth_llm(monkeypatch, responses):
    m = llm.LLM.__new__(llm.LLM)
    m.provider, m.model = "anthropic", "claude-haiku-5-5"
    seq = list(responses)

    def create(**kw):
        assert kw["tools"] == [prompts.TOOL]
        assert kw["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    m.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return m, seq


def _resp(blocks):
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(model_dump=lambda: {
        "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}))


class _HTTPErr(Exception):
    def __init__(self, code):
        self.status_code = code
        self.response = SimpleNamespace(status_code=code, headers={})


def test_anthropic_tool_use_and_usage(monkeypatch):
    before = dict(llm.USAGE)
    m, _ = _anth_llm(monkeypatch, [_resp([SimpleNamespace(type="text", text="hm"),
                                          SimpleNamespace(type="tool_use", input={"value": "YES", "evidence": []})])])
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "YES"
    assert llm.USAGE["cache_read_input_tokens"] - before.get("cache_read_input_tokens", 0) == 100


def test_anthropic_retries_5xx_then_succeeds(monkeypatch):
    m, seq = _anth_llm(monkeypatch, [_HTTPErr(529), _resp([SimpleNamespace(type="tool_use", input={"value": "NO"})])])
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "NO" and not seq


def test_anthropic_does_not_retry_400(monkeypatch):
    m, seq = _anth_llm(monkeypatch, [_HTTPErr(400), _resp([SimpleNamespace(type="tool_use", input={})])])
    with pytest.raises(_HTTPErr):
        m.call("s", "c", "u", prompts.TOOL)
    assert len(seq) == 1


def test_anthropic_missing_tool_use_raises_after_retries(monkeypatch):
    m, _ = _anth_llm(monkeypatch, [_resp([SimpleNamespace(type="text", text="no tool")])] * llm.MAX_ATTEMPTS)
    with pytest.raises(RuntimeError, match="no tool_use"):
        m.call("s", "c", "u", prompts.TOOL)


def _openai_llm(monkeypatch, bodies):
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "http://local/v1")
    monkeypatch.setattr(config, "OPENAI_TOOLS", False)
    m = llm.LLM.__new__(llm.LLM)
    m.provider, m.model = "openai", "qwen"
    seq = list(bodies)

    def post(url, json=None, headers=None, timeout=None):
        b = seq.pop(0)
        return httpx.Response(b[0], json=b[1], request=httpx.Request("POST", url))
    monkeypatch.setattr(llm.httpx, "post", post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return m, seq


def _ok(content):
    return (200, {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def test_openai_native_tool_call(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_TOOLS", True)
    call = {"choices": [{"message": {"tool_calls": [{"function": {"name": "submit_evaluation",
                                                                  "arguments": json.dumps({"value": "10"})}}]}}]}
    m, seq = _openai_llm(monkeypatch, [(200, call)])
    monkeypatch.setattr(config, "OPENAI_TOOLS", True)
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "10" and not seq


def test_openai_tools_400_falls_back_to_json_mode(monkeypatch):
    m, seq = _openai_llm(monkeypatch, [(400, {"error": "tools unsupported"}), _ok('{"value": "YES"}')])
    monkeypatch.setattr(config, "OPENAI_TOOLS", True)
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "YES" and not seq


def test_openai_json_in_prose(monkeypatch):
    m, _ = _openai_llm(monkeypatch, [_ok('Sure: {"value": "15", "evidence": []} done')])
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "15"


def test_openai_malformed_json_retried(monkeypatch):
    m, seq = _openai_llm(monkeypatch, [_ok("{not json"), (503, {}), _ok(json.dumps({"value": "NA"}))])
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "NA" and not seq


def test_csv_formula_injection_guard():
    from cqa import store
    doc = {"dialogs": [{"conversation_id": "=HYPERLINK(1)", "channel": "@x", "language": "RU", "checklist_id": "c",
                        "criteria": [], "total": 1, "gross_error": False}]}
    line = store.to_csv(doc).splitlines()[1]
    assert line.startswith("'=HYPERLINK(1),'@x,RU")


def test_wrong_turn_index_prefers_operator_turn():
    from cqa import data
    from cqa.evaluator import validate_evidence
    r = data.normalize({"transcript": [{"speaker": "client", "text": "спасибо"}, {"speaker": "operator", "text": "спасибо"}]}, 0)
    v, _ = validate_evidence(r, [{"turn": 7, "quote": "спасибо"}])
    assert v[0]["turn"] == 1 and v[0]["speaker"] == "operator"


def test_anthropic_text_json_fallback(monkeypatch):
    m, _ = _anth_llm(monkeypatch, [_resp([SimpleNamespace(type="text", text='Ответ: {"value": "NA", "evidence": []}')])])
    assert m.call("s", "c", "u", prompts.TOOL)["value"] == "NA"
