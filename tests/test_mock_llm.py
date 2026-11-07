from cqa import config
from cqa.data import load_json
from cqa.llm import LLM


def test_mock_is_deterministic_and_offline():
    assert config.PROVIDER == "mock"
    from cqa.checklist import load
    from cqa.evaluator import Evaluator
    rec = load_json([{"conversation_id": "m", "channel": "agent chat", "transcript": [
        {"timestamp": "00:00", "speaker": "client", "text": "hi"},
        {"timestamp": "00:05", "speaker": "operator", "text": "Hello, my name is Ann. How can I help?"},
        {"timestamp": "00:09", "speaker": "operator", "text": "Goodbye and have a nice day!"}]}])[0]
    cl = load("freedom_chat_v1")
    ev = Evaluator(use_cache=False)
    a = ev.evaluate_criterion(rec, cl, cl["criteria"][0])
    b = ev.evaluate_criterion(rec, cl, cl["criteria"][0])
    assert a["value"] == b["value"] == "YES" and a["evidence"][0]["turn"] == 1
    assert LLM().model == "mock-heuristic"
