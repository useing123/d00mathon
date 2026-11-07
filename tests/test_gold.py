"""Gold pipeline + metrics run offline (mock). Checks plumbing, not accuracy."""
import json

from cqa import checklist as ckl, scoring
from cqa.gold import compute_metrics, load_gold, run_gold


def test_gold_files_are_consistent():
    gold = load_gold()
    assert 12 <= len(gold) <= 16
    langs = {g["dialog"]["language"] for g in gold}
    assert {"RU", "KZ", "EN"} <= langs and sum("call" in g["dialog"]["channel"] for g in gold) >= 2
    for g in gold:
        cl = ckl.load("freedom_call_v1" if "call" in g["dialog"]["channel"] else "freedom_chat_v1")
        assert set(g["expected"]) == {c["id"] for c in cl["criteria"]}
        for c in cl["criteria"]:
            assert ckl.strict_value(c, g["expected"][c["id"]]) is not None, (g["file"], c["id"])
        t = scoring.compute_total(cl["criteria"], g["expected"], cl["assumptions"]["rounding"], cl["assumptions"]["decimals"])
        assert t["total"] == g["expected_total"], g["file"]


def test_gold_mock_run(tmp_path):
    m = run_gold(mock=True, out=tmp_path / "r.json")
    assert m["dialogs"] == len(load_gold()) and 0 <= m["overall_agreement"] <= 1
    assert set(m["per_criterion"]) == {f"0{i}" for i in range(1, 9)} and m["total_mae"] is not None
    assert json.loads((tmp_path / "r.json").read_text())["metrics"]["overall_agreement"] == m["overall_agreement"]


def test_metrics_perfect_and_failed():
    g = [{"dialog": {"conversation_id": "a"}, "expected": {"01": "YES", "04": "NA", "08": "YES"},
          "expected_total": 50.0, "expected_gross_error": True}]
    perfect = [{"conversation_id": "a", "total": 50.0, "gross_error": True,
                "criteria": [{"criterion_id": "01", "value": "YES"}, {"criterion_id": "04", "value": "NA"},
                             {"criterion_id": "08", "value": "YES"}]}]
    m = compute_metrics(g, perfect)
    assert m["overall_agreement"] == 1.0 and m["total_mae"] == 0 and m["gross"]["recall"] == 1.0
    failed = [{"conversation_id": "a", "total": None, "provisional_total": 40.0, "status": "incomplete", "gross_error": False,
               "criteria": [{"criterion_id": "01", "value": None}, {"criterion_id": "04", "value": 10},
                            {"criterion_id": "08", "value": "NO"}]}]
    m = compute_metrics(g, failed)
    assert m["overall_agreement"] == 0 and m["gross"]["recall"] == 0 and m["incomplete"] == ["a"]
    assert m["na_confusion"]["exp_NA/pred_applicable"] == 1 and m["na_confusion"]["exp_applicable/pred_FAILED"] == 1
