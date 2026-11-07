"""Gold pipeline + metrics run offline (mock). Checks plumbing, not accuracy."""
import json

import pytest

from cqa import checklist as ckl, scoring
from cqa.gold import GOLD_DIR, compute_metrics, load_gold, run_gold


@pytest.mark.parametrize("d,lo,hi", [(GOLD_DIR, 12, 16), (GOLD_DIR.parent / "gold_holdout", 8, 10)])
def test_gold_files_are_consistent(d, lo, hi):
    gold = load_gold(d)
    assert lo <= len(gold) <= hi
    langs = {g["dialog"]["language"] for g in gold}
    assert {"RU", "KZ", "EN"} <= langs and sum("call" in g["dialog"]["channel"] for g in gold) >= 2
    ids = {g["dialog"]["conversation_id"] for g in load_gold(GOLD_DIR)}
    if d != GOLD_DIR:
        assert not ids & {g["dialog"]["conversation_id"] for g in gold}   # held-out split is disjoint
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


def test_label_dialog_and_labels_roundtrip(tmp_path, monkeypatch):
    from cqa import data
    from cqa.gold import label_dialog, load_labels, run_gold
    rec = data.normalize(load_gold()[0]["dialog"], 0)
    cl = ckl.load("freedom_chat_v1")
    answers = iter(["7.5", "YES", "", "NA", "", "NA", "", "20", "", "YES", "", "10", "", "YES", "", "NO", "ok"])
    out = []
    lab = label_dialog(rec, cl, ask=lambda _: next(answers), say=out.append)
    assert lab["expected"]["01"] == "YES" and lab["expected"]["04"] == 20 and lab["notes"] == {"08": "ok"}
    assert any("invalid" in str(x) for x in out)          # 7.5 was rejected and re-asked
    monkeypatch.setattr(data, "load_group", lambda g, refresh=False: [rec])
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({rec["conversation_id"]: lab}))
    items = load_labels(p, "X")
    assert items[0]["expected_total"] == 100.0
    m = run_gold(mock=True, labels=p, group="X", out=tmp_path / "r.json")
    assert m["dialogs"] == 1 and m["source"].startswith("labels:")
