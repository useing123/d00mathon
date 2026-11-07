"""scripts/agreement.py + scripts/labeling_kit.py on synthetic fixtures (no real labels, no LLM)."""
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ag = _load("agreement")
kit = _load("labeling_kit")


def test_kappa_known_values():
    assert ag.cohen_kappa(["YES", "NO", "YES", "NO"], ["YES", "NO", "YES", "NO"]) == 1.0
    # classic 2x2: po=0.7, pe=0.5 → 0.4
    a = ["Y"] * 5 + ["N"] * 5
    b = ["Y"] * 4 + ["N"] + ["Y"] * 2 + ["N"] * 3
    assert ag.cohen_kappa(a, b) == pytest.approx(0.4)
    assert ag.qw_kappa([0, 5, 10, 20], [0, 5, 10, 20], ag.SCALES["04"]) == 1.0
    near = ag.qw_kappa([0, 5, 10, 15, 20], [5, 5, 10, 15, 20], ag.SCALES["04"])
    far = ag.qw_kappa([0, 5, 10, 15, 20], [20, 5, 10, 15, 0], ag.SCALES["04"])
    assert 1 > near > far


def test_total_matches_cqa_scoring():
    from cqa import checklist as ckl, scoring
    cl = ckl.load(ckl.pick_for_channel("chat"))
    rnd = random.Random(0)
    for _ in range(50):
        v = {"01": rnd.choice(["YES", "NO"]), "02": rnd.choice(["YES", "NO", "NA"]), "03": rnd.choice(["YES", "NO", "NA"]),
             "04": rnd.choice([0, 5, 10, 15, 20, "NA"]), "05": rnd.choice(["YES", "NO"]), "06": rnd.choice([0, 4, 7, 10]),
             "07": rnd.choice(["YES", "NO", "NA"]), "08": rnd.choice(["YES", "NO"])}
        assert ag.total(v) == pytest.approx(scoring.compute_total(cl["criteria"], v)["total"], abs=0.01)


def _fixture(tmp_path):
    rnd = random.Random(1)
    ids = [f"d{i:02d}" for i in range(30)]
    A, B, S = {}, {}, []
    for i in ids:
        a = {"01": rnd.choice(["YES", "NO"]), "02": rnd.choice(["YES", "NA"]), "03": "NA", "04": rnd.choice([10, 15, 20]),
             "05": "YES", "06": rnd.choice([7, 10]), "07": rnd.choice(["YES", "NO"]), "08": "YES" if i in ("d00", "d01") else "NO"}
        b = dict(a)
        if i in ("d03", "d04"):
            b["01"] = "NO" if a["01"] == "YES" else "YES"
        if i == "d05":
            b["04"] = 0
        A[i] = {"expected": a, "notes": {}}
        B[i] = {"expected": b, "notes": {}}
        sv = dict(a)
        if i in ("d06", "d07"):
            sv["07"] = "NO" if a["07"] == "YES" else "YES"
        crit = [{"criterion_id": code, "value": (None if sv[k] == "NA" else sv[k] == "YES" if k not in ag.SCALES else sv[k]),
                 "rationale": f"why {k}"} for code, k in ag.CODES.items()]
        S.append({"conversation_id": i, "criteria": crit,
                  "score": {"final_score": ag.total(sv), "critical_flag": i in ("d00", "d02")}})
    for n, d in (("A", A), ("B", B)):
        (tmp_path / f"labels_{n}.json").write_text(json.dumps(d))
    (tmp_path / "sys.json").write_text(json.dumps({"results": S}))
    (tmp_path / "sample.json").write_text(json.dumps({"ids": ids}))
    (tmp_path / "held.json").write_text(json.dumps({"ids": ids[-5:]}))
    return ids


def test_end_to_end_report(tmp_path):
    _fixture(tmp_path)
    md = tmp_path / "H.md"
    rc = ag.main(["--a", str(tmp_path / "labels_A.json"), "--b", str(tmp_path / "labels_B.json"),
                  "--system", str(tmp_path / "sys.json"), "--sample", str(tmp_path / "sample.json"),
                  "--heldout", str(tmp_path / "held.json"), "--consensus", str(tmp_path / "none.json"),
                  "--out-dir", str(tmp_path / "out"), "--md", str(md), "--reps", "200"])
    assert rc == 0
    r = json.loads((tmp_path / "out" / "agreement.json").read_text())
    assert r["meta"]["n_ids"] == 25                      # sealed held-out excluded by default
    ia = r["inter_annotator"]["per_criterion"]
    assert ia["01"]["agreement"] == pytest.approx(23 / 25)
    assert ia["05"]["kappa"] == 1.0 and ia["04"]["mae"] == pytest.approx(ia["04"]["mae"])
    assert ia["04"]["ci"]["qw_kappa"][0] <= ia["04"]["qw_kappa"] <= 1
    sc = r["system_vs_consensus"]["per_criterion"]
    assert sc["07"]["agreement"] == pytest.approx(23 / 25) and sc["03"]["agreement"] == 1.0
    g = r["gross_pr_vs_consensus"]
    assert (g["tp"], g["fp"], g["fn"]) == (1, 1, 1) and g["precision"] == 0.5
    dis = json.loads((tmp_path / "out" / "disagreements.json").read_text())
    assert {(d["conversation_id"], d["criterion_id"]) for d in dis if d["criterion_id"] == "07"} == {("d06", "07"), ("d07", "07")}
    assert all(d["system_rationale"] for d in dis)
    todo = json.loads((tmp_path / "out" / "adjudicate.json").read_text())
    assert {(t["conversation_id"], t["criterion_id"]) for t in todo} == {("d03", "01"), ("d04", "01"), ("d05", "04")}
    assert "held-out (15 dialogs) EXCLUDED" in md.read_text()


def test_missing_labels_refuses(tmp_path):
    assert ag.main(["--a", str(tmp_path / "x.json"), "--b", str(tmp_path / "y.json"), "--md", str(tmp_path / "m.md")]) == 2
    assert not (tmp_path / "m.md").exists()


def test_kit_sample_is_stratified(tmp_path):
    recs, scores = [], {}
    for lg, n in (("RU", 30), ("KZ", 15), ("EN", 15)):
        for i in range(n):
            recs.append({"conversation_id": f"{lg}-{i}", "kind": "chat", "language": lg})
            scores[f"{lg}-{i}"] = (40 + i * 4, i == 3)
    for i in range(5):
        recs.append({"conversation_id": f"call-{i}", "kind": "call", "language": "RU"})
    s = kit.pick_sample(recs, scores)
    assert len(s) == 50 == len(set(s))
    assert all(f"call-{i}" in s for i in range(5)) and all(f"{lg}-3" in s for lg in ("RU", "KZ", "EN"))
    h = kit.pick_heldout(s, recs, scores)
    assert len(h) == 15 and set(h) <= set(s) and any(x.startswith("call") for x in h)
    assert {x.split("-")[0] for x in h} >= {"RU", "KZ", "EN"}
    assert kit.pick_sample(recs, scores) == s             # deterministic


def test_fast_labeller_one_key_per_criterion():
    from cqa import checklist as ckl
    from cqa.__main__ import fast_label_dialog
    cl = ckl.load(ckl.pick_for_channel("chat"))
    keys = iter(["y", "?", "x", "b", "n", "x", "4", "y", "z", "3", "n", "n"])
    rec = {"transcript": [{"index": 1, "speaker": "client", "text": "hi"}]}
    lab = fast_label_dialog(rec, cl, key=lambda _p: next(keys), say=lambda *_: None)
    assert lab["expected"] == {"01": "YES", "02": "NO", "03": "NA", "04": 20, "05": "YES", "06": 10, "07": "NO", "08": "NO"}
