"""External validation (MAIA-DQE): loader, label derivation, sampling, statistics — offline, no network."""
import importlib.util
import json
from pathlib import Path

from cqa import checklist as ckl, data, ext_data
from cqa.evaluator import Evaluator
from cqa.llm import LLM

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("external_bench", ROOT / "scripts" / "external_bench.py")
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)


def _dialog(i, ts=5, dc=0, pol=(1, 1), sens=(1, 1)):
    turns = [{"floor": "inbound", "text_src": ["Hallo"], "text_mt": [f"Hello {i}"], "Politeness": None}]
    for p, s in zip(pol, sens, strict=True):
        turns.append({"floor": "outbound", "text_src": ["Hi #NAME#,", "how can I help?"], "text_mt": ["Hallo"],
                      "Politeness": p, "Sensibleness": s, "Understanding": 1})
        turns.append({"floor": "inbound", "text_src": ["Danke"], "text_mt": ["Thanks"], "Politeness": 1})
    return {"id": f"d{i}", "turns": turns, "dialog": {"Dropped conversation": dc, "Task Sucess": ts}}


def test_record_format_and_labels():
    r = ext_data.maia_to_record(_dialog(1, ts=3, dc=1, pol=(1, 0), sens=(1, 1)), "x")
    assert [t["speaker"] for t in r["transcript"]] == ["client", "operator", "client", "operator", "client"]
    assert r["transcript"][0]["text"] == "Hello 1"            # customer: English MT
    assert r["transcript"][1]["text"] == "Hi #NAME#, how can I help?"   # agent: English original
    assert r["human_labels"] == {"task_success": 3, "dropped": 1, "agent_polite": 0, "agent_sensible": 1,
                                 "agent_understanding": 1}
    n = data.normalize(r, 0)
    assert n["kind"] == "chat" and n["transcript"][1]["speaker"] == "operator"
    # invalid (-1) source labels are dropped, not coerced
    bad = ext_data.maia_labels({"turns": [], "dialog": {"Dropped conversation": -1, "Task Sucess": -1}})
    assert bad["task_success"] is None and bad["dropped"] is None and bad["agent_polite"] is None


def test_load_and_stratified_sample_is_deterministic(tmp_path):
    ds = [_dialog(i, ts=1 + i % 5, dc=i % 2, pol=(1, 0) if i % 7 == 0 else (1, 1)) for i in range(60)]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(ds))
    recs = ext_data.load_maia([p])
    a = ext_data.stratified_sample(recs, n=20, seed=1)
    b = ext_data.stratified_sample(recs, n=20, seed=1)
    assert [r["conversation_id"] for r in a] == [r["conversation_id"] for r in b] and len(a) == 20
    imp = [r for r in a if r["human_labels"]["agent_polite"] == 0]
    assert len(imp) == 5   # rare negatives included up to n//4
    dup = ext_data.stratified_sample(recs + recs, n=20, seed=1)   # duplicated ids are dropped after sampling
    assert len({r["conversation_id"] for r in dup}) == len(dup)


def test_ext_checklist_mirrors_source_and_runs_mock():
    c = ckl.load("ext_maia_dqe")
    assert [cr["id"] for cr in c["criteria"]] == ["TS", "DC", "POL", "SENS"]
    assert c["criteria"][0]["scale"] == [1, 2, 3, 4, 5]
    assert "Was the agent able to fulfil the customer's request?" in c["criteria"][0]["rubric"]
    rec = data.normalize(ext_data.maia_to_record(_dialog(2), "x"), 0)
    out, _ = Evaluator(llm=LLM(provider="mock"), use_cache=False, ensemble=False).evaluate_many(
        [rec], checklist_override="ext_maia_dqe")
    assert {x["criterion_id"] for x in out[0]["criteria"]} == {"TS", "DC", "POL", "SENS"}


def test_statistics():
    assert eb.kappa(["Y", "N", "Y", "N"], ["Y", "N", "Y", "N"]) == 1.0
    assert abs(eb.kappa(["Y", "Y", "N", "N"], ["Y", "N", "Y", "N"])) < 1e-9
    assert eb.qwk([1, 2, 3], [1, 2, 3], [1, 2, 3, 4, 5]) == 1.0
    assert abs(eb.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert eb.spearman([1, 2, 3, 4], [4, 3, 2, 1]) < -0.99
    p, r = eb.pr(["NO", "NO", "YES", "YES"], ["NO", "YES", "NO", "YES"], "NO")
    assert p == 0.5 and r == 0.5
    s = eb.stat_block([("NO", "NO"), ("YES", "YES"), ("YES", "NO"), ("YES", "YES")], "binary", neg="NO")
    assert s["accuracy"] == 0.75 and s["neg_recall"] == 0.5 and s["neg_precision"] == 1.0
    lo, hi = s["accuracy_ci"]
    assert 0 <= lo <= 0.75 <= hi <= 1
