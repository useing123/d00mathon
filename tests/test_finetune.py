"""Fine-tuning pipeline tests: run WITHOUT a GPU and without torch/transformers/peft/trl (those paths are skipped)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cqa import checklist as ckl, config, data, prompts
from cqa.finetune import augment, dataset as ds, train_utils as tu

ROOT = Path(__file__).resolve().parent.parent


def _dialog(cid, lang, text_op="Здравствуйте! Меня зовут Алина. Ваш баланс 12345 тенге."):
    return {"conversation_id": cid, "channel": "agent chat", "language": lang, "transcript": [
        {"timestamp": "00:01", "speaker": "client", "text": "Сәлеметсіз бе, баланс қанша?" if lang == "KZ" else "Hello, balance?"},
        {"timestamp": "00:05", "speaker": "operator", "text": text_op}]}


def _jury_doc(recs):
    dialogs = []
    for r in recs:
        cl = ckl.load(ckl.pick_for_channel(r["kind"]))
        crit = []
        for cr in cl["criteria"]:
            v = "YES" if cr["type"] == "binary" else str(cr["scale"][-1])
            crit.append({"criterion_id": cr["id"], "value": v, "justification": "ok", "reasoning": "r",
                         "evidence": [{"turn": 2, "quote": "Здравствуйте!", "speaker": "operator"}], "confidence": 0.95,
                         "votes": [v, v], "decided_by": "consensus", "criterion_hash": ckl.criterion_hash(cl, cr)})
        crit[0].update(votes=["YES", "NO"], decided_by="judge", confidence=0.5)   # disagreement + low conf → dropped
        dialogs.append({"conversation_id": r["conversation_id"], "checklist_id": cl["id"], "criteria": crit})
    return {"meta": {"reproducibility": {"prompt_version": prompts.PROMPT_VERSION}, "turn_base": 1}, "dialogs": dialogs}


@pytest.fixture
def fixture_data(tmp_path):
    raw = [_dialog(f"ru-{i}", "RU") for i in range(12)] + [_dialog(f"kz-{i}", "KZ") for i in range(4)] + \
          [_dialog(f"en-{i}", "EN") for i in range(5)]
    raw.append(dict(_dialog("ru-dup", "RU"), conversation_id="ru-dup"))   # identical transcript to ru-0 (other id)
    recs = data.load_json(raw)
    jp = tmp_path / "jury.json"
    jp.write_text(json.dumps(_jury_doc(recs), ensure_ascii=False), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"ru-1": {"expected": {"01": "NO"}, "notes": {"01": "no name"}}}), encoding="utf-8")
    return recs, jp, labels


def _build(fixture_data, tmp_path, **kw):
    recs, jp, labels = fixture_data
    return ds.build(tmp_path / "ft", recs=recs, jury=[jp], labels=labels, val_frac=0.25, **kw)


def test_build_no_leakage_and_outputs(fixture_data, tmp_path):
    rep, tr, va = _build(fixture_data, tmp_path)
    assert tr and va
    assert rep["leakage_conversations"] == []
    tr_hash = {r["meta"]["dialog_hash"] for r in tr}
    assert not tr_hash & {r["meta"]["dialog_hash"] for r in va}           # identical transcripts stay together
    assert (tmp_path / "ft" / "train.jsonl").exists() and (tmp_path / "ft" / "stats.json").exists()
    assert set(rep["train"]["by_source"]) >= {"gold", "distill", "human"}
    assert rep["skipped"]["jury.json"]["low_agreement"] >= 1               # 01 with split votes + conf .5 was dropped
    for r in tr + va:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert json.loads(r["messages"][2]["content"])["value"] == r["meta"]["value"]


def test_human_label_beats_distill(fixture_data, tmp_path):
    _, tr, va = _build(fixture_data, tmp_path)
    rows = [r for r in tr + va if r["meta"]["conversation_id"] == "ru-1" and r["meta"]["criterion_id"] == "01"]
    assert rows and all(r["meta"]["source"] == "human" and r["meta"]["value"] == "NO" for r in rows)


def test_language_balance_upsamples_kz(fixture_data, tmp_path):
    rep, tr, _ = _build(fixture_data, tmp_path, use_gold=False)
    by = rep["train"]["by_language"]
    assert by["KZ"] >= 0.95 * by["RU"]            # KZ target 1.0 of the largest language
    assert by["EN"] >= 0.55 * by["RU"]
    assert rep["train"]["upsampled"] > 0
    assert rep["val"]["upsampled"] == 0           # val never upsampled


def test_prompt_identical_to_inference(monkeypatch):
    """Capture what the evaluator actually sends to the backend and compare with the dataset's prompt."""
    from cqa import evaluator, llm
    rec = data.load_json([_dialog("p-1", "RU")])[0]
    cl = ckl.load(ckl.pick_for_channel(rec["kind"]))
    seen = []
    orig = llm.MockBackend.call

    def spy(self, system, cached_user, user, tool, temperature=0.0):
        seen.append((system, cached_user, user, tool))
        return orig(self, system, cached_user, user, tool, temperature)

    monkeypatch.setattr(llm.MockBackend, "call", spy)
    ev = evaluator.Evaluator(use_cache=False, samples=1)
    cr = cl["criteria"][0]
    ev.evaluate_criterion(rec, cl, cr)
    system, cached_user, user, tool = seen[0]
    s, u = ds.build_prompt(rec, cl, cr)
    # cqa.llm.LLM._openai JSON mode: system + suffix + schema; user = cached_user + "\n\n" + user
    assert s == system + ds.JSON_MODE_SUFFIX + json.dumps(tool["input_schema"], ensure_ascii=False)
    assert u == cached_user + "\n\n" + user


def test_gold_holdout_refused():
    with pytest.raises(ValueError):
        ds.from_gold(ROOT / "examples" / "gold_holdout")


def test_tokenize_masks_prompt_and_collator():
    tok = tu.CharTokenizer()
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U" * 50},
                        {"role": "assistant", "content": '{"value": "YES"}'}], "meta": {"criterion_id": "01", "value": "YES"}}
    t = tu.tokenize_example(row, tok, max_len=4096)
    p = tu.prompt_ids(tok, row["messages"])
    assert t["input_ids"][:len(p)] == p and all(x == tu.IGNORE for x in t["labels"][:len(p)])
    assert tok.decode([x for x in t["labels"] if x != tu.IGNORE]) == '{"value": "YES"}<|im_end|>'
    assert tu.tokenize_example(row, tok, max_len=20) is None                     # drop policy for real training
    assert len(tu.tokenize_example(row, tok, max_len=40, truncate="middle")["input_ids"]) <= 40
    short = tu.tokenize_example(dict(row, messages=row["messages"][:1] + row["messages"][2:]), tok)
    b = tu.PadCollator(pad_id=0, as_tensors=False)([t, short])
    assert len({len(x) for x in b["input_ids"]}) == 1 and b["labels"][1][-1] == tu.IGNORE


def test_agreement_metric():
    rows = [{"meta": {"criterion_id": "01", "value": "YES"}}, {"meta": {"criterion_id": "04", "value": "15"}},
            {"meta": {"criterion_id": "04", "value": "NA"}}]
    m = tu.agreement(['bla {"value": "yes"} x', '{"value": "15"}', "not json"], rows)
    assert m["per_criterion"] == {"01": 1.0, "04": 0.5} and m["json_parse_rate"] == round(2 / 3, 4)


def test_mock_pipeline_end_to_end(fixture_data, tmp_path):
    _, tr, va = _build(fixture_data, tmp_path)
    res = tu.run_mock(tr, va, steps=2, log=lambda *_: None)
    assert len(res["history"]) == 2 and res["history"][0]["loss_after"] < res["history"][0]["loss_before"]
    assert res["eval"]["json_parse_rate"] == 1.0 and res["eval"]["n"] == len(va)


def test_cost_estimate():
    e = tu.estimate_cost(1_000_000, "Qwen/Qwen3-14B", epochs=1)
    assert e["size_class"] == "14b" and e["gpus"]["A40"]["usd"] > 0
    assert e["gpus"]["A40"]["hours"] > tu.estimate_cost(1_000_000, "Qwen/Qwen3-8B")["gpus"]["A40"]["hours"]


def test_augment_swaps_consistently(fixture_data, tmp_path):
    recs, jp, labels = fixture_data
    exs, _ = ds.from_jury(jp, {r["conversation_id"]: r for r in recs})
    kz = [e for e in exs if e["language"] == "KZ"][:3]
    new = augment.Augmenter().augment_examples(kz)
    assert new and all(e["augmented"] and e["dialog_hash"] != kz[0]["dialog_hash"] for e in new)
    txt = new[0]["_rec"]["transcript"][1]["text"]
    assert "Алина" not in txt and "12345" not in txt and augment.evidence_ok(new[0]["_rec"], new[0]["target"])
    assert augment.Augmenter().back_translate(kz[0]) is None                  # default translator = no-op


def test_cli_mock_runs(tmp_path, fixture_data):
    _build(fixture_data, tmp_path)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune_lora.py"), "--mock",
                          "--train", str(tmp_path / "ft" / "train.jsonl"), "--val", str(tmp_path / "ft" / "val.jsonl")],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "[mock] step 2/2" in out.stdout and "agreement=" in out.stdout


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None,
                    reason="transformers/torch not installed (GPU-free CI)")
def test_dry_run_tiny_model(tmp_path, fixture_data):
    _build(fixture_data, tmp_path)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune_lora.py"), "--dry-run", "--max-len", "1024",
                          "--train", str(tmp_path / "ft" / "train.jsonl"), "--val", str(tmp_path / "ft" / "val.jsonl"),
                          "--out", str(tmp_path / "run")], capture_output=True, text=True, timeout=900)
    if out.returncode != 0 and ("OSError" in out.stderr or "ConnectionError" in out.stderr):
        pytest.skip("tiny model not downloadable (offline)")
    assert out.returncode == 0, out.stderr[-3000:]


def test_config_isolated():
    assert "results" in str(config.RESULTS_DIR)
