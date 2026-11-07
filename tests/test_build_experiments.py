"""build_experiments.py runs on whatever results exist and writes the report + SVG."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_experiments.py"


def test_build_experiments_writes_report(tmp_path):
    out, svg = tmp_path / "EXP.md", tmp_path / "p.svg"
    r = subprocess.run([sys.executable, str(SCRIPT), "--reps", "200", "--out", str(out), "--svg", str(svg)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    for h in ("## 1. Executive summary", "## 2. Evaluation protocol", "## 3. Statistics", "## 4. Master leaderboard",
              "## 6. Ablations", "## 7. Threats to validity", "## 8. Reproducibility"):
        assert h in text


def _mod():
    spec = importlib.util.spec_from_file_location("build_experiments", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_shared_schema_per_dialog_is_scored(tmp_path):
    m = _mod()
    gold = m.GOLD["holdout"]
    per = {cid: {"criteria": dict(g["expected"]), "total": g.get("expected_total"),
                 "gross_error": bool(g.get("expected_gross_error"))} for cid, g in gold.items()}
    f = tmp_path / "litjev__qwen.json"
    f.write_text(json.dumps({"method": "litjev", "model": "qwen", "suite": "holdout", "per_dialog": per}))
    run = m.score(f, "holdout")
    a = m.agg(run)
    assert a["acc"] == 1.0 and a["dialogs"] == len(gold) and a["mae"] == 0
    lo, hi = m.boot_ci(list(m.dialog_acc([run]).values()), 100)
    assert lo == hi == 1.0
