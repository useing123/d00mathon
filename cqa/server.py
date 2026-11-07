"""FastAPI backend (no login). Run: python -m cqa serve  ->  http://127.0.0.1:8090"""
import json
import re
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import checklist as ckl, config, data, store
from .evaluator import Evaluator, rescore
from .llm import USAGE, cost_usd
from .util import atomic_write

_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


def safe_id(value, what="identifier"):
    """Strict allowlist for identifiers that become file names (group / run / checklist)."""
    v = str(value or "").strip()
    if not _ID_RE.fullmatch(v) or ".." in v:
        raise HTTPException(400, f"invalid {what}")
    return v


def safe_path(base, name, suffix=".json"):
    """Resolve base/name+suffix and make sure it stays inside base."""
    base = Path(base).resolve()
    p = (base / f"{name}{suffix}").resolve()
    if p.parent != base:
        raise HTTPException(400, "invalid path")
    return p


app = FastAPI(title="CQA — Freedom Broker quality assessment")
STATIC = Path(__file__).parent / "static"
JOBS = {}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/static/{name}")
def static_file(name: str):
    if name not in ("app.js", "app.css"):
        raise HTTPException(404, "not found")
    return FileResponse(STATIC / name, media_type="text/javascript" if name.endswith(".js") else "text/css")


@app.get("/models")
def models_page():
    return FileResponse(STATIC / "models.html")


@app.get("/api/models/bench")
def models_bench():
    """Benchmark matrix (scripts/bench_matrix.py matrix) + presets; see docs/MODELS.md."""
    from . import presets
    p = config.RESULTS_DIR / "bench_matrix.json"
    if not p.exists():
        raise HTTPException(404, "results/bench_matrix.json not found: run scripts/bench_matrix.py matrix")
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["presets"] = {k: {**v, "default": k == presets.DEFAULT} for k, v in presets.PRESETS.items()}
    doc["active_preset"] = getattr(config, "PRESET", "")
    return doc


@app.get("/api/health")
def health():
    return {"ok": True, "provider": config.PROVIDER, "model": config.MODEL,
            "fetch_configured": bool(config.get("CQA_API_URL"))}


# ---------- checklists ----------
@app.get("/api/checklists")
def checklists():
    return ckl.list_checklists()


@app.get("/api/checklists/{cid}")
def get_checklist(cid: str):
    cid = safe_id(cid, "checklist id")
    try:
        c = ckl.load(cid)
    except FileNotFoundError as e:
        raise HTTPException(404) from e
    return {"checklist": c, "yaml": safe_path(config.CHECKLIST_DIR, cid, ".yaml").read_text(encoding="utf-8")}


@app.put("/api/checklists/{cid}")
def put_checklist(cid: str, body: dict = Body(...)):
    cid = safe_id(cid, "checklist id")
    try:
        c = yaml.safe_load(body["yaml"]) if "yaml" in body else body["checklist"]
        c["id"] = c.get("id") or cid
        safe_id(c["id"], "checklist id")
        c = ckl.save(c)
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "id": c["id"]}


# ---------- data ----------
def _group_records(group):
    p = safe_path(config.DATA_DIR, safe_id(group, "group"))
    if not p.exists():
        raise HTTPException(404, f"no data for group {group}")
    return data.load_file(p)


@app.get("/api/groups")
def groups():
    out = []
    for p in sorted(config.DATA_DIR.glob("*.json")):
        g = p.stem
        if g == "dev_raw":
            continue
        res = store.results_path(g)
        out.append({"group": g, "dialogs": len(data.load_file(p)), "evaluated": res.exists()})
    return out


@app.post("/api/fetch")
def fetch(body: dict = Body(...)):
    g = safe_id(body.get("group_id"), "group")
    try:
        recs = data.fetch_group(g)
    except Exception as e:
        raise HTTPException(502, f"fetch failed: {e}") from e
    return {"group": g, "dialogs": len(recs)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), group: str = "UPLOAD"):
    group = safe_id(group, "group")
    try:
        obj = json.loads((await file.read()).decode("utf-8"))
        recs = data.load_json(obj)
    except Exception as e:
        raise HTTPException(400, f"bad JSON: {e}") from e
    _save_group(group, recs)
    return {"group": group, "dialogs": len(recs)}


@app.post("/api/paste")
def paste(body: dict = Body(...)):
    group = safe_id(body.get("group") or "PASTE", "group")
    rec = data.parse_paste(body["text"], body.get("channel", "agent chat"), body.get("language", "RU"))
    p = safe_path(config.DATA_DIR, group)
    existing = json.loads(p.read_text())["data"]["records"] if p.exists() else []
    existing = [r for r in existing if r["conversation_id"] != rec["conversation_id"]] + [rec]
    _save_group(group, existing)
    return {"group": group, "conversation_id": rec["conversation_id"]}


def _save_group(group, recs):
    config.DATA_DIR.mkdir(exist_ok=True)
    clean = [{k: r[k] for k in ("conversation_id", "channel", "language", "audio_url", "transcript") if k in r} for r in recs]
    for r in clean:
        r["transcript"] = [{k: t.get(k) for k in ("timestamp", "speaker", "text")} for t in r["transcript"]]
    atomic_write(safe_path(config.DATA_DIR, group), json.dumps({"data": {"records": clean}}, ensure_ascii=False))


@app.get("/api/dialogs/{group}")
def dialogs(group: str):
    group = safe_id(group, "group")
    return _group_records(group)


# ---------- evaluation jobs ----------
def _run_job(jid, group, ids, fresh, checklist_override, fetch_first):
    j = JOBS[jid]
    try:
        if fetch_first:
            j["status"] = "fetching"
            data.fetch_group(group)
        recs = _group_records(group)
        if ids:
            recs = [r for r in recs if r["conversation_id"] in set(ids)]
        j["status"] = "running"
        j["dialogs"] = len(recs)
        def prog(d, t):
            j["done"], j["total"] = d, t
        before = dict(USAGE)
        dl, meta = Evaluator(use_cache=not fresh).evaluate_many(recs, checklist_override=checklist_override, progress=prog)
        meta["usage"] = {k: USAGE[k] - before.get(k, 0) for k in USAGE}
        meta["cost_usd"] = cost_usd(meta["usage"])
        old = store.load_results(group)
        if old:
            new_ids = {d["conversation_id"] for d in dl}
            dl = [d for d in old["dialogs"] if d["conversation_id"] not in new_ids] + dl
        store.save_results(group, dl, meta)
        j.update(status="done", meta=meta)
    except Exception as e:
        j.update(status="error", error=str(e))
    j["finished_at"] = time.time()


@app.post("/api/evaluate")
def evaluate(body: dict = Body(...)):
    group = safe_id(body.get("group"), "group")
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"id": jid, "group": group, "status": "queued", "done": 0, "total": 0, "started_at": time.time()}
    threading.Thread(target=_run_job, args=(jid, group, body.get("ids"), bool(body.get("fresh")),
                                            body.get("checklist") or None, bool(body.get("fetch"))), daemon=True).start()
    return JOBS[jid]


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    j = dict(JOBS[jid])
    j["elapsed_s"] = round((j.get("finished_at") or time.time()) - j["started_at"], 1)
    return j


# ---------- results / review ----------
def _runs(group):
    """All result files for a group (results/*.json whose doc.group matches), newest first."""
    out = []
    for p in config.RESULTS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if str(d.get("group", "")).lower() != group.lower() or not isinstance(d.get("dialogs"), list):
            continue
        m = d.get("meta") or {}
        out.append({"run": p.stem, "created_at": d.get("created_at", ""), "dialogs": len(d["dialogs"]),
                    "model": m.get("model", ""), "cost_usd": m.get("cost_usd"), "mtime": p.stat().st_mtime})
    return sorted(out, key=lambda r: (r["created_at"], r["mtime"]), reverse=True)


def _load(group, run=None):
    """Load a run with overrides applied. Default: newest run for the group."""
    group = safe_id(group, "group")
    if run:
        run = safe_id(run, "run")
    if not run:
        rs = _runs(group)
        run = rs[0]["run"] if rs else None
    if not run or run == store.results_path(group).stem:
        return store.load_results(group)
    p = safe_path(config.RESULTS_DIR, run)
    if not p.exists():
        return None
    from .evaluator import apply_overrides
    doc = store.upgrade_turns(json.loads(p.read_text()))
    ov = store.load_overrides(group)
    for d in doc["dialogs"]:
        apply_overrides(d, ckl.load(d["checklist_id"]), ov.get(d["conversation_id"], {}))
    doc["run"] = run
    return doc


@app.get("/api/runs/{group}")
def runs(group: str):
    return _runs(safe_id(group, "group"))


@app.get("/api/results/{group}")
def results(group: str, run: str = ""):
    doc = _load(group, run)
    if not doc:
        raise HTTPException(404, "not evaluated yet")
    return doc


@app.put("/api/overrides/{group}/{conv}/{crit}")
def override(group: str, conv: str, crit: str, body: dict = Body(...), run: str = ""):
    group = safe_id(group, "group")
    doc = _load(group, run)
    if not doc:
        raise HTTPException(404, "not evaluated yet")
    d = next((d for d in doc["dialogs"] if d["conversation_id"] == conv), None)
    if d is None:
        raise HTTPException(404, "unknown conversation")
    try:
        store.save_override(group, conv, crit, body.get("value"), body.get("justification", ""), body.get("reviewer", ""),
                            checklist_id=d["checklist_id"])
        doc = _load(group, run)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return next(d for d in doc["dialogs"] if d["conversation_id"] == conv)


@app.post("/api/rescore/{group}")
def do_rescore(group: str):
    group = safe_id(group, "group")
    doc = rescore(store.load_results(group))
    store.save_results(group, doc["dialogs"], doc["meta"])
    return {"ok": True}


@app.get("/api/export/{group}.json")
def export_json(group: str, run: str = ""):
    doc = _load(group, run)
    if not doc:
        raise HTTPException(404, "not evaluated yet")
    return JSONResponse(doc, headers={"Content-Disposition": f'attachment; filename="{group}.json"'})


@app.get("/api/export/{group}/results.json")
def export_v1(group: str, run: str = ""):
    """Submission format v1.0 (cqa/submission.py), validated before download."""
    from . import submission
    doc = _load(group, run)
    if not doc:
        raise HTTPException(404, "not evaluated yet")
    obj = submission.to_v1(doc, source_run=doc.get("run") or run or group.lower())
    errs = submission.validate(obj)
    if errs:
        raise HTTPException(500, "v1 export failed validation: " + "; ".join(errs[:5]))
    return JSONResponse(obj, headers={"Content-Disposition": 'attachment; filename="results.json"'})


@app.get("/api/export/{group}.csv")
def export_csv(group: str, which: str = "ai", run: str = ""):
    if which not in ("ai", "human", "both", "organizer"):
        raise HTTPException(400, "which must be ai|human|both|organizer")
    doc = _load(group, run)
    if not doc:
        raise HTTPException(404, "not evaluated yet")
    txt = store.to_organizer_csv(doc) if which == "organizer" else store.to_csv(doc, which)
    return PlainTextResponse(txt, media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{group}_{which}.csv"'})
