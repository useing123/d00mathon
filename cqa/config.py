"""Config: reads <repo>/.env (KEY=VALUE, optional quotes) + environment variables (env wins)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKLIST_DIR = ROOT / "checklists"
CACHE_DIR = ROOT / ".cache" / "criteria"


def _load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


_load_env()


def get(key: str, default: str | None = None) -> str | None:
    """Read a setting from the environment (``.env`` values are loaded as defaults)."""
    return os.environ.get(key, default)


PROVIDER = get("CQA_PROVIDER", "anthropic")          # anthropic | openai | mock (offline heuristic)
MODEL = get("CQA_MODEL", "claude-sonnet-5-5")
# Ensemble ("jury") mode: two jurors vote in parallel, the frontier judge adjudicates disagreements.
ENSEMBLE = get("CQA_ENSEMBLE", "1") == "1"
JURORS = [m.strip() for m in get("CQA_JURORS", "claude-haiku-5-5,claude-sonnet-5-5").split(",") if m.strip()]
JUDGE = get("CQA_JUDGE", "claude-opus-5-5")
OPENAI_BASE_URL = get("CQA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")  # Ollama default
OPENAI_API_KEY = get("CQA_OPENAI_API_KEY", "ollama")
CONCURRENCY = int(get("CQA_CONCURRENCY", "24"))
BATCH_DIALOGS = int(get("CQA_BATCH_DIALOGS", "6"))  # dialog-major batches keep prompt-cache hits within TTL
JURORS_EXPLICIT = get("CQA_OPENAI_JURY", "0") == "1"  # opt-in jury on local/OpenAI-compatible models (CQA_JURORS/CQA_JUDGE = local ids)
OPENAI_TOOLS = get("CQA_OPENAI_TOOLS", "1") == "1"  # use native tool calling on OpenAI-compatible servers
CONFIRM = get("CQA_CONFIRM", "")  # e.g. "03,04": judge re-checks unanimous jury verdicts on these criteria (08=YES always)


def confirm_criteria():
    return {x.strip() for x in CONFIRM.split(",") if x.strip()}
# replacements for configured ids missing at the provider (checked by Evaluator.preflight), strongest first
FALLBACK_MODELS = [m.strip() for m in get("CQA_FALLBACK_MODELS", "claude-sonnet-4-6,claude-opus-4-7,claude-haiku-4-5-20251001").split(",")
                   if m.strip()]
SCALE_SAMPLES = int(get("CQA_SCALE_SAMPLES", "1"))   # >1: each juror's verdict on SCALE_SAMPLE_CRITERIA = median of N samples
SCALE_SAMPLE_CRITERIA = {x.strip() for x in get("CQA_SCALE_SAMPLE_CRITERIA", "04,06").split(",") if x.strip()}

# `evaluate --final`: control-run profile (stronger jury, judge re-checks unanimous verdicts, median of 3 on 04/06)
FINAL_PROFILE = {"JURORS": ["claude-sonnet-5-5", "claude-opus-5-5"], "JUDGE": "claude-opus-5-5",
                 "CONFIRM": "02,04,06", "SCALE_SAMPLES": 3}


def apply_final_profile():
    global JURORS, JUDGE, CONFIRM, SCALE_SAMPLES
    JURORS, JUDGE = list(FINAL_PROFILE["JURORS"]), FINAL_PROFILE["JUDGE"]
    CONFIRM, SCALE_SAMPLES = FINAL_PROFILE["CONFIRM"], FINAL_PROFILE["SCALE_SAMPLES"]
BLIND_JUDGE = get("CQA_BLIND_JUDGE", "1") == "1"   # on juror disagreement the judge first decides without seeing them

# Model preset (cqa/presets.py, chosen from the benchmark matrix — docs/MODELS.md). Applied as the default only
# when no explicit CQA_MODEL / CQA_ENSEMBLE / CQA_JURORS / CQA_JUDGE is configured; CQA_PRESET forces one.
PRESET = get("CQA_PRESET", "")


def _apply_default_preset():
    from . import presets
    explicit = any(os.environ.get(k) for k in ("CQA_MODEL", "CQA_ENSEMBLE", "CQA_JURORS", "CQA_JUDGE"))
    if PROVIDER == "anthropic" and (PRESET or not explicit):
        presets.apply(PRESET or presets.DEFAULT, globals())


_apply_default_preset()
