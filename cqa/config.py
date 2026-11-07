"""Config: reads /root/cqa/.env style file (KEY=VALUE, optional quotes) + environment."""
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
