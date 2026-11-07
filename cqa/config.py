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


def get(key, default=None):
    return os.environ.get(key, default)


PROVIDER = get("CQA_PROVIDER", "anthropic")          # anthropic | openai
MODEL = get("CQA_MODEL", "claude-sonnet-4-5-20250929")
OPENAI_BASE_URL = get("CQA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")  # Ollama default
OPENAI_API_KEY = get("CQA_OPENAI_API_KEY", "ollama")
CONCURRENCY = int(get("CQA_CONCURRENCY", "24"))
