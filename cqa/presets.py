"""Model presets chosen from the benchmark matrix (results/bench_matrix.json, docs/MODELS.md).

Selection rule for ``max`` (optional preset; the product default is ``jury``): highest exact agreement on the HELD-OUT gold set (mean of
repeated runs), tie-break lower total MAE, then higher gross recall, then lower $/dialog.
A preset only sets the model setup (single model or jury) and the prompt variant; checklists, evidence
validation and scoring are the same for every preset. Explicit CQA_MODEL / CQA_ENSEMBLE / CQA_JURORS /
CQA_JUDGE in the environment override the default preset.
"""
import importlib
import os

PRESETS = {
    "econom": {"label": "gpt-6-luna · few-shot (v1)", "bench_config": "gpt-6-luna@v1",
               "ensemble": False, "model": "gpt-6-luna", "prompt": "v1",
               "why": "Граница Парето: ~20× дешевле max, на 1 оценку из 80 ниже на отложенном (1 прогон)."},
    "balanced": {"label": "claude-sonnet-5-5 (только Anthropic)", "bench_config": "sonnet",
                 "ensemble": False, "model": "claude-sonnet-5-5", "prompt": "v0",
                 "why": "Один вендор (Anthropic), без ключа OpenAI. Не на границе Парето — выбор по требованиям к поставщику."},
    "max": {"label": "gpt-5.3-codex", "bench_config": "gpt-5.3-codex",
            "ensemble": False, "model": "gpt-5.3-codex", "prompt": "v0",
            "why": ("Лучший на отложенном наборе (2 прогона), но отрыв 2 метки из 80 в пределах шума; "
                    "не прогонялся на всём DEV. Опционально.")},
    "jury": {"label": "жюри haiku+sonnet → opus (по умолчанию)", "bench_config": "jury-haiku+sonnet>opus", "default": True,
             "ensemble": True, "jurors": ["claude-haiku-5-5", "claude-sonnet-5-5"], "judge": "claude-opus-5-5", "prompt": "v0",
             "why": ("Режим по умолчанию: им сделан results/dev/results.json и им же оцениваются контрольные диалоги. "
                     "Различия с max в пределах шума отложенного набора.")},
}
DEFAULT = "jury"


def apply(name, cfg):
    """Apply preset ``name`` to a config namespace (``vars(cqa.config)`` or its globals())."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose one of {', '.join(PRESETS)}")
    p = PRESETS[name]
    cfg["PROVIDER"] = "anthropic"   # "cloud" mode: the vendor is picked per model id (cqa.llm.cloud_route)
    cfg["ENSEMBLE"] = p["ensemble"]
    if p["ensemble"]:
        cfg["JURORS"], cfg["JUDGE"] = list(p["jurors"]), p["judge"]
    else:
        cfg["MODEL"] = p["model"]
    cfg["PRESET"] = name
    os.environ["CQA_PROMPT_VARIANT"] = p.get("prompt", "v0")
    import sys
    if "cqa.prompts" in sys.modules:   # re-apply the prompt variant on the base prompts
        importlib.reload(sys.modules["cqa.prompts"])
    return p
