# Architecture

```mermaid
flowchart TB
  subgraph Input
    API[Organizer API<br/>CQA_API_URL&group_id=ID]
    FILE[JSON file / upload]
    PASTE[Pasted text]
  end
  API & FILE & PASTE --> DATA[data.py<br/>normalize: roles, turn index,<br/>hold/consultation marking, audio offsets, dialog hash]
  DATA --> CKL[checklist.py<br/>pick by channel, validate, criterion hash]
  CKL --> EV[evaluator.py<br/>ThreadPool over dialog × criterion]
  EV --> PR[prompts.py<br/>system prompt (cached) + one criterion block]
  PR --> LLM[llm.py]
  LLM -->|anthropic| A[Claude tool use]
  LLM -->|openai| O[Ollama / vLLM JSON mode]
  LLM -->|mock| M[MockBackend heuristic]
  EV --> VAL[validate_evidence<br/>fix turn / retry / reject]
  EV --> CACHE[(.cache/ per-criterion)]
  EV --> SC[scoring.py]
  SC --> ST[store.py<br/>results/&lt;group&gt;.json + .csv]
  OV[(results/overrides/&lt;group&gt;.json)] --> ST
  ST --> SRV[server.py FastAPI] --> UI[static/index.html]
  CLI[__main__.py] --> EV
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Reads `.env` + env vars: provider, models, jury, fallback models, blind judge, confirm/final profile, paths, concurrency. |
| `data.py` | Accepts API envelope / list / single record; normalises speaker aliases to `operator`/`client`/`bot`/`system` (unknown → `unknown` + warning), parses mm:ss / hh:mm:ss / ISO-with-tz timestamps, marks operator turns during hold/consultation as internal, computes audio offsets, hashes dialog content incl. timestamps, and computes deterministic FACTS (first operator turn, client-language heuristic, longest wait after the first reply and whether it was announced, holds, ending). |
| `checklist.py` | Loads/validates YAML checklists (unique ids, integer scale steps, finite non-negative weights, id allowlist), resolves `extends` + `criteria_overrides` (the call checklist inherits the chat one and overrides 06), saves edits, picks checklist by channel, hashes each criterion's text + prompt content for the cache. |
| `prompts.py` | System prompt (role rules, evidence format, untrusted-data boundary) + one criterion block (with its allowed values) + ONE stable tool schema shared by all criteria (`value` is a string; exact values enforced by `checklist.strict_value` + retry). The transcript block (delimited, ASR caveat for calls, code-computed FACTS) is sent as a cacheable prefix. |
| `llm.py` | `LLM.call()` → dict. Anthropic tool use (forced `tool_choice` for Haiku, `auto` for Sonnet/Opus 5-5 which reject forced choice), OpenAI-compatible JSON mode, or offline `MockBackend`. Token/cost accounting. Cloud routing by model id (`cloud_route`): Anthropic, OpenAI Responses API, Gemini OpenAI-compatible endpoint. Retries 429/5xx with Retry-After + jitter; actionable 404 for unknown model ids. |
| `evaluator.py` | Per criterion: jury (parallel jurors, judge on disagreement; a failing juror → judge reviews the surviving verdict; all jurors fail → judge alone) or single model with self-consistency on low confidence; strict value check + evidence validation (verbatim, operator quotes, 08 needs-review) with retry; dialog-major batches for prompt-cache hits; cache read/write; fail-closed aggregation (`incomplete`, `total: null`) and score. |
| `scoring.py` | Pure deterministic scoring (see [SCORING.md](SCORING.md)). |
| `store.py` | Results JSON (atomic writes), overrides kept separately, human totals and AI→Human diff, CSV export (ai / human / both), organizer submission CSV + diagnostics CSV. |
| `server.py` | FastAPI routes (see [API.md](API.md)), background jobs with progress. |
| `__main__.py` | CLI `evaluate` (`--final`, `--confirm`, `--organizer`), `stats`, `rescore`, `serve`, `gold` (`--dir`, `--labels`), `label` (human labelling of real dialogs), `export` (organizer CSV). |
| `gold.py` | Gold / held-out / human-label agreement metrics (per criterion, MAE, gross P/R, NA confusion). |
| `util.py` | Atomic writes, identifier allowlist + path containment. |
| `pricing.py` | Per-model price table used for cost reporting. |

## Design decisions
- **One criterion per call** — narrow context and a criterion-specific decision procedure beat one big “score everything” prompt; it also makes per-criterion caching and re-evaluation possible.
- **LLM decides, code scores** — models output only the categorical value + evidence; arithmetic, NA redistribution and rounding are in tested Python.
- **Evidence is verified, not trusted** — quotes must be substrings of the cited turn.
- **Process correctness** — there are no stored answers; the cache key includes the dialog content hash, so any new transcript always goes to the model. `--fresh` disables the cache entirely.
- **Provider-agnostic** — the same prompts/schema run on Claude, a local model, or the mock backend.
