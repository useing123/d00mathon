# HTTP API

Served by `python -m cqa serve --port 8090` (FastAPI, no auth). Interactive docs: `/docs` (Swagger) and `/openapi.json`.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/` | — | Web UI (`cqa/static/index.html`) |
| GET | `/api/health` | — | `{ok, provider, model, fetch_configured}` |
| GET | `/api/checklists` | — | List of checklists `[{id, name, channel, criteria}]` |
| GET | `/api/checklists/{cid}` | — | `{checklist, yaml}`; 404 if unknown |
| PUT | `/api/checklists/{cid}` | checklist JSON, or `{"yaml": "..."}` | `{ok, id}`; 400 on validation error |
| GET | `/api/groups` | — | Loaded groups `[{group, dialogs, evaluated}]` |
| POST | `/api/fetch` | `{"group_id": "ID"}` | Fetches from `CQA_API_URL` → `{group, dialogs}`; 502 on failure |
| POST | `/api/upload?group=NAME` | multipart `file` (JSON) | `{group, dialogs}`; 400 on bad JSON |
| POST | `/api/paste` | `{text, channel?, language?, group?}` — lines `speaker: text` | `{group, conversation_id}` |
| GET | `/api/dialogs/{group}` | — | Normalised dialogs (turns, roles, offsets, internal flags) |
| POST | `/api/evaluate` | `{group, ids?, fresh?, checklist?, fetch?}` | Background job `{id, status, done, total, ...}` |
| GET | `/api/jobs/{jid}` | — | Job progress `{status: queued/running/done/error, done, total, elapsed_s, ...}` |
| GET | `/api/results/{group}` | — | Results document with AI output + `human` block per dialog |
| PUT | `/api/overrides/{group}/{conv}/{crit}` | `{value, justification, reviewer}` (`value: null` removes) | Updated dialog with human total/diff |
| POST | `/api/rescore/{group}` | — | Recompute totals with current weights (no LLM) → `{ok}` |
| GET | `/api/export/{group}.json` | — | Download results JSON |
| GET | `/api/export/{group}.csv` | `which=ai\|human\|both` | Download CSV |

Example (offline):
```bash
CQA_PROVIDER=mock python -m cqa serve --port 8091 &
curl -F file=@examples/sample_dialogs.json 'http://127.0.0.1:8091/api/upload?group=DEMO'
curl -X POST -H 'Content-Type: application/json' -d '{"group":"DEMO"}' http://127.0.0.1:8091/api/evaluate
curl http://127.0.0.1:8091/api/results/DEMO
```
All routes are exercised in `tests/test_e2e_mock.py::test_http_api_end_to_end` (except `/api/fetch`, which needs the organizer URL).
