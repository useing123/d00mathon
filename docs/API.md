# HTTP API

Served by `python -m cqa serve --port 8090` (FastAPI, no auth). Interactive docs: `/docs` (Swagger) and `/openapi.json`.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/` | — | Web UI (`cqa/static/index.html`) |
| GET | `/static/app.js`, `/static/app.css` | — | UI script / styles (only these two names) |
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
| GET | `/api/runs/{group}` | — | Result files for the group, newest first: `[{run, created_at, dialogs, model, cost_usd, mtime}]` |
| GET | `/api/results/{group}` | `run=<name>` (optional; default = newest run) | Results document with AI output + `human` block per dialog |
| PUT | `/api/overrides/{group}/{conv}/{crit}` | `{value, justification, reviewer}` (`value: null` removes); `?run=` optional | Updated dialog with human total/diff; **400** if the value is not an allowed step/`YES`/`NO`/`NA` for that criterion, 404 for unknown run/dialog |
| POST | `/api/rescore/{group}` | — | Recompute totals with current weights (no LLM) → `{ok}` |
| GET | `/api/export/{group}.json` | `run=` optional | Download results JSON |
| GET | `/api/export/{group}.csv` | `which=ai\|human\|both`, `run=` optional | Download CSV |

Identifiers that become file names (`group`, `run`, checklist `cid`) must match `[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}` without `..`, and the resolved path must stay inside its directory → otherwise **400** (`tests/test_review_hardening.py::test_path_traversal_rejected`).

Result dialogs carry `status` (`complete`/`incomplete`), `failed_criteria`, `provisional_total`, `needs_review` (criterion ids) and `warnings`; criterion results may carry `evidence_warning`, `needs_review`, and `jury` keyed `"<index>:<model>"`.

Example (offline):
```bash
CQA_PROVIDER=mock python -m cqa serve --port 8091 &
curl -F file=@examples/sample_dialogs.json 'http://127.0.0.1:8091/api/upload?group=DEMO'
curl -X POST -H 'Content-Type: application/json' -d '{"group":"DEMO"}' http://127.0.0.1:8091/api/evaluate
curl http://127.0.0.1:8091/api/results/DEMO
```
All routes are exercised in `tests/test_e2e_mock.py::test_http_api_end_to_end` (except `/api/fetch`, which needs the organizer URL).

## Submission files (CLI)
`python -m cqa evaluate --group <ID> --organizer out.csv` writes:
- `out.csv` — only scored columns: `conversation_id, 01..08, total, gross_error` (criterion values `YES`/`NO`/`NA` or the
  scale step, total with 2 decimals, gross_error `YES`/`NO`). The organizer's exact column spec is not published to us;
  this layout is an assumption. Dialogs still incomplete after the rescue pass have an empty total unless
  `--fill-provisional` is given.
- `out.diagnostics.csv` — status, failed / needs-review criteria, provisional total, warnings.
