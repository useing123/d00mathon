# Reviewer UI (`cqa/static/index.html`, served at `/`)

Vanilla JS app without a build step (`index.html` markup, `app.js` logic, `app.css` styles) on top of the HTTP API (`docs/API.md`). No login, by design.

## Screens
| Screen | What a reviewer sees / does |
|---|---|
| **Overview** | Group and run selectors; mean / median total, score histogram, per-criterion failure rates, insights (most failed criterion, gross-error dialogs as chips, **incomplete evaluations**, **dialogs needing human review**, low totals); run metadata (models, cost). |
| **Dialog table** | One row per dialog: channel, language, a badge per criterion (YES/NO/NA/scale step; ⚖ when the judge decided), total (or a red «неполная» badge with failed criteria and provisional total in the tooltip), human total if reviewed. Filters (gross / judge-decided / low / reviewed), sorting by any column, `j`/`k` navigation. |
| **Dialog card** | Transcript with turn numbers, roles, times, internal hold turns greyed; turns cited as evidence are highlighted with criterion tags. Header: total, provisional total for incomplete dialogs, «Неполная оценка» and ⚠-warning tags, gross-error tag, prev/next, re-evaluate this dialog. For calls: audio player, click a turn to seek. |
| **Criterion cards** | Value badge, justification, NA reason / "expected at turn N", verbatim quotes (click → scroll to the turn), jury chips (each juror's vote, judge decision), evidence warnings, «требует проверки» for an unconfirmed gross flag, evaluation error text, and for 04 the client-request ledger (✓ closed / ◐ partial / ✗ open). |
| **Human override** | Per criterion: value (only allowed values offered; the server also rejects invalid ones with HTTP 400), justification, reviewer. Stored separately from AI output; human total and AI→Human diff recomputed; restore AI value. |
| **Checklist editor** | Form or YAML editor for criteria (type, scale, weight, NA, rubric, rules); validation errors from the server are shown; reweighting → rescore without LLM. |
| **Data in** | Fetch a group from the organizer API, upload JSON (evaluation starts automatically), paste a dialog as text; in-page dialogs instead of browser prompts; background evaluation with a progress bar. |
| **AI↔human agreement** | Overview card: share of reviewed criteria where the human kept the AI value, per criterion. |
| **Export** | JSON, CSV AI / Human / both (CSV contains `status`, `failed_criteria`, `needs_review`, `provisional_total`). |

## Automated browser check
`python scripts/ui_smoke.py http://127.0.0.1:8090 DEV` (Playwright, headless Chromium). Output of the last run
against the DEV results (2026-11-07):
```json
{
 "table_rows": 65,
 "criteria_cards": 8,
 "jury_chips": 16,
 "incomplete_badge": 1,
 "provisional_shown": true,
 "ask_modal_visible": true,
 "invalid_override_status": 400,
 "console_errors": []
}
```
The script injects an incomplete dialog client-side to check the badge and provisional total rendering, and
sends an invalid override (`04 = 4.5`) to check the server rejects it.
