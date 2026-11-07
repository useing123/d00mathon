# Assumptions

All are explicit and most are configurable in the checklist YAML (`assumptions:` block or rubric/rules text). Changing them changes the criterion hash, so affected criteria are re-evaluated.

| # | Topic | Assumption | Rationale | Where |
|---|---|---|---|---|
| 1 | Criterion 01 “first reply” | The first reply of the **human operator** (`speaker=operator`), even if a bot answered earlier. A bot greeting does not count. Two consecutive operator messages without a client reply in between are treated together. | The checklist evaluates the operator, not the bot. | `assumptions.first_reply: first_human_operator_turn`, rules of 01 |
| 2 | Criterion 07 closing | Scored **strictly**: the operator must close properly (farewell / offer further help). NA only when the client or system ended the dialog with no chance for the operator to close. | Mirrors organizer rubric; lenient reading inflated scores. | rubric/rules of 07 |
| 3 | First-response time | The 3-minute first-response SLA, tags and attachments are **not scored**; queue waiting before the operator’s first reply does not lower 06. | Not observable reliably / not in the scored criteria. | rules of 06 |
| 4 | Calls (organizer note) | Separate checklist `freedom_call_v1`. Criterion 06 assesses logic, clarity and hold management (warned before hold, thanked for waiting) instead of written literacy. In all criteria (incl. 04 consultation) ASR recognition errors are ignored and not held against the operator. | Transcripts are noisy ASR. | `checklists/freedom_call_v1.yaml` |
| 5 | Internal hold turns | Operator turns between “consultation initiated / client on hold” and “client no longer on hold” are internal agent talk; marked `[ВНУТРЕННИЕ ПЕРЕГОВОРЫ НА УДЕРЖАНИИ]` and not judged as client communication. | Client cannot hear them. | `assumptions.ignore_internal_hold_turns`, `cqa/data.py` |
| 6 | Audio offset | ISO timestamps → offset = turn time − first turn time; used to seek the audio player. Approximate: the recording may not start exactly at the first transcript turn. | No alignment metadata in input. | `cqa/data.py` |
| 7 | Rounding | `Decimal` ROUND_HALF_UP, 2 decimals. | Avoid float/banker’s rounding surprises (50.625 → 50.63). | `assumptions.rounding/decimals` |
| 8 | Languages | One checklist for RU/KZ/EN; criterion 01 requires greeting in the **client’s** language. | Organizer spec. | rules of 01 |
