# Evaluation

## Current version (v7) — numbers used in the README

Code: rubric aligned with the official checklist text (docs/ASSUMPTIONS.md, “Rubric alignment”), wait threshold
180 s from the checklist YAML, judge decides scale criteria (04, 06) by the **median of 3 samples**, `needs_review`
flags and per-dialog `stability`. Preset `jury` (jurors claude-haiku-5-5 + claude-sonnet-5-5, judge claude-opus-5-5).

**DEV** (`results/dev_jury_v7.json` → `results/dev/results.json`): 65 dialogs, 0 errors, 0 incomplete, mean **80.41**
(v6: 78.45), 5 gross flags (the same 5 dialogs), 294 s, **$10.28 = $0.158 per dialog** (1285 API calls, current
`cqa/pricing.py` table; v6 was $7.64 — the extra cost is the 3 judge samples on scale criteria). Decided by:
consensus 444, blind judge (binary) 41, judge with juror opinions 30 (29 of them scale criteria, 3 samples each),
judge-confirmed 08=YES 5.

Value distribution, v6 → v7 (one run each; changed verdicts: **34 of 520** — 06 ×11, 04 ×10, 07 ×6, 02 ×3, 03 ×2, 05 ×2):

| Crit. | v6 (before) | v7 (after) | Main transitions |
|---|---|---|---|
| 01 | YES 60 / NO 5 | YES 60 / NO 5 | — |
| 02 | NA 43 / YES 15 / NO 7 | NA 43 / YES 14 / NO 8 | 3 changes, no direction |
| 03 | NA 53 / YES 8 / NO 4 | NA 53 / YES 8 / NO 4 | 2 swaps (NA↔NO) |
| 04 | 20: 37 / 15: 11 / 10: 9 / 5: 2 / 0: 5 / NA 1 | 20: 42 / 15: 10 / 10: 4 / 5: 3 / 0: 5 / NA 1 | 15→20 ×5, 10→15 ×4 (next step = 20 without a deadline) |
| 05 | YES 52 / NO 13 | YES 54 / NO 11 | NO→YES ×2 |
| 06 | 10: 45 / 7: 12 / 4: 8 | 10: 53 / 7: 3 / 4: 9 | 7→10 ×8 (no “extra template” glitch), 7→4 ×2 (re-ask = 4) |
| 07 | NO 49 / YES 11 / NA 5 | NO 45 / YES 13 / NA 7 | NO→YES ×3, NO→NA ×2, YES→NO ×1 |
| 08 | NO 60 / YES 5 | NO 60 / YES 5 | — |

07 stays mostly NO after alignment: most DEV operators end with a bare farewell («Хорошего дня!») with neither a
summary nor an offer of help, which is NO under the literal official text. 02 = NA follows the official N/A condition.

**Variance (measured in this run).** 29 scale verdicts were decided by the judge with 3 samples: 26 unanimous, 3 split
by one step (0/4/4, 20/15/15, 7/4/4) → the median was used; i.e. a single judge sample would have differed from the
median in ≈ 3 of 29 cases. Per-dialog `stability` (field in the internal results): share of criteria where every vote
(jurors + judge samples) agreed — mean **0.863**; score range obtained by replacing each final value with any value a
model actually voted for — spread mean **11.3** points, median 7.7, max 47.8, 0 in 23/65 dialogs. The range is a
worst case over model disagreement, not a repeated-run interval. Repeated full runs of the held-out set earlier gave
0.925–0.938 (see HISTORY); this run 0.95.

**Review flags.** `needs_review` on **95 of 520 criteria (18.3%)** in **49 of 65 dialogs (75%)**; reasons (a criterion
can have several): judge decided 76, jurors disagreed 71, low-confidence scale value (< 0.7) 36, judge samples split 3.
By criterion: 04 ×31, 07 ×23, 06 ×17, 02 ×8, 05 ×6, 03 ×5, 08 ×5 (every 08=YES is judge-confirmed and flagged).
`needs_review` is in the internal results/UI only; the v1.0 export schema is unchanged.

**Held-out regression** (`examples/gold_holdout`, 10 synthetic dialogs, run once, nothing tuned on it):
**0.95 (76/80)**, total MAE **4.43**, gross P/R 1.0 / 1.0, $1.14. All 4 disagreements are in hold-10 (contradictory
answers): 02 NA→NO, 04 5→0, 05 YES→NO, 06 4→0. The tuned gold set was not re-run for this version.

**Call metrics** (informational, not scored): operator words/min, estimated overlap rate and operator response pauses
from ISO turn start times (`call_metrics` in the internal results; added to `dev_jury_v7.json` after the run with
`data.call_metrics`, no LLM). DEV calls: 77–145 wpm, median response 3.0–9.3 s, max 432 s (p4-ru-002). Overlap is an
estimate (turn duration unknown; assumes 2.5 words/s) — 31–49% of speaker changes, likely overstated.

History of earlier versions: [HISTORY.md](HISTORY.md).


## Gold regression set (labelled, reproducible)
`examples/gold/*.json` — **16 synthetic dialogs** (RU 8 / KZ 3 / EN 5; 14 chats + 2 calls) written by us to cover the
edge cases of the organizer rubric: bot greets then operator joins, greeting in the wrong language, general tariff
question (03 NA), balance with/without verification, multi-question request partially answered (04=10), long
pause/hold without warning (06=7), client leaves before formulating a request (04 NA, 07 NA), farewell without
summary/offer (07 NO), complaint with/without empathy, private paid service and transfer to a personal card
(08 YES), insult (08 YES), and a prompt-injection attempt inside the transcript. Each file has hand-assigned values
for all 8 criteria, a one-line rationale for non-obvious labels and `expected_total` computed by `cqa/scoring.py`
(checked by `tests/test_gold.py`).

```bash
python -m cqa gold          # real models → results/gold_report.json
python -m cqa gold --mock   # offline plumbing check (used in CI; not an accuracy number)
```
Reported: per-criterion exact-match agreement, overall agreement, total MAE, 08 gross-flag precision/recall,
NA-vs-applicable confusion, list of disagreements.

## Held-out split (not used to write rules; see contamination note below)
`examples/gold_holdout/` — 10 further synthetic dialogs (RU 4 / KZ 2 / EN 4 incl. 2 calls) written after the tuning
rounds, labelled strictly by the organizer rubric, and **not used to change any rule**. Scenarios: needed
clarification asked, protected action (phone change) without ID, greeting without a name, secondary detail missed,
angry complaint handled well, wrong key recommendation, call ending before the request, third-party data
disclosure, long pause without warning, contradictory answers. Caveat: written by the same team (agent) that wrote
the rules, so it is less independent than real labelled data.

| Metric (prompt v8, jury) | Tuned set (16) | **Held-out (10)** |
|---|---|---|
| Overall exact agreement | 0.992 | **0.925** (74/80) |
| 01 / 02 / 03 / 04 | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 0.9 / 1.0 / 0.8 |
| 05 / 06 / 07 / 08 | 1.0 / 0.938 / 1.0 / 1.0 | 0.9 / 0.8 / 1.0 / 1.0 |
| Total MAE | 0.43 | **7.32** |
| Gross flag P / R | 1.0 / 1.0 | 1.0 / 1.0 (1 TP) |
| Cost / time | $2.45 / 55 s | $1.77 / 47 s |

Held-out disagreements: hold-10 (contradictory answers) — model is harsher on 04 (0 vs 5), 06 (0 vs 4), 05, 02;
hold-02 04 = 0 vs 20 (the model treats changing the phone without ID as a wrong key action, which our label keeps in
03 only); hold-08 06 = 7 vs 10. The gap between 0.992 and 0.925 is the honest estimate of how much the tuned-set
number is inflated; real agreement with the organizer's labels may be lower still.

### Human labels for real DEV dialogs
`python -m cqa label --group DEV -n 10 --labeller NAME` shows each transcript and asks for the 8 values (only
allowed values accepted), saving to `examples/dev_labels.json` (git-ignored: real data). Then
`python -m cqa gold --labels examples/dev_labels.json --group DEV` reports the same metrics on those dialogs.
We did **not** create these labels ourselves — they need a human labeller.

### Optional: judge re-checks unanimous verdicts on 03/04 (`--confirm 03,04`)
By default a unanimous haiku+sonnet verdict is final (except 08=YES, which the judge always confirms).
`python -m cqa evaluate … --confirm 03,04` (or `CQA_CONFIRM=03,04`) sends those verdicts to opus too.

| | default | `--confirm 03,04` |
|---|---|---|
| Tuned set: agreement / MAE / cost / time | 0.992 / 0.43 / $2.45 / 55 s | 0.992 / 0.43 / $2.98 / 63 s |
| Held-out: agreement / MAE / cost / time | 0.925 / 7.32 / $1.77 / 47 s | 0.938 / 6.88 / $1.92 / 46 s |
| 03 / 04 agreement on held-out | 1.0 / 0.8 | 1.0 / 0.8 |
On these sets it changed no 03/04 verdict (the held-out difference is in 02, i.e. run-to-run noise) and cost +8–22%.
We recommend it for a final control run where cost is secondary, but cannot claim an accuracy gain from it.

## Prompt-cache measurement (3 DEV dialogs, `--fresh`, jury)
| | Per-criterion tool schema (v2) | One stable tool schema + dialog-major batches (v3) |
|---|---|---|
| cache_read input tokens | 24,708 | **120,955** |
| cache_creation input tokens | 78,763 | 20,711 |
| uncached input tokens | 39,965 | 50,101 |
| calls / cost | 50 / $0.608 | 55 / $0.605 |
The cached prefix (tool → system → transcript) is now shared by all 8 criteria. Cost is roughly flat because output
tokens dominate and the new evidence checks cause a few extra retries (55 vs 50 calls).

## Manual review method
DEV has no labelled ground truth, so accuracy was checked by hand on **8 dialogs** covering chats in RU/KZ/EN, calls, high and low totals, gross-error flags and jury disagreements:
1. Open each dialog in the UI; for each of the 8 criteria read the verdict, justification and highlighted evidence.
2. Check the quote is the right evidence (not only that it exists — that is verified automatically).
3. Where we disagree, record a Human override with a justification (stored in `results/overrides/`, AI output untouched).
4. Export CSV `both` to see AI vs Human per criterion and total; recurring disagreements were turned into explicit rules in the YAML (which re-evaluates only the edited criteria thanks to the criterion hash).

## Known disagreement areas
| Criterion | Why it is hard | Mitigation |
|---|---|---|
| 07 Closing | Where a dialog “ends” (client leaves, auto-close) and how strict a farewell must be. | Strict reading + explicit NA rule (assumption 2). |
| 04 Consultation (scale) | Adjacent steps (10 vs 15) depend on completeness judgement. | Step-by-step rubric per scale value; judge resolves juror splits. |
| 05 Empathy | Subjective; formal politeness vs real acknowledgement of emotion. | Rules on when empathy is required vs NA. |

## Offline check
`make demo` / `python -m cqa evaluate --mock --input examples/sample_dialogs.json` runs the whole pipeline on synthetic dialogs with a deterministic heuristic backend — it validates plumbing (I/O, evidence validation, scoring, export), **not** accuracy.
