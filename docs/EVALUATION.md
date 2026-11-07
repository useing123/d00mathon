# Evaluation

## Current version (prompt v9) — numbers used in the README
| Set | Exact agreement | Total MAE | Gross P/R | Cost / time |
|---|---|---|---|---|
| Tuned synthetic gold (16) | 0.992 (127/128) | 0.43 | 1.0 / 1.0 | $1.16 / 60 s (partly cached) |
| Held-out synthetic (10) | 0.938 (75/80) | 3.86 | 1.0 / 1.0 (1 TP) | $0.87 / 42 s (partly cached) |
| Held-out, `--final` profile | 0.925 (74/80) | 6.55 | 1.0 / 1.0 | $3.45 / 78 s (fresh) |
| Real DEV, human labels | not measured | | | needs `cqa label` |

DEV (`results/dev_jury_v5.json`, prompt v9): 65 dialogs, 0 errors, 0 incomplete, 0 rescued, mean **78.53**, 5 gross
flags (the same 5 dialogs as in every version), judge decided 78 of 520 verdicts, 37 evidence warnings, 02 = NA in
44/65, 07 = NO 47 / YES 12 / NA 6. Changed verdicts vs v1: 40 (06 ×13, 04 ×12, 02 ×9, 05 ×4, 07 ×2); vs v4: 19.
Two full runs were made for v5 ($13.89, then $12.14 — the second was meant to re-evaluate only 07 after the 07 rule
change, but adding the scale-sampling setting to the cache key invalidated every entry).

**Held-out contamination note:** after the first held-out run (0.925), a reviewer pointed at hold-02 (04 = 0 for an
action done without ID) and we added the general rule "missing identification is a 03 issue, not a wrong 04
recommendation". That fixed hold-02, so the held-out set is no longer fully untouched for criterion 04. No other rule
was changed in response to held-out results.

**`--final` profile:** jurors sonnet-5-5 + opus-5-5, judge re-checks unanimous 02/04/06, median of 3 samples per juror
on 04/06. On the held-out set it was not better (0.925 vs 0.938, MAE 6.55 vs 3.86) at ~2–4× the cost; it stays opt-in.

## History

### DEV set results (jury mode, v1 — before the review changes)
Real run on the organizer DEV group (raw results are git-ignored because they contain client text):

| Metric | Value |
|---|---|
| Dialogs | 65 (60 chats: RU 30 / KZ 15 / EN 15; 5 RU calls) |
| Errors | 0 |
| Mean total | 75.95 |
| Gross-error flags | 5 |
| Criterion verdicts | 520 (65 × 8) |
| Jurors (haiku-5-5 + sonnet-5-5) agreed | 84.4% |
| Decided by judge (opus-5-5) | 81 |
| Wall time / cost, full set | ~5 min / ~$13 |
| Wall time / cost, 10 dialogs | ~1 min / ~$2 |

Reproduce: `python -m cqa evaluate --group DEV` then `python -m cqa stats --group DEV`.

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

### Real results (jury haiku-5-5 + sonnet-5-5, judge opus-5-5; 2026-11-07)
| Metric | Before tuning | After one 02 rule change |
|---|---|---|
| Overall exact agreement (128 verdicts) | 0.945 | **0.969** |
| 01 / 03 / 04 / 07 / 08 | 1.0 each | 1.0 each |
| 02 Clarification | 0.812 (13/16) | **1.0** |
| 05 Empathy | 0.875 | 0.875 |
| 06 Dialog management | 0.875 | 0.875 |
| Total MAE (points) | 5.45 | **3.64** |
| 08 gross flag precision / recall | 1.0 / 1.0 (3 TP) | 1.0 / 1.0 |
| NA predicted for applicable / applicable for NA | 0 / 3 | 0 / 0 |
| Cost / wall time | $2.45 / 68 s | $0.27 / 20 s (only 02 re-evaluated, criterion cache) |

### Later rounds (after review round 1; still the same 16 dialogs, nothing dialog-specific)
| Change | Agreement | Total MAE | Remaining disagreements |
|---|---|---|---|
| + 06 rule: a skipped question counts only in 04, not 06 | 0.969 | 3.41 | 06 gold-05, 06 gold-13, 05 gold-12, 05 gold-16 |
| + code-computed FACTS block (first operator turn, language heuristic, longest wait & warning, holds, ending), 01 naming examples, 06 literacy rule (prompt v4, full re-run, $2.30) | 0.992 | 1.44 | 05 gold-12 (model treats "takes too long" as a complaint needing acknowledgement) |
| + round-2 changes: 04 request ledger (`client_requests`, consistency check), 05 boundary (incidental inconvenience inside a how-to question needs no apology), 03/07 rule additions, better FACTS (multi-message first reply, warning window, ack-only gaps excluded, auth marker) (prompt v5, full re-run, $2.54) | **0.992** | **0.43** | 06 gold-13 (7 vs 10; model docks 06 for the rude repetition — arguably a defensible reading) |

| + round-3 changes: rescue pass, judge confirms unanimous 08=YES, two-way ledger consistency, internal hold turns excluded from operator evidence, 01 mixed KZ/RU, 03 "ID done, nothing disclosed" = YES, 07 "offer asked, client silent" = NA (prompt v6) | 0.969 / **0.984** (two identical runs) | 1.71 / 0.91 | 04 gold-11 (5 vs 0) in both; 04 gold-10, 06 gold-05/13 in one run only |

**Run-to-run noise:** two `--fresh` runs of the identical v6 code differed by 2 of 128 verdicts (0.969 vs 0.984) —
Anthropic sampling is provider default (the endpoint rejects `temperature`). Differences of ±2 verdicts on this set
are noise, not signal; that is also why we stopped tuning on it.

Note: the 05 boundary rule resolved gold-12; we report it, but treat it with suspicion as it was suggested by a reviewer after seeing that mismatch. The gold set is now saturated and no longer discriminative — new, harder labelled cases (ideally real dialogs labelled from the organizer's checklist text, `python -m cqa gold --dir <private dir>`) are the next step.

**Tuning (one change, generic, not dialog-specific):** before tuning, all three 02 errors were `NA→NO`, and the
justifications penalised the *answer* (an ignored question, a private paid offer) under 02. We added one rule to 02:
it judges only the pre-answer stage; answer quality belongs to 04/08; if the client's data were sufficient → NA.
Remaining disagreements (not tuned, to avoid overfitting): 06 on gold-05 (7 vs 10) and gold-10 (4 vs 10 — the client
re-asked an ignored question, which the model attributes to 06; defensible either way) and 05 on gold-12/gold-16
(model treats "takes too long" / "why is my account blocked??" as a complaint that needed acknowledgement).

**Caveat (honest):** the gold labels were written by the same team that wrote the rubric rules, on synthetic
dialogs with clear-cut cases. This is a regression/consistency check, not an estimate of agreement with the
organizer's hidden ground truth, which is likely lower on 04/05/06.

## Held-out split (never used for tuning)
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

## DEV re-evaluation after the review changes (v2 vs v1)
`python -m cqa evaluate --group DEV --out results/dev_jury_v2.json` — 65 dialogs, 0 errors, 0 incomplete, 245 s, $11.60.

| | v1 (`dev_jury.json`) | v2 (`dev_jury_v2.json`) |
|---|---|---|
| Mean total | 75.95 | 76.57 |
| Gross-error flags | 5 | 5 (same 5 dialogs) |
| Incomplete dialogs / needs-review flags | n/a | 0 / 0 |
| Changed verdicts (of 520) | — | 28: 04 ×12, 02 ×8, 06 ×3, 07 ×3, 05 ×2; 01/03/08 unchanged |
Most frequent changes: 04 20→15 (7; explicit "minor detail missed" anchor), 02 NO→NA (4) and NO→YES (3) (new 02 scope).

### DEV v4 (prompt v8, after review round 4) — history
`results/dev_jury_v4.json` — 65 dialogs, 0 errors, 0 incomplete, 0 rescued, 267 s, $13.37. Mean 78.08, 5 gross flags
(same 5 dialogs as v1), judge decided 75 verdicts, 27 evidence warnings, 02 = NA in 43/65. Changed verdicts: 43 vs v1
(04 ×14, 06 ×13, 02 ×8, 05 ×5, 07 ×2, 03 ×1), 31 vs v3.

### DEV v3 (prompt v5: FACTS block, request ledger, round-1/2 rule changes)
`results/dev_jury_v3.json` — 65 dialogs, 0 errors, 0 incomplete, 0 juror failures, 275 s, $13.03.

| | v1 | v2 | v3 |
|---|---|---|---|
| Mean total | 75.95 | 76.57 | 78.08 (v4 also 78.08 by coincidence: 31 verdicts differ between v3 and v4) |
| Gross-error flags | 5 | 5 | 5 (same dialogs) |
| Changed verdicts vs v1 (of 520) | — | 28 | 34: 04 ×11, 06 ×10, 02 ×8, 07 ×3, 05 ×2 |
| Decided by the judge | 81 | — | 72 |
| `evidence_warning` (positive verdict kept without an operator quote after one retry) | n/a | 0 | 24 (05 YES ×7, 06 ×9, 04 ×6, 02 ×2) — shown in the UI, not hidden |
Most frequent v1→v3 changes: 06 4→7 (6; pauses now computed by code, skipped questions moved to 04), 04 20→15 (5), 02 NO→YES (4) / NO→NA (3). 02 is NA in 42 of 65 dialogs.

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
