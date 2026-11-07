# Evaluation

## DEV set results (jury mode, v1 — before the review changes)
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
| + code-computed FACTS block (first operator turn, language heuristic, longest wait & warning, holds, ending), 01 naming examples, 06 literacy rule (prompt v4, full re-run, $2.30) | **0.992** | **1.44** | 05 gold-12 (model treats "takes too long" as a complaint needing acknowledgement) |

**Tuning (one change, generic, not dialog-specific):** before tuning, all three 02 errors were `NA→NO`, and the
justifications penalised the *answer* (an ignored question, a private paid offer) under 02. We added one rule to 02:
it judges only the pre-answer stage; answer quality belongs to 04/08; if the client's data were sufficient → NA.
Remaining disagreements (not tuned, to avoid overfitting): 06 on gold-05 (7 vs 10) and gold-10 (4 vs 10 — the client
re-asked an ignored question, which the model attributes to 06; defensible either way) and 05 on gold-12/gold-16
(model treats "takes too long" / "why is my account blocked??" as a complaint that needed acknowledgement).

**Caveat (honest):** the gold labels were written by the same team that wrote the rubric rules, on synthetic
dialogs with clear-cut cases. This is a regression/consistency check, not an estimate of agreement with the
organizer's hidden ground truth, which is likely lower on 04/05/06.

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
