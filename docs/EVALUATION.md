# Evaluation

## DEV set results (jury mode)
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
