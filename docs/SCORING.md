# Scoring specification

Implemented in `cqa/scoring.py` (pure functions, no LLM).

## Per criterion
| Type | Value | Points |
|---|---|---|
| binary | `YES` | `weight` |
| binary | `NO` | `0` |
| scale | step `s` from `scale` | `weight × s / max(scale)` |
| any | `NA` | excluded (weight redistributed) |
| any | failed evaluation (`None`, `error`) | **not** NA → dialog `status: incomplete`, `total: null` (fail-closed) |
| `gross: true` or `weight == 0` | — | not scored (flag only) |

## Total
```
total = 100 × Σ points(applicable) / Σ weight(applicable)
```
rounded with `Decimal.quantize(..., ROUND_HALF_UP)` to `assumptions.decimals` (default 2). `rounding: bankers` switches to ROUND_HALF_EVEN.
If no weighted criterion is applicable → `total = null`.
`gross_error = true` iff a `gross` criterion has value `YES`; it does not change `total`.

## Fail-closed evaluation
If any criterion evaluation fails (API error, no valid value after retries), the dialog result gets
`status: "incomplete"`, `total: null`, `failed_criteria: [...]` and an informational `provisional_total`
(score over the remaining criteria). It is never silently scored as NA. The UI shows an «неполная» badge,
the CSV has `status` / `failed_criteria` / `needs_review` columns, and a human override can complete the
dialog (`human.total` is then computed). `scoring.compute_total` itself still treats `None` like NA — the
fail-closed rule lives one level up in `evaluator.build_dialog_result`.

## Strict values
`checklist.strict_value` accepts only the exact declared values: `YES`/`NO`, `NA` only if `allow_na`, and
scale steps that are listed in `scale` (`"15"`, `15`, `15.0` ok; `4.5`, `7`, `"15 points"` rejected — no
truncation). The same function validates LLM output (invalid → retry with a hint) and human overrides
(invalid → HTTP 400). Checklists are validated on load: finite non-negative weights, unique non-negative
scale steps, top step > 0.

## Worked example (chat checklist)
01 YES (10/10), 02 NA, 03 NA, 04 = 15 of 20 → 15, 05 NO → 0, 06 = 7 of 10 → 10.5, 07 NA, 08 NO.
Applicable weight = 10 + 20 + 15 + 15 = 60; points = 35.5; total = 100 × 35.5 / 60 = 59.1666… → **59.17**.

## Edge cases → tests
| Edge case | Test |
|---|---|
| All YES → 100 | `tests/test_scoring.py::test_perfect` |
| All NO → 0 | `test_scoring.py::test_all_no` |
| NA redistributes weight | `test_scoring.py::test_na_redistribution`, `test_scoring_edge.py::test_na_excluded_from_denominator` |
| Scale proportional | `test_scoring.py::test_scale_proportional`, `test_scoring_edge.py::test_scale_step_points` |
| Half-up rounding (x.xx5 rounds up, not banker's) | `test_scoring.py::test_half_up_rounding`, `test_rounding_two_decimals` |
| Configurable decimals | `test_scoring_edge.py::test_decimals_configurable` |
| All NA → `null` | `test_scoring.py::test_all_na_is_null` |
| Gross criterion not scored, separate flag | `test_scoring.py::test_gross_not_scored`, `test_scoring_edge.py::test_gross_flag_independent_of_total`, `test_zero_weight_and_gross_skip` |
| Missing value = NA at the pure-scoring level | `test_scoring.py::test_missing_value_treated_as_na`, `test_scoring_edge.py::test_none_value_like_na` |
| Failed criterion → dialog incomplete, total null | `test_review_hardening.py::test_failed_criterion_marks_dialog_incomplete`, `test_llm_error_is_not_na` |
| Strict values / override 400 / checklist invariants | `test_review_hardening.py::test_strict_value`, `test_override_validation_http_400`, `test_checklist_invariants` |
| Invalid value / step rejected | `test_scoring.py::test_invalid_values` |
| README example | `test_scoring_edge.py::test_readme_worked_example` |
| Reweighting does not invalidate LLM cache | `test_checklist.py::test_criterion_hash_tracks_text_not_weight` |
| Human override recomputes a separate total | `test_e2e_mock.py::test_pipeline_end_to_end` |
