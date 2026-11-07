# Scoring specification

Implemented in `cqa/scoring.py` (pure functions, no LLM).

## Per criterion
| Type | Value | Points |
|---|---|---|
| binary | `YES` | `weight` |
| binary | `NO` | `0` |
| scale | step `s` from `scale` | `weight × s / max(scale)` |
| any | `NA` / missing (`None`) | excluded |
| `gross: true` or `weight == 0` | — | not scored (flag only) |

## Total
```
total = 100 × Σ points(applicable) / Σ weight(applicable)
```
rounded with `Decimal.quantize(..., ROUND_HALF_UP)` to `assumptions.decimals` (default 2). `rounding: bankers` switches to ROUND_HALF_EVEN.
If no weighted criterion is applicable → `total = null`.
`gross_error = true` iff a `gross` criterion has value `YES`; it does not change `total`.

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
| Missing value = NA | `test_scoring.py::test_missing_value_treated_as_na`, `test_scoring_edge.py::test_none_value_like_na` |
| Invalid value / step rejected | `test_scoring.py::test_invalid_values` |
| README example | `test_scoring_edge.py::test_readme_worked_example` |
| Reweighting does not invalidate LLM cache | `test_checklist.py::test_criterion_hash_tracks_text_not_weight` |
| Human override recomputes a separate total | `test_e2e_mock.py::test_pipeline_end_to_end` |
