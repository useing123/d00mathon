# Model matrix: cost × accuracy × latency

Which model setup should CQA use? Every configuration below ran through the **same** pipeline: the same checklists,
prompts (unless a prompt variant is named), tool schema, evidence (quote) validation, fail-closed rules and scoring.
The only thing that changes is the model ids (`CQA_MODEL` or `CQA_JURORS` + `CQA_JUDGE`) and, on the prompt axis,
`CQA_PROMPT_VARIANT`. The vendor is chosen from the model id (`cqa.llm.cloud_route`): Anthropic Messages API,
OpenAI Responses API (needed for `gpt-5.3-codex`, and for tool calls on `gpt-6-luna`), or Gemini's
OpenAI-compatible endpoint. All values are model defaults. No per-model prompt tweaks were made, and the
per-criterion cache was off.

**Reproduce:** `python scripts/bench_matrix.py snapshot && python scripts/bench_matrix.py run --suites gold,holdout && python scripts/bench_matrix.py matrix`
(see the script docstring for reruns, `@v1` prompt variants and `dev`/`dev20`). Raw per-run JSON (metrics, mismatches,
per-model token usage, dialogs) is in `results/bench/`; the aggregate is `results/bench_matrix.json`, served at
`GET /api/models/bench` and shown at **`/models`**. Every run came from one frozen snapshot of the working tree
(`same_snapshot_and_base_prompts: true`, prompt version v9). That way,
edits other agents made during the benchmark could not reach some configurations and not others.

## Data sets and metrics

| Set | Size | Role |
|---|---|---|
| Tuned gold (`examples/gold`) | 16 synthetic dialogs × 8 criteria = 128 labels | Used while the prompts/rules were developed, so scores here are **optimistic** |
| Held-out gold (`examples/gold_holdout`) | 10 dialogs × 8 = 80 labels | Not used to write rules, but one 04 rule was later influenced by a held-out mismatch (see EVALUATION.md), and it was used to choose between models, so it is no longer a clean test set. Differences of ≤2/80 labels are within run-to-run noise. |
| DEV | 65 real dialogs | No human labels. Agreement is measured against our own reference jury run (`results/dev_jury_v4.json`, haiku+sonnet→opus) and is **not ground truth** |

Metrics:
- exact per-criterion agreement
- overall agreement
- MAE of the 0–100 total
- gross-error (08) precision/recall
- NA confusion
- incomplete (fail-closed) dialogs
- $ cost = provider-reported tokens × the price table below
- wall time with 16 parallel requests per run (s/dialog)
- tokens and calls per dialog

## Full matrix

Sorted by held-out score. "runs" lists the individual held-out scores when the config was run twice.
$/dialog and s/dialog are measured over gold + held-out (26 dialogs).

| Config | Prompt | Held-out (80) | runs | Tuned gold (128) | MAE held-out | MAE tuned | Gross R/P (held+tuned) | Incomplete | DEV vs jury (65) | DEV-20 | $/dialog | s/dialog | tokens/dialog | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-5.3-codex | v0 | 97.5% | 97.5/97.5 | 97.7% | 1.54 | 0.91 | 1.0/1.0 · 1.0/1.0 | 0 | — | 90.6% | 0.05823 | 2.85 | 18772 | ✓ |
| gemini-4-argon | v0 | 96.9% | 97.5/96.3 | 100.0% | 3.01 | 0.0 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.06674 | 7.84 | 22552 |  |
| gpt-6-luna | v1 | 96.3% | 1 | 98.4% | 2.46 | 0.85 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.00297 | 3.65 | 19676 | ✓ |
| sonnet | v0 | 96.2% | 95.0/97.5 | 99.2% | 2.41 | 0.43 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.08047 | 3.9 | 47122 |  |
| jury-haiku+sonnet>opus | v0 | 95.0% | 95.0/95.0 | 99.2% | 4.21 | 0.43 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.07276 | 4.33 | 75587 |  |
| gpt-6-luna | v0 | 94.4% | 95.0/93.8 | 98.4% | 4.67 | 1.5 | 1.0/1.0 · 1.0/1.0 | 0 | 88.5% | 87.5% | 0.00238 | 3.46 | 19173 | ✓ |
| gemini-3.8-flash | v0 | 93.8% | 1 | 98.4% | 4.87 | 0.54 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.03717 | 7.29 | 25497 |  |
| gpt-6-luna | v2 | 93.8% | 1 | 98.4% | 6.64 | 0.48 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.00333 | 3.98 | 20972 |  |
| jury-sonnet+luna>opus | v0 | 93.8% | 1 | 100.0% | 4.67 | 0.0 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.0742 | 4.76 | 59198 |  |
| opus | v0 | 93.8% | 1 | 98.4% | 4.96 | 0.79 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.1858 | 5.2 | 51594 |  |
| fable | v0 | 92.5% | 1 | 94.5% | 5.49 | 2.97 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.54365 | 8.82 | 49950 |  |
| haiku | v2 | 92.5% | 1 | 97.7% | 8.18 | 2.48 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.0053 | 2.94 | 52955 |  |
| gpt-5.4-mini | v0 | 91.2% | 1 | 93.8% | 5.73 | 4.03 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.02237 | 2.13 | 18839 |  |
| haiku | v0 | 91.2% | 1 | 96.9% | 7.59 | 0.83 | 1.0/1.0 · 1.0/1.0 | 0 | 87.1% | 88.1% | 0.00465 | 2.95 | 48928 |  |
| jury-sonnet+opus>opus | v0 | 91.2% | 1 | 98.4% | 8.9 | 0.79 | 1.0/1.0 · 1.0/1.0 | 0 | — | — | 0.18512 | 4.87 | 72818 |  |
| gpt-6-luna | v3 | — | — | 96.9% | — | 1.35 | — · 1.0/1.0 | 0 | — | — | — | — | — |  |
| haiku | v1 | — | — | 95.3% | — | 2.62 | — · 0.67/1.0 | 0 | — | — | — | — | — |  |
| haiku | v3 | — | — | 94.5% | — | 4.12 | — · 1.0/1.0 | 0 | — | — | — | — | — |  |

Per-criterion agreement on the held-out set (%):

| Config | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 |
|---|---|---|---|---|---|---|---|---|
| gpt-5.3-codex | 100 | 100 | 100 | 80 | 100 | 100 | 100 | 100 |
| gemini-4-argon | 100 | 90 | 100 | 100 | 100 | 90 | 100 | 100 |
| gpt-6-luna@v1 | 100 | 90 | 100 | 100 | 100 | 80 | 100 | 100 |
| sonnet | 100 | 90 | 100 | 80 | 100 | 90 | 100 | 100 |
| jury-haiku+sonnet>opus | 100 | 90 | 100 | 90 | 90 | 90 | 100 | 100 |
| gpt-6-luna | 100 | 90 | 90 | 90 | 100 | 90 | 100 | 100 |
| gemini-3.8-flash | 100 | 90 | 100 | 100 | 90 | 70 | 100 | 100 |
| gpt-6-luna@v2 | 100 | 80 | 90 | 90 | 100 | 90 | 100 | 100 |
| jury-sonnet+luna>opus | 100 | 80 | 100 | 90 | 100 | 80 | 100 | 100 |
| opus | 100 | 90 | 100 | 90 | 90 | 80 | 100 | 100 |
| fable | 100 | 90 | 100 | 90 | 90 | 70 | 100 | 100 |
| haiku@v2 | 100 | 100 | 90 | 70 | 90 | 90 | 100 | 100 |
| gpt-5.4-mini | 100 | 80 | 90 | 80 | 100 | 100 | 80 | 100 |
| haiku | 100 | 100 | 90 | 70 | 90 | 90 | 90 | 100 |
| jury-sonnet+opus>opus | 100 | 80 | 90 | 90 | 90 | 80 | 100 | 100 |

NA handling on the held-out set (only off-diagonal cells are listed):

| Config | NA errors on held-out (expected/predicted) | NA correct |
|---|---|---|
| gpt-5.3-codex |  | 17 |
| gemini-4-argon | NA/applicable=1 | 16 |
| gpt-6-luna@v1 | NA/applicable=1 | 16 |
| sonnet | NA/applicable=1 | 16 |
| jury-haiku+sonnet>opus | NA/applicable=1 | 16 |
| gpt-6-luna | NA/applicable=1 | 16 |
| gemini-3.8-flash | NA/applicable=1 | 16 |
| gpt-6-luna@v2 | NA/applicable=2 | 15 |
| jury-sonnet+luna>opus | NA/applicable=2 | 15 |
| opus | NA/applicable=1 | 16 |
| fable | NA/applicable=1 | 16 |
| haiku@v2 | NA/applicable=1 | 16 |
| gpt-5.4-mini | NA/applicable=4 | 13 |
| haiku | NA/applicable=1 | 16 |
| jury-sonnet+opus>opus | NA/applicable=2 | 15 |

### Local model (open weights, external run)

| Config | Held-out | Tuned gold | MAE held-out | DEV vs jury | $/dialog (GPU time) | s/dialog |
|---|---|---|---|---|---|---|
| Qwen3-32B-AWQ on vLLM, 1× A40 at $0.59/h | 88.7% | 88.3% | 6.38 | 81.0% (vs `dev_jury_v5.json`) | 0.00227 | 13.86 |

- It sits on the Pareto frontier only because it is about as cheap as `gpt-6-luna` ($0.0023 vs $0.0024 per dialog).
- It is 6–9 points less accurate than luna and 4–5× slower.
- Use it only when data must not leave the perimeter.

### DEV (65 real dialogs) — agreement with the reference jury

| Config | Set | Dialogs | Agreement vs jury v4 | Total MAE | gross tp,fp,fn | Source |
|---|---|---|---|---|---|---|
| gpt-5.3-codex | dev20 | 20 | 90.6% | 7.44 | 0,0,0 | run |
| gpt-6-luna | dev | 65 | 88.5% | 7.47 | 5,0,0 | run |
| gpt-6-luna | dev20 | 20 | 87.5% | 6.11 | 0,0,0 | derived from full DEV run |
| haiku | dev | 65 | 87.1% | 6.95 | 5,0,0 | run |
| haiku | dev20 | 20 | 88.1% | 4.72 | 0,0,0 | derived from full DEV run |

The DEV checks were cut by budget: the full set only for the two cheapest models, and a 20-dialog subset for the winner.
Gemini-4-argon, sonnet and the juries were **not** run on DEV (each would have cost $2.5–12). DEV-20 rows marked
"derived" are the first 20 dialogs of the full DEV run, re-scored offline (no API calls), so they compare on the same
subset as the `gpt-5.3-codex` DEV-20 run. Read DEV as "how close is this to the previous production jury". It does
not measure correctness.

## Prompt axis (protocol against overfitting)

Variants live in `cqa/prompt_variants/`. The base `cqa/prompts.py` is unchanged apart from a 6-line loader hook, and
the tool schema, rubric and rules are identical in every variant. Each variant is selected by `CQA_PROMPT_VARIANT` or a
preset's `prompt` key.

| Variant | What it changes |
|---|---|
| v0 | Base prompts |
| v1 | Few-shot: 1–2 short **synthetic** examples per criterion, written from the checklist rules. Not taken from held-out or DEV; `tests/test_models_bench.py` checks for verbatim held-out lines |
| v2 | Checklist-first: an explicit 5-step decision procedure (NA check → relevant turns → rules top-down → rubric check → quotes) |
| v3 | Compact system prompt (473 chars instead of 1,830) |

Protocol:
1. Variants were developed and scored **only on the tuned gold set**.
2. The top variant combos there were then run **once** on the held-out set: luna v1 and v2 (both tied at 98.4%) and haiku v2 (the only haiku variant above v0).
3. Nothing was edited after the held-out results were seen.

The budget (≈$40 total) allowed the prompt search only on the two cheapest models (≈$0.05–0.09 per tuned-gold run).
A prompt search on codex/argon/sonnet (≈$0.6–1.3 per run) was not affordable.

Tuned gold, overall agreement (MAE):

| Model | v0 base | v1 few-shot | v2 step-by-step | v3 compact | Picked for held-out |
|---|---|---|---|---|---|
| gpt-6-luna | 98.4% (MAE 1.5) | 98.4% (MAE 0.85) | 98.4% (MAE 0.48) | 96.9% (MAE 1.35) | v1, v2 |
| haiku | 96.9% (MAE 0.83) | 95.3% (MAE 2.62) | 97.7% (MAE 2.48) | 94.5% (MAE 4.12) | v2 |

Held-out, run once:

| Combo | Held-out (single run, decided before looking) | MAE |
|---|---|---|
| gpt-6-luna@v1 | 96.3% | 2.46 |
| gpt-6-luna@v2 | 93.8% | 6.64 |
| haiku@v2 | 92.5% | 8.18 |

Result:
- **v1 (few-shot)** helped `gpt-6-luna` on the held-out set: 96.3% vs a 94.4% two-run mean for v0, i.e. +1.5 labels out of 80. That is within run-to-run noise, so read it as "probably not worse".
- v2 and v3 gave no held-out gain.
- On haiku, every variant was neutral or worse on tuned gold, and v2 fell to 92.5% on held-out.

## Winner (automatic selection rule)

**Rule:** highest exact agreement on the held-out set (mean of repeated runs). Ties are broken by lower total MAE, then
higher gross recall, then lower $/dialog. If the top configs were within ±2 labels, the top ones were run a second
time and averaged. Here, codex, argon, sonnet, luna and the old jury were all re-run, which cost $2.6.

| Rank | Config | Held-out (mean) | Runs | $/dialog |
|---|---|---|---|---|
| 1 | gpt-5.3-codex | 97.50% | 2 | 0.05823 |
| 2 | gemini-4-argon | 96.90% | 2 | 0.06674 |
| 3 | gpt-6-luna@v1 | 96.30% | 1 | 0.00297 |
| 4 | sonnet | 96.25% | 2 | 0.08047 |
| 5 | jury-haiku+sonnet>opus | 95.00% | 2 | 0.07276 |
| 6 | gpt-6-luna | 94.40% | 2 | 0.00238 |

- **Winner: `gpt-5.3-codex` (prompt v0).**
  - Held-out: 97.5% in both runs (78/80 each time). Every miss was criterion 04 (consultation scale).
  - Held-out total MAE 1.54, the lowest of all configs. Gross recall and precision are 1.0 on both gold sets.
  - Tuned gold 97.7%, DEV-20 agreement with the old jury 90.6% (the highest of the three models measured on that subset).
  - $0.058/dialog, 2.85 s/dialog (the fastest, since it uses no reasoning tokens by default), 0 incomplete.
  - It is now `--preset max` and the product default.
- **Runner-up: `gemini-4-argon`**: 97.5% then 96.3% (mean 96.9%), MAE 3.01, $0.067/dialog, 7.8 s/dialog.
  - It is within noise of codex. Codex wins on mean score, MAE, cost and latency all at once.
  - Note: Gemini prices double after 2026-11-30.
- The previous default (jury haiku+sonnet→opus) scored 95.0% in both runs, with MAE 4.21 and $0.073/dialog. On the
  held-out set it is dominated by the winner.

## Presets (`python -m cqa evaluate --preset econom|balanced|max|jury`, or `CQA_PRESET=...`)

| Preset | Setup | Held-out | $/dialog | Why |
|---|---|---|---|---|
| **econom** | `gpt-6-luna` + prompt v1 | 96.3% (1 run) | 0.0030 | On the Pareto frontier. About 20× cheaper than max, 1 label of 80 lower (within noise) |
| **balanced** | `claude-sonnet-5-5` | 96.25% (2 runs) | 0.080 | Single vendor: needs only an Anthropic key. **Not** on the frontier; choose it for vendor or data-processing reasons |
| **max** (default) | `gpt-5.3-codex` | 97.5% (2 runs) | 0.058 | Selected by the rule above. On the frontier |
| jury | haiku+sonnet → opus | 95.0% (2 runs) | 0.073 | The previous default, kept for comparison and continuity |

**Pareto frontier** (held-out score vs $/dialog): gpt-5.3-codex, gpt-6-luna@v1, gpt-6-luna, local:qwen3-32b-awq. Every other config is beaten by at least
one of these on both cost and accuracy.

Default behaviour: if no explicit `CQA_MODEL` / `CQA_ENSEMBLE` / `CQA_JURORS` / `CQA_JUDGE` is set, `cqa.config`
applies the `max` preset. `max` needs `OPENAI_API_KEY_QUORUM` (or `CQA_OPENAI_CLOUD_KEY` / `OPENAI_API_KEY`), and the
whole cloud mode still runs with `CQA_PROVIDER=anthropic`. Setting the old variables explicitly, or `CQA_PRESET=jury`,
restores the jury.

## Price table (assumption)

$ per 1M tokens, standard tier, retrieved 2026-11-07 (`cqa/pricing.py`). The cost columns depend on these
numbers. Token counts are the provider-reported usage, cached input included.

Sources:
- Anthropic: platform.claude.com/docs/en/about-claude/pricing
- OpenAI: developers.openai.com/api/docs/pricing
- Google: ai.google.dev/gemini-api/docs/pricing

| Model | input | output | cache read | cache write |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.1 | 0.5 | 0.01 | 0.125 |
| claude-sonnet-5-5 | 2.0 | 10.0 | 0.1 | 2.5 |
| claude-opus-5-5 | 4.0 | 20.0 | 0.2 | 5.0 |
| claude-fable-5-1 | 10.0 | 50.0 | 0.25 | 12.5 |
| gpt-6-luna | 0.1 | 0.5 | 0.01 | 0.125 |
| gpt-5.4-mini | 0.75 | 4.5 | 0.075 | 0.0 |
| gpt-5.3-codex | 1.75 | 14.0 | 0.175 | 0.0 |
| gemini-4-argon | 2.0 | 10.0 | 0.1 | 0.0 |
| gemini-3.8-flash | 0.75 | 3.75 | 0.075 | 0.0 |

Notes:
- gemini-4-argon: promo until 2026-11-30; $4/$20 from 2026-12-01; gemini-3.8-flash: promo until 2026-12-31; $1.50/$7.50 from 2027-01-01; claude-haiku-5-5: prompts <=100k tokens.
- The old in-code table (haiku $1/$5, sonnet $3/$15, opus $5/$25) overstated Anthropic costs and was replaced.
  Earlier `cost_usd` figures in older results files used the old table.

## Spend (real, this benchmark)

- **Total: $40.13**, against a budget of about $40.
- From the result files: Anthropic $32.08, OpenAI $4.26, Google $3.35.
- Plus about $0.44 for a pipeline smoke test whose result files were discarded: haiku $0.04, luna $0.03, gemini-flash $0.36 on the held-out set.
- `claude-fable-5-1` alone cost $14.1 for 26 dialogs and ranked low (92.5% held-out). It is the main reason DEV coverage had to be cut.

## Caveats

- **Synthetic gold, labelled by the rule author.** It measures how faithfully the rules are applied, not agreement with
  Freedom Broker auditors. Human labels on DEV (`python -m cqa label`) would be the real test.
- **Small held-out set.** 80 labels means one label is 1.25 points. Run-to-run noise was 0–2.5 points (e.g. sonnet
  95.0→97.5, argon 97.5→96.3), so ranks 1–5 (95–97.5%) are close. The ranking is supported by: codex's identical
  repeats, its lowest MAE, and its lead on DEV-20.
- **Tuned vs held-out gap.** Most configs lose 1–5 points from tuned gold to held-out, the old jury included
  (99.2→95.0). The tuned set flatters whatever the prompts were tuned with, which was Anthropic models.
- **DEV is agreement with our own jury,** not accuracy. A model that disagrees with the jury may be right.
- **Latency** is wall time at 16 concurrent requests from a single host, including retries. It is not a per-call
  benchmark.
- **Local Qwen (vLLM)** comes from another agent's run (`results/bench_qwen3-32b-awq*.json`, docs/LOCAL_LLM.md). It was read, not re-run.
  The matrix shows it as `local:qwen3-32b-awq` with `external_run: true` and leaves it out of the winner selection,
  because it ran on a different prompt/checklist state.
- **Skipped:** `gpt-5.3-codex` is Responses-API only (it is not on chat completions), so it ran through the Responses API rather than being skipped.
  Prompt variants were not run on expensive models, and DEV was not run for argon, sonnet or the juries, both because of budget.
