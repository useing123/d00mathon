# AI review quorum

The repository was reviewed by independent LLM "hackathon judges" (`.quorum/run_quorum.py`, git-ignored
outputs). Each receives the full repo and returns scores, risks and concrete improvements. We implemented the
suggestions that are correct and do not contradict the organizer rubric, and re-ran the review.

## Score history (overall, 0–100)
| Round | claude-fable-5-1 | claude-opus-5-5 | gemini-4-argon | gpt-6-luna | gemini-3.8-flash |
|---|---|---|---|---|---|
| 0 (initial repo) | 77 | — | — | 65 | 86 |
| 1 (after round-0 changes) | 76 | — | 93 | — | — |
| 2 (after round-1 changes) | 79 | — | 95 | — | — |
| 3 (after round-2 changes) | 80 | — | 97 | — | — |
| 4 (after round-3 changes) | 77 | — | 96 | 66 | — |
| 5 (after round-4 changes) | 77 | 78 | 94 | 72 | — |

Reviewer set changed over time (user decision): round 0 fable + luna + gemini-3.8-flash; rounds 1–3 fable + argon;
round 4 adds luna (run separately on the same commit); round 5 fable + opus + argon + luna; from round 6 also
gpt-5.3-codex. Stop rule (user): fable ≥ 90 and opus ≥ 90, at most 5 rounds after round 5, or 2 rounds without gain. Round-4 fable output was truncated JSON; overall (77) and risks were read from the raw
text, its improvement list was lost. Up to round 4 the review script truncated `cqa/static/index.html` to 14k chars
(a limitation of the script, not the repo); from round 5 it is sent in full.

## Round 0 → changes
### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| Fail closed on evaluation errors (luna) | `status: incomplete`, `total: null`, `failed_criteria`, `provisional_total`; CSV status column; UI badge |
| Gold set + agreement metric (fable, luna) | `examples/gold/` (16), `python -m cqa gold`, `docs/EVALUATION.md` |
| Evidence: empty allowed for NA / omission; operator quotes; 08 needs review (luna, gemini) | `evaluator._check_evidence`, tool schema + system prompt |
| Strict values, override validation, checklist invariants (luna) | `checklist.strict_value`, HTTP 400, `checklist.validate` |
| Speaker normalisation, timestamps, hash incl. timestamps (luna, gemini) | `data.normalize_speaker`, `data.parse_timestamp` |
| Untrusted-transcript boundary, ASR caveat, 04 anchors, 02 clarification (luna, gemini) | `prompts.py`, checklist rules |
| Jury vote key collision (gemini) | votes keyed `"<index>:<model>"` |
| Atomic writes, 429/5xx Retry-After + jitter, path-safe ids (luna, gemini) | `util.atomic_write`, `llm.retryable/backoff_delay`, `server.safe_id/safe_path` |
| Stable tool schema + dialog-major batches for prompt cache (fable, gemini) | cache_read 24.7k → 121k tokens on 3 dialogs |

### Rejected
| Suggestion | Why |
|---|---|
| 02: score YES instead of NA when the request was clear (fable) | Contradicts the organizer rubric: "NA = data sufficient from the start". |
| 06 scale 0/5/10/15 (fable) | Organizer scale for 06 is 0/4/7/10 (weight 15); points are weight × step / 10. |
| Alias model ids to claude-3-* (gemini) | The ids are real: verified via the provider's `/v1/models` listing (below). |

## Verified model IDs
Listed via the provider models API (`GET /v1/models`) on 2026-11-07: `claude-haiku-5-5`, `claude-sonnet-5-5`,
`claude-opus-5-5` (used as jurors / judge) are present in the listing, along with `claude-fable-5-1` and others.

## Round 1 → changes
Reviewers: claude-fable-5-1 (76), gemini-4-argon (93). fable's first response was malformed JSON; it was re-run
(the review script now saves raw output and falls back to `json_repair`).

### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| English hold bug: "on hold" ⊂ "no longer on hold" re-opened the hold (fable) | END checked first (`elif`), bare "on hold" removed; test |
| Code-computed facts instead of LLM arithmetic (fable) | `data.compute_facts/format_facts` → FACTS section in the transcript block |
| Jury partial-failure tolerance (fable) | failing juror → judge reviews survivor; all fail → judge alone; then fail-closed; tests |
| Text-JSON fallback when no tool_use (fable) | `_anthropic` parses a JSON object with `value` from a text block; test |
| CSV shows the provisional score for incomplete dialogs (fable) | `provisional_total` column (`total` stays empty — fail-closed) |
| 01 naming examples, language tie-break by metadata (fable) | rules of 01 |
| 06: skipped questions belong to 04 (argon); pauses from FACTS; gross literacy errors (fable) | rules of 06 |
| Prefer operator turns when correcting a wrong quote index (argon) | `validate_evidence` |
| Kazakh/RU hold markers (argon) | `HOLD_START/HOLD_END` |
| Native tool calling for OpenAI-compatible servers with JSON-mode fallback; jury on local models (argon) | `_openai_tools`, `CQA_OPENAI_TOOLS`, jury when `CQA_JURORS`+`CQA_JUDGE` set |
| Self-consistency temperature 1.0 → 0.4 (argon) | `SC_TEMPERATURE` |
| CSV formula-injection guard (argon) | `store._cell` |
| Gold mock run in CI (argon) | workflow step |
| Reproducibility metadata (luna, round 0) | `meta.reproducibility` (prompt version, checklist/criterion hashes, models, sampling) |
| Docs drift (enum schema, test count) (fable) | README / ARCHITECTURE |

### Rejected / not done
| Suggestion | Why |
|---|---|
| 02: YES whenever the operator understood the need; NA only if nothing referenced (fable) | Organizer rubric: NA = data sufficient from the start. Our YES already covers explicit use of known details. |
| 05: urgent factual questions ("why blocked??") need no acknowledgement (argon) | Would contradict our existing rule that anxiety about money/account is a trigger, and it was proposed to fit the two gold mismatches — overfitting risk. |
| 06: raise the pause threshold to ~5 min (fable) | No organizer basis for either number; kept 2 min, but the gap is now computed by code. |
| Include jury config in `criterion_hash` (argon) | Already part of the cache key (`msig` = jurors + judge); cosmetic. |
| Never-null total (fable) | Fail-closed was an explicit requirement; mitigated by the fallback chain + `provisional_total` column. |

## Round 2 → changes
### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| 04 structured request ledger (fable) | optional `client_requests` tool field (schema still shared), rule asks to fill it first; essential `open` + value > 10 → one retry; shown in the UI |
| 02: referencing earlier client details in the answer is YES (fable) | examples added to the existing YES rule (NA rule unchanged) |
| 03: client-provided ID without a check → NO; system-reported authentication counts (fable, argon) | rules of 03 + `system_says_client_authenticated` FACT |
| 07: "если появятся вопросы — обращайтесь" = offer of help; system timeout after operator farewell is not a break (fable) | rules of 07 |
| 05: incidental inconvenience inside a how-to question needs no separate apology (argon, 2nd time) | rules of 05 (see EVALUATION caveat) |
| FACTS: multi-message first reply, wait warning in any operator turn since the client message, ack-only client turns don't start a "wait", short Kazakh words (argon) | `data.compute_facts`, `detect_language`; tests |
| Strip copied `[N] (mm:ss) SPEAKER:` prefixes from quotes (argon) | `validate_evidence`; test |
| Judge sees jurors' quotes (argon) | `_jury` opinions |
| Actionable 404 message for unknown model ids (argon, fable) | `LLM.call`; test |
| `max_tokens` 1500 → 3000 to avoid truncated tool calls on long calls (fable) | `_anthropic` |
| OpenAI-provider jury only when explicitly enabled (fable: `.env.example` pitfall) | `CQA_OPENAI_JURY=1` |
| `gold --dir` for a private labelled DEV subset (fable) | CLI |

### Rejected / not done
| Suggestion | Why |
|---|---|
| Configurable NA mode (full credit / zero) (fable) | The organizer rubric fixes it: NA weight is redistributed proportionally. |
| 07 strictness switch (farewell alone = YES) (fable) | Rubric: (summary OR offer) AND goodbye. |
| Hand-label real DEV dialogs (fable) | No organizer labels exist; labelling them ourselves would repeat the self-labelling bias. `--dir` support added instead. |
| Criterion hash incl. jury models (argon) | Already in the cache key (`msig`). |

## Round 3 → changes
### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| Rescue pass before nulling a total (fable) | failed criteria get 2 fresh single-model attempts with the judge model; only then `incomplete`; test |
| Double-check gross flags (fable) | a unanimous 08=YES is confirmed by the judge (`decided_by: judge_confirm:…`); test |
| Two-way 04 ledger consistency (argon) | essential partial → not 20; all closed → not 5/10; test |
| Internal hold turns are not operator evidence (argon) | `internal` flag on evidence, excluded in `_check_evidence`; test |
| Wait FACT distinguishes reply-to-client vs pause between operator messages (argon) | `kind` field |
| Mixed KZ/RU clients (argon) | `client_mixed_languages` FACT; 01 rule: greeting in either language = YES |
| 03: ID done, nothing disclosed because the dialog ended → YES (argon) | rules of 03 |
| 07: operator asked "чем ещё помочь?", client silent, chat closed → NA (argon) | rules of 07 (consistent with rubric NA = chat broke off) |
| Judge sees jurors' missing_where / na_reason / request ledger (argon) | `_jury` |

### Rejected / not done
| Suggestion | Why |
|---|---|
| Narrow 02 NA further (fable, 3rd time) | Rubric: NA = data sufficient from the start; our YES already covers explicit use of known details. |
| Write provisional score into `total` for submission (fable) | Conflicts with the fail-closed requirement; `provisional_total` column + rescue pass instead. |
| "Contest" jury sonnet+opus (fable) | Not measured; we don't claim untested accuracy. Configurable via `CQA_JURORS`. |

## Round 4 → changes
### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| 05 rule contradiction (anxiety trigger vs plain "why blocked?") (fable) | trigger narrowed to explicitly expressed anxiety; precedence rule + contrasting examples |
| 02 YES/NA precedence (fable) | YES only when the latest client message lacked needed data and the operator got it (asked or took it from bot/history); repeating given details = NA |
| 04: wrong / bypass advice → strictly 0 with ledger `open` (argon); contrasting 0/5, 10/15, escalation examples (luna) | rules of 04 |
| Call 06 without chat artefacts (argon) | call rubric/rules |
| Wait warning survives client pings (argon) | `compute_facts`; test |
| 02: operator clarifying specifics after a generic bot topic = YES (argon); 07 closing across two operator turns (argon) | rules |
| Holistic evidence (05/06) must cite a representative operator turn (argon) | system prompt |
| Negation-aware auth fact (luna) | `data.says_authenticated`; tests |
| Cache key hashes the actual system prompt + tool schema (luna) | `criterion_hash`; test |
| Quote fragments must appear in order (luna) | `validate_evidence`; test |
| Language heuristic: Kazakh name in Russian text ≠ KZ; per-message (luna) | `detect_language`; test |
| Integer-only scale steps, checklist id allowlist in `load` (luna) | `checklist.validate/load`; tests |
| Model payload shape hygiene (confidence, ledger, text fields) (luna) | `Evaluator._finish`; test |
| Per-run usage/cost instead of process totals (luna) | `evaluate_many`; test |
| Independent accuracy set (luna, fable) | held-out split `examples/gold_holdout` + `cqa label` / `cqa gold --labels` (see EVALUATION) |
| Opus never sees consensus verdicts (fable) | optional `--confirm 03,04` / `CQA_CONFIRM`; measured in EVALUATION |

### Rejected / not done
| Suggestion | Why |
|---|---|
| `--allow-provisional` total (argon) | Conflicts with fail-closed; `provisional_total` column exists. |
| Labelling real DEV dialogs ourselves | Would be self-labelling; tooling is ready for a human labeller. |

## Round 5 → changes (decisions by the user on Opus/Fable suggestions)
### Accepted (implemented)
| Suggestion (source) | Implementation |
|---|---|
| All unwarned waits > 2 min in FACTS; anchored warning regex (opus) | `waits_over_2min`, `WAIT_RE` («в данный момент», «в течение 5 минут» no longer count as warnings); 06 rule uses the list; tests |
| `evaluate --final` control profile (opus) | jurors sonnet + opus, judge confirms unanimous 02/04/06, median of 3 on 04/06; measured on held-out (not better, opt-in) |
| Never give up on the first failing model (opus, fable) | rescue pass: judge, then each juror; only then `incomplete` |
| Model-id preflight + optional fallback (fable) | `Evaluator.preflight`, `CQA_FALLBACK_MODELS`; test |
| 07: «чем ещё помочь?», client silent, closed without the operator's goodbye → NO (opus; user decision) | rule of 07 — the rubric requires a goodbye and the operator had the chance; NA only if the chat broke off before the operator could close |
| 04 vs 03 double penalty (fable) | rule: missing ID is 03, not a wrong 04 recommendation |
| 02: ≥2 plausible readings answered without clarifying → NO (opus) | rule of 02 (NA definition unchanged) |
| 08 scope: rudeness → 05, wrong advice → 04; confidential-info and multilingual insult examples (opus) | rules of 08 |
| Checklist dedupe (opus) | `freedom_call_v1` = `extends: freedom_chat_v1` + override of 06 only; test |
| Organizer submission export (fable) | `evaluate --organizer FILE`, `cqa export --format organizer`; test |
| One result shape for all paths; no inner imports; dead `--reeval` flag removed (opus, fable) | `_result_dict`, CLI |
| UI: no `window.prompt/confirm`; upload starts evaluation; AI↔human agreement card; JS/CSS in separate files (opus, fable) | `static/app.js`, `static/app.css`, in-page `ask()` dialog; Playwright smoke updated |
| Docs consistency: one current version, CI badge wired to the workflow, no static test count (fable, opus) | README / EVALUATION |

### Rejected
| Suggestion | Why |
|---|---|
| Make 02 YES whenever the operator understood/used the stated need; reduce the 44/65 NA rate (fable, opus) | The organizer rubric defines 02 NA as «данных хватало с самого начала». A request answerable as stated is NA by definition; YES is reserved for cases where data were missing and the operator obtained them (asked, or took them from the bot dialogue instead of re-asking). Changing this would contradict the rubric, so the NA rate is a property of the DEV data, not a bug. |
| Hand-label real DEV dialogs ourselves (fable, opus) | Self-labelling; the tooling (`cqa label`, `cqa gold --labels`) is ready for a human labeller. |
| Measure the local-LLM path (fable, opus) | No local GPU/LLM server in this environment; not claimed. |
