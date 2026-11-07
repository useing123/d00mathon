# AI review quorum

The repository was reviewed by independent LLM "hackathon judges" (`.quorum/run_quorum.py`, git-ignored
outputs). Each receives the full repo and returns scores, risks and concrete improvements. We implemented the
suggestions that are correct and do not contradict the organizer rubric, and re-ran the review.

## Score history (overall, 0–100)
| Round | claude-fable-5-1 | gpt-6-luna | gemini-3.8-flash | gemini-4-argon |
|---|---|---|---|---|
| 0 (initial repo) | 77 | 65 | 86 | — |
| 1 (after round-0 changes) | 76 | — | — | 93 |
| 2 (after round-1 changes) | 79 | — | — | 95 |
| 3 (after round-2 changes) | 80 | — | — | 97 |

From round 1 the quorum is claude-fable-5-1 + gemini-4-argon.

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
