# AI review quorum

The repository was reviewed by independent LLM "hackathon judges" (`.quorum/run_quorum.py`, git-ignored
outputs). Each receives the full repo and returns scores, risks and concrete improvements. We implemented the
suggestions that are correct and do not contradict the organizer rubric, and re-ran the review.

## Score history (overall, 0–100)
| Round | claude-fable-5-1 | gpt-6-luna | gemini-3.8-flash | gemini-4-argon |
|---|---|---|---|---|
| 0 (initial repo) | 77 | 65 | 86 | — |
| 1 (after round-0 changes) | 76 | — | — | 93 |

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
