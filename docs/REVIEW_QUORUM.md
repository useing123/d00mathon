# AI review quorum

The repository was reviewed by independent LLM "hackathon judges" (`.quorum/run_quorum.py`, git-ignored
outputs). Each receives the full repo and returns scores, risks and concrete improvements. We implemented the
suggestions that are correct and do not contradict the organizer rubric, and re-ran the review.

## Score history (overall, 0–100)
| Round | claude-fable-5-1 | gpt-6-luna | gemini-3.8-flash | gemini-4-argon |
|---|---|---|---|---|
| 0 (initial repo) | 77 | 65 | 86 | — |

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
