"""Per-criterion LLM evaluation with evidence validation, self-consistency and a per-criterion cache."""
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import checklist as ckl, config, data, prompts, scoring
from .llm import LLM, USAGE, cost_usd
from .util import atomic_write as store_atomic


def _norm(s):
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[«»\"'“”„`’]", "", s)
    s = re.sub(r"[\s\u00a0]+", " ", s)
    return s.strip(" .,…")


def validate_evidence(rec, evidence):
    """Return (valid, invalid). A quote is valid if it is a (normalised) substring of the cited turn;
    if it is found in a different turn, the turn index is corrected."""
    turns = rec["transcript"]
    valid, invalid = [], []
    if isinstance(evidence, str):          # sonnet/opus in tool_choice=auto sometimes send JSON as a string
        try:
            evidence = json.loads(evidence)
        except Exception:
            evidence = []
    if isinstance(evidence, dict):
        evidence = [evidence]
    for e in evidence or []:
        if not isinstance(e, dict):
            continue
        raw_q = re.sub(r"^\s*(?:\[\d+\]\s*)?(?:\([\d:.?—-]*\)\s*)?(?:(?:OPERATOR|CLIENT|BOT|SYSTEM|UNKNOWN)\s*:\s*)?", "",
                       str(e.get("quote", "")))   # models sometimes copy the "[N] (mm:ss) SPEAKER:" line prefix
        q = _norm(raw_q)
        try:
            ti = int(e.get("turn"))
        except Exception:
            ti = -1
        if not q:
            continue
        q = q.strip("… .")
        parts = [p.strip() for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip()]
        def ok(t, parts=parts):
            tn = _norm(t.get("text", ""))
            return all(p in tn for p in parts)
        if 0 <= ti < len(turns) and ok(turns[ti]):
            valid.append({"turn": ti, "quote": e["quote"], "speaker": turns[ti].get("speaker")})
            continue
        # wrong turn index: prefer an operator turn containing the quote, then any turn
        hit = next((t for t in turns if t.get("speaker") == "operator" and ok(t)), None) or next((t for t in turns if ok(t)), None)
        if hit:
            valid.append({"turn": hit["index"], "quote": e["quote"], "speaker": hit.get("speaker"), "turn_corrected": True})
        else:
            invalid.append(e)
    return valid, invalid


def _coerce(cr, v):
    """Strict: accept only the exact declared YES/NO(/NA) or an exact declared scale step. Anything else -> None."""
    return ckl.strict_value(cr, v)


def _positive(cr, v):
    return v == "YES" if cr["type"] == "binary" else (isinstance(v, int) and v > 0)


def _check_evidence(cr, val, evidence, out):
    """Return a problem string (for a retry hint) or ''. Empty evidence is fine for NA / NO-by-omission
    if missing_where / na_reason is filled."""
    ops = [e for e in evidence if e.get("speaker") == "operator"]
    reqs = out.get("client_requests")
    if cr.get("request_ledger") and isinstance(reqs, list) and reqs and isinstance(val, int):
        ess_open = [r for r in reqs if isinstance(r, dict) and r.get("essential") and r.get("status") == "open"]
        if ess_open and val > 10:
            return "ledger_inconsistent"
    if cr.get("gross") and val == "YES" and not ops:
        return "gross_no_operator_quote"
    if not evidence:
        if val == "NA" and not str(out.get("na_reason") or "").strip():
            return "na_without_reason"
        if not _positive(cr, val) and val != "NA" and not str(out.get("missing_where") or "").strip():
            return "omission_without_missing_where"
        if _positive(cr, val) and val != "NA":
            return "positive_without_evidence"
        return ""
    if _positive(cr, val) and not ops and cr.get("evidence_from", "operator") == "operator":
        return "no_operator_evidence"
    return ""


_HINTS = {
    "gross_no_operator_quote": ("Значение YES для грубой ошибки требует хотя бы одной ДОСЛОВНОЙ цитаты из реплики OPERATOR. "
                                "Если такой цитаты нет — пересмотри значение."),
    "na_without_reason": "При value=NA заполни na_reason.",
    "omission_without_missing_where": "Если evidence пустой, укажи в missing_where, где ожидалось действие оператора.",
    "positive_without_evidence": "Положительная оценка требует цитат из реплик OPERATOR.",
    "ledger_inconsistent": ("В client_requests есть существенный вопрос со статусом open, а значение выше 10. "
                            "Согласуй список запросов и значение по якорям критерия."),
    "no_operator_evidence": "Критерий оценивает поведение оператора: приведи цитаты из реплик OPERATOR (а не клиента/бота).",
}


SC_TEMPERATURE = 0.4   # extra self-consistency samples (OpenAI-compatible providers; Anthropic endpoint ignores it)


class Evaluator:
    def __init__(self, llm=None, use_cache=True, samples=None, ensemble=None):
        self.llm = llm or LLM()
        self.ensemble = config.ENSEMBLE if ensemble is None else ensemble
        jury_ok = self.llm.provider == "anthropic" or (self.llm.provider == "openai" and config.JURORS_EXPLICIT)
        if self.ensemble and jury_ok and llm is None:
            self.jurors = [LLM(model=m) for m in config.JURORS]
            self.judge = LLM(model=config.JUDGE)
        else:
            self.ensemble = False
        self.use_cache = use_cache
        self.samples = samples

    # ---- single sample with quote validation + retry ----
    def _sample(self, rec, tx_block, checklist, cr, temperature, llm=None, extra=""):
        llm = llm or self.llm
        tool = prompts.tool_schema(cr)
        base = prompts.criterion_block(checklist, cr) + extra
        user = base
        out = None
        ev_retry_used = False
        last_valid = None
        for attempt in range(3):
            out = llm.call(prompts.SYSTEM, tx_block, user, tool, temperature)
            val = _coerce(cr, out.get("value"))
            valid, invalid = validate_evidence(rec, out.get("evidence"))
            if val is None:
                user = base + f"\n\nВНИМАНИЕ: значение {out.get('value')!r} недопустимо. Используй только: {prompts.allowed_values(cr)}."
                continue
            last_valid = (out, val, valid, invalid, attempt)
            if invalid and attempt < 2:
                bad = "; ".join(f"[{e.get('turn')}] «{e.get('quote')}»" for e in invalid)
                user = base + (
                    f"\n\nВНИМАНИЕ: в прошлой попытке цитаты не найдены в транскрипте дословно: {bad}. "
                    "Цитируй ТОЛЬКО точные подстроки текста указанной реплики.")
                continue
            problem = _check_evidence(cr, val, valid, out)
            if problem and not ev_retry_used and attempt < 2:
                ev_retry_used = True
                user = base + "\n\nВНИМАНИЕ: " + _HINTS[problem]
                continue
            return self._finish(cr, out, val, valid, invalid, attempt + 1, problem)
        if last_valid is not None:
            out, val, valid, invalid, attempt = last_valid
            return self._finish(cr, out, val, valid, invalid, attempt + 1, _check_evidence(cr, val, valid, out))
        raise RuntimeError(f"criterion {cr['id']}: model failed to return a valid value (last: {out and out.get('value')!r})")

    @staticmethod
    def _finish(cr, out, val, valid, invalid, attempts, problem):
        out = dict(out)
        out["value"], out["evidence"], out["rejected_quotes"], out["attempts"] = val, valid, invalid, attempts
        if problem == "gross_no_operator_quote":
            out["needs_review"] = True   # do NOT silently flip YES -> NO; a human must decide
            out["evidence_warning"] = "08=YES without a verified operator quote"
        elif problem:
            out["evidence_warning"] = problem
        return out

    def _jury(self, rec, tx_block, checklist, cr, key):
        """Jurors (haiku + sonnet) vote in parallel; on disagreement the frontier judge (opus) decides."""
        def safe(m):
            try:
                return self._sample(rec, tx_block, checklist, cr, 0.0, llm=m), None
            except Exception as e:   # one juror failing must not fail the criterion
                return None, f"{m.model}: {e}"
        with ThreadPoolExecutor(max_workers=len(self.jurors)) as ex:
            pairs = list(ex.map(safe, self.jurors))
        juror_errors = [err for _, err in pairs if err]
        live = [(i, m, o) for i, (m, (o, _)) in enumerate(zip(self.jurors, pairs, strict=True)) if o is not None]
        outs = [o for _, _, o in live]
        votes = {f"{i}:{m.model}": o["value"] for i, m, o in live}
        agree = len(outs) >= 2 and len(set(map(str, votes.values()))) == 1
        if not outs:   # all jurors failed → the judge decides alone (raises if it fails too → fail-closed)
            chosen, decided_by = self._sample(rec, tx_block, checklist, cr, 0.0, llm=self.judge), f"judge_only:{self.judge.model}"
            chosen["evidence_warning"] = "jurors_failed"
        elif len(outs) == 1:   # one juror failed → the judge reviews the surviving verdict
            o = outs[0]
            extra = (f"\n\nМнение эксперта: значение {o['value']!r}. Обоснование: {o.get('justification','')}\n"
                     "Проверь транскрипт сам, строго по правилам критерия, и вынеси окончательное решение.")
            try:
                chosen, decided_by = self._sample(rec, tx_block, checklist, cr, 0.0, llm=self.judge, extra=extra), self.judge.model
            except Exception as e:
                juror_errors.append(f"{self.judge.model}: {e}")
                chosen, decided_by = o, f"single:{live[0][1].model}"
            chosen["evidence_warning"] = chosen.get("evidence_warning") or "juror_failed"
        elif agree:
            chosen, decided_by = outs[-1], "consensus"   # strongest juror's explanation
        else:
            opinions = "\n".join(
                f"- Эксперт {i+1}: значение {o['value']!r}. Обоснование: {o.get('justification','')} Рассуждение: {o.get('reasoning','')}"
                + (" Цитаты: " + "; ".join(f"[{e['turn']}] «{e['quote']}»" for e in o.get("evidence") or []) if o.get("evidence") else "")
                for i, o in enumerate(outs))
            extra = ("\n\nДва независимых эксперта разошлись во мнениях:\n" + opinions +
                     "\nПроверь транскрипт сам, строго по правилам критерия, и вынеси окончательное решение. "
                     "Не усредняй и не выбирай по большинству: решай по фактам.")
            chosen, decided_by = self._sample(rec, tx_block, checklist, cr, 0.0, llm=self.judge, extra=extra), self.judge.model
        final = chosen["value"]
        if juror_errors:
            chosen = {**chosen, "juror_errors": juror_errors}
        all_votes = list(votes.values()) + ([final] if not agree else [])
        return {
            "criterion_id": cr["id"], "name": cr["name"], "type": cr["type"], "value": final,
            "is_na": final == "NA", "scale_step": final if cr["type"] == "scale" else None,
            "justification": chosen.get("justification", ""), "reasoning": chosen.get("reasoning", ""),
            "evidence": chosen.get("evidence", []), "rejected_quotes": chosen.get("rejected_quotes", []),
            "missing_where": chosen.get("missing_where", ""), "na_reason": chosen.get("na_reason", ""),
            "confidence": float(chosen.get("confidence") or 0), "votes": all_votes,
            "jury": votes, "decided_by": decided_by,
            "evidence_warning": chosen.get("evidence_warning", ""), "needs_review": bool(chosen.get("needs_review")),
            "client_requests": chosen.get("client_requests") or [],
            "self_consistency_disagreement": not agree, "juror_errors": chosen.get("juror_errors", []),
            "criterion_hash": key.split("_")[0], "cached": False,
        }

    def evaluate_criterion(self, rec, checklist, cr):
        import hashlib
        msig = (f"J{config.JURORS}{config.JUDGE}" if self.ensemble else self.llm.model)
        key = f"{ckl.criterion_hash(checklist, cr)}{hashlib.sha1(msig.encode()).hexdigest()[:6]}_{rec['hash']}"
        cp = config.CACHE_DIR / f"{key}.json"
        if self.use_cache and cp.exists():
            res = json.loads(cp.read_text())
            res["cached"] = True
            return res
        a = checklist["assumptions"]
        n = int(self.samples or a.get("self_consistency_samples", 3))
        thr = float(a.get("confidence_threshold", 0.8))
        tx_block = prompts.transcript_block(rec, data.format_transcript(rec, a.get("ignore_internal_hold_turns", True)),
                                            data.format_facts(data.compute_facts(rec)))
        if self.ensemble:
            res = self._jury(rec, tx_block, checklist, cr, key)
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            store_atomic(cp, json.dumps(res, ensure_ascii=False))
            return res
        first = self._sample(rec, tx_block, checklist, cr, 0.0)
        samples = [first]
        conf = float(first.get("confidence") or 0)
        if n > 1 and conf < thr:
            with ThreadPoolExecutor(max_workers=n - 1) as ex2:   # extra samples in parallel
                samples += list(ex2.map(lambda _: self._sample(rec, tx_block, checklist, cr, SC_TEMPERATURE), range(n - 1)))
        votes = [s["value"] for s in samples]
        cnt = Counter(votes)
        top_n = cnt.most_common(1)[0][1]
        winners = [v for v, c in cnt.items() if c == top_n]
        if len(winners) == 1:
            final = winners[0]
        elif cr["type"] == "scale" and all(isinstance(v, int) for v in votes):
            s = sorted(votes)
            final = s[len(s) // 2]   # median on tie
        else:
            final = first["value"]
        chosen = next(s for s in samples if s["value"] == final)
        res = {
            "criterion_id": cr["id"], "name": cr["name"], "type": cr["type"], "value": final,
            "is_na": final == "NA", "scale_step": final if cr["type"] == "scale" else None,
            "justification": chosen.get("justification", ""), "reasoning": chosen.get("reasoning", ""),
            "evidence": chosen.get("evidence", []), "rejected_quotes": chosen.get("rejected_quotes", []),
            "missing_where": chosen.get("missing_where", ""), "na_reason": chosen.get("na_reason", ""),
            "confidence": conf, "votes": votes, "self_consistency_disagreement": len(cnt) > 1,
            "evidence_warning": chosen.get("evidence_warning", ""), "needs_review": bool(chosen.get("needs_review")),
            "client_requests": chosen.get("client_requests") or [],
            "criterion_hash": key.split("_")[0], "cached": False,
        }
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        store_atomic(cp, json.dumps(res, ensure_ascii=False))
        return res

    def evaluate_many(self, records, checklist_override=None, progress=None, concurrency=None, batch_dialogs=None):
        """Flatten (dialog x criterion) into one thread pool -> ~10 dialogs in a couple of minutes."""
        t0 = time.time()
        jobs = []
        cls = {}
        for rec in records:
            cid = checklist_override or ckl.pick_for_channel(rec["kind"])
            if cid not in cls:
                cls[cid] = ckl.load(cid)
            for cr in cls[cid]["criteria"]:
                jobs.append((rec, cid, cr))
        results = {r["conversation_id"]: {} for r in records}
        done = 0
        errors = []
        # Dialog-major batches; inside a batch two phases: the first criterion of every dialog warms the
        # prompt cache (stable tool schema + system + transcript block), then the remaining criteria of
        # that dialog read it at ~10% of the input price while the cache is still within its TTL.
        first_ids = {}
        for rec, _cid, cr in jobs:
            first_ids.setdefault(rec["conversation_id"], cr["id"])
        bs = max(1, int(batch_dialogs or config.BATCH_DIALOGS))
        order = [r["conversation_id"] for r in records]
        batches = [set(order[i:i + bs]) for i in range(0, len(order), bs)]
        with ThreadPoolExecutor(max_workers=concurrency or config.CONCURRENCY) as ex:
            for batch in batches:
                bjobs = [j for j in jobs if j[0]["conversation_id"] in batch]
                phase1 = [j for j in bjobs if first_ids[j[0]["conversation_id"]] == j[2]["id"]]
                phase2 = [j for j in bjobs if first_ids[j[0]["conversation_id"]] != j[2]["id"]]
                for phase in (phase1, phase2):
                    futs = {ex.submit(self.evaluate_criterion, rec, cls[cid], cr): (rec, cid, cr) for rec, cid, cr in phase}
                    for f in as_completed(futs):
                        rec, cid, cr = futs[f]
                        try:
                            results[rec["conversation_id"]][cr["id"]] = f.result()
                        except Exception as e:
                            errors.append(f"{rec['conversation_id']}/{cr['id']}: {e}")
                            results[rec["conversation_id"]][cr["id"]] = {
                                "criterion_id": cr["id"], "name": cr["name"], "type": cr["type"], "value": None,
                                "error": str(e), "failed": True, "evidence": [],
                                "justification": "ERROR: evaluation failed — dialog marked incomplete (fail-closed)"}
                        done += 1
                        if progress:
                            progress(done, len(jobs))
        out = []
        for rec in records:
            cid = checklist_override or ckl.pick_for_channel(rec["kind"])
            out.append(build_dialog_result(rec, cls[cid], [results[rec["conversation_id"]][c["id"]] for c in cls[cid]["criteria"]]))
        meta = {"elapsed_s": round(time.time() - t0, 1), "usage": dict(USAGE), "cost_usd": cost_usd(),
                "provider": self.llm.provider,
                "model": (f"jury[{'+'.join(config.JURORS)}] judge={config.JUDGE}" if self.ensemble else self.llm.model),
                "reproducibility": {
                    "prompt_version": prompts.PROMPT_VERSION,
                    "jurors": list(config.JURORS) if self.ensemble else [self.llm.model],
                    "judge": config.JUDGE if self.ensemble else None,
                    "sampling": "provider default (Anthropic endpoint rejects temperature)" if self.llm.provider == "anthropic"
                    else "temperature 0 (first sample)",
                    "self_consistency_samples": self.samples,
                    "batch_dialogs": batch_dialogs or config.BATCH_DIALOGS,
                    "checklists": {c: {"version_hash": hashlib.sha256(json.dumps(
                        {cr["id"]: ckl.criterion_hash(cl, cr) for cr in cl["criteria"]}, sort_keys=True).encode()).hexdigest()[:12],
                        "criterion_hashes": {cr["id"]: ckl.criterion_hash(cl, cr) for cr in cl["criteria"]},
                        "weights": {cr["id"]: cr.get("weight") for cr in cl["criteria"]}} for c, cl in cls.items()},
                },
                "errors": errors, "incomplete_dialogs": [d["conversation_id"] for d in out if d["status"] != "complete"]}
        return out, meta


def _failed(r):
    return bool(r.get("error") or r.get("failed")) or r.get("value") is None


def build_dialog_result(rec, checklist, crit_results, overrides=None):
    """Fail-closed: if any criterion failed, the dialog is `incomplete` and `total` is None
    (a failed criterion is NEVER treated as NA). `provisional_total` shows the score over the rest."""
    a = checklist["assumptions"]
    ai_vals = {r["criterion_id"]: r.get("value") for r in crit_results}
    ai = scoring.compute_total(checklist["criteria"], ai_vals, a["rounding"], a["decimals"])
    failed = [r["criterion_id"] for r in crit_results if _failed(r)]
    flags = [r["criterion_id"] for r in crit_results if r.get("needs_review")]
    warns = [f"{r['criterion_id']}: {r['evidence_warning']}" for r in crit_results if r.get("evidence_warning")]
    res = {"conversation_id": rec["conversation_id"], "channel": rec.get("channel"), "language": rec.get("language"),
           "kind": rec["kind"], "dialog_hash": rec["hash"], "checklist_id": checklist["id"],
           "criteria": crit_results, "status": "incomplete" if failed else "complete",
           "failed_criteria": failed, "needs_review": flags,
           "warnings": list(rec.get("warnings") or []) + warns,
           "total": None if failed else ai["total"], "provisional_total": ai["total"] if failed else None,
           "gross_error": ai["gross_error"]}
    return apply_overrides(res, checklist, overrides or {})


def validate_override(cr, value):
    """Return the canonical value or raise ValueError (-> HTTP 400)."""
    v = ckl.strict_value(cr, value)
    if v is None:
        raise ValueError(f"invalid value {value!r} for criterion {cr['id']}; allowed: {prompts.allowed_values(cr)}")
    return v


def apply_overrides(res, checklist, overrides):
    """Human overrides live separately; recompute the human total and a diff.
    A human value can complete a failed criterion; otherwise the human total stays None too."""
    a = checklist["assumptions"]
    vals = {}
    diff = []
    missing = []
    for r in res["criteria"]:
        o = overrides.get(r["criterion_id"])
        v = r.get("value")
        if o and "value" in o and o["value"] is not None:
            cr = next(c for c in checklist["criteria"] if c["id"] == r["criterion_id"])
            hv = validate_override(cr, o["value"])
            if hv != v:
                diff.append({"criterion_id": r["criterion_id"], "ai": v, "human": hv})
            v = hv
        elif _failed(r):
            missing.append(r["criterion_id"])
        vals[r["criterion_id"]] = v
    hum = scoring.compute_total(checklist["criteria"], vals, a["rounding"], a["decimals"])
    res["human"] = {"overrides": overrides, "values": vals, "total": None if missing else hum["total"],
                    "gross_error": hum["gross_error"], "diff": diff, "reviewed": bool(overrides),
                    "status": "incomplete" if missing else "complete"}
    return res


def rescore(results_doc):
    """Recompute totals after weight changes without calling the LLM."""
    for d in results_doc["dialogs"]:
        cl = ckl.load(d["checklist_id"])
        vals = {r["criterion_id"]: r.get("value") for r in d["criteria"]}
        t = scoring.compute_total(cl["criteria"], vals, cl["assumptions"]["rounding"], cl["assumptions"]["decimals"])
        failed = [r["criterion_id"] for r in d["criteria"] if _failed(r)]
        d["status"], d["failed_criteria"] = ("incomplete" if failed else "complete"), failed
        d["total"], d["provisional_total"] = (None, t["total"]) if failed else (t["total"], None)
        d["gross_error"] = t["gross_error"]
        apply_overrides(d, cl, d.get("human", {}).get("overrides", {}))
    return results_doc
