"""Per-criterion LLM evaluation with evidence validation, self-consistency and a per-criterion cache."""
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import checklist as ckl, config, data, prompts, scoring
from .llm import LLM, USAGE, cost_usd


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
        q = _norm(str(e.get("quote", "")))
        try:
            ti = int(e.get("turn"))
        except Exception:
            ti = -1
        if not q:
            continue
        q = q.strip("… .")
        parts = [p.strip() for p in re.split(r"\s*(?:\.\.\.|…)\s*", q) if p.strip()]
        def ok(t):
            tn = _norm(t.get("text", ""))
            return all(p in tn for p in parts)
        if 0 <= ti < len(turns) and ok(turns[ti]):
            valid.append({"turn": ti, "quote": e["quote"], "speaker": turns[ti].get("speaker")})
            continue
        hit = next((t for t in turns if ok(t)), None)
        if hit:
            valid.append({"turn": hit["index"], "quote": e["quote"], "speaker": hit.get("speaker"), "turn_corrected": True})
        else:
            invalid.append(e)
    return valid, invalid


def _coerce(cr, v):
    v = str(v).strip().upper()
    if v == "NA" and cr.get("allow_na"):
        return "NA"
    if cr["type"] == "scale":
        try:
            return int(float(v))
        except Exception:
            return None
    return v if v in ("YES", "NO") else None


class Evaluator:
    def __init__(self, llm=None, use_cache=True, samples=None, ensemble=None):
        self.llm = llm or LLM()
        self.ensemble = config.ENSEMBLE if ensemble is None else ensemble
        if self.ensemble and self.llm.provider == "anthropic" and llm is None:
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
        for attempt in range(3):
            out = llm.call(prompts.SYSTEM, tx_block, user, tool, temperature)
            val = _coerce(cr, out.get("value"))
            valid, invalid = validate_evidence(rec, out.get("evidence"))
            if val is None:
                user = base + f"\n\nВНИМАНИЕ: значение {out.get('value')!r} недопустимо. Используй только: {prompts.allowed_values(cr)}."
                continue
            if invalid and attempt < 2:
                bad = "; ".join(f"[{e.get('turn')}] «{e.get('quote')}»" for e in invalid)
                user = base + (
                    f"\n\nВНИМАНИЕ: в прошлой попытке цитаты не найдены в транскрипте дословно: {bad}. "
                    "Цитируй ТОЛЬКО точные подстроки текста указанной реплики.")
                continue
            out["value"] = val
            out["evidence"] = valid
            out["rejected_quotes"] = invalid
            out["attempts"] = attempt + 1
            return out
        if out is not None and _coerce(cr, out.get("value")) is not None:
            out["value"] = _coerce(cr, out.get("value"))
            out["evidence"], out["rejected_quotes"] = validate_evidence(rec, out.get("evidence"))
            out["attempts"] = 3
            return out
        raise RuntimeError(f"criterion {cr['id']}: model failed to return a valid value")

    def _jury(self, rec, tx_block, checklist, cr, key):
        """Jurors (haiku + sonnet) vote in parallel; on disagreement the frontier judge (opus) decides."""
        with ThreadPoolExecutor(max_workers=len(self.jurors)) as ex:
            outs = list(ex.map(lambda m: self._sample(rec, tx_block, checklist, cr, 0.0, llm=m), self.jurors))
        votes = {m.model: o["value"] for m, o in zip(self.jurors, outs)}
        agree = len(set(map(str, votes.values()))) == 1
        if agree:
            chosen, decided_by = outs[-1], "consensus"   # strongest juror's explanation
        else:
            opinions = "\n".join(
                f"- Эксперт {i+1}: значение {o['value']!r}. Обоснование: {o.get('justification','')} Рассуждение: {o.get('reasoning','')}"
                for i, o in enumerate(outs))
            extra = ("\n\nДва независимых эксперта разошлись во мнениях:\n" + opinions +
                     "\nПроверь транскрипт сам, строго по правилам критерия, и вынеси окончательное решение. "
                     "Не усредняй и не выбирай по большинству: решай по фактам.")
            chosen, decided_by = self._sample(rec, tx_block, checklist, cr, 0.0, llm=self.judge, extra=extra), self.judge.model
        final = chosen["value"]
        all_votes = list(votes.values()) + ([final] if not agree else [])
        return {
            "criterion_id": cr["id"], "name": cr["name"], "type": cr["type"], "value": final,
            "is_na": final == "NA", "scale_step": final if cr["type"] == "scale" else None,
            "justification": chosen.get("justification", ""), "reasoning": chosen.get("reasoning", ""),
            "evidence": chosen.get("evidence", []), "rejected_quotes": chosen.get("rejected_quotes", []),
            "missing_where": chosen.get("missing_where", ""), "na_reason": chosen.get("na_reason", ""),
            "confidence": float(chosen.get("confidence") or 0), "votes": all_votes,
            "jury": votes, "decided_by": decided_by,
            "self_consistency_disagreement": not agree,
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
        tx_block = prompts.transcript_block(rec, data.format_transcript(rec, a.get("ignore_internal_hold_turns", True)))
        if self.ensemble:
            res = self._jury(rec, tx_block, checklist, cr, key)
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(res, ensure_ascii=False))
            return res
        first = self._sample(rec, tx_block, checklist, cr, 0.0)
        samples = [first]
        conf = float(first.get("confidence") or 0)
        if n > 1 and conf < thr:
            with ThreadPoolExecutor(max_workers=n - 1) as ex2:   # extra samples in parallel
                samples += list(ex2.map(lambda _: self._sample(rec, tx_block, checklist, cr, 1.0), range(n - 1)))
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
            "criterion_hash": key.split("_")[0], "cached": False,
        }
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(res, ensure_ascii=False))
        return res

    def evaluate_many(self, records, checklist_override=None, progress=None, concurrency=None):
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
        # Two phases: first criterion of every dialog warms the prompt cache (transcript block),
        # then the remaining criteria of that dialog read it at 10% of the input price.
        first_ids = {}
        for rec, cid, cr in jobs:
            first_ids.setdefault(rec["conversation_id"], cr["id"])
        phase1 = [j for j in jobs if first_ids[j[0]["conversation_id"]] == j[2]["id"]]
        phase2 = [j for j in jobs if first_ids[j[0]["conversation_id"]] != j[2]["id"]]
        with ThreadPoolExecutor(max_workers=concurrency or config.CONCURRENCY) as ex:
          for phase in (phase1, phase2):
            futs = {ex.submit(self.evaluate_criterion, rec, cls[cid], cr): (rec, cid, cr) for rec, cid, cr in phase}
            for f in as_completed(futs):
                  rec, cid, cr = futs[f]
                  try:
                      results[rec["conversation_id"]][cr["id"]] = f.result()
                  except Exception as e:
                      errors.append(f"{rec['conversation_id']}/{cr['id']}: {e}")
                      results[rec["conversation_id"]][cr["id"]] = {"criterion_id": cr["id"], "name": cr["name"],
                          "type": cr["type"], "value": None, "error": str(e), "evidence": [], "justification": "ERROR"}
                  done += 1
                  if progress:
                      progress(done, len(jobs))
        out = []
        for rec in records:
            cid = checklist_override or ckl.pick_for_channel(rec["kind"])
            out.append(build_dialog_result(rec, cls[cid], [results[rec["conversation_id"]][c["id"]] for c in cls[cid]["criteria"]]))
        meta = {"elapsed_s": round(time.time() - t0, 1), "usage": dict(USAGE), "cost_usd": cost_usd(),
                "provider": self.llm.provider, "model": (f"jury[{'+'.join(config.JURORS)}] judge={config.JUDGE}" if self.ensemble else self.llm.model), "errors": errors}
        return out, meta


def build_dialog_result(rec, checklist, crit_results, overrides=None):
    a = checklist["assumptions"]
    ai_vals = {r["criterion_id"]: r.get("value") for r in crit_results}
    ai = scoring.compute_total(checklist["criteria"], ai_vals, a["rounding"], a["decimals"])
    res = {"conversation_id": rec["conversation_id"], "channel": rec.get("channel"), "language": rec.get("language"),
           "kind": rec["kind"], "dialog_hash": rec["hash"], "checklist_id": checklist["id"],
           "criteria": crit_results, "total": ai["total"], "gross_error": ai["gross_error"]}
    return apply_overrides(res, checklist, overrides or {})


def apply_overrides(res, checklist, overrides):
    """Human overrides live separately; recompute the human total and a diff."""
    a = checklist["assumptions"]
    vals = {}
    diff = []
    for r in res["criteria"]:
        o = overrides.get(r["criterion_id"])
        v = r.get("value")
        if o and "value" in o and o["value"] is not None:
            hv = o["value"]
            cr = next(c for c in checklist["criteria"] if c["id"] == r["criterion_id"])
            hv = hv if hv in ("YES", "NO", "NA") else int(hv)
            if cr["type"] == "scale" and hv not in ("NA",) and hv not in cr["scale"]:
                raise ValueError("bad override value")
            if hv != v:
                diff.append({"criterion_id": r["criterion_id"], "ai": v, "human": hv})
            v = hv
        vals[r["criterion_id"]] = v
    hum = scoring.compute_total(checklist["criteria"], vals, a["rounding"], a["decimals"])
    res["human"] = {"overrides": overrides, "values": vals, "total": hum["total"], "gross_error": hum["gross_error"],
                    "diff": diff, "reviewed": bool(overrides)}
    return res


def rescore(results_doc):
    """Recompute totals after weight changes without calling the LLM."""
    for d in results_doc["dialogs"]:
        cl = ckl.load(d["checklist_id"])
        vals = {r["criterion_id"]: r.get("value") for r in d["criteria"]}
        t = scoring.compute_total(cl["criteria"], vals, cl["assumptions"]["rounding"], cl["assumptions"]["decimals"])
        d["total"], d["gross_error"] = t["total"], t["gross_error"]
        apply_overrides(d, cl, d.get("human", {}).get("overrides", {}))
    return results_doc
