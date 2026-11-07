"""Cascade router (preset ``cascade``): three tiers, cheapest first.

  tier 1  open-weight model on RunPod/vLLM (or Jev, CQA_ROUTER_FIRST=jev) scores every criterion. Confidence =
          probability of the emitted value tokens from vLLM logprobs (JSON-schema constrained decoding, so the
          value is always one of the allowed options); self-reported confidence only as a fallback. It also returns
          evidence quotes, which are validated against the transcript like on the cloud path.
  tier 2  claude-haiku-5-5 re-scores a criterion when tier 1 is unsure or unusable:
            local_error / local_invalid  — call failed, invalid value
            evidence_failed              — a quote failed validation or the evidence check complained
            low_conf                     — p(value) < T1
            hard_criterion               — 04/05/06 (CQA_ROUTER_HARD) unless p(value) >= HARD_CONF
            gross_yes                    — 08 = YES (high stakes)
  tier 3  claude-sonnet-5-5 decides when Haiku is still unsure: Haiku failed, Haiku confidence < HAIKU_LOW, or
          Haiku disagrees with tier 1 and its confidence < T2.
  speed mode (CQA_ROUTER_MODE=speed): tier 1 down (health check) or slower than LOCAL_TIMEOUT s → straight to Haiku.

Every escalated verdict carries Haiku's/Sonnet's evidence. Each criterion result has ``routing`` = {tier, route,
reasons, local/haiku/sonnet value+confidence, local_s, models}; ``evaluate_many`` adds ``meta.routing``
(share per tier, $ by model). Raw samples are cached in .cache/router/ so thresholds can be tuned offline
(scripts/bench_oss.py tune). A LoRA adapter (CQA_LOCAL_ADAPTER, served by vLLM as ``cqa-adapter``) replaces the
base model id when set. Settings: see ``settings()`` and docs/OSS_ROUTER.md.
"""
import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

import httpx

from . import checklist as ckl, config, data, evaluator as _ev, prompts
from .llm import LLM, USAGE_BY_MODEL, _add_usage

BaseEvaluator = _ev.Evaluator   # captured before install() patches the module attribute
CACHE = config.ROOT / ".cache" / "router"
ADAPTER_NAME = "cqa-adapter"


def _f(key, default):
    v = config.get(key)
    return float(default if v in (None, "") else v)


def settings(**over):
    g = config.get
    s = {"first": g("CQA_ROUTER_FIRST") or "jev",            # local (vLLM) | jev (OpenRouter decision model)
         "local_model": g("CQA_ROUTER_LOCAL_MODEL") or "qwen3-14b-awq",
         "adapter": g("CQA_LOCAL_ADAPTER") or "",             # LoRA path / HF repo; vLLM serves it as cqa-adapter
         "escalate": g("CQA_ROUTER_ESCALATE") or "claude-haiku-5-5",
         "judge": g("CQA_ROUTER_JUDGE", "claude-sonnet-5-5"),  # '' disables tier 3
         "t1": _f("CQA_ROUTER_T1", 0.7), "hard": g("CQA_ROUTER_HARD") or "04,05,06",
         "hard_conf": _f("CQA_ROUTER_HARD_CONF", 0.9), "t2": _f("CQA_ROUTER_T2", 0.7),
         "haiku_low": _f("CQA_ROUTER_HAIKU_LOW", 0.0), "gross_yes": g("CQA_ROUTER_GROSS_YES", "1") == "1",
         "mode": g("CQA_ROUTER_MODE") or "quality", "local_timeout": _f("CQA_ROUTER_LOCAL_TIMEOUT", 0),
         "logprobs": g("CQA_ROUTER_LOGPROBS", "1") == "1",
         "force": g("CQA_ROUTER_FORCE") or ""}                 # "2" / "3": call higher tiers always (tuning data)
    s.update({k: v for k, v in over.items() if v is not None})
    s["hard_set"] = {x.strip() for x in str(s["hard"]).split(",") if x.strip()}
    return s


def tier1_reasons(cr, local, s):
    """Pure: why (if at all) a tier-1 verdict goes to Haiku. ``local``: sample dict or {"error": ...}."""
    if local is None or local.get("error"):
        return ["local_error"]
    why = []
    conf = float(local.get("confidence") or 0)
    if local.get("value") is None:
        why.append("local_invalid")
    if local.get("rejected_quotes") or local.get("evidence_warning"):
        why.append("evidence_failed")
    if conf < s["t1"]:
        why.append("low_conf")
    if cr["id"] in s["hard_set"] and conf < s["hard_conf"]:
        why.append("hard_criterion")
    if local.get("decision_disagree"):
        why.append("decision_disagree")
    if s.get("gross_yes") and cr.get("gross") and local.get("value") == "YES":
        why.append("gross_yes")
    return sorted(set(why))


def tier2_reasons(local, haiku, s):
    """Pure: why (if at all) Haiku's verdict goes to Sonnet."""
    if haiku is None or haiku.get("error") or haiku.get("value") is None:
        return ["haiku_error"]
    hc = float(haiku.get("confidence") or 0)
    why = []
    if hc < s["haiku_low"]:
        why.append("haiku_low_conf")
    if local and local.get("value") is not None and str(local["value"]) != str(haiku["value"]) and hc < s["t2"]:
        why.append("disagree_low_conf")
    return why


def value_prob(content, logprobs):
    """Joint probability of the tokens spelling the ``value`` field in a JSON answer (vLLM logprobs)."""
    m = re.search(r'"value"\s*:\s*"?([^",}\s]+)', content or "")
    toks = (logprobs or {}).get("content") or []
    if not m or not toks:
        return None
    a, b, pos, lp, hit = m.start(1), m.end(1), 0, 0.0, False
    for t in toks:
        n = len(t.get("token") or "")
        if pos < b and pos + n > a:
            lp += float(t.get("logprob") or 0.0)
            hit = True
        pos += n
    return round(math.exp(lp), 4) if hit else None


class LogprobLLM(LLM):  # noqa: D101
    """Tier-1 client: JSON-schema constrained decoding on vLLM with logprobs → p(value) as confidence."""

    def __init__(self, model):
        super().__init__(provider="openai", model=model)

    def _openai(self, system, cached_user, user, tool, temperature):
        body = {"model": self.model, "temperature": temperature, "max_tokens": 3000, "logprobs": True, "top_logprobs": 1,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": cached_user + "\n\n" + user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": tool["name"],
                                                                           "schema": tool["input_schema"]}}}
        r = httpx.post(str(config.OPENAI_BASE_URL).rstrip("/") + "/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"}, timeout=600)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        _add_usage({"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}, self.model)
        ch = j["choices"][0]
        txt = ch["message"].get("content") or ""
        m = re.search(r"\{.*\}", txt, re.S)
        out = json.loads(m.group(0) if m else txt)
        out["value_token_p"] = value_prob(txt, ch.get("logprobs"))   # logged only: ~1.0 after reasoning
        return out


def options(cr):
    """Allowed values of a criterion as strings (decision-call enum)."""
    base = [str(x) for x in cr["scale"]] if cr.get("type") == "scale" else ["YES", "NO"]
    return base + (["NA"] if cr.get("allow_na") else [])


def option_probs(toks, opts):
    """Distribution over ``opts`` from the logprobs of a constrained answer ``{"value": "<opt>"}``.

    Reads top_logprobs at the first value token (all options' first tokens) and, for options that share the
    emitted first token (e.g. "10"/"15"), the next position. Raw (pre-mask) logprobs, renormalised over options."""
    txt, start = "", None
    for i, t in enumerate(toks):
        txt += t.get("token") or ""
        m = re.search(r'"value"\s*:\s*"', txt)
        if m and start is None and len(txt) >= m.end():
            start = i + 1 if len(txt) == m.end() else i
            break
    if start is None or start >= len(toks):
        return None
    first = {x["token"]: math.exp(x["logprob"]) for x in toks[start].get("top_logprobs") or []}
    chosen1 = toks[start]["token"]
    nxt = {x["token"]: math.exp(x["logprob"]) for x in (toks[start + 1].get("top_logprobs") or [])} if start + 1 < len(toks) else {}
    P = {}
    for o in opts:
        f = next((t for t in first if t and o.startswith(t)), None)
        if f is None:
            P[o] = 0.0
            continue
        rest = o[len(f):]
        if not rest:
            P[o] = first[f]
        elif f == chosen1:
            P[o] = first[f] * next((pv for t, pv in nxt.items() if t and rest.startswith(t)), 0.0)
        else:
            P[o] = first[f] / max(1, sum(1 for o2 in opts if o2.startswith(f)))   # unknown split: share evenly
    z = sum(P.values())
    return {k: round(v / z, 4) for k, v in P.items()} if z > 0 else None


class _Decider:
    """Tier-1 'decision' call: the same prompt, but the model must answer ONLY the value (JSON-schema enum, a few
    tokens, prefix-cached transcript) — so the option probabilities are read before any reasoning text commits
    the answer. (After the reasoning field, p(value) is ~1.0 and useless for routing — measured.)"""

    def __init__(self, model):
        self.model = model

    def probs(self, system, tx, user, cr):
        opts = options(cr)
        body = {"model": self.model, "temperature": 0, "max_tokens": 48, "logprobs": True, "top_logprobs": 20,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": tx + "\n\n" + user + "\n\nОтветь сразу: только поле value."}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "decide", "schema": {
                    "type": "object", "properties": {"value": {"type": "string", "enum": opts}}, "required": ["value"]}}}}
        r = httpx.post(str(config.OPENAI_BASE_URL).rstrip("/") + "/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        _add_usage({"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}, self.model)
        ch = j["choices"][0]
        P = option_probs((ch.get("logprobs") or {}).get("content") or [], opts)
        m = re.search(r'"value"\s*:\s*"([^"]*)"', ch["message"].get("content") or "")
        emitted = m.group(1) if m else None
        if P is not None:   # alignment check: the argmax option must be the value the decision call emitted
            P = {**P, "_emitted": emitted, "_aligned": emitted == max(opts, key=lambda o: P.get(o, 0))}
        return P


class _Health:
    """Cached liveness of the local endpoint (speed mode)."""
    def __init__(self, ttl=30.0):
        self.ttl, self.t, self.ok, self.lock = ttl, 0.0, True, threading.Lock()

    def up(self):
        with self.lock:
            if time.time() - self.t < self.ttl:
                return self.ok
            try:
                r = httpx.get(str(config.OPENAI_BASE_URL).rstrip("/") + "/models", timeout=3,
                              headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"})
                self.ok = r.status_code == 200
            except Exception:   # noqa: BLE001
                self.ok = False
            self.t = time.time()
            return self.ok


def _brief(o):
    return {"value": o.get("value"), "confidence": o.get("confidence"), "error": o.get("error")} if o else None


class CascadeEvaluator(BaseEvaluator):
    def __init__(self, llm=None, use_cache=True, samples=None, ensemble=None, **over):
        super().__init__(llm=llm, use_cache=use_cache, samples=samples, ensemble=False)
        self.s = settings(**over)
        model = ADAPTER_NAME if self.s["adapter"] else self.s["local_model"]
        if config.PROVIDER == "mock" or llm is not None:
            self.local = self.esc = self.llm
            self.judge_llm = self.llm if self.s["judge"] else None
        else:
            if self.s["adapter"]:   # fine-tuned adapters are trained on the JSON-only answer format (docs/FINETUNE.md)
                config.OPENAI_TOOLS = False
                self.local = LLM(provider="openai", model=model)
            else:
                self.local = LogprobLLM(model) if self.s["logprobs"] else LLM(provider="openai", model=model)
            self.esc = LLM(provider="anthropic", model=self.s["escalate"])
            self.judge_llm = LLM(provider="anthropic", model=self.s["judge"]) if self.s["judge"] else None
        self.tier1_name = "jev-1.14" if self.s["first"] == "jev" else self.local.model
        self.health = _Health()
        self.jev_memo, self._jev_lock = {}, threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=64)
        self.llm = self.esc      # rescue pass in evaluate_many uses a cloud model
        self.ensemble = False

    def _raw(self, which, llm, rec, tx, checklist, cr):
        key = hashlib.sha1(f"{llm.provider}:{llm.model}:{type(llm).__name__}:{ckl.criterion_hash(checklist, cr)}:"
                           f"{rec['hash']}{':kz' if os.environ.get('CQA_KZ_SIGNALS') == '1' else ''}".encode()).hexdigest()[:20]
        p = CACHE / f"{which}_{key}.json"
        if self.use_cache and p.exists():
            return json.loads(p.read_text())
        t = time.time()
        try:
            o = {**self._sample(rec, tx, checklist, cr, 0.0, llm=llm), "elapsed_s": round(time.time() - t, 2)}
        except Exception as e:   # noqa: BLE001
            o = {"error": str(e)[:300], "value": None, "elapsed_s": round(time.time() - t, 2)}
        if self.use_cache and not o.get("error"):
            CACHE.mkdir(parents=True, exist_ok=True)
            _ev.store_atomic(p, json.dumps(o, ensure_ascii=False))
        return o

    def _jev(self, rec, checklist, cr):
        from . import jev
        k = rec["conversation_id"]
        with self._jev_lock:
            if k not in self.jev_memo:
                t = time.time()
                try:
                    self.jev_memo[k] = jev.evaluate(rec, checklist)
                except Exception as e:   # noqa: BLE001
                    self.jev_memo[k] = {"error": str(e)[:300], "criteria": []}
                self.jev_memo[k].setdefault("latency_s", round(time.time() - t, 2))
            d = self.jev_memo[k]
        c = next((c for c in d.get("criteria") or [] if c["criterion_id"] == cr["id"]), None)
        if not c or c.get("value") is None:
            return {"error": d.get("error") or "jev: no answer", "value": None}
        return {"value": _ev._coerce(cr, c["value"]), "confidence": c.get("confidence") or 0, "evidence": [],
                "justification": c.get("justification", ""), "elapsed_s": d.get("latency_s"), "confidence_source": "jev"}

    def _tier1(self, rec, tx, checklist, cr):
        if self.s["first"] == "jev":
            return self._jev(rec, checklist, cr)
        if self.s["mode"] == "speed" and not self.health.up():
            return {"error": "local endpoint down", "value": None, "skipped": True}
        if self.s["local_timeout"] > 0:
            f = self._pool.submit(self._raw, "local", self.local, rec, tx, checklist, cr)
            try:
                return f.result(timeout=self.s["local_timeout"])
            except FutTimeout:
                return {"error": f"local slower than {self.s['local_timeout']}s", "value": None, "timeout": True}
        return self._with_probs(self._raw("local", self.local, rec, tx, checklist, cr), rec, tx, checklist, cr)

    def _with_probs(self, o, rec, tx, checklist, cr):
        """Confidence of the full answer's value = its probability under the decision call's distribution."""
        if o.get("error") or not self.s["logprobs"] or self.s["first"] == "jev":
            return o
        key = hashlib.sha1(f"decide:{self.local.model}:{ckl.criterion_hash(checklist, cr)}:{rec['hash']}".encode()).hexdigest()[:20]
        p = CACHE / f"decide_{key}.json"
        if self.use_cache and p.exists():
            P = json.loads(p.read_text())
        else:
            try:
                P = _Decider(self.local.model).probs(prompts.SYSTEM, tx, prompts.criterion_block(checklist, cr), cr)
            except Exception:   # noqa: BLE001
                P = None
            if P and self.use_cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                _ev.store_atomic(p, json.dumps(P))
        if not P:
            return {**o, "confidence_source": "self"}
        if not P.get("_aligned", True):   # misaligned token mapping → don't trust the probabilities
            return {**o, "option_probs": P, "confidence": 0.0, "confidence_source": "logprob_misaligned"}
        conf = P.get(str(o.get("value")), 0.0)   # p(parsed reasoned value) under the snap-decision distribution
        return {**o, "self_confidence": o.get("self_confidence", o.get("confidence")), "option_probs": P,
                "decision_value": P.get("_emitted"), "decision_disagree": P.get("_emitted") != str(o.get("value")),
                "confidence": conf, "confidence_source": "logprob"}

    def evaluate_criterion(self, rec, checklist, cr):
        a = checklist["assumptions"]
        tx = prompts.transcript_block(rec, data.format_transcript(rec, a.get("ignore_internal_hold_turns", True)),
                                      data.format_facts(data.compute_facts(rec, a.get("long_wait_seconds"))))
        s = self.s
        local = self._tier1(rec, tx, checklist, cr)
        why1 = tier1_reasons(cr, local, s)
        if local.get("timeout") or local.get("skipped"):
            why1 = ["speed_" + ("timeout" if local.get("timeout") else "local_down")]
        if s["first"] == "jev":   # Jev has no quotes: an empty evidence list is expected, not a failure
            why1 = [w for w in why1 if w != "evidence_failed"]
        tier, chosen, haiku, sonnet, why2 = 1, local, None, None, []
        models = [self.tier1_name]
        if why1 or "2" in s["force"] or "3" in s["force"]:
            haiku = self._raw("escalate", self.esc, rec, tx, checklist, cr)
            models.append(self.esc.model)
            why2 = tier2_reasons(local, haiku, s) if self.judge_llm else []
            if self.judge_llm and (why2 or "3" in s["force"]):
                sonnet = self._raw("judge", self.judge_llm, rec, tx, checklist, cr)
                models.append(self.judge_llm.model)
            if why1:
                if why2 and sonnet and not sonnet.get("error"):
                    tier, chosen = 3, sonnet
                elif not haiku.get("error"):
                    tier, chosen = 2, haiku
                elif local.get("value") is None:
                    raise RuntimeError(f"all tiers failed: {local.get('error')} / {haiku.get('error')}")
        routing = {"tier": tier, "route": {1: "local", 2: "haiku", 3: "sonnet"}[tier], "reasons": why1,
                   "tier2_reasons": why2 if why1 else [], "local_value": local.get("value"),
                   "local_conf": local.get("confidence"), "local_conf_source": local.get("confidence_source", "self"),
                   "local_option_probs": local.get("option_probs"), "decision_value": local.get("decision_value"),
                   "local_s": local.get("elapsed_s"), "local_error": local.get("error"),
                   "local_evidence_ok": not (local.get("rejected_quotes") or local.get("evidence_warning")),
                   "haiku": _brief(haiku), "sonnet": _brief(sonnet), "models": models}
        decided = {1: self.tier1_name, 2: self.esc.model, 3: self.judge_llm.model if self.judge_llm else None}[tier]
        res = _ev._result_dict(cr, chosen, chosen["value"], decided_by=f"cascade:tier{tier}:{decided}",
                               votes=[o["value"] for o in (local, haiku, sonnet) if o and o.get("value") is not None],
                               routing=routing)
        return _ev._flag(cr, res)

    def evaluate_many(self, records, *args, **kw):
        before = {m: dict(u) for m, u in USAGE_BY_MODEL.items()}
        out, meta = super().evaluate_many(records, *args, **kw)
        crit = [c for d in out for c in d["criteria"] if c.get("routing")]
        tiers, reasons = {1: 0, 2: 0, 3: 0}, {}
        for c in crit:
            tiers[c["routing"]["tier"]] += 1
            for r in c["routing"]["reasons"] + c["routing"]["tier2_reasons"]:
                reasons[r] = reasons.get(r, 0) + 1
        usd = {m: round(u["usd"] - before.get(m, {}).get("usd", 0.0), 5) for m, u in USAGE_BY_MODEL.items()
               if m.startswith("claude") and u["usd"] - before.get(m, {}).get("usd", 0.0) > 0}
        n = max(len(crit), 1)
        meta["routing"] = {"settings": {k: v for k, v in self.s.items() if k != "hard_set"}, "criteria": len(crit),
                           "tier_counts": tiers, "tier_share": {k: round(v / n, 3) for k, v in tiers.items()},
                           "escalated_share": round((tiers[2] + tiers[3]) / n, 3), "reasons": reasons,
                           "cloud_usd_by_model": usd, "cloud_usd": round(sum(usd.values()), 5)}
        meta["model"] = (f"cascade[{self.tier1_name}→{self.esc.model}"
                         + (f"→{self.judge_llm.model}" if self.judge_llm else "") + "]")
        meta["cost_usd"] = meta["routing"]["cloud_usd"]
        return out, meta


class HybridEvaluator(CascadeEvaluator):
    """Preset ``hybrid``: cascade + KZ-aware parser signals in FACTS (CQA_KZ_SIGNALS=1); an 08 red flag forces Sonnet."""

    def evaluate_criterion(self, rec, checklist, cr):
        from . import rules_kz
        facts = data.compute_facts(rec, checklist["assumptions"].get("long_wait_seconds"))
        flags = rules_kz.red_flags(rec) if cr["id"] == "08" else []
        if flags:   # rules never say NO on 08; red flags force the strongest model and attach the turns
            llm = self.judge_llm or self.esc
            rec_tx = prompts.transcript_block(rec, data.format_transcript(rec, checklist["assumptions"].get(
                "ignore_internal_hold_turns", True)), data.format_facts(facts))
            o = self._raw("judge" if self.judge_llm else "escalate", llm, rec, rec_tx, checklist, cr)
            if not o.get("error"):
                res = _ev._result_dict(cr, o, o["value"], decided_by=f"hybrid:redflag:{llm.model}",
                                       routing={"tier": 3, "route": "sonnet", "reasons": ["rules_kz_red_flag"],
                                                "tier2_reasons": [], "red_flag_turns": flags, "models": [llm.model]})
                return _ev._flag(cr, res)
        return super().evaluate_criterion(rec, checklist, cr)

    def evaluate_many(self, records, *args, **kw):
        out, meta = super().evaluate_many(records, *args, **kw)
        crit = [c for d in out for c in d["criteria"] if c.get("routing")]
        r = meta.setdefault("routing", {})
        r["red_flag_escalations"] = sum("rules_kz_red_flag" in c["routing"].get("reasons", []) for c in crit)
        meta["model"] = "kz_signals+" + meta.get("model", "cascade")
        return out, meta


if os.environ.get("CQA_ROUTER_POLICY"):   # advanced per-criterion policy (docs/ROUTER.md); unset = unchanged
    from . import routing_policy as _rp
    _rp.patch(globals(), os.environ["CQA_ROUTER_POLICY"])


def vllm_lora_args(adapter=None):
    """Extra vLLM flags for CQA_LOCAL_ADAPTER (a fine-tuned LoRA; see cqa/finetune/)."""
    adapter = adapter if adapter is not None else (config.get("CQA_LOCAL_ADAPTER") or "")
    return ["--enable-lora", "--max-lora-rank 64", f"--lora-modules {ADAPTER_NAME}={adapter}"] if adapter else []


def install(cls=None):
    """Make ``cqa.evaluator.Evaluator`` (and modules that already imported it) the cascade (or hybrid) evaluator."""
    import sys
    cls = cls or CascadeEvaluator
    _ev.Evaluator = cls
    for name in ("cqa.gold", "cqa.__main__", "cqa.server"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "Evaluator", None) in (BaseEvaluator, CascadeEvaluator, HybridEvaluator):
            m.Evaluator = cls
