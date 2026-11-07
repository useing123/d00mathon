"""LLM backends.

* ``anthropic`` — Claude via tool use (strict JSON schema, prompt caching).
* ``openai``    — any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio) in JSON mode.
* ``mock``      — deterministic offline heuristic (:class:`MockBackend`) for tests, CI and demos
  without an API key. It is NOT an accuracy baseline.
"""
import json
import random
import re
import threading
import time

import httpx

from . import config, pricing

USAGE = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "calls": 0, "usd": 0.0}
USAGE_BY_MODEL: dict = {}   # model id -> same keys as USAGE (process-global; used by scripts/bench_matrix.py)


def _price(model: str | None) -> tuple[float, float]:
    """$/MTok (input, output) for a model name — see cqa/pricing.py for the full table and sources."""
    return pricing.price(model)[:2]


_lock = threading.Lock()


def _add_usage(u, model=None):
    """``u``: Anthropic-style usage (input_tokens = uncached input). Cost uses the cqa/pricing.py table."""
    toks = {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    usd = 0.0 if (model or "").startswith("mock") else pricing.usd(
        model, toks["input_tokens"], toks["output_tokens"], toks["cache_read_input_tokens"], toks["cache_creation_input_tokens"])
    with _lock:
        per = USAGE_BY_MODEL.setdefault(model or "?", {**{k: 0 for k in toks}, "calls": 0, "usd": 0.0})
        for d in (USAGE, per):
            d["usd"] += usd
            d["calls"] += 1
            for k, v in toks.items():
                d[k] += v


def cost_usd(usage=None, model=None):
    u = usage or USAGE
    if config.PROVIDER != "anthropic":   # local OpenAI-compatible servers: GPU time, not tokens
        return 0.0
    return round(float(u.get("usd", 0.0)), 4)


def cloud_route(model: str | None) -> str:
    """Cloud vendor for a model id when CQA_PROVIDER=anthropic (the "cloud" mode): lets jury members mix vendors."""
    m = (model or "").lower()
    if m.startswith(("gpt-", "o3", "o4")) or "codex" in m:
        return "openai_cloud"     # OpenAI Responses API (needed for gpt-5.3-codex and tool use on gpt-6-luna)
    if m.startswith("gemini-"):
        return "gemini"           # Gemini OpenAI-compatible endpoint
    return "anthropic"


_TURN_RE = re.compile(r"^\[(\d+)\] \([^)]*\) ([A-Z]+)(?: \[[^\]]*\])?: (.*)$", re.M)
_GREET = ("здравств", "добр", "привет", "hello", "hi ", "good ", "сәлем", "қайырлы")
_NAME = ("меня зовут", "my name", "this is", "атым", "менің")
_BYE = ("до свидан", "всего доброго", "хорошего", "goodbye", "have a nice", "сау болыңыз", "көріскенше")
_RUDE = ("дурак", "идиот", "stupid", "заткн")


class MockBackend:
    """Deterministic keyword heuristic that mimics the tool-call contract of a real LLM.

    Parses the numbered transcript from the prompt, picks a value from the allowed enum and
    returns verbatim quotes (so evidence validation passes). Used with ``CQA_PROVIDER=mock``.
    """

    model = "mock-heuristic"

    def call(self, system: str, cached_user: str, user: str, tool: dict, temperature: float = 0.0) -> dict:
        turns = [(int(i), sp, tx) for i, sp, tx in _TURN_RE.findall(cached_user)]
        ops = [t for t in turns if t[1] == "OPERATOR"]
        enum = tool["input_schema"]["properties"]["value"].get("enum")
        if not enum:
            mv = re.search(r"Допустимые значения[^:]*: (.*)", user)
            enum = [x.strip() for x in mv.group(1).split("|")] if mv else ["YES", "NO"]
        m = re.search(r"КРИТЕРИЙ (\S+): (.*)", user)
        cid, cname = (m.group(1), m.group(2).lower()) if m else ("?", "")
        low = [(i, tx.lower()) for i, _, tx in ops]
        ev, value = [], None
        if not ops:
            value = "NA" if "NA" in enum else ("NO" if "NO" in enum else enum[0])
        elif "груб" in cname or cid == "08":
            hit = next(((i, tx) for i, _, tx in ops if any(r in tx.lower() for r in _RUDE)), None)
            value = "YES" if hit else "NO"
            ev = [hit] if hit else []
        elif cid == "01":
            i, tx = low[0]
            value = "YES" if any(g in tx for g in _GREET) and any(n in tx for n in _NAME) else "NO"
            ev = [(i, ops[0][2])]
        elif cid == "07":
            hit = next(((i, tx) for i, _, tx in reversed(ops) if any(b in tx.lower() for b in _BYE)), None)
            value = "YES" if hit else "NO"
            ev = [hit] if hit else []
        elif "YES" in enum:
            value = "NA" if "NA" in enum and cid in ("02", "03") and len(ops) < 2 else "YES"
            ev = [(ops[-1][0], ops[-1][2])]
        else:
            nums = sorted(int(v) for v in enum if v != "NA")
            value = str(nums[-1] if len(ops) >= 2 else nums[len(nums) // 2])
            ev = [(ops[-1][0], ops[-1][2])]
        evidence = [{"turn": i, "quote": tx[:60]} for i, tx in ev if tx]
        _add_usage({"input_tokens": len(cached_user) // 4, "output_tokens": 50}, self.model)
        return {"reasoning": f"mock heuristic for criterion {cid}", "value": value,
                "justification": f"[mock] эвристика по ключевым словам, критерий {cid}",
                "evidence": evidence, "missing_where": "", "na_reason": "", "confidence": 0.95}


MAX_ATTEMPTS = 5


def _status(e):
    for obj in (e, getattr(e, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def retryable(e):
    """429 / 5xx / 529 overload / network errors are retried; other 4xx are not."""
    code = _status(e)
    if code is None:
        return True   # timeouts, connection resets, malformed responses
    return code == 429 or code == 408 or code >= 500


def backoff_delay(e, attempt, base=2.0, cap=60.0):
    """Honour Retry-After when present, else exponential backoff with full jitter."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    ra = headers.get("retry-after") if hasattr(headers, "get") else None
    if ra:
        try:
            return min(cap, float(ra)) + random.uniform(0, 1)
        except ValueError:
            pass
    return random.uniform(0, min(cap, base * (2 ** attempt)))


class LLM:
    """Provider-agnostic structured-output client: ``call()`` returns the tool-call arguments dict."""

    def __init__(self, provider=None, model=None):
        self.provider = provider or config.PROVIDER
        self.model = model or config.MODEL
        self.route = cloud_route(self.model) if self.provider == "anthropic" else self.provider
        if self.route == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"), max_retries=2)  # + own Retry-After/jitter loop
        elif self.provider == "mock":
            self.model = MockBackend.model
            self._mock = MockBackend()

    def call(self, system: str, cached_user: str, user: str, tool: dict, temperature: float = 0.0) -> dict:
        """Run one structured evaluation call with up to 4 attempts (exponential-ish backoff)."""
        if self.provider == "mock":
            return self._mock.call(system, cached_user, user, tool, temperature)
        for attempt in range(MAX_ATTEMPTS):
            route = getattr(self, "route", None) or self.provider
            try:
                if route == "anthropic":
                    return self._anthropic(system, cached_user, user, tool, temperature)
                if route == "openai_cloud":
                    return self._openai_responses(system, cached_user, user, tool)
                if route == "gemini":
                    return self._gemini(system, cached_user, user, tool)
                return self._openai(system, cached_user, user, tool, temperature)
            except Exception as e:
                if _status(e) == 404:
                    raise RuntimeError(
                        f"model {self.model!r} not found at provider {self.provider!r} (HTTP 404). Check CQA_MODEL / "
                        "CQA_JURORS / CQA_JUDGE in .env against the provider's model list "
                        "(e.g. GET /v1/models) or set CQA_ENSEMBLE=0.") from e
                if attempt == MAX_ATTEMPTS - 1 or not retryable(e):
                    raise
                time.sleep(backoff_delay(e, attempt))

    def _anthropic(self, system, cached_user, user, tool, temperature):
        r = self.client.messages.create(
            model=self.model, max_tokens=3000,  # NB: this API endpoint rejects `temperature`; sampling is provider default
            system=[{"type": "text", "text": system}],
            tools=[tool], tool_choice=({"type": "tool", "name": tool["name"]} if "haiku" in self.model
                                       else {"type": "auto"}),  # sonnet/opus 5-5 reject forced tool_choice
            messages=[{"role": "user", "content": [
                {"type": "text", "text": cached_user, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user + f"\n\nОтветь ТОЛЬКО вызовом инструмента `{tool['name']}`."}]}])
        _add_usage(r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage), self.model)
        for b in r.content:
            if b.type == "tool_use":
                return b.input
        # tool_choice=auto models sometimes answer with the JSON as text: accept it if it parses
        for b in r.content:
            if b.type == "text":
                m = re.search(r"\{.*\}", b.text or "", re.S)
                if m:
                    try:
                        js = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(js, dict) and "value" in js:
                        return js
        raise RuntimeError("no tool_use in response")

    def _openai_tools(self, system, cached_user, user, tool, temperature):
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": cached_user + "\n\n" + user}],
                "tools": [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                                            "parameters": tool["input_schema"]}}],
                "tool_choice": {"type": "function", "function": {"name": tool["name"]}}}
        r = httpx.post(config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"}, timeout=600)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        _add_usage({"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})
        calls = j["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            raise RuntimeError("no tool_calls in response")
        args = calls[0]["function"]["arguments"]
        return json.loads(args) if isinstance(args, str) else args

    def _openai(self, system, cached_user, user, tool, temperature):
        if config.OPENAI_TOOLS:   # native function calling (vLLM/Ollama/OpenAI); falls back to JSON mode on 400
            try:
                return self._openai_tools(system, cached_user, user, tool, temperature)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 400:
                    raise
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system + "\nВерни ТОЛЬКО JSON-объект по схеме: " +
                              json.dumps(tool["input_schema"], ensure_ascii=False)},
                             {"role": "user", "content": cached_user + "\n\n" + user}],
                "response_format": {"type": "json_object"}}
        r = httpx.post(config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"}, timeout=600)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        _add_usage({"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")})
        txt = j["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0) if m else txt)

    # ---- cloud vendors other than Anthropic (selected per model id, see cloud_route) ----
    def _openai_responses(self, system, cached_user, user, tool):
        """OpenAI Responses API with a forced function call; sampling/reasoning = model defaults (no tweaks)."""
        key = config.get("CQA_OPENAI_CLOUD_KEY") or config.get("OPENAI_API_KEY_QUORUM") or config.get("OPENAI_API_KEY")
        body = {"model": self.model, "instructions": system,
                "input": [{"role": "user", "content": cached_user + "\n\n" + user}],
                "tools": [{"type": "function", "name": tool["name"], "description": tool["description"],
                           "parameters": tool["input_schema"]}],
                "tool_choice": {"type": "function", "name": tool["name"]}, "prompt_cache_key": "cqa"}
        r = httpx.post("https://api.openai.com/v1/responses", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=600)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        _add_usage({"input_tokens": int(u.get("input_tokens") or 0) - cached, "cache_read_input_tokens": cached,
                    "output_tokens": u.get("output_tokens")}, self.model)
        for o in j.get("output") or []:
            if o.get("type") == "function_call":
                return json.loads(o["arguments"]) if isinstance(o.get("arguments"), str) else o["arguments"]
        raise RuntimeError("no function_call in response")

    def _gemini(self, system, cached_user, user, tool):
        """Gemini via its OpenAI-compatible endpoint, forced function call. Billed output includes thinking tokens."""
        key = config.get("GOOGLE_API_KEY") or config.get("GEMINI_API_KEY")
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": cached_user + "\n\n" + user}],
                "tools": [{"type": "function", "function": {"name": tool["name"], "description": tool["description"],
                                                            "parameters": tool["input_schema"]}}],
                "tool_choice": {"type": "function", "function": {"name": tool["name"]}}}
        r = httpx.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", json=body,
                       headers={"Authorization": f"Bearer {key}"}, timeout=600)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        pt = int(u.get("prompt_tokens") or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        out = max(int(u.get("total_tokens") or 0) - pt, int(u.get("completion_tokens") or 0))   # + thinking tokens
        _add_usage({"input_tokens": pt - cached, "cache_read_input_tokens": cached, "output_tokens": out}, self.model)
        calls = (j.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
        if not calls:
            raise RuntimeError("no tool_calls in response")
        args = calls[0]["function"]["arguments"]
        return json.loads(args) if isinstance(args, str) else args
