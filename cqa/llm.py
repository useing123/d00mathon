"""LLM backends.

* ``anthropic`` — Claude via tool use (strict JSON schema, prompt caching).
* ``openai``    — any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio) in JSON mode.
* ``mock``      — deterministic offline heuristic (:class:`MockBackend`) for tests, CI and demos
  without an API key. It is NOT an accuracy baseline.
"""
import json
import re
import threading
import time

import httpx

from . import config

USAGE = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "calls": 0, "usd": 0.0}
PRICES = {"haiku": (1.0, 5.0), "sonnet": (3.0, 15.0), "opus": (5.0, 25.0)}  # $/MTok in/out (approx.)


def _price(model: str | None) -> tuple[float, float]:
    """Approximate $/MTok (input, output) for a model name."""
    return next((v for k, v in PRICES.items() if k in (model or "")), (3.0, 15.0))


_lock = threading.Lock()


def _add_usage(u, model=None):
    pin, pout = (0.0, 0.0) if (model or "").startswith("mock") else _price(model)
    usd = (int(u.get("input_tokens") or 0) * pin + int(u.get("cache_creation_input_tokens") or 0) * pin * 1.25 +
           int(u.get("cache_read_input_tokens") or 0) * pin * 0.1 + int(u.get("output_tokens") or 0) * pout) / 1e6
    with _lock:
        USAGE["usd"] += usd
        USAGE["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            USAGE[k] += int(u.get(k) or 0)


def cost_usd(usage=None, model=None):
    u = usage or USAGE
    if config.PROVIDER != "anthropic":
        return 0.0
    return round(float(u.get("usd", 0.0)), 4)


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
        enum = tool["input_schema"]["properties"]["value"]["enum"]
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


class LLM:
    """Provider-agnostic structured-output client: ``call()`` returns the tool-call arguments dict."""

    def __init__(self, provider=None, model=None):
        self.provider = provider or config.PROVIDER
        self.model = model or config.MODEL
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"), max_retries=6)
        elif self.provider == "mock":
            self.model = MockBackend.model
            self._mock = MockBackend()

    def call(self, system: str, cached_user: str, user: str, tool: dict, temperature: float = 0.0) -> dict:
        """Run one structured evaluation call with up to 4 attempts (exponential-ish backoff)."""
        if self.provider == "mock":
            return self._mock.call(system, cached_user, user, tool, temperature)
        for attempt in range(4):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, cached_user, user, tool, temperature)
                return self._openai(system, cached_user, user, tool, temperature)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    def _anthropic(self, system, cached_user, user, tool, temperature):
        r = self.client.messages.create(
            model=self.model, max_tokens=1500,  # NB: this API endpoint rejects `temperature`; sampling is provider default
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
        raise RuntimeError("no tool_use in response")

    def _openai(self, system, cached_user, user, tool, temperature):
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
