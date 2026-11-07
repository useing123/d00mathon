"""LLM backends: Anthropic (tool use, strict JSON) and any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio)."""
import json
import re
import threading
import time

import httpx

from . import config

USAGE = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "calls": 0, "usd": 0.0}
PRICES = {"haiku": (1.0, 5.0), "sonnet": (3.0, 15.0), "opus": (5.0, 25.0)}  # $/MTok in/out (approx.)


def _price(model):
    return next((v for k, v in PRICES.items() if k in (model or "")), (3.0, 15.0))
_lock = threading.Lock()


def _add_usage(u, model=None):
    pin, pout = _price(model)
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


class LLM:
    def __init__(self, provider=None, model=None):
        self.provider = provider or config.PROVIDER
        self.model = model or config.MODEL
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"), max_retries=6)

    def call(self, system, cached_user, user, tool, temperature=0.0):
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
