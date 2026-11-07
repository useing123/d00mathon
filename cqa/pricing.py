"""Per-model API price table, USD per 1M tokens — ASSUMPTIONS used for cost reporting only.

Columns: input (uncached), output (incl. reasoning/thinking tokens), cache_read, cache_write (5-minute TTL).
Sources, retrieved 2026-11-07 (standard tier, no batch/flex/fast discounts):
  * Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (Haiku 5.5: prompts <=100k tokens)
  * OpenAI:    https://developers.openai.com/api/docs/pricing (standard tier; gpt-5.4-mini from the same page)
  * Google:    https://ai.google.dev/gemini-api/docs/pricing (promo prices valid until 2026-11-30 / 2026-12-31;
               list prices double afterwards — see ``PRICE_NOTES``)
Local models (vLLM etc.) have no per-token price; their cost is GPU $/hour × wall time (see docs/LOCAL_LLM.md).
"""

PRICES_AS_OF = "2026-11-07"

PRICES = {
    # model id prefix: (input, output, cache_read, cache_write_5m)
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.50),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-fable-5-1": (10.0, 50.0, 0.25, 12.50),
    "gpt-6-luna": (0.10, 0.50, 0.01, 0.125),
    "gpt-5.4-mini": (0.75, 4.50, 0.075, 0.0),
    "gpt-5.3-codex": (1.75, 14.0, 0.175, 0.0),
    "gemini-4-argon": (2.0, 10.0, 0.10, 0.0),
    "gemini-3.8-flash": (0.75, 3.75, 0.075, 0.0),
}
PRICE_NOTES = {
    "gemini-4-argon": "promo until 2026-11-30; $4/$20 from 2026-12-01",
    "gemini-3.8-flash": "promo until 2026-12-31; $1.50/$7.50 from 2027-01-01",
    "claude-haiku-5-5": "prompts <=100k tokens",
}
DEFAULT = (3.0, 15.0, 0.30, 3.75)   # unknown cloud model: conservative guess


def price(model):
    """(input, output, cache_read, cache_write) $/MTok for a model id (longest matching prefix)."""
    m = model or ""
    hits = [k for k in PRICES if m.startswith(k)]
    return PRICES[max(hits, key=len)] if hits else DEFAULT


def usd(model, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
    pin, pout, pcr, pcw = price(model)
    return (input_tokens * pin + output_tokens * pout + cache_read * pcr + cache_write * pcw) / 1e6
