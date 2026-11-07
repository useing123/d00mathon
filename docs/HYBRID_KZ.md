# Preset `hybrid`: Kazakh-aware parser signals → cascade

Code: `cqa/rules_kz.py`, hook in `cqa/data.py` (`compute_facts`/`format_facts`), `HybridEvaluator` in `cqa/router.py`,
preset in `cqa/presets.py`, eval `scripts/eval_hybrid_kz.py`, tests `tests/test_rules_kz.py`.

## Why
Kazakh words, KZ/RU code-switching, Latin-script Kazakh and colloquial forms are where models misread
(e.g. "Сау болыңыз" not recognised as a farewell, a KZ client answered in RU). A deterministic parser is free,
explainable and cites turn numbers, so the models don't judge blind.

## What it does
* **Rules do not decide verdicts.** `rules_kz.signals(rec)` emits hints `{type, phrase, turn, lang, gloss}`:
  greeting / introduction (operator's first reply block), farewell / help_offer / summary (last two operator turns),
  id_verification (ЖСН/ИИН, кодовое слово, SMS-код, туған күні…), wait_announcement (күте тұрыңыз, бір минут,
  подождите…), empathy (кешіріңіз, түсінемін, извините…), red_flag_08 (жеке нөмір, на мою карту, за плату, insults).
  Plus per-turn language (KZ/RU/EN/MIXED) from KZ-only letters (ә ғ қ ң ө ұ ү һ і) + KZ function words, Latin KZ.
  KZ phrases carry a gloss ("Сау болыңыз = goodbye").
* With `CQA_KZ_SIGNALS=1` the section `СИГНАЛЫ (парсинг, RU/KZ/EN)` is appended to the FACTS block, which every
  model gets — Haiku/Sonnet prompts and Jev's `state` (both go through `data.format_facts`). It says the text wins
  if it contradicts a hint. Flag off → byte-identical FACTS (tested). The `hybrid` preset sets the flag; default `jury` doesn't.
* Only rule action: an operator red flag on criterion 08 forces the strongest tier (Sonnet) and logs
  `red_flag_turns`; it never decides 08 by itself.
* Cascade cache keys include the flag, so ON/OFF answers never mix.

## Results (1 run each, 2026-11-08; `results/hybrid_kz/summary.json`)
Exact agreement with gold labels (8 criteria × dialogs). "Jev alone" = the tier-1 answer before escalation.

| set | labels | cascade (off) | hybrid (on) | Jev alone off | Jev alone on |
|---|---|---|---|---|---|
| tuned gold | 128 | 0.984 | 0.977 | 0.984 | 0.953 |
| tuned gold, KZ subset (3 dialogs) | 24 | 1.000 | 1.000 | 1.000 | 0.958 |
| held-out | 80 | 0.950 | 0.950 | 0.925 | 0.950 |
| held-out, KZ subset (2 dialogs) | 16 | 1.000 | 1.000 | 0.938 | 0.938 |

**Honest reading: no measurable gain.** Hybrid = cascade on held-out; −1 label on tuned gold; Jev alone +2 labels on
held-out but −4 on tuned gold. All within single-run noise; the KZ subsets are 5 dialogs total and the cascade
already gets them all right. Nothing was tuned on held-out. Cost: ≈$0.14 Anthropic for these runs, Jev ≈ $0.05.

## Limitations
* Tiny synthetic gold, 1 run → cannot separate a few-label effect from noise; needs real KZ chats.
* Lexicons are hand-written regexes: misses paraphrases; "пишите"-style offers can be ambiguous.
* The semantic (embedding) backup (`alphaedge-ai/Qwen3-Embedding-kaz-32768`) was not built — out of time.
* An earlier variant where rules decided 01/07 directly hit 100 % precision on held-out (01: 9/10 coverage,
  07: 8/10) but was dropped by design: verdicts stay with the models.
