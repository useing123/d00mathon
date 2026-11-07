"""Prompt templates (one focused prompt per criterion)."""

PROMPT_VERSION = "v2"

SYSTEM = """Ты — строгий и последовательный эксперт по контролю качества (QA) клиентской поддержки брокера Freedom Broker.
Ты оцениваешь работу ЧЕЛОВЕКА-ОПЕРАТОРА (speaker=OPERATOR) по ОДНОМУ критерию чек-листа.
Принципы:
- Реплики BOT (AI-ассистент / IVR) и SYSTEM не являются действиями оператора и не засчитываются оператору (но дают контекст: что клиент уже сообщил, что было раньше).
- Опирайся только на текст транскрипта. Не додумывай то, чего нет.
- Каждое утверждение подкрепляй цитатой: номер реплики [N] и ТОЧНЫЙ фрагмент текста этой реплики (дословно, копированием, без перевода и правок; можно взять подстроку).
- Если оценка NO/низкая из-за отсутствия действия — укажи, в какой момент (номер реплики) действие ожидалось, в поле missing_where.
- Язык диалога может быть русский, казахский или английский; чек-лист одинаков для всех языков.
- Пиши обоснование по-русски, кратко (1–3 предложения).
- Укажи confidence (0..1) — насколько ты уверен, что эксперт-аудитор поставил бы то же значение.
Ответ — ТОЛЬКО вызовом инструмента submit_evaluation."""


def allowed_values(cr):
    vals = [str(x) for x in cr["scale"]] if cr["type"] == "scale" else ["YES", "NO"]
    if cr.get("allow_na"):
        vals.append("NA")
    return vals


def tool_schema(cr):
    return {
        "name": "submit_evaluation",
        "description": "Submit the evaluation of one checklist criterion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Пошаговый разбор: какие реплики оператора релевантны, что требовалось правилами, вывод."},
                "value": {"type": "string", "enum": allowed_values(cr)},
                "justification": {"type": "string", "description": "Краткое обоснование для отчёта (по-русски)."},
                "evidence": {"type": "array", "items": {"type": "object", "properties": {
                    "turn": {"type": "integer"}, "quote": {"type": "string"}}, "required": ["turn", "quote"]},
                    "description": "Дословные цитаты из транскрипта."},
                "missing_where": {"type": "string", "description": "Где ожидалось отсутствующее действие (номер реплики/момент), если применимо; иначе пусто."},
                "na_reason": {"type": "string", "description": "Почему NA, если value=NA; иначе пусто."},
                "confidence": {"type": "number"},
            },
            "required": ["reasoning", "value", "justification", "evidence", "confidence"],
        },
    }


def transcript_block(rec, transcript_text):
    return (f"ДИАЛОГ {rec['conversation_id']} | канал: {rec.get('channel')} | язык (метаданные): {rec.get('language')}\n"
            f"Формат строки: [номер реплики] (время от начала) СПИКЕР: текст\n\n<transcript>\n{transcript_text}\n</transcript>")


def criterion_block(checklist, cr):
    a = checklist["assumptions"]
    notes = []
    if a.get("first_reply") == "first_human_operator_turn":
        notes.append("«Первая реплика/первый ответ» = первая реплика ЧЕЛОВЕКА-оператора (OPERATOR), даже если бот отвечал раньше.")
    if a.get("ignore_internal_hold_turns"):
        notes.append("Реплики, помеченные [ВНУТРЕННИЕ ПЕРЕГОВОРЫ НА УДЕРЖАНИИ], клиент не слышит — это не общение с клиентом.")
    vals = " | ".join(allowed_values(cr))
    return f"""ЧЕК-ЛИСТ: {checklist.get('name', checklist['id'])}
{checklist.get('description', '')}
Допущения: {' '.join(notes)}

КРИТЕРИЙ {cr['id']}: {cr['name']}
Допустимые значения: {vals}

Рубрика:
{cr.get('rubric', '').strip()}

Правила принятия решения:
{cr.get('rules', '').strip()}

Оцени ТОЛЬКО этот критерий. Сначала найди все релевантные реплики оператора, затем примени рубрику и правила буквально, затем вызови submit_evaluation."""
