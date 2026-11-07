"""Kazakh-aware SIGNAL EXTRACTOR (preset ``hybrid``; flag ``CQA_KZ_SIGNALS=1``).

Rules do NOT decide verdicts. They parse each dialog into hints {type, phrase, turn, lang, gloss} — greetings,
self-introductions, farewells, help offers, summaries, ID verification, wait/hold announcements, empathy,
08 red flags — with per-turn language (KZ / RU / EN / MIXED, incl. Latin-script KZ). The hints are appended
to the FACTS block every model (Jev, Haiku, Sonnet) already receives, so models don't misread Kazakh,
code-switching and colloquial forms. The only rule action: an 08 red flag forces escalation to Sonnet.
See docs/HYBRID_KZ.md.
"""
import re

KZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
KZ_WORDS = {"мен", "сіз", "сізге", "сізді", "бар", "жоқ", "иә", "рақмет", "рахмет", "қалай", "керек", "болады",
            "бе", "ма", "ме", "па", "пе", "және", "үшін", "деп", "емес", "кешіріңіз", "салем", "сәлем", "жақсы",
            "менің", "атым", "шот", "шотты", "ақша", "көмек", "оператор", "мархабат",
            # Latin-script Kazakh
            "salem", "salemetsiz", "rakhmet", "rahmet", "raqmet", "qalai", "qalay", "kerek", "jok", "zhok",
            "iya", "men", "sizge", "boladi", "bolady", "atym", "menin", "jaqsy", "zhaksy"}
RU_WORDS = {"и", "в", "не", "на", "что", "как", "это", "у", "меня", "мне", "вы", "вам", "можно", "здравствуйте",
            "спасибо", "пожалуйста", "счет", "счёт", "да", "нет", "почему", "когда", "где", "добрый", "день"}
EN_WORDS = {"the", "and", "is", "my", "i", "you", "hello", "hi", "please", "thanks", "account", "how", "what",
            "can", "to", "of", "a", "it", "for", "with", "good"}

_W = re.compile(r"[\wәғқңөұүһіё']+", re.I)


def detect_lang(text):
    """'KZ' | 'RU' | 'EN' | 'MIXED' (KZ+RU code-switching) | None. KZ-only letters or KZ function words → KZ."""
    t = (text or "").lower()
    words = _W.findall(t)
    if not words:
        return None
    kz_letters = sum(c in KZ_LETTERS for c in t)
    cyr = sum("а" <= c <= "я" or c in "ё" or c in KZ_LETTERS for c in t)
    lat = sum("a" <= c <= "z" for c in t)
    kz = sum(w in KZ_WORDS for w in words) + (2 if kz_letters >= 2 or (cyr and kz_letters / cyr > 0.02) else 0)
    kz += min(kz_letters, 3) if kz_letters else 0
    ru = sum(w in RU_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    if lat > cyr:   # Latin script: English vs Latin-Kazakh
        lkz = sum(w in KZ_WORDS for w in words)
        return "KZ" if lkz > en else ("EN" if en or lat else None)
    if kz >= 2 and ru >= 2:
        return "MIXED"
    if kz >= 1 and (kz_letters or kz >= 2):
        return "KZ"
    return "RU" if cyr else None


def lang_of(texts):
    seen = {detect_lang(x) for x in texts} - {None}
    if "MIXED" in seen or {"KZ", "RU"} <= seen:
        return "MIXED"
    for L in ("KZ", "RU", "EN"):
        if seen == {L}:
            return L
    return next(iter(seen), None) if len(seen) == 1 else ("MIXED" if seen else None)


def _rx(*alts):
    return re.compile("|".join(alts), re.I)


GREETING = _rx(r"с[әа]лем(етсіз|етсиз)?", r"ассаля?уа?ма?ғ?[ае]л[еи]йкум", r"қайырлы\s+(күн|таң|кеш)",
               r"\bsalem(etsiz)?\b", r"здравствуй", r"добр(ый|ое|ого)\s+(день|утро|вечер|времени)", r"приветству",
               r"\bпривет\b", r"\bhello\b", r"\bhi\b", r"good\s+(morning|afternoon|evening|day)", r"\bgreetings\b")
INTRO = _rx(r"менің\s+атым\s+\w+", r"\bмен\s+[А-ЯӘҒҚҢӨҰҮҺІA-Z]\w+", r"меня\s+зовут\s+\w+",
            r"\bэто\s+[А-Я]\w+", r"\bна\s+связи\s+\w+", r"my\s+name\s+is\s+\w+", r"\b(this\s+is|i\s+am|i'm)\s+[A-Z]\w+",
            r"\bmenin\s+atym\s+\w+", r"\bmen\s+[A-Z]\w+", r"оператор\s+[А-ЯA-Z]\w+")
FAREWELL = _rx(r"сау\s+бол", r"қош\s+бол", r"жақсы\s+күн", r"күніңіз\s+сәтті", r"до\s+свидания", r"всего\s+(доброго|хорошего)",
               r"хорошего\s+(дня|вечера)", r"удачи", r"good\s*bye", r"\bbye\b", r"have\s+a\s+(nice|good|great)",
               r"\bsau\s+bol", r"take\s+care", r"всех\s+благ")
HELP_OFFER = _rx(r"тағы\s+(қандай\s+)?сұрақ", r"басқа\s+сұрақ", r"тағы\s+(не|қалай)\s+көмек", r"сұрақтарыңыз\s+болса",
                 r"ещ[её]\s+(какие|есть)?\s*вопрос", r"чем\s+ещ[её]", r"ещ[её]\s+чем", r"если\s+(появятся|возникнут|будут)\s+(ещ[её]\s+)?вопрос",
                 r"обращайтесь", r"пишите", r"anything\s+else", r"other\s+questions", r"further\s+(help|questions)",
                 r"feel\s+free", r"жазыңыз", r"хабарласыңыз")
SUMMARY = _rx(r"\bитак\b", r"подвед[её]м", r"резюмир", r"to\s+sum(marize)?", r"in\s+summary", r"\bқорыта\s+келе",
              r"\bсонымен\b", r"как\s+договорились")
RED_FLAGS = _rx(r"личн(ый|ого)\s+номер", r"жеке\s+нөмір", r"переведите\s+мне", r"на\s+(мою|мой)\s+(карту|сч[её]т|номер)",
                r"маған\s+аударыңыз", r"за\s+плату", r"за\s+[\d\s]+\s*(тенге|тг|₸|руб)", r"telegram|телеграм|whatsapp|ватсап", r"ақыға", r"to\s+my\s+(personal\s+)?(card|account)",
                r"туп(ой|ая|ые)", r"идиот", r"дура[кч]", r"ақымақ", r"\bступид", r"stupid", r"idiot", r"\bнадоел")
# a "full" operator reply (not a one-word ack): ending without farewell then is a real omission
_MIN_FULL = 25


VERIFY = _rx(r"ЖСН", r"\bИИН\b", r"\bIIN\b", r"кодов(ое|ого)\s+слов", r"код(тық)?\s+сөз", r"sms[- ]?код", r"смс[- ]?код",
             r"туған\s+(күн|жыл)", r"дат[ау]\s+рождения", r"верификац", r"паспорт", r"жеке\s+куәлік", r"удостоверени",
             r"date\s+of\s+birth", r"code\s*word", r"verify\s+your")
WAIT = _rx(r"күте\s+тұрыңыз", r"күтіңіз", r"бір\s+минут", r"аз\s+уақыт", r"подождите", r"минут(к|оч)", r"ожидайте",
           r"уточню", r"нақтылап", r"one\s+moment", r"please\s+wait", r"hold\s+on", r"a\s+minute")
EMPATHY = _rx(r"кешіріңіз", r"түсінемін", r"өкінішке", r"ренжіме", r"извините", r"прошу\s+прощения", r"понимаю",
              r"сожалени", r"к\s+сожалению", r"sorry", r"apologi", r"i\s+understand")
GLOSS = {"сәлеметсіз бе": "hello (formal)", "сәлем": "hi", "қайырлы күн": "good afternoon", "қайырлы таң": "good morning",
         "қайырлы кеш": "good evening", "менің атым": "my name is", "сау болыңыз": "goodbye", "қош болыңыз": "farewell",
         "жақсы күн": "have a good day", "тағы сұрақ": "any more questions?", "басқа сұрақ": "other questions?",
         "сұрақтарыңыз болса": "if you have questions", "жазыңыз": "write (to us)", "хабарласыңыз": "contact us",
         "сонымен": "so / to sum up", "жсн": "IIN (personal ID number)", "кодтық сөз": "code word", "туған күн": "birthday",
         "жеке куәлік": "ID card", "күте тұрыңыз": "please wait", "күтіңіз": "wait", "бір минут": "one minute",
         "аз уақыт": "a little time", "нақтылап": "clarifying", "кешіріңіз": "sorry / excuse me", "түсінемін": "I understand",
         "өкінішке": "unfortunately", "жеке нөмір": "personal number", "маған аударыңыз": "transfer to me",
         "ақыға": "for a fee", "ақымақ": "fool (insult)", "салем": "hi", "рақмет": "thank you", "сау бол": "goodbye", "қош бол": "farewell"}
TYPES = (("greeting", GREETING, "operator"), ("introduction", INTRO, "operator"), ("farewell", FAREWELL, "operator"),
         ("help_offer", HELP_OFFER, "operator"), ("summary", SUMMARY, "operator"), ("id_verification", VERIFY, None),
         ("wait_announcement", WAIT, "operator"), ("empathy", EMPATHY, "operator"), ("red_flag_08", RED_FLAGS, "operator"))


def gloss(phrase):
    p = (phrase or "").lower()
    return next((v for k, v in GLOSS.items() if k in p or (len(p) > 5 and k.startswith(p))), "")


def _ops(rec):
    return [t for t in rec["transcript"] if t["speaker"] == "operator" and not t.get("internal")]


def red_flags(rec):
    from .data import GROSS_SIGNALS
    return [t["index"] for t in _ops(rec) if RED_FLAGS.search(t.get("text") or "") or GROSS_SIGNALS.search(t.get("text") or "")]


def signals(rec):
    """Per-dialog hints. greeting/introduction only in the operator's FIRST reply block; farewell/help/summary only in
    the last two operator turns; the rest anywhere. Each: {type, phrase, turn (1-based), lang, gloss?, speaker}."""
    tr = rec["transcript"]
    ops = _ops(rec)
    first, closing = set(), {t["index"] for t in ops[-2:]}
    if ops:
        for t in tr[tr.index(ops[0]):]:
            if t["speaker"] == "client":
                break
            if t in ops:
                first.add(t["index"])
    out = []
    for t in tr:
        if t["speaker"] not in ("operator", "client") or t.get("internal"):
            continue
        text = t.get("text") or ""
        for typ, rx, who in TYPES:
            if who and t["speaker"] != who:
                continue
            if typ in ("greeting", "introduction") and t["index"] not in first:
                continue
            if typ in ("farewell", "help_offer", "summary") and t["index"] not in closing:
                continue
            m = rx.search(text)
            if m:
                ph = m.group(0) + re.match(r"[\wәғқңөұүһі]*", text[m.end():]).group(0)   # whole last word
                s = {"type": typ, "phrase": ph, "turn": t["index"], "lang": detect_lang(ph) or detect_lang(text),
                     "speaker": t["speaker"]}
                g = gloss(ph)
                if g:
                    s["gloss"] = g
                out.append(s)
    langs = {"client": [], "operator": []}
    for t in tr:
        if t["speaker"] in langs and not t.get("internal"):
            langs[t["speaker"]].append((t["index"], detect_lang(t.get("text"))))
    return {"signals": out, "turn_langs": langs,
            "client_lang": lang_of([t.get("text") or "" for t in tr if t["speaker"] == "client"
                                    and len((t.get("text") or "").split()) > 1]),
            "operator_first_lang": lang_of([t.get("text") or "" for t in tr if t["index"] in first])}


def format_signals(sig):
    """FACTS sub-section (Russian, like the rest of the block). Hints only: the text wins on contradiction."""
    L = ["СИГНАЛЫ (парсинг, RU/KZ/EN) — подсказки парсера; если текст реплики им противоречит, верь тексту:",
         f"- Язык клиента (KZ-aware): {sig['client_lang'] or 'не определён'}; язык первой реплики оператора: "
         f"{sig['operator_first_lang'] or 'не определён'} (MIXED = смешение KZ/RU; любой из двух языков подходит)."]
    for k, who in (("client", "клиент"), ("operator", "оператор")):
        tl = [f"№{n}:{l}" for n, l in sig["turn_langs"][k] if l]
        if tl:
            L.append(f"- Языки реплик ({who}): " + ", ".join(tl[:20]))
    for s in sig["signals"]:
        L.append(f"- {s['type']}: «{s['phrase']}» (№{s['turn']}, {s['lang'] or '?'})"
                 + (f" = {s['gloss']}" if s.get("gloss") else ""))
    return "\n".join(L)
