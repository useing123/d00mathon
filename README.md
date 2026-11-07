# CQA — AI-оценка качества чатов и звонков Freedom Broker

> **Жюри из LLM проверяет каждый критерий отдельно, приводит дословные цитаты-доказательства и считает балл детерминированным кодом.**
> LLM jury scores every checklist criterion separately, backs each verdict with verified verbatim quotes, and computes the score in deterministic code.

[![CI](https://github.com/useing123/d00mathon/actions/workflows/ci.yml/badge.svg)](https://github.com/useing123/d00mathon/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)
![ruff](https://img.shields.io/badge/lint-ruff-261230)
![license](https://img.shields.io/badge/license-MIT-green)

[English version ↓](#english) · [Pitch deck (PDF)](docs/CQA_Pitch_Deck.pdf) · [Presentation (HTML)](docs/presentation.html) · [Architecture](docs/ARCHITECTURE.md) · [Scoring](docs/SCORING.md) · [Assumptions](docs/ASSUMPTIONS.md) · [Evaluation](docs/EVALUATION.md) · [API](docs/API.md)

## Архитектура за 30 секунд

```mermaid
flowchart LR
    D[Диалог: чат / звонок] --> M{Режим}
    M -->|по умолчанию| J["Жюри: haiku-5-5 + sonnet-5-5<br/>судья opus-5-5, медиана 3 на 04/06"]
    M -->|--preset cascade| C1["Jev 1.14<br/>73–82% критериев"]
    C1 -->|низкая уверенность| C2["Haiku<br/>15–19%"]
    C2 -->|спор / грубая ошибка| C3["Sonnet<br/>3–7%"]
    M -->|своя GPU| L["Qwen3-8B + LoRA<br/>vLLM, данные в контуре"]
    J --> V[Проверка цитат → балл в коде → needs_review]
    C1 --> V
    C2 --> V
    C3 --> V
    L --> V
```

| Режим | Held-out | $ / диалог | Когда использовать |
|---|---|---|---|
| **Жюри Claude** (по умолчанию) | 0.95 | $0.16 | сдача, максимум доказательств: цитата на каждый критерий, 18% критериев → `needs_review` |
| **Каскад** Jev 1.14 → Haiku → Sonnet (`--preset cascade`) | 0.963 (DEV: 0.948 совпадения с жюри, грубые 5/5) | $0.004–0.015 | поток в продакшене: в 10–40 раз дешевле; у неэскалированных критериев нет цитат |
| Гибрид: KZ-сигналы парсера в FACTS + каскад (`--preset hybrid`, [docs/HYBRID_KZ.md](docs/HYBRID_KZ.md)) | 0.95 (= каскад в этом прогоне, KZ 16/16) | ≈ каскад | казахский/смешанный чат: подсказки с номерами реплик и глоссами |
| Jev 1.14 в одиночку | 0.934 | $0.00053 | массовый скрининг без цитат |
| **Своя модель** Qwen3-8B + QLoRA (16 шагов ≈ 0.32 эпохи) | 0.787 (база 0.787; MAE 12.17 → 10.43) | ≈ $0.001 (своя GPU) | данные не покидают контур; адаптер на [HF](https://huggingface.co/useing/cqa-qwen3-8b-lora); честно: пока не лучше базы по совпадению |

Подробно: [демо-сценарий](docs/DEMO.md) · [каскад](docs/OSS_ROUTER.md) · [Jev](docs/JEV.md) · [матрица моделей](docs/MODELS.md) (страница `/models`) · [дообучение](docs/FINETUNE.md) · модель на HF: [useing/cqa-qwen3-8b-lora](https://huggingface.co/useing/cqa-qwen3-8b-lora) .

## Результаты (текущая версия, v7)

| Набор | Совпадение значений | MAE итога | Грубые P/R | Прочее |
|---|---|---|---|---|
| **Held-out**, 10 синтетических диалогов (по ним не настраивались) | **0.95** (76/80, 1 прогон) | **4.43** | 1.0 / 1.0 | прежние прогоны 0.925–0.938 — это уровень шума |
| **DEV**, 65 реальных/синтетических диалогов организаторов | не измерено (нет разметки человека) | — | 5 флагов | средний балл 80.41; 0 ошибок; 294 с |
| Стоимость | **$0.16 за диалог** (DEV: $10.28 за 65, жюри haiku+sonnet → opus) | | | ≈ $1.6 за 10 контрольных |
| Флаги `needs_review` (DEV) | 18.3% критериев, 75% диалогов | | | разошлись присяжные / решал судья / шкала с низкой уверенностью |

Оба gold-набора синтетические и размечены нами — это проверка регрессий, а не совпадение со скрытой разметкой.
Подробно: [docs/EVALUATION.md](docs/EVALUATION.md); история версий: [docs/HISTORY.md](docs/HISTORY.md).

## Submission / Сдача результатов (формат JSON 1.0)

| Файл | Что внутри | Как получить |
|---|---|---|
| [`results/dev/results.json`](results/dev/results.json) | все 65 записей DEV (60 чатов + 5 звонков), прогон `dev_jury_v7` | `make submit-dev RUN=dev_jury_v7` (без вызовов LLM) |
| `results/prod/results.json` | ровно 10 записей контрольного набора, включая ошибки | в контрольный час: `make submit GROUP=<id>` → `git push` |

- Формат — приложение к студенческому гайду «Как сдать результаты»: `{"schema_version":"1.0","team_name":"казачи капай","model":{"mode","name"},"results":[…]}`; на диалог — `status:"ok"` с 8 критериями (`contact, discovery, identification, consultation, empathy, dialogue, closure, critical`; `true/false/null` или шаг шкалы до веса) и `score:{final_score, critical_flag}`, либо `status:"error"`, `criteria:[]`, `score:null`, `error:"<причина>"` (ошибка никогда не заменяется нулём).
- `evidence_turns` — **нумерация с 1** по исходному массиву транскрипта (клиент, бот, оператор, системные события); вся система (промпт, UI, API, экспорт) использует ту же нумерацию.
- Все правила — в одном месте: [`cqa/submission.py`](cqa/submission.py); JSON-схема — [`docs/results.schema.json`](docs/results.schema.json); проверка: `python -m cqa validate results/prod/results.json` (схема, домены значений, 8 критериев по одному разу, `critical_flag == critical`, 2 знака, без дублей ID, в prod ровно 10 записей; `--group <id>` — ID совпадают с API).
- `make submit GROUP=<id>`: загрузка из API → оценка (пресет по умолчанию) → экспорт JSON 1.0 → валидация → `git add -f` + commit; push — вручную. В UI: «Ещё → Выгрузить results.json (v1.0)».
- Заменённый в UI критерий выгружается под своим `id` (или `code:` из YAML); стандартные 01–08 → официальные коды.
- **Модель и стоимость.** `results/dev/results.json`: жюри `claude-haiku-5-5` + `claude-sonnet-5-5`, судья `claude-opus-5-5` (`external_api`), $10.28 за 65 диалогов ≈ **$0.16 за диалог**, ≈ 5 мин. Тот же режим (`--preset jury`) — по умолчанию для новых прогонов, в т.ч. `make submit` для контрольных диалогов. Альтернатива `--preset max` (`gpt-5.3-codex`, ≈ $0.058) опциональна: отрыв на отложенном наборе 2 метки из 80, это в пределах шума (см. [docs/MODELS.md](docs/MODELS.md)). Тихих подмен на старые модели нет: запасные модели выключены по умолчанию (`CQA_FALLBACK_MODELS`); фактическая модель записывается в `model.name`.
- **Апелляции** (два отдельных действия): «Оспорить» на карточке критерия (обязательный комментарий) → вкладка «Апелляции»: проверяющий подтверждает или меняет оценку с обязательной причиной → итог пересчитывается; карточка показывает исходную оценку, решение и итог. API: `POST /api/appeals`, `GET /api/appeals?group=&status=`, `POST /api/appeals/{id}/resolve`.
- Точность на реальных DEV-диалогах **не измерена** человеком (нет разметки); измерено только на синтетических gold-наборах (см. [docs/EVALUATION.md](docs/EVALUATION.md)).

Система оценивает работу **оператора-человека** (не бота) по гибкому чек-листу: чаты и голосовые звонки, RU / KZ / EN. Для каждого критерия — значение, обоснование, дословная цитата с номером реплики (или «где ожидалось действие»), итог 0–100 и отдельный флаг грубой ошибки. Веб-интерфейс без логина: дашборд, карточка диалога с подсветкой доказательств, аудиоплеер, ручная проверка (override), редактор чек-листов.

## Быстрый старт (3 команды)

```bash
make install                       # venv + зависимости
make demo                          # офлайн-прогон на синтетических диалогах (mock-LLM, без ключа)
make serve                         # UI: http://127.0.0.1:8090
```

С реальной моделью: `cp .env.example .env`, вписать `ANTHROPIC_API_KEY` и `CQA_API_URL`, затем `make eval-dev`.
Без make: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m cqa evaluate --mock --input examples/sample_dialogs.json`.
Docker: `docker compose up --build` (секреты из `.env`, в образ не попадают).

### ⏱ Контрольный набор (час 23)

```bash
python -m cqa evaluate --group <ID> --organizer results/<id>_submission.csv   # или: make eval-group GROUP=<ID>
```
Опционально `--final` — усиленный профиль (присяжные sonnet + opus, судья перепроверяет единогласные 02/04/06, медиана 3 сэмплов на 04/06). На held-out наборе он **не** дал выигрыша (0.925 против 0.925–0.938 у профиля по умолчанию) при цене ×2–4, поэтому по умолчанию не включён; см. EVALUATION.
Загрузка `CQA_API_URL&group_id=<ID>` → авто-выбор чек-листа по каналу (чат → `freedom_chat_v1`, звонок → `freedom_call_v1`) → параллельная оценка → `results/<id>.json` + `results/<id>.csv`. В UI то же — кнопка **Fetch group & evaluate**. Каждый новый диалог реально анализируется LLM: заготовленных ответов нет, кэш ключуется хэшем диалога.

## Архитектура

```mermaid
flowchart LR
  A[API организаторов / JSON / вставка текста] --> N[data.py<br/>нормализация, роли,<br/>удержание, аудио-смещения]
  N --> C{checklist.py<br/>выбор по каналу}
  C --> T[(диалог × критерий)]
  T --> P[prompts.py<br/>1 критерий = 1 промпт]
  P --> L[llm.py<br/>Anthropic / OpenAI-compat / Mock]
  L --> V[evaluator.py<br/>проверка цитат, жюри, кэш]
  V --> S[scoring.py<br/>Decimal ROUND_HALF_UP]
  S --> R[(results JSON/CSV)]
  R --> U[server.py + UI<br/>дашборд, override, редактор]
  O[(overrides отдельно)] --> U
```

### Жюри (режим по умолчанию)

```mermaid
sequenceDiagram
  participant E as Evaluator
  participant H as Juror: claude-haiku-5-5
  participant S as Juror: claude-sonnet-5-5
  participant J as Judge: claude-opus-5-5
  par параллельно
    E->>H: критерий k + транскрипт
  and
    E->>S: критерий k + транскрипт
  end
  H-->>E: value, цитаты, обоснование
  S-->>E: value, цитаты, обоснование
  alt вердикты совпали
    E->>E: принять вердикт жюри
  else расхождение
    E->>J: критерий + транскрипт + оба мнения
    J-->>E: финальный вердикт
  end
  E->>E: валидация цитат → кэш → скоринг кодом
```

## Как считается балл

`итог = 100 × Σ набранное / Σ весов применимых критериев`, где YES = вес, NO = 0, шкала = `вес × шаг / максимальный_шаг`, NA исключается из числителя **и** знаменателя. Округление `Decimal ROUND_HALF_UP` до 2 знаков. Критерий 08 «Грубые ошибки» без веса — отдельный флаг `gross_error`. Нет применимых критериев → `total = null`.

**Пример с NA** (чат-чек-лист, тест `test_readme_worked_example`):

| Критерий | Вес | Оценка | Балл |
|---|---|---|---|
| 01 Контакт | 10 | YES | 10 |
| 02 Потребность | 15 | NA | — |
| 03 Идентификация | 20 | NA | — |
| 04 Консультация | 20 | 15 из 20 | 15 |
| 05 Эмпатия | 15 | NO | 0 |
| 06 Управление/грамотность | 15 | 7 из 10 | 10.5 |
| 07 Завершение | 5 | NA | — |
| **Итого** | **60 применимо** | | **100 × 35.5 / 60 = 59.17** |

Все крайние случаи и соответствующие тесты: [docs/SCORING.md](docs/SCORING.md).

## Как максимизируется точность
- **Один критерий — один запрос.** В промпте только рубрика и правила принятия решения этого критерия.
- **Жюри + судья.** Haiku и Sonnet голосуют параллельно; при расхождении решает Opus. На DEV (v7) жюри согласилось в 85.4% из 520 вердиктов, судья решил 76.
- **Меньше шума на шкалах.** Если судья решает шкальный критерий (04, 06), он отвечает 3 раза и берётся **медиана** шага (`CQA_JUDGE_SAMPLES=3`); бинарные — большинство (судья разбивает ничью 1:1). Каждый диалог получает поле `stability` (согласие голосов, диапазон итога по фактическим голосам).
- **Флаг «на проверку».** `needs_review` + причины на каждом критерии, где присяжные разошлись, решал судья или шкальное значение с низкой уверенностью; счётчик в интерфейсе (во внутреннем JSON, не в выгрузке v1.0).
- **Строгий JSON** через tool use: одна стабильная схема инструмента для всех критериев (общий кэш промпта); допустимые значения перечислены в блоке критерия и проверяются в коде (`checklist.strict_value`) — недопустимое значение → повтор с подсказкой; NA только там, где разрешён.
- **Факты, вычисленные кодом** (первая реплика оператора, эвристика языка клиента, самая долгая пауза после первого ответа и было ли предупреждение, удержания, как закончился диалог) передаются модели, чтобы она не считала время и роли сама.
- **Проверка цитат.** Цитата обязана быть дословной подстрокой указанной реплики (нормализация пробелов/кавычек/ё, фрагменты через «…»); неверный номер реплики исправляется, выдуманная цитата → повтор запроса, затем отбрасывается (`rejected_quotes`).
- **Роли и удержание.** Бот/система исключены; реплики во время консультации агентов помечены `[ВНУТРЕННИЕ ПЕРЕГОВОРЫ НА УДЕРЖАНИИ]`; ошибки ASR не наказываются.
- **Детерминированный скоринг** в коде, а не в LLM.
- **Кэш по критерию**: ключ = хэш(текст критерия + допущения + версия промпта) × модель × хэш диалога. Правка рубрики → переоценивается только этот критерий; правка веса → пересчёт без LLM.

## Интерфейс
- Дашборд: средний балл, гистограмма, распределение по критериям, флаги грубых ошибок, сортировка.
- Карточка диалога: подсветка реплик-доказательств, клик по цитате → прокрутка; NA с причиной; «ожидалось в реплике N»; ⚖ — разногласие жюри.
- Звонки: плеер `audio_url`, клик по реплике → перемотка.
- Ручная проверка: значение + обоснование + ревьюер; хранится отдельно, AI-вывод не меняется; итог «человек» и diff AI→Human; экспорт JSON / CSV AI / Human / оба.
- Загрузка JSON-файла, вставка диалога текстом, прогресс-бар оценки.
- Статусы надёжности: бейдж «неполная» (оценка критерия упала → итог не выставлен, показан предварительный), ⚠ предупреждения (неизвестные роли, доказательства не из реплик оператора), «требует проверки» для грубой ошибки без цитаты оператора; для 04 — список запросов клиента со статусами ✓/◐/✗.
- Недопустимая ручная оценка (например, 4.5 или NA там, где NA не разрешён) отклоняется сервером (HTTP 400).
- Проверка в браузере: `python scripts/ui_smoke.py http://127.0.0.1:8090 DEV` (Playwright, headless) — таблица, карточка диалога, жюри, бейдж «неполная», отказ 400 на неверную правку, 0 ошибок консоли.

## Гибкие чек-листы
Вкладка «Чек-листы» в UI или YAML в `checklists/`. Добавить критерий:
```yaml
  - id: "09"
    name: Предложение продукта
    type: binary            # или type: scale + scale: [0, 5, 10]
    weight: 5
    allow_na: true
    rubric: |
      YES — оператор уместно предложил релевантный продукт ...
    rules: |
      - ...
```
Сохранить → «Re-evaluate with current checklist» (к LLM уйдёт только новый критерий) или «Пересчитать веса» (без LLM). Валидация: уникальные id, тип, шкала (`tests/test_checklist.py`).

## Run on your own GPU (fine-tuned adapter, prototype)

QLoRA adapter for Qwen3-8B: https://huggingface.co/useing/cqa-qwen3-8b-lora (mirror:
https://github.com/useing123/d00mathon/releases/download/finetune-v1/cqa-qwen3-8b-lora.tar.gz). You need one 24 GB GPU.

```bash
vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules cqa=./cqa-qwen3-8b-lora --max-lora-rank 16 \
     --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}'
export CQA_PROVIDER=openai CQA_OPENAI_BASE_URL=http://127.0.0.1:8000/v1 CQA_OPENAI_API_KEY=x CQA_OPENAI_TOOLS=0
CQA_ENSEMBLE=0 CQA_MODEL=cqa python -m cqa gold --dir examples/gold_holdout              # single model
CQA_LOCAL_ADAPTER=cqa CQA_ROUTER_LOCAL_MODEL=cqa python -m cqa evaluate --preset cascade ...  # cascade tier 1
```

Measured after 0.32 epoch (16 steps, A5000, ≈$0.22): held-out agreement **0.787 → 0.787** (no gain), MAE 12.17 → 10.43.
Jev 1.14 scores 0.934 / 4.83 on the same set, so the adapter does not beat it yet. Details: [docs/FINETUNE.md](docs/FINETUNE.md).

## Локальная LLM (Ollama / vLLM)
```bash
CQA_PROVIDER=openai CQA_OPENAI_BASE_URL=http://127.0.0.1:11434/v1 CQA_MODEL=qwen2.5:14b \
  python -m cqa evaluate --input examples/sample_dialogs.json
```
Те же промпты, JSON-режим, та же схема и валидация цитат. `CQA_PROVIDER=mock` (или `--mock`) — детерминированная эвристика без сети для тестов, CI и демо (не для точности).

## Допущения

| # | Допущение | Где задано |
|---|---|---|
| 1 | Критерий 01: «первая реплика» = первая реплика **человека-оператора**; приветствие бота не засчитывается | `assumptions.first_reply` |
| 2 | Критерий 07: прощание + (итог ИЛИ предложение помощи), буквально по чек-листу | рубрика 07 |
| 3 | Время первого ответа, теги, вложения не оцениваются; «явное ожидание» позже — > 3 мин (180 с), задаётся только в YAML (`assumptions.long_wait_seconds`) | docs/ASSUMPTIONS.md |
| 4 | Звонки (по замечанию организаторов): критерий 06 оценивает логику, ясность, отсутствие неуместных выражений, корректность речи и управление паузами; в 04 и всех остальных ошибки ASR не наказываются | `freedom_call_v1.yaml` |
| 5 | Реплики во время удержания — внутренние, не оцениваются | `ignore_internal_hold_turns` |
| 6 | Аудио-смещение = время реплики − время первой реплики (приближение) | `cqa/data.py` |
| 7 | Округление ROUND_HALF_UP до 2 знаков | `assumptions.rounding` |

Подробно (включая таблицу сверки каждого правила с текстом официального чек-листа): [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md#rubric-alignment-official-checklist-pdf-vs-our-yaml).

## Выбор модели (пресеты)
Матрица цена/точность/скорость по 19 конфигурациям (Anthropic, OpenAI, Gemini, локальная Qwen; одиночные модели, жюри, варианты промптов): [docs/MODELS.md](docs/MODELS.md), страница **`/models`** в интерфейсе.
Победитель выбран автоматически по отложенному gold-набору: **`gpt-5.3-codex` — 97.5% (2 прогона), MAE 1.54, $0.058/диалог**; второе место — `gemini-4-argon` (96.9%), жюри Claude — 95.0%. Разница 1–2 метки из 80 в пределах шума, поэтому режим по умолчанию оставлен `jury` (им сделан DEV-файл), а `max` доступен как опция.
`python -m cqa evaluate --preset econom|balanced|max|jury` (или `CQA_PRESET=...`; явные `CQA_MODEL`/`CQA_JURORS`/`CQA_JUDGE` важнее пресета). Воспроизвести: `python scripts/bench_matrix.py snapshot && python scripts/bench_matrix.py run && python scripts/bench_matrix.py matrix`.

## Результаты
См. таблицу [в начале README](#результаты-текущая-версия-v7). Распределения значений до/после выравнивания рубрики,
шум и флаги проверки — [docs/EVALUATION.md](docs/EVALUATION.md); прежние версии — [docs/HISTORY.md](docs/HISTORY.md).

## Ограничения (честно)
- Для DEV нет размеченного ground truth; gold-набор синтетический и размечен нами — это регрессионная проверка, а не оценка точности на скрытой разметке.
- Если оценка критерия упала (ошибка API), диалог помечается `incomplete` и итог не выставляется (fail-closed).
- Наиболее неоднозначные критерии — 07, 04, 05 (субъективность рубрики).
- Облачный режим жюри стоит денег (≈ $0.16 на диалог); точность локальной модели мы не измеряли.
- Смещение аудио приблизительное: запись может стартовать не синхронно с транскриптом.
- Фоновые задачи UI живут в памяти процесса; авторизации нет (по условию).
- Mock-провайдер — эвристика для тестов, не бейзлайн качества.

## Структура проекта
```
cqa/
  config.py      .env + переменные окружения
  data.py        загрузка/fetch, нормализация, удержание, аудио-смещения
  checklist.py   YAML-чек-листы, валидация, хэш критерия, выбор по каналу
  prompts.py     системный промпт + промпт на каждый критерий + JSON-схема
  llm.py         Anthropic | OpenAI-совместимый | MockBackend
  evaluator.py   жюри/судья, проверка цитат, кэш, пул потоков
  scoring.py     детерминированный скоринг
  store.py       результаты, overrides, CSV
  server.py      FastAPI; static/index.html — UI
  __main__.py    CLI: evaluate / stats / rescore / serve
checklists/      freedom_chat_v1.yaml, freedom_call_v1.yaml
examples/        sample_dialogs.json — 4 синтетических диалога (RU/EN/KZ чат + RU звонок)
tests/           pytest (offline, mock): скоринг, цитаты, чек-листы, кэш-ключ, fail-closed, строгие значения, роли/время, факты, жюри, провайдеры, безопасность путей, gold, mock e2e, HTTP API
docs/            ARCHITECTURE, SCORING, ASSUMPTIONS, EVALUATION, API, презентация + PDF
EXPERIMENTS.md   все эксперименты: CI, парные тесты, лидерборд, Pareto, абляции (scripts/build_experiments.py)
Makefile · pyproject.toml · Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## CLI
`python -m cqa evaluate [--group G | --input file.json] [--ids a,b] [--limit N] [--checklist ID] [--fresh] [--mock] [--confirm 03,04] [--final] [--organizer out.csv] [--concurrency N] [--refresh]` · `python -m cqa stats --group G` · `python -m cqa rescore --group G` · `python -m cqa serve --port 8090` · `python -m cqa gold [--dir D | --labels F]` · `python -m cqa label` · `python -m cqa export --group G --format organizer`.

## Публичный репозиторий
`.env`, `data/` (сырые диалоги), `results/` (цитаты клиентов), `.cache/` — в `.gitignore`. Данные загружаются во время работы через `CQA_API_URL`. В `examples/` только выдуманные диалоги.

---

<a id="english"></a>
## English

**CQA** evaluates the *human operator* in Freedom Broker chats and voice calls (RU/KZ/EN) against a flexible YAML checklist. Each criterion is a separate LLM call; a jury (claude-haiku-5-5 + claude-sonnet-5-5) votes in parallel and claude-opus-5-5 adjudicates disagreements. Every verdict carries a justification and verbatim, turn-referenced quotes validated against the transcript. The 0–100 score is computed in code (NA weight redistribution, Decimal ROUND_HALF_UP); gross errors are a separate flag.

**Quickstart:** `make install && make demo && make serve` (offline, mock LLM). Real run: fill `.env` from `.env.example`, then `python -m cqa evaluate --group <ID>` → `results/<id>.json|csv`.

**Highlights:** per-criterion prompts · LLM jury + judge · quote validation with auto-correction / retry · deterministic scoring · per-criterion cache (editing one rubric re-evaluates only that criterion; reweighting needs no LLM) · no-login web UI with evidence highlighting, audio seek, human override (stored separately, AI vs Human diff, CSV/JSON export) and checklist editor · cloud (Anthropic), local (Ollama/vLLM via OpenAI-compatible API) or offline mock · Docker · CI on 3.11/3.12 without secrets.

**Results (current version, v7):** held-out synthetic set **0.95 / MAE 4.43** (one run; earlier runs 0.925–0.938 = noise floor); real-DEV agreement not measured (needs human labels: `cqa label`). DEV: 65 dialogs, 0 errors, mean 80.41, 5 gross-error flags, ~5 min, **$0.16 per dialog** ($10.28 for 65). `needs_review` on 18.3% of criteria. History: [docs/HISTORY.md](docs/HISTORY.md).

**Limitations:** no labelled ground truth for real DEV dialogs; both gold sets are synthetic and self-labelled; criteria 04/05/06 are the most subjective; audio offsets are approximate; local-model accuracy not measured (no local GPU run in this repo).

**Robustness:** fail-closed `incomplete` status after a rescue pass over other models, strict values, operator-quote checks, untrusted-transcript boundary, code-computed facts (waits, holds, language, ending), path-safe ids, atomic writes.

UI: [docs/UI.md](docs/UI.md).

Docs: [Architecture](docs/ARCHITECTURE.md) · [Scoring](docs/SCORING.md) · [Assumptions](docs/ASSUMPTIONS.md) · [Evaluation](docs/EVALUATION.md) · [API](docs/API.md) · [Deck](docs/CQA_Pitch_Deck.pdf). License: [MIT](LICENSE).
