# Контрольный прогон (prod) · runbook

Цель: за час после выдачи `group_id` получить `results/prod/results.json` (JSON 1.0, ровно 10 записей) и запушить до стоп-кода.

## Система для prod

**Жюри Claude (пресет `jury`, по умолчанию):** `claude-haiku-5-5` + `claude-sonnet-5-5` голосуют, при расхождении решает `claude-opus-5-5` (медиана из 3 на шкалах 04/06).
Им же сделан `results/dev/results.json`, он не зависит от GPU и OpenRouter. Отложенный эталон 0.95.
Цена ≈ $0.16 за диалог, 10 диалогов ≈ $1.6 и 1–2 минуты.

Альтернативы (только если жюри недоступно):
- `CQA_PRESET=cascade`: Jev → Haiku → Sonnet, 0.963, ≈ $0.004–0.015/диалог (нужен ключ OpenRouter с балансом).
- Локально: Qwen3.8-27B-FP8 на H100 через vLLM, 0.963 на отложенном наборе, без облака (см. docs/KZ_MODELS.md, pod остановлен, не удалён).

## Шаги (≈ 5 минут)

```bash
cd /root/cqa
git pull --ff-only                       # актуальная версия
grep -c CQA_API_URL .env                 # 1 = ссылка API на месте
make submit GROUP=<group_id>             # fetch → evaluate → export JSON 1.0 → validate → commit
git push origin HEAD:main
curl -s -o /dev/null -w "%{http_code}\n" https://raw.githubusercontent.com/useing123/d00mathon/main/results/prod/results.json
```

`make submit` сам снимает переменные `CQA_MODEL/CQA_ENSEMBLE/CQA_JURORS/CQA_JUDGE`, чтобы старые значения из шелла не подменили модель.

## Проверки перед push
- `python -m cqa validate results/prod/results.json --group <group_id>`: ровно 10 записей, ID совпадают с API, все 8 критериев, `critical_flag == critical`.
- Ошибочный диалог остаётся записью `status: "error"`, не нулём.
- Если в контроле заменили критерий через UI, экспорт берёт его код (валидатор выдаёт предупреждение, не ошибку).

## Если что-то сломалось
- Anthropic 429/5xx: повторы с backoff встроены. Повторить `make submit` (кэш не пересчитывает готовое).
- Нет ответа API организаторов: сохранить ответ вручную в `data/<group_id>.json` (формат как у DEV) и запустить без `--refresh`.
- Совсем нет облака: локальный режим на H100 (docs/KZ_MODELS.md, команды запуска pod).

## Репетиция (сделана 07.11)
10 диалогов DEV как фиктивная группа `REH`: evaluate → export → validate прошли, 10 записей, 0 ошибок, валидатор: «valid JSON 1.0 (10 records, prod: exactly 10)». Прогон с нуля (`--fresh`, без кэша): 59 с, $1.79, 0 ошибок.
