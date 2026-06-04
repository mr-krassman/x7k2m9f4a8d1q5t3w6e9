# crypto_research

Автономный проект: отчёты по эффектам дня недели на минутных Bybit JSONL (`*_klines_1m.jsonl`).

Внешняя зависимость — только **данные** (папка с JSONL). Код `statistic` / crypto_bot не нужен.

## Установка

```bash
cd crypto_research
pip install -r requirements.txt
```

Папка репозитория должна называться `crypto_research` (имя Python-пакета), либо запускайте из родителя:

```bash
cd /path/to/parent
python3 crypto_research/report_generator.py ...
```

## Данные

Положите `*_klines_1m.jsonl` в `data/` внутри репозитория или укажите путь:

```bash
export CRYPTO_DATA_DIR=/path/to/jsonl_folder
python3 report_generator.py --data-dir "$CRYPTO_DATA_DIR" ...
```

Если `crypto_research` лежит внутри crypto_bot и `data/` пустая, по умолчанию берётся
`../load_data_from_bybit/data` (когда там есть JSONL).

## Запуск

```bash
python3 report_generator.py \
  --from-date 2022-01-01 \
  --to-date 2026-05-31 \
  --max-pair-start 2022-01-01
```

Артефакты:

- `research_outputs/day_of_week/statistics/weekday_statistics_{N}pairs_{from}_{to}.log`
- `research_outputs/day_of_week/statistics/plots/dow_intraday_session_nav_{N}pairs_{from}_{to}.png`

## GitHub

Репозиторий: https://github.com/mr-krassman/x7k2m9f4d1q5t3w6e9

```bash
git remote add origin git@github.com:mr-krassman/x7k2m9f4d1q5t3w6e9.git
git push -u origin main
```

## Структура

```
report_generator.py
docs/
  journal.md        # хронология и планы
utils/
  weekday/          # расчёт: returns, bands, table, repeatability
  pipeline/         # загрузка JSONL, графики, CLI, сборка отчёта
data/               # ваши JSONL (не в git)
research_outputs/   # генерируется
```
