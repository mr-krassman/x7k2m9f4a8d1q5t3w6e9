# crypto_research

Отчёты по эффектам дня недели на минутных Bybit JSONL (`*_klines_1m.jsonl`).

## Зависимости

- Python 3.10+
- Пакеты: `pip install -r requirements.txt`
- Модуль **`statistic`** из репозитория [crypto_bot](https://github.com/mr-krassman/crypto_bot) (или вашего форка): таблицы weekday, пороги μ, повторяемость по годам.

Переменные окружения (опционально):

| Переменная | Назначение |
|------------|------------|
| `CRYPTO_BOT_ROOT` | Корень репозитория crypto_bot (где лежит `statistic/`). По умолчанию — родительская папка `crypto_research`. |
| `CRYPTO_DATA_DIR` | Папка с `*_klines_1m.jsonl`. По умолчанию: `$CRYPTO_BOT_ROOT/load_data_from_bybit/data`. |

Пример рядом с crypto_bot:

```bash
export CRYPTO_BOT_ROOT=/home_admin/crypto_bot
export CRYPTO_DATA_DIR=$CRYPTO_BOT_ROOT/load_data_from_bybit/data
```

Клон только этого репозитория (папка должна называться `crypto_research`):

```bash
git clone git@github.com:YOUR_USER/crypto_research.git ~/crypto_research
export CRYPTO_BOT_ROOT=/path/to/crypto_bot
```

## Запуск

Из каталога репозитория:

```bash
python3 report_generator.py \
  --from-date 2022-01-01 \
  --to-date 2026-05-31 \
  --max-pair-start 2022-01-01
```

Артефакты:

- `research_outputs/day_of_week/statistics/weekday_statistics_{N}pairs_{from}_{to}.log`
- `research_outputs/day_of_week/statistics/plots/dow_intraday_session_nav_{N}pairs_{from}_{to}.png`

## GitHub (первый push)

1. На GitHub: **New repository** → имя `crypto_research` → без README (уже есть локально).
2. Локально:

```bash
cd /path/to/crypto_research
git remote add origin git@github.com:YOUR_USER/crypto_research.git
git branch -M main
git push -u origin main
```

SSH: `git@github.com:...` · HTTPS: `https://github.com/YOUR_USER/crypto_research.git`

## Структура

```
report_generator.py   # оркестратор
utils/                # загрузка, пулы, таблицы, графики
research_outputs/     # генерируется, в git не попадает
```
