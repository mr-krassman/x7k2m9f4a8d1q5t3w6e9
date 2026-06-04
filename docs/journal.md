# Journal — crypto_research

Хронология решений и планов. Техническая документация по запуску — в [README](../README.md).

## Цель проекта

Автономный исследовательский пайплайн по минутным Bybit JSONL (`*_klines_1m.jsonl`):

- дневная доходность UTC (open → close);
- таблицы условий с порогами μ×0.5 / μ×1.5 по каждой паре;
- повторяемость по годам `(X/Y)` и согласие пар `[N]`;
- отчёты и графики в `research_outputs/`.

Сейчас реализован блок **day of week**; далее — перенос и расширение логики из `crypto_bot/statistic/load_jsonl.py` (серии дней, RSI, EMA, объём и др.).

Внешняя зависимость — только **данные** (JSONL).

---

## 2026-06-04

### Сделано

- Вынесен отдельный репозиторий `crypto_research`, без зависимости от `statistic/`.
- Пакет `utils/weekday/` (returns, bands, table, repeatability) и `utils/pipeline/` (загрузка, CLI, графики, сборка лога).
- Оркестратор `report_generator.py`: пул пар → mean bands → таблицы + PNG.
- Метрика на графике: **Cumulative Simple Return (%)**, не compounded NAV.
- Имена артефактов с тегом `{N}pairs_{from}_{to}`.
- Пути к данным: `CRYPTO_DATA_DIR`, `data/`, fallback на `../load_data_from_bybit/data`.
- Локальный git (`main`), подготовка к публикации на GitHub.

### Заметки

- `(X/Y)` в ячейке — согласие знака Δ по **годам** на объединённой выборке; `[N]` — число **пар** с тем же знаком при разборе по паре. Метрики независимы.

### Дальше

- [ ] Подключить `git remote` и `push` на GitHub.
- [ ] Обновить README под общее описание репозитория (не только DOW).
- [ ] Перенести из `load_jsonl.py`: transition tables, RSI, EMA, train/validate split.
- [ ] Единый оркестратор отчётов или подкоманды CLI.

---

## Шаблон записи

```markdown
## YYYY-MM-DD

### Сделано
- ...

### Решения
- ...

### Дальше
- [ ] ...
```
