"""Сборка log-отчёта price_sequences."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import (
    TRAIN_MAX_PAIR_START,
    VAL_MAX_PAIR_START,
    price_sequences_stats_log_path,
)
from crypto_research.utils.pipeline.weekday_effects import format_pairs_lines
from crypto_research.utils.price_sequences.table import build_price_sequence_table
from crypto_research.utils.weekday.bands import MeanBands
from crypto_research.utils.weekday.repeatability import MIN_YEAR_BASE_DAYS, MIN_YEAR_ROW_DAYS

import polars as pl

log = get_logger("price_sequences_report")

_REPORT_FOOTER = [
    "=== Пояснение к строкам таблицы ===",
    "",
    "Строка, например «После 3д падения (n=4677)»:",
    "",
    "  После Nд падения/роста — перед текущим днём была непрерывная серия дней с закрытием "
    "ниже/выше open; N=1…6 (для серий ≥6 дней используется бакет 6д).",
    "  n=4677 — число дней в выборке (все пары × все такие эпизоды).",
    "",
    "Пороги μ×0.5 / μ×1.5 в колонках — отдельно по каждой паре "
    "(среднее дней роста / падения после обрезки перцентилей 5–95%).",
    "",
    "=== Пояснение к ячейкам таблиц ===",
    "",
    "В ячейке, например 50.7 (2/5) [36]:",
    "",
    "  50.7     — доля дней с условием колонки (%)",
    "  (2/5)    — в Y годах с достаточной историей знак отклонения доли колонки от BASE "
    "совпал с общим; в X — совпал",
    "  [36]     — 36 пар из отобранных, у которых на своих данных тот же знак Δ",
    "",
    f"Год в знаменатель (X/Y) попадает только если в этом году ≥ {MIN_YEAR_BASE_DAYS} дней для BASE "
    f"и ≥ {MIN_YEAR_ROW_DAYS} дней для строки сценария.",
    f"Для [пар] аналогично: ≥ {MIN_YEAR_BASE_DAYS} / ≥ {MIN_YEAR_ROW_DAYS} дней по каждой паре.",
]


def build_report_header(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    max_pair_start: datetime | None = None,
    split: str | None = None,
) -> list[str]:
    lines = [
        "=== Отчёт: прогностическая сила последовательных движений цены ===",
        f"Пар: {len(pairs)}",
        f"Период теста (UTC): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
    ]
    if split is not None:
        lines.append(f"Split: {split}")
        lines.append(
            f"Фильтр пар ({split}): "
            f"train-cohort ≤ {TRAIN_MAX_PAIR_START}, val = пары пула {VAL_MAX_PAIR_START} вне train"
            if split == "val"
            else f"первая свеча не позже {TRAIN_MAX_PAIR_START} (UTC)"
        )
    elif max_pair_start is not None:
        lines.append(f"Фильтр пар: первая свеча не позже {max_pair_start:%Y-%m-%d} (UTC)")
    lines.extend(format_pairs_lines(pairs))
    lines.append("")
    return lines


def assemble_price_sequences_report(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    table_lines: list[str],
    max_pair_start: datetime | None = None,
    split: str | None = None,
) -> str:
    return "\n".join(
        build_report_header(pairs, from_date, to_date, max_pair_start, split)
        + table_lines
        + _REPORT_FOOTER
    )


def run_price_sequences_report(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    max_pair_start: datetime | None = None,
    split: str | None = None,
) -> Path:
    table_lines = build_price_sequence_table(daily, pair_bands)
    text = assemble_price_sequences_report(
        pairs,
        from_date,
        to_date,
        table_lines,
        max_pair_start,
        split,
    )
    path = price_sequences_stats_log_path(len(pairs), from_date, to_date, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[price_sequences] отчёт сохранён: %s", path)
    return path
