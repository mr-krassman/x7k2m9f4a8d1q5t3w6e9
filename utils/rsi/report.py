"""Сборка и сохранение отчёта rsi_spreads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.ema_spreads.report import format_pairs_lines
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import rsi_spreads_stats_log_path
from crypto_research.utils.rsi.conditional_table import build_rsi_period_table_lines
from crypto_research.utils.rsi.constants import RSI_BUCKET_THRESHOLDS_NOTE
from crypto_research.utils.rsi.value_summary import build_rsi_level_summary_lines
from crypto_research.utils.weekday.bands import MeanBands

log = get_logger("rsi_spreads_report")


def build_report_header(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> list[str]:
    lines = [
        "=== Отчёт: rsi_spreads (RSI вчера → return сегодня) ===",
        f"Пар: {len(pairs)}",
        f"Период (UTC): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
        f"Периоды RSI: {', '.join(str(p) for p in periods)}",
    ]
    if max_pair_start is not None:
        lines.append(
            f"Фильтр пар: первая свеча не позже {max_pair_start:%Y-%m-%d} (UTC)"
        )
    lines.extend(format_pairs_lines(pairs))
    lines.append("")
    lines.append(
        "Методика: RSI Wilder(N) по дневному close UTC; 6 квантилей по RSI вчера. "
        "Сегодняшний return = (close−open)/open×100%. "
        "Без look-ahead: границы квантилей по выборке, RSI считается только по истории пары."
    )
    lines.append("")
    return lines


def _bucket_legend_lines() -> list[str]:
    lines = ["=== Пояснение к таблицам RSI ===", "", RSI_BUCKET_THRESHOLDS_NOTE, ""]
    for i in range(6):
        lines.append(f"  b{i}: Q{i + 1}")
    lines.append("")
    return lines


_REPORT_FOOTER = [
    *_bucket_legend_lines(),
    "Колонки — условные доли дневного return сегодня (close vs open и intraday high/low); "
    "пороги mean — отдельно по каждой паре (μ×0.5 / μ×1.5 дней роста и падения).",
    "",
    "В ячейке (X/Y) [N]: X/Y — годы со согласованным знаком Δ к BASE; "
    "[N] — число пар с тем же знаком Δ.",
]


def build_rsi_spreads_table_lines(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
) -> list[str]:
    lines = build_rsi_level_summary_lines(daily, periods)
    for period in periods:
        lines.extend(build_rsi_period_table_lines(daily, period, pair_bands))
    return lines


def run_rsi_spreads_report(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> Path:
    header = build_report_header(pairs, from_date, to_date, periods, max_pair_start)
    table_lines = build_rsi_spreads_table_lines(daily, pair_bands, periods)
    text = "\n".join(header + table_lines + _REPORT_FOOTER)
    path = rsi_spreads_stats_log_path(len(pairs), from_date, to_date, periods)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[rsi_spreads] отчёт сохранён: %s", path)
    return path
