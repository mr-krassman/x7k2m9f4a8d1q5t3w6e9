"""Сборка и сохранение отчёта ema_spreads."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.ema_spreads.conditional_table import build_ema_period_table_lines
from crypto_research.utils.ema_spreads.constants import EMA_BUCKET_THRESHOLDS_NOTE, EMA_SCENARIO_ROWS
from crypto_research.utils.ema_spreads.deviation_summary import build_deviation_summary_lines
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import ema_spreads_stats_log_path
from crypto_research.utils.weekday.bands import MeanBands

log = get_logger("ema_spreads_report")
_PAIRS_PER_LINE = 11


def format_pairs_lines(pairs: list[str], per_line: int = _PAIRS_PER_LINE) -> list[str]:
    sorted_pairs = sorted(pairs)
    if not sorted_pairs:
        return ["Пары: —"]
    out: list[str] = []
    for i in range(0, len(sorted_pairs), per_line):
        chunk = sorted_pairs[i : i + per_line]
        prefix = "Пары: " if i == 0 else "      "
        out.append(f"{prefix}{', '.join(chunk)}")
    return out


def build_report_header(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> list[str]:
    lines = [
        "=== Отчёт: ema_spreads (отклонение close от EMA) ===",
        f"Пар: {len(pairs)}",
        f"Период (UTC): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
        f"Периоды EMA: {', '.join(str(p) for p in periods)}",
    ]
    if max_pair_start is not None:
        lines.append(
            f"Фильтр пар: первая свеча не позже {max_pair_start:%Y-%m-%d} (UTC)"
        )
    lines.extend(format_pairs_lines(pairs))
    lines.append("")
    lines.append(
        "Методика: dev вчера = (close−EMA)/EMA×100 на дневном close UTC; "
        "сегодняшний return = (close−open)/open×100%. "
        "7 бакетов b0–b6 — диапазоны dev по порогам t1⁺/t2⁺/t1⁻/t2⁻/near своей пары. "
        "Без look-ahead: пороги и EMA считаются только по истории пары до текущего дня."
    )
    lines.append("")
    return lines


def _bucket_legend_lines() -> list[str]:
    lines = ["=== Пояснение к таблицам EMA ===", "", EMA_BUCKET_THRESHOLDS_NOTE, ""]
    for i, label in enumerate(EMA_SCENARIO_ROWS):
        lines.append(f"  b{i}: {label}")
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


def build_ema_spreads_table_lines(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
) -> list[str]:
    lines = build_deviation_summary_lines(daily, periods)
    for period in periods:
        lines.extend(build_ema_period_table_lines(daily, period, pair_bands, periods))
    return lines


def assemble_ema_spreads_report(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    table_lines: list[str],
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> str:
    parts = (
        build_report_header(pairs, from_date, to_date, periods, max_pair_start)
        + table_lines
        + _REPORT_FOOTER
    )
    return "\n".join(parts)


def save_ema_spreads_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[ema_spreads] отчёт сохранён: %s", path)
    return path


def run_ema_spreads_report(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> Path:
    table_lines = build_ema_spreads_table_lines(daily, pair_bands, periods)
    text = assemble_ema_spreads_report(
        pairs,
        from_date,
        to_date,
        table_lines,
        periods,
        max_pair_start,
    )
    path = ema_spreads_stats_log_path(len(pairs), from_date, to_date, periods)
    return save_ema_spreads_report(text, path)
