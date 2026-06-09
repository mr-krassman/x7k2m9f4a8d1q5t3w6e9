"""Сводка отклонения close от EMA по периодам."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.ema_spreads.ema import (
    average_deviation_summaries,
    summarize_ema_deviation_per_pair,
)
from crypto_research.utils.weekday.bands import TRIM_HI_PCT, TRIM_LO_PCT


def _fmt(v: float, width: int = 7) -> str:
    if v != v:
        return "n/a".rjust(width)
    return f"{v:+.2f}".rjust(width)


def build_deviation_summary_lines(
    daily: pl.DataFrame,
    periods: tuple[int, ...],
) -> list[str]:
    if "day_close" not in daily.columns:
        return ["=== Отклонение close от EMA: нет колонки day_close ===", ""]

    per_pair = summarize_ema_deviation_per_pair(daily, periods)
    if not per_pair:
        return ["=== Отклонение close от EMA: недостаточно данных ===", ""]

    n_pairs = len(per_pair)
    if n_pairs == 1:
        summaries = next(iter(per_pair.values()))
        title_note = "одна пара"
    else:
        summaries = average_deviation_summaries(per_pair, periods)
        title_note = f"среднее по {n_pairs} парам"

    col_w = 8
    header = (
        f"{'EMA':<8} | {'mean':>{col_w}} | {'median':>{col_w}} | "
        f"{'|dev|':>{col_w}} | {'std':>{col_w}} | {'min':>{col_w}} | "
        f"{'max':>{col_w}} | {'n':>6}"
    )
    sep = "-" * len(header)

    lines: list[str] = []
    lines.append("=== Отклонение close от EMA (%) ===")
    lines.append(
        f"dev = (close − EMA) / EMA × 100 на дневном close UTC; периоды "
        f"{', '.join(str(p) for p in periods)}. "
        f"Счёт по каждой паре отдельно; ниже — {title_note}."
    )
    lines.append(
        f"Обрезка внутри каждой пары: перцентили {TRIM_LO_PCT:g}–{TRIM_HI_PCT:g}% "
        f"(как в блоке дней роста/падения). "
        "mean — со знаком; |dev| — средняя дистанция без знака."
    )
    lines.append(header)
    lines.append(sep)
    for period in periods:
        if period not in summaries:
            continue
        s = summaries[period]
        lines.append(
            f"EMA({period:<3}) | {_fmt(s.mean_dev_pct, col_w)} | "
            f"{_fmt(s.median_dev_pct, col_w)} | {_fmt(s.mean_abs_dev_pct, col_w)} | "
            f"{_fmt(s.std_dev_pct, col_w)} | {_fmt(s.min_dev_pct, col_w)} | "
            f"{_fmt(s.max_dev_pct, col_w)} | {s.kept_days:>6}"
        )
    lines.append("")
    return lines
