"""Сводка уровней RSI по периодам."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.rsi.rsi import build_rsi_work_frame, rsi_prev_column


def build_rsi_level_summary_lines(
    daily: pl.DataFrame,
    periods: tuple[int, ...],
) -> list[str]:
    if "day_close" not in daily.columns or daily.height == 0:
        return ["=== Уровни RSI: недостаточно данных ===", ""]

    col_w = 8
    header = (
        f"{'RSI':<8} | {'mean':>{col_w}} | {'median':>{col_w}} | "
        f"{'std':>{col_w}} | {'min':>{col_w}} | {'max':>{col_w}} | {'n':>6}"
    )
    sep = "-" * len(header)
    lines = [
        "=== Уровни RSI (вчера) ===",
        "RSI Wilder(N) по дневному close UTC; pooled по всем парам и дням выборки.",
        header,
        sep,
    ]
    for period in periods:
        work = build_rsi_work_frame(daily, period)
        prev_col = rsi_prev_column(period)
        if prev_col not in work.columns:
            continue
        vals = work[prev_col].to_numpy().astype(np.float64, copy=False)
        clean = vals[np.isfinite(vals)]
        if clean.size == 0:
            continue
        lines.append(
            f"RSI({period:<3}) | {clean.mean():>{col_w}.1f} | "
            f"{np.median(clean):>{col_w}.1f} | {clean.std():>{col_w}.1f} | "
            f"{clean.min():>{col_w}.1f} | {clean.max():>{col_w}.1f} | {clean.size:>6}"
        )
    lines.append("")
    return lines
