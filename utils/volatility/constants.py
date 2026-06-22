"""Константы исследования volatility."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import (
    EMA_BUCKET_THRESHOLDS_NOTE,
    EMA_SCENARIO_ROWS,
    SCREEN_MIN_POOLED_DELTA_PP,
    SCREEN_STATS_COLS,
)

DEFAULT_SCREEN_RANGE_SMA_PERIODS: tuple[int, ...] = (5, 9, 12, 20, 50, 100, 200)
SELECTED_RANGE_SMA_PERIOD = 9

N_VOLATILITY_BUCKETS = len(EMA_SCENARIO_ROWS)

VOLATILITY_SCENARIO_ROWS: tuple[str, ...] = tuple(
    row.replace("dev", "ratio−1") for row in EMA_SCENARIO_ROWS
)

VOLATILITY_BUCKET_THRESHOLDS_NOTE = (
    "range_pct = (H−L)/C×100; ratio = range_pct / SMA(range_pct, N); "
    "dev = ratio−1 на день сигнала (вчера). "
    + EMA_BUCKET_THRESHOLDS_NOTE.replace("dev", "ratio−1")
)

__all__ = [
    "DEFAULT_SCREEN_RANGE_SMA_PERIODS",
    "N_VOLATILITY_BUCKETS",
    "SCREEN_MIN_POOLED_DELTA_PP",
    "SCREEN_STATS_COLS",
    "SELECTED_RANGE_SMA_PERIOD",
    "VOLATILITY_BUCKET_THRESHOLDS_NOTE",
    "VOLATILITY_SCENARIO_ROWS",
]
