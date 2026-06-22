"""Константы исследования volume_ema_period_screen."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import (
    DEFAULT_SCREEN_EMA_PERIODS,
    SCREEN_MIN_POOLED_DELTA_PP,
    SCREEN_STATS_COLS,
)

DEFAULT_SCREEN_VOLUME_EMA_PERIODS: tuple[int, ...] = DEFAULT_SCREEN_EMA_PERIODS
SELECTED_VOLUME_EMA_PERIOD = 50

VOLUME_QUANTILE_PROBS: tuple[float, ...] = (0.30, 0.70, 0.85, 0.95)
N_VOLUME_BUCKETS = 5

VOLUME_BUCKET_LABELS: tuple[str, ...] = (
    "LOW(<=q30)",
    "NORMAL(q30-q70)",
    "HIGH(q70-q85)",
    "VERY_HIGH(q85-q95)",
    "SPIKE(>q95)",
)

VOLUME_BUCKET_THRESHOLDS_NOTE = (
    "ln(day_volume/EMA(volume,N)) вчера; пороги q30/q70/q85/q95 per-pair на train. "
    "Бакеты — см. строки таблицы."
)

__all__ = [
    "DEFAULT_SCREEN_VOLUME_EMA_PERIODS",
    "SELECTED_VOLUME_EMA_PERIOD",
    "N_VOLUME_BUCKETS",
    "SCREEN_MIN_POOLED_DELTA_PP",
    "SCREEN_STATS_COLS",
    "VOLUME_BUCKET_LABELS",
    "VOLUME_BUCKET_THRESHOLDS_NOTE",
    "VOLUME_QUANTILE_PROBS",
]
