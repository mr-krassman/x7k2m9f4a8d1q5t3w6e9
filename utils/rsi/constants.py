"""Константы исследования rsi_period_screen."""

from __future__ import annotations

DEFAULT_SCREEN_RSI_PERIODS: tuple[int, ...] = (5, 9, 14, 21, 50)
SELECTED_RSI_PERIOD = 9
DEFAULT_RSI_PERIODS: tuple[int, ...] = (SELECTED_RSI_PERIOD,)
N_RSI_QUANTILES = 6
SCREEN_MIN_POOLED_DELTA_PP = 2.0

RSI_BUCKET_THRESHOLDS_NOTE = (
    "RSI Wilder(N) по дневному close; 6 квантилей по RSI вчера "
    "(границы по выборке, без look-ahead на return). "
    "Бакеты b0–b5 — Q1 (низкий RSI) … Q6 (высокий RSI)."
)
