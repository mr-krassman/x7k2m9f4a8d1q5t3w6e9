"""Константы исследования harami."""

from __future__ import annotations

SCENARIO_ROWS: tuple[str, ...] = (
    "Bullish Harami",
    "Bearish Harami",
)

BUCKET_BULLISH = 0
BUCKET_BEARISH = 1

PATTERN_NOTE = (
    "Bullish Harami: день t−1 медвежий, день t — тело внутри тела t−1, тело t < тело t−1; "
    "сигнал long на следующий день (t+1). "
    "Bearish Harami: день t−1 бычий, день t — тело внутри тела t−1; сигнал short на t+1."
)
