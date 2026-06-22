"""Детекция Bullish / Bearish Harami по дневным OHLC."""

from __future__ import annotations

import numpy as np

from crypto_research.utils.candlestick_patterns.harami.constants import BUCKET_BEARISH, BUCKET_BULLISH


def _body_bounds(open_: float, close: float) -> tuple[float, float, float]:
    low = min(open_, close)
    high = max(open_, close)
    return low, high, high - low


def is_bullish_harami(open1: float, close1: float, open2: float, close2: float) -> bool:
    if close1 >= open1:
        return False
    b1_low, b1_high, b1_size = _body_bounds(open1, close1)
    b2_low, b2_high, b2_size = _body_bounds(open2, close2)
    if b1_size <= 0.0:
        return False
    if b2_size <= 0.0 or b2_size >= b1_size:
        return False
    return b2_low >= b1_low and b2_high <= b1_high


def is_bearish_harami(open1: float, close1: float, open2: float, close2: float) -> bool:
    if close1 <= open1:
        return False
    b1_low, b1_high, b1_size = _body_bounds(open1, close1)
    b2_low, b2_high, b2_size = _body_bounds(open2, close2)
    if b1_size <= 0.0:
        return False
    if b2_size <= 0.0 or b2_size >= b1_size:
        return False
    return b2_low >= b1_low and b2_high <= b1_high


def is_bullish_harami_confirmed(
    open1: float,
    close1: float,
    open2: float,
    close2: float,
    open3: float,
    close3: float,
) -> bool:
    """Harami (t−2,t−1) + подтверждение на t: close(t) > open(t−2)."""
    return is_bullish_harami(open1, close1, open2, close2) and close3 > open1


def is_bearish_harami_confirmed(
    open1: float,
    close1: float,
    open2: float,
    close2: float,
    open3: float,
    close3: float,
) -> bool:
    """Harami (t−2,t−1) + подтверждение на t: close(t) < open(t−2)."""
    return is_bearish_harami(open1, close1, open2, close2) and close3 < open1


def confirmed_harami_on_signal_day(
    opens: np.ndarray,
    closes: np.ndarray,
    j: int,
) -> int | None:
    """На индексе j (= t+1, сегодня): паттерн (t−2, t−1), подтверждение на t (= j−1)."""
    if j < 3:
        return None
    o1, c1 = float(opens[j - 3]), float(closes[j - 3])
    o2, c2 = float(opens[j - 2]), float(closes[j - 2])
    o3, c3 = float(opens[j - 1]), float(closes[j - 1])
    if is_bullish_harami_confirmed(o1, c1, o2, c2, o3, c3):
        return BUCKET_BULLISH
    if is_bearish_harami_confirmed(o1, c1, o2, c2, o3, c3):
        return BUCKET_BEARISH
    return None


def assign_harami_buckets(
    opens: np.ndarray,
    closes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """bucket на индексе i+1, если паттерн завершился на свечах (i−1, i)."""
    n = opens.size
    buckets = np.full(n, -1, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    if n < 3:
        return buckets, valid
    for i in range(1, n - 1):
        o1, c1 = float(opens[i - 1]), float(closes[i - 1])
        o2, c2 = float(opens[i]), float(closes[i])
        if is_bullish_harami(o1, c1, o2, c2):
            buckets[i + 1] = BUCKET_BULLISH
            valid[i + 1] = True
        elif is_bearish_harami(o1, c1, o2, c2):
            buckets[i + 1] = BUCKET_BEARISH
            valid[i + 1] = True
    return buckets, valid
