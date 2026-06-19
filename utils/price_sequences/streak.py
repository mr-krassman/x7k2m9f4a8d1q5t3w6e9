"""Подписанная длина серии роста/падения перед текущим днём."""

from __future__ import annotations

import numpy as np
import polars as pl

STREAK_SIGNED_PREV_COL = "streak_signed_prev"


def signed_streak_before_returns(returns: np.ndarray) -> np.ndarray:
    """Для индекса i≥1: знак×длина серии, закончившейся вчера; иначе nan."""
    n = returns.size
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        prev = float(returns[i - 1])
        if prev == 0:
            continue
        sign = 1.0 if prev > 0 else -1.0
        streak = 1
        j = i - 2
        while j >= 0:
            rj = float(returns[j])
            if rj == 0:
                break
            same_sign = (rj > 0 and sign > 0) or (rj < 0 and sign < 0)
            if not same_sign:
                break
            streak += 1
            j -= 1
        out[i] = sign * float(streak)
    return out


def attach_streak_signed_prev(daily: pl.DataFrame) -> pl.DataFrame:
    if daily.is_empty():
        return daily.with_columns(pl.Series(STREAK_SIGNED_PREV_COL, [], dtype=pl.Float64))
    if "pair" not in daily.columns:
        sub = daily.sort("day_utc")
        signed = signed_streak_before_returns(
            sub["return_pct"].to_numpy().astype(np.float64, copy=False)
        )
        return sub.with_columns(pl.Series(STREAK_SIGNED_PREV_COL, signed, dtype=pl.Float64))
    chunks: list[pl.DataFrame] = []
    for pair in daily["pair"].unique().to_list():
        sub = daily.filter(pl.col("pair") == pair).sort("day_utc")
        signed = signed_streak_before_returns(
            sub["return_pct"].to_numpy().astype(np.float64, copy=False)
        )
        chunks.append(
            sub.with_columns(pl.Series(STREAK_SIGNED_PREV_COL, signed, dtype=pl.Float64))
        )
    return pl.concat(chunks, how="vertical")
