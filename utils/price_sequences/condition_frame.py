"""Подготовка фрейма: серия роста/падения → bucket сценария."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.price_sequences.constants import MAX_STREAK_DAYS, SCENARIO_ROWS
from crypto_research.utils.weekday.bands import MeanBands

_ROW_TO_IDX = {name: i for i, name in enumerate(SCENARIO_ROWS)}


def _row_bucket(day_before_sign: int, streak_len: int) -> str | None:
    if day_before_sign not in (-1, 1) or streak_len <= 0:
        return None
    bucket = min(streak_len, MAX_STREAK_DAYS)
    if day_before_sign == -1:
        return f"После {bucket}д падения"
    return f"После {bucket}д роста"


def _assign_streak_buckets(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = returns.size
    buckets = np.full(n, -1, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev = float(returns[i - 1])
        if prev == 0:
            continue
        sign = 1 if prev > 0 else -1
        streak = 1
        j = i - 2
        while j >= 0:
            rj = float(returns[j])
            if rj == 0:
                break
            same_sign = (rj > 0 and sign == 1) or (rj < 0 and sign == -1)
            if not same_sign:
                break
            streak += 1
            j -= 1
        row_name = _row_bucket(sign, streak)
        if row_name is None:
            continue
        buckets[i] = _ROW_TO_IDX[row_name]
        valid[i] = True
    return buckets, valid


def prepare_price_sequence_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    **_kwargs,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    del period
    if daily.is_empty():
        return None
    work = daily
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    chunks: list[pl.DataFrame] = []
    for pair in work["pair"].unique().to_list():
        sub = work.filter(pl.col("pair") == pair).sort("day_utc")
        returns = sub["return_pct"].to_numpy().astype(np.float64, copy=False)
        buckets, valid = _assign_streak_buckets(returns)
        chunks.append(
            sub.with_columns(
                pl.Series("bucket", buckets),
                pl.Series("scenario_valid", valid),
            )
        )
    merged = pl.concat(chunks, how="vertical")
    buckets = merged["bucket"].to_numpy().astype(np.int8, copy=False)
    valid = merged["scenario_valid"].to_numpy()
    if not valid.any():
        return None

    pair_arr = merged["pair"].to_numpy().astype(object, copy=False)
    hit_matrix = build_return_hit_matrix(
        merged["return_pct"].to_numpy().astype(np.float64, copy=False),
        merged["day_open"].to_numpy().astype(np.float64, copy=False),
        merged["day_high"].to_numpy().astype(np.float64, copy=False),
        merged["day_low"].to_numpy().astype(np.float64, copy=False),
        pair_arr,
        pair_bands,
    )
    return merged, buckets, valid, hit_masks_from_matrix(hit_matrix)
