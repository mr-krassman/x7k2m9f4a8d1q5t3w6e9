"""EMA dev вчера + Harami с подтверждением (t−2,t−1,t) → сигнал на t+1."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.candlestick_patterns.harami.detection import confirmed_harami_on_signal_day
from crypto_research.utils.ema_harami.constants import EMA_HARAMI_HARAMI_BY_BUCKET
from crypto_research.utils.ema_spreads.ema import (
    assign_ema_dev_buckets_vectorized,
    build_ema_work_frame,
    build_pair_thresholds_frame,
    ema_dev_prev_column,
)
from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.weekday.bands import MeanBands


def _harami_on_signal_day(opens: np.ndarray, closes: np.ndarray, j: int) -> int | None:
    return confirmed_harami_on_signal_day(opens, closes, j)


def prepare_ema_harami_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    *,
    periods: tuple[int, ...] | None = None,
    **_kwargs,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    ema_periods = periods or (period,)
    work = build_ema_work_frame(daily, ema_periods)
    prev_col = ema_dev_prev_column(period)
    if work.is_empty() or prev_col not in work.columns:
        return None
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    thresholds = build_pair_thresholds_frame(work, prev_col)
    if thresholds.is_empty():
        return None

    merged = work.join(thresholds, on="pair", how="inner")
    dev = merged[prev_col].to_numpy().astype(np.float64, copy=False)
    ema_buckets = assign_ema_dev_buckets_vectorized(
        dev,
        merged["t1_up"].to_numpy(),
        merged["t2_up"].to_numpy(),
        merged["t1_down"].to_numpy(),
        merged["t2_down"].to_numpy(),
        merged["near_abs"].to_numpy(),
    )
    valid = ema_buckets >= 0
    if not valid.any():
        return None

    n = merged.height
    combined = np.full(n, -1, dtype=np.int8)
    opens = merged["day_open"].to_numpy().astype(np.float64, copy=False)
    closes = merged["day_close"].to_numpy().astype(np.float64, copy=False)

    for j in range(n):
        ema_b = int(ema_buckets[j])
        required_harami = EMA_HARAMI_HARAMI_BY_BUCKET.get(ema_b)
        if required_harami is None:
            continue
        harami = _harami_on_signal_day(opens, closes, j)
        if harami == required_harami:
            combined[j] = ema_b

    pair_arr = merged["pair"].to_numpy().astype(object, copy=False)
    hit_matrix = build_return_hit_matrix(
        merged["return_pct"].to_numpy().astype(np.float64, copy=False),
        opens,
        merged["day_high"].to_numpy().astype(np.float64, copy=False),
        merged["day_low"].to_numpy().astype(np.float64, copy=False),
        pair_arr,
        pair_bands,
    )
    return merged, combined, valid, hit_masks_from_matrix(hit_matrix)
