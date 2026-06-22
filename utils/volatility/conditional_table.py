"""range_dev вчера → return сегодня (per-pair бакеты b0–b6, как EMA)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.ema import (
    assign_ema_dev_buckets_vectorized,
    build_pair_thresholds_frame,
)
from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.volatility.range import build_range_work_frame, range_dev_prev_column
from crypto_research.utils.weekday.bands import MeanBands


def prepare_volatility_condition_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...] | None = None,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    use_periods = periods if periods is not None else (period,)
    work = build_range_work_frame(daily, period, use_periods)
    prev_col = range_dev_prev_column(period)
    if work.is_empty() or prev_col not in work.columns:
        return None
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    thresholds = build_pair_thresholds_frame(work, prev_col)
    if thresholds.is_empty():
        return None

    merged = work.join(thresholds, on="pair", how="inner")
    dev = merged[prev_col].to_numpy().astype(np.float64, copy=False)
    buckets = assign_ema_dev_buckets_vectorized(
        dev,
        merged["t1_up"].to_numpy(),
        merged["t2_up"].to_numpy(),
        merged["t1_down"].to_numpy(),
        merged["t2_down"].to_numpy(),
        merged["near_abs"].to_numpy(),
    )
    valid = buckets >= 0
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


def volatility_bucket_labels() -> list[str]:
    from crypto_research.utils.volatility.constants import VOLATILITY_SCENARIO_ROWS

    return list(VOLATILITY_SCENARIO_ROWS)
