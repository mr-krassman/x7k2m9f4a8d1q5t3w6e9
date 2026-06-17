"""RSI(N) вчера → return сегодня (векторно)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.rsi.constants import N_RSI_QUANTILES
from crypto_research.utils.rsi.rsi import (
    assign_rsi_buckets,
    build_rsi_work_frame,
    quantile_edges,
    rsi_prev_column,
)
from crypto_research.utils.weekday.bands import MeanBands


def prepare_rsi_condition_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
    work = build_rsi_work_frame(daily, period)
    prev_col = rsi_prev_column(period)
    if work.height == 0 or prev_col not in work.columns:
        return None

    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    rsi_prev = work[prev_col].to_numpy().astype(np.float64, copy=False)
    edges = quantile_edges(rsi_prev)
    buckets = assign_rsi_buckets(rsi_prev, edges)
    valid = buckets >= 0
    if not valid.any():
        return None

    pair_arr = work["pair"].to_numpy().astype(object, copy=False)
    hit_matrix = build_return_hit_matrix(
        work["return_pct"].to_numpy().astype(np.float64, copy=False),
        work["day_open"].to_numpy().astype(np.float64, copy=False),
        work["day_high"].to_numpy().astype(np.float64, copy=False),
        work["day_low"].to_numpy().astype(np.float64, copy=False),
        pair_arr,
        pair_bands,
    )
    return work, buckets, valid, hit_masks_from_matrix(hit_matrix), edges


def rsi_bucket_labels(edges: np.ndarray) -> list[str]:
    from crypto_research.utils.rsi.rsi import rsi_bucket_label

    return [rsi_bucket_label(b, edges) for b in range(N_RSI_QUANTILES)]
