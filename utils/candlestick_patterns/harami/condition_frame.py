"""Подготовка фрейма: Harami на (t−1, t) → сигнал на доходность дня t+1."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.candlestick_patterns.harami.detection import assign_harami_buckets
from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.weekday.bands import MeanBands


def prepare_harami_frame(
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
        opens = sub["day_open"].to_numpy().astype(np.float64, copy=False)
        closes = sub["day_close"].to_numpy().astype(np.float64, copy=False)
        buckets, valid = assign_harami_buckets(opens, closes)
        chunks.append(
            sub.with_columns(
                pl.Series("bucket", buckets),
                pl.Series("pattern_valid", valid),
            )
        )
    merged = pl.concat(chunks, how="vertical")
    buckets = merged["bucket"].to_numpy().astype(np.int8, copy=False)
    valid = merged["pattern_valid"].to_numpy()
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
