"""ln(volume/EMA(volume,N)) вчера → return сегодня (per-pair квантили)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.volume.buckets import (
    assign_volume_buckets_vectorized,
    build_pair_volume_thresholds_frame,
)
from crypto_research.utils.volume.constants import N_VOLUME_BUCKETS, VOLUME_BUCKET_LABELS
from crypto_research.utils.volume.volume import build_volume_work_frame, vol_log_rel_prev_column
from crypto_research.utils.weekday.bands import MeanBands


def prepare_volume_condition_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    work = build_volume_work_frame(daily, period)
    prev_col = vol_log_rel_prev_column(period)
    if work.height == 0 or prev_col not in work.columns:
        return None

    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    thresholds = build_pair_volume_thresholds_frame(work, prev_col)
    if thresholds.height == 0:
        return None

    work = work.join(thresholds, on="pair", how="left")
    log_prev = work[prev_col].to_numpy().astype(np.float64, copy=False)
    buckets = assign_volume_buckets_vectorized(
        log_prev,
        work["q30"].to_numpy().astype(np.float64, copy=False),
        work["q70"].to_numpy().astype(np.float64, copy=False),
        work["q85"].to_numpy().astype(np.float64, copy=False),
        work["q95"].to_numpy().astype(np.float64, copy=False),
    )
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
    return work, buckets, valid, hit_masks_from_matrix(hit_matrix)


def volume_bucket_labels() -> list[str]:
    return list(VOLUME_BUCKET_LABELS)
