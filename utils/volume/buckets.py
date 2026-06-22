"""Per-pair квантили ln(volume/EMA) и 5 бакетов объёма."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.volume.constants import VOLUME_QUANTILE_PROBS


def build_pair_volume_thresholds_frame(work: pl.DataFrame, prev_col: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in work["pair"].unique().to_list():
        vals = (
            work.filter(pl.col("pair") == pair)[prev_col]
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        clean = vals[np.isfinite(vals)]
        if clean.size < 10:
            continue
        q30, q70, q85, q95 = np.quantile(clean, VOLUME_QUANTILE_PROBS)
        rows.append(
            {
                "pair": pair,
                "q30": float(q30),
                "q70": float(q70),
                "q85": float(q85),
                "q95": float(q95),
            }
        )
    if not rows:
        return pl.DataFrame(schema={"pair": pl.Utf8, "q30": pl.Float64, "q70": pl.Float64, "q85": pl.Float64, "q95": pl.Float64})
    return pl.DataFrame(rows)


def assign_volume_buckets_vectorized(
    log_prev: np.ndarray,
    q30: np.ndarray,
    q70: np.ndarray,
    q85: np.ndarray,
    q95: np.ndarray,
) -> np.ndarray:
    buckets = np.full(log_prev.shape[0], -1, dtype=np.int8)
    ok = (
        np.isfinite(log_prev)
        & np.isfinite(q30)
        & np.isfinite(q70)
        & np.isfinite(q85)
        & np.isfinite(q95)
    )
    x = log_prev
    buckets[ok & (x <= q30)] = 0
    buckets[ok & (x > q30) & (x <= q70)] = 1
    buckets[ok & (x > q70) & (x <= q85)] = 2
    buckets[ok & (x > q85) & (x <= q95)] = 3
    buckets[ok & (x > q95)] = 4
    return buckets
