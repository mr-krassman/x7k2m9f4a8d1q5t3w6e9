"""Дневной RSI Wilder и колонка rsi{N}_prev (вчера → return сегодня)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.rsi.constants import N_RSI_QUANTILES


def rsi_prev_column(period: int) -> str:
    return f"rsi{period}_prev"


def rsi_wilder(close: np.ndarray, period: int) -> np.ndarray:
    """RSI Wilder по дневным close; первые `period` значений — NaN."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out
    delta = np.diff(close.astype(np.float64, copy=False))
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_g = float(gain[:period].mean())
    avg_l = float(loss[:period].mean())
    if avg_l == 0.0:
        out[period] = 100.0 if avg_g > 0.0 else 50.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(period, len(delta)):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        if avg_l == 0.0:
            out[i + 1] = 100.0 if avg_g > 0.0 else 50.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def attach_rsi_prev_columns(daily: pl.DataFrame, period: int) -> pl.DataFrame:
    if daily.height == 0:
        return daily
    col = f"rsi{period}"
    prev_col = rsi_prev_column(period)
    close = daily["day_close"].to_numpy().astype(np.float64, copy=False)
    rsi = rsi_wilder(close, period)
    return daily.with_columns(pl.Series(col, rsi)).with_columns(
        pl.col(col).shift(1).alias(prev_col),
    )


def build_rsi_work_frame(daily: pl.DataFrame, period: int) -> pl.DataFrame:
    if daily.height == 0:
        return daily
    if "pair" in daily.columns:
        parts = []
        for pair in daily["pair"].unique().to_list():
            sub = daily.filter(pl.col("pair") == pair)
            if "day_utc" in sub.columns:
                sub = sub.sort("day_utc")
            parts.append(attach_rsi_prev_columns(sub, period))
        return pl.concat(parts)
    work = daily.sort("day_utc") if "day_utc" in daily.columns else daily
    return attach_rsi_prev_columns(work, period)


def quantile_edges(values: np.ndarray) -> np.ndarray:
    clean = values[np.isfinite(values)]
    if clean.size < N_RSI_QUANTILES:
        return np.array([], dtype=np.float64)
    probs = np.linspace(0.0, 1.0, N_RSI_QUANTILES + 1)[1:-1]
    return np.quantile(clean, probs)


def assign_rsi_buckets(rsi_prev: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(rsi_prev.shape[0], -1, dtype=np.int8)
    valid = np.isfinite(rsi_prev)
    if edges.size == 0 or not valid.any():
        return out
    out[valid] = np.searchsorted(edges, rsi_prev[valid], side="right").astype(np.int8)
    return out


def rsi_bucket_label(bucket: int, edges: np.ndarray) -> str:
    if edges.size == 0:
        return f"Q{bucket + 1}"
    if bucket == 0:
        return f"Q1 RSI ≤{edges[0]:.1f}"
    if bucket == N_RSI_QUANTILES - 1:
        return f"Q{N_RSI_QUANTILES} RSI >{edges[-1]:.1f}"
    return f"Q{bucket + 1} RSI ({edges[bucket - 1]:.1f}..{edges[bucket]:.1f}]"
