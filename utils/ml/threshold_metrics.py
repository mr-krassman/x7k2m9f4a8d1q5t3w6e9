"""Signal metrics at policy thresholds t_long / t_short."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.ml.model_compare import signal_rates


def threshold_signal_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    t_long: float,
    t_short: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=float)
    long_m = y_prob >= t_long
    short_m = y_prob <= t_short
    flat_m = ~(long_m | short_m)
    active = long_m | short_m
    n_long = int(long_m.sum())
    n_short = int(short_m.sum())
    n_flat = int(flat_m.sum())
    rates = signal_rates(y_prob, t_long, t_short)
    long_hit_rate = float(y_true[long_m].mean()) if n_long else float("nan")
    short_hit_rate = float((1 - y_true[short_m]).mean()) if n_short else float("nan")
    if active.any():
        correct = (long_m & (y_true == 1)) | (short_m & (y_true == 0))
        signal_accuracy = float(correct.sum() / active.sum())
    else:
        signal_accuracy = float("nan")
    return {
        "t_long": float(t_long),
        "t_short": float(t_short),
        "long_rate": float(rates["long"]),
        "flat_rate": float(rates["flat"]),
        "short_rate": float(rates["short"]),
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": n_flat,
        "n_total": int(y_prob.size),
        "long_hit_rate": long_hit_rate,
        "short_hit_rate": short_hit_rate,
        "signal_accuracy": signal_accuracy,
    }


def threshold_signal_metrics_optional(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    t_long: float | None,
    t_short: float | None,
) -> dict[str, float | int] | None:
    if t_long is None or t_short is None:
        return None
    return threshold_signal_metrics(y_true, y_prob, t_long=t_long, t_short=t_short)


def threshold_signal_metrics_from_oos(
    oos: pl.DataFrame,
    *,
    t_long: float | None,
    t_short: float | None,
) -> dict[str, float | int] | None:
    if t_long is None or t_short is None:
        return None
    return threshold_signal_metrics(
        oos["y_true"].to_numpy(),
        oos["y_prob"].to_numpy(),
        t_long=t_long,
        t_short=t_short,
    )


def weekday_threshold_signal_metrics(
    oos: pl.DataFrame,
    *,
    t_long: float | None,
    t_short: float | None,
) -> dict[str, dict[str, float | int]]:
    if t_long is None or t_short is None:
        return {}
    oos_wd = oos.with_columns(((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday"))
    out: dict[str, dict[str, float | int]] = {}
    for wd, name in enumerate(WEEKDAY_NAMES):
        sub = oos_wd.filter(pl.col("weekday") == wd)
        if sub.is_empty():
            continue
        metrics = threshold_signal_metrics(
            sub["y_true"].to_numpy(),
            sub["y_prob"].to_numpy(),
            t_long=t_long,
            t_short=t_short,
        )
        if metrics is not None:
            out[name] = metrics
    return out


def core_threshold_signal_metrics(
    metrics: dict[str, float | int] | None,
) -> dict[str, float | int] | None:
    if metrics is None:
        return None
    keys = (
        "t_long",
        "t_short",
        "long_rate",
        "flat_rate",
        "short_rate",
        "long_hit_rate",
        "short_hit_rate",
        "signal_accuracy",
        "n_long",
        "n_short",
        "n_flat",
    )
    return {k: metrics[k] for k in keys if k in metrics}
