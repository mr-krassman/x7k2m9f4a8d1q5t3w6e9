"""Нормировка вчерашнего EMA-dev относительно train-диапазона пары (trim 5–95%)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.ema import (
    _trim_dev_array,
    attach_ema_columns,
    ema_dev_prev_column,
)
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_ema_dev_norm")


@dataclass(frozen=True)
class PairEmaDevBounds:
    min_p: float
    max_p: float


def compute_pair_ema_dev_bounds(dev: np.ndarray) -> PairEmaDevBounds | None:
    """min/max dev после обрезки 5–95% внутри пары (как в ema_spreads)."""
    clean = _trim_dev_array(dev)
    if clean.size < 2:
        return None
    min_p = float(clean.min())
    max_p = float(clean.max())
    if max_p <= 0.0:
        return None
    if min_p >= 0.0:
        min_p = -max_p
    return PairEmaDevBounds(min_p=min_p, max_p=max_p)


def normalize_ema_dev(dev: np.ndarray, bounds: PairEmaDevBounds) -> np.ndarray:
    """max_p → +1, min_p → −1, dev=0 → 0; за пределами — линейная экстраполяция."""
    out = np.full(dev.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(dev)
    if not np.any(ok):
        return out
    d = dev[ok].astype(np.float64, copy=False)
    min_p = bounds.min_p
    max_p = bounds.max_p
    min_abs = abs(min_p)
    scores = np.zeros(d.shape, dtype=np.float64)

    zero = d == 0.0
    pos = d > 0.0
    neg = d < 0.0
    scores[zero] = 0.0

    if np.any(pos):
        dp = d[pos]
        within = dp <= max_p
        rp = np.empty(dp.shape, dtype=np.float64)
        rp[within] = dp[within] / max_p
        rp[~within] = 1.0 + (dp[~within] - max_p) / max_p
        scores[pos] = rp

    if np.any(neg):
        dn = d[neg]
        within = dn >= min_p
        rn = np.empty(dn.shape, dtype=np.float64)
        rn[within] = dn[within] / min_abs
        rn[~within] = -1.0 + (dn[~within] - min_p) / min_abs
        scores[neg] = rn

    out[ok] = scores
    return out


def fit_pair_ema_dev_bounds_from_daily(
    daily: pl.DataFrame,
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> dict[str, PairEmaDevBounds]:
    """Границы min/max dev (trim 5–95%) на train для каждой пары."""
    if "day_close" not in daily.columns:
        raise ValueError("daily должен содержать day_close для EMA-dev")
    work = daily.select("pair", "day_utc", "day_close").sort(["pair", "day_utc"])
    work = attach_ema_columns(work, (ema_period,))
    prev_col = ema_dev_prev_column(ema_period)
    bounds: dict[str, PairEmaDevBounds] = {}
    for pair in work["pair"].unique().to_list():
        dev = (
            work.filter(pl.col("pair") == pair)[prev_col]
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        pair_bounds = compute_pair_ema_dev_bounds(dev)
        if pair_bounds is None:
            log.warning("[ml] pair %s: недостаточно dev для границ нормировки", pair)
            continue
        bounds[str(pair)] = pair_bounds
    log.info("[ml] ema_dev_pair_norm bounds: pairs=%s", len(bounds))
    return bounds


def apply_pair_ema_dev_norm(
    dev: np.ndarray,
    pairs: np.ndarray,
    bounds: dict[str, PairEmaDevBounds],
) -> np.ndarray:
    out = np.full(dev.shape, np.nan, dtype=np.float64)
    for pair in np.unique(pairs):
        key = str(pair)
        pair_bounds = bounds.get(key)
        if pair_bounds is None:
            continue
        mask = pairs == pair
        out[mask] = normalize_ema_dev(dev[mask], pair_bounds)
    return out


def pair_ema_dev_bounds_to_dict(bounds: dict[str, PairEmaDevBounds]) -> dict[str, dict[str, float]]:
    return {pair: {"min_p": b.min_p, "max_p": b.max_p} for pair, b in bounds.items()}


def pair_ema_dev_bounds_from_dict(raw: dict[str, dict[str, float]]) -> dict[str, PairEmaDevBounds]:
    return {
        pair: PairEmaDevBounds(min_p=float(v["min_p"]), max_p=float(v["max_p"]))
        for pair, v in raw.items()
    }
