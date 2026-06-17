"""Общие границы и нормировка per-pair для ML-фич."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.weekday.bands import TRIM_HI_PCT, TRIM_LO_PCT, _trim_by_percentile


@dataclass(frozen=True)
class PairBounds:
    min_p: float
    max_p: float


def trim_array(values: np.ndarray) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return valid
    trimmed, _ = _trim_by_percentile(pl.Series(valid), TRIM_LO_PCT, TRIM_HI_PCT)
    return trimmed.to_numpy().astype(np.float64, copy=False)


def compute_linear_bounds(values: np.ndarray) -> PairBounds | None:
    clean = trim_array(values)
    if clean.size < 2:
        return None
    min_p = float(clean.min())
    max_p = float(clean.max())
    if max_p <= min_p:
        return None
    return PairBounds(min_p=min_p, max_p=max_p)


def normalize_linear(values: np.ndarray, bounds: PairBounds) -> np.ndarray:
    """min_p → −1, max_p → +1; линейная экстраполяция за пределами."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(values)
    if not np.any(ok):
        return out
    span = bounds.max_p - bounds.min_p
    if span <= 0.0:
        return out
    scaled = 2.0 * (values[ok].astype(np.float64, copy=False) - bounds.min_p) / span - 1.0
    out[ok] = scaled
    return out


def fit_bounds_per_pair(
    daily: pl.DataFrame,
    *,
    raw_column: str,
) -> dict[str, PairBounds]:
    if "pair" not in daily.columns or raw_column not in daily.columns:
        return {}
    bounds: dict[str, PairBounds] = {}
    for pair in daily["pair"].unique().to_list():
        raw = (
            daily.filter(pl.col("pair") == pair)[raw_column]
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        pair_bounds = compute_linear_bounds(raw)
        if pair_bounds is not None:
            bounds[str(pair)] = pair_bounds
    return bounds


def apply_bounds_per_pair(
    values: np.ndarray,
    pairs: np.ndarray,
    bounds: dict[str, PairBounds],
    *,
    normalize,
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for pair in np.unique(pairs):
        key = str(pair)
        pair_bounds = bounds.get(key)
        if pair_bounds is None:
            continue
        mask = pairs == pair
        out[mask] = normalize(values[mask], pair_bounds)
    return out


def bounds_to_dict(bounds: dict[str, PairBounds]) -> dict[str, dict[str, float]]:
    return {pair: {"min_p": b.min_p, "max_p": b.max_p} for pair, b in bounds.items()}


def bounds_from_dict(raw: dict[str, dict[str, float]]) -> dict[str, PairBounds]:
    return {
        pair: PairBounds(min_p=float(v["min_p"]), max_p=float(v["max_p"]))
        for pair, v in raw.items()
    }
