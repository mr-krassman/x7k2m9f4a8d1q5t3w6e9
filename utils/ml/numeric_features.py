"""Реестр числовых ML-фич и единая подготовка датасета."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.ema import attach_ema_columns, ema_dev_prev_column
from crypto_research.utils.ml.ema_dev_norm import (
    PairEmaDevBounds,
    apply_pair_ema_dev_norm,
    fit_pair_ema_dev_bounds_from_daily,
    pair_ema_dev_bounds_from_dict,
    pair_ema_dev_bounds_to_dict,
)
from crypto_research.utils.ml.pair_bounds import (
    PairBounds,
    apply_bounds_per_pair,
    bounds_from_dict,
    bounds_to_dict,
    fit_bounds_per_pair,
    normalize_linear,
)
from crypto_research.utils.ml.registry import (
    FEATURE_EMA_DEV_PAIR_NORM,
    FEATURE_RSI_PAIR_NORM,
)
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.rsi.constants import SELECTED_RSI_PERIOD
from crypto_research.utils.rsi.rsi import attach_rsi_prev_columns, build_rsi_work_frame, rsi_prev_column

log = get_logger("ml_numeric_features")

NormKind = Literal["signed_dev", "linear"]


@dataclass(frozen=True)
class NumericFeatureSpec:
    column: str
    bundle_key: str
    plot_slug: str
    period: int
    norm: NormKind
    raw_column: Callable[[int], str]
    attach: Callable[[pl.DataFrame, int], pl.DataFrame]
    plot_x_label: str | None = None
    plot_bin_width: float = 0.1


def _attach_ema(frame: pl.DataFrame, period: int) -> pl.DataFrame:
    return attach_ema_columns(frame.sort(["pair", "day_utc"]), (period,))


def _attach_rsi(frame: pl.DataFrame, period: int) -> pl.DataFrame:
    if "pair" in frame.columns:
        parts = []
        for pair in frame["pair"].unique().to_list():
            sub = frame.filter(pl.col("pair") == pair).sort("day_utc")
            parts.append(attach_rsi_prev_columns(sub, period))
        return pl.concat(parts)
    return attach_rsi_prev_columns(frame.sort("day_utc"), period)


NUMERIC_FEATURE_SPECS: dict[str, NumericFeatureSpec] = {
    FEATURE_EMA_DEV_PAIR_NORM: NumericFeatureSpec(
        column=FEATURE_EMA_DEV_PAIR_NORM,
        bundle_key="pair_ema_dev_bounds",
        plot_slug="ema_dev",
        period=SELECTED_EMA_PERIOD,
        norm="signed_dev",
        raw_column=ema_dev_prev_column,
        attach=_attach_ema,
    ),
    FEATURE_RSI_PAIR_NORM: NumericFeatureSpec(
        column=FEATURE_RSI_PAIR_NORM,
        bundle_key="pair_rsi_bounds",
        plot_slug="rsi",
        period=SELECTED_RSI_PERIOD,
        norm="linear",
        raw_column=rsi_prev_column,
        attach=_attach_rsi,
    ),
}


def active_numeric_specs(feature_columns: tuple[str, ...] | list[str]) -> list[NumericFeatureSpec]:
    return [NUMERIC_FEATURE_SPECS[c] for c in feature_columns if c in NUMERIC_FEATURE_SPECS]


def needs_day_close(feature_columns: tuple[str, ...] | list[str]) -> bool:
    return bool(active_numeric_specs(feature_columns))


def fit_bounds_for_features(
    daily: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> dict[str, dict[str, PairBounds | PairEmaDevBounds]]:
    out: dict[str, dict[str, PairBounds | PairEmaDevBounds]] = {}
    for spec in active_numeric_specs(feature_columns):
        if spec.norm == "signed_dev":
            out[spec.column] = fit_pair_ema_dev_bounds_from_daily(daily, ema_period=spec.period)
        else:
            work = build_rsi_work_frame(daily, spec.period) if "pair" in daily.columns else daily
            out[spec.column] = fit_bounds_per_pair(work, raw_column=spec.raw_column(spec.period))
        log.info("[ml] bounds %s: pairs=%s", spec.column, len(out[spec.column]))
    return out


def attach_normalized_features(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    bounds_map: dict[str, dict[str, PairBounds | PairEmaDevBounds]] | None,
) -> pl.DataFrame:
    work = frame
    resolved = bounds_map or {}
    for spec in active_numeric_specs(feature_columns):
        work = spec.attach(work, spec.period)
        raw_col = spec.raw_column(spec.period)
        raw = work[raw_col].to_numpy().astype(np.float64, copy=False)
        pairs_arr = work["pair"].to_numpy().astype(object, copy=False)
        bounds = resolved.get(spec.column)
        if bounds is None:
            if spec.norm == "signed_dev":
                bounds = fit_pair_ema_dev_bounds_from_daily(work, ema_period=spec.period)
            else:
                bounds = fit_bounds_per_pair(work, raw_column=raw_col)
        if spec.norm == "signed_dev":
            norm = apply_pair_ema_dev_norm(raw, pairs_arr, bounds)  # type: ignore[arg-type]
        else:
            norm = apply_bounds_per_pair(raw, pairs_arr, bounds, normalize=normalize_linear)  # type: ignore[arg-type]
        work = work.with_columns(pl.Series(spec.column, norm, dtype=pl.Float64))
    return work


def bounds_map_to_bundle(
    bounds_map: dict[str, dict[str, PairBounds | PairEmaDevBounds]] | None,
) -> dict[str, object]:
    if not bounds_map:
        return {}
    out: dict[str, object] = {}
    for column, bounds in bounds_map.items():
        spec = NUMERIC_FEATURE_SPECS[column]
        if spec.norm == "signed_dev":
            out[spec.bundle_key] = pair_ema_dev_bounds_to_dict(bounds)  # type: ignore[arg-type]
        else:
            out[spec.bundle_key] = bounds_to_dict(bounds)  # type: ignore[arg-type]
        out[f"{column}_period"] = spec.period
    return out


def bounds_map_from_bundle(
    bundle: dict[str, object],
    feature_columns: tuple[str, ...] | list[str],
) -> dict[str, dict[str, PairBounds | PairEmaDevBounds]]:
    out: dict[str, dict[str, PairBounds | PairEmaDevBounds]] = {}
    for spec in active_numeric_specs(feature_columns):
        raw_bounds = bundle.get(spec.bundle_key)
        if not isinstance(raw_bounds, dict):
            continue
        if spec.norm == "signed_dev":
            out[spec.column] = pair_ema_dev_bounds_from_dict(raw_bounds)
        else:
            out[spec.column] = bounds_from_dict(raw_bounds)
    return out


def resolve_predictive(specs: list[NumericFeatureSpec]) -> tuple[str | None, str | None]:
    if not specs:
        return None, None
    last = specs[-1]
    return last.column, last.plot_slug
