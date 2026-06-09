"""Расчёт EMA и отклонения close от EMA (%) — векторно через Polars/NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.weekday.bands import TRIM_HI_PCT, TRIM_LO_PCT, _trim_by_percentile


def ema_period_column(period: int) -> str:
    return f"ema{period}"


def ema_dev_column(period: int) -> str:
    return f"ema{period}_dev_pct"


def ema_dev_prev_column(period: int) -> str:
    return f"ema{period}_dev_prev"


def _ema_exprs(periods: tuple[int, ...], over_pair: bool) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for period in periods:
        ema_col = ema_period_column(period)
        base = pl.col("day_close").ewm_mean(span=period, adjust=False, min_periods=period)
        exprs.append(base.over("pair").alias(ema_col) if over_pair else base.alias(ema_col))
    return exprs


def attach_ema_columns(daily: pl.DataFrame, periods: tuple[int, ...]) -> pl.DataFrame:
    if daily.height == 0 or "day_close" not in daily.columns:
        return daily
    over_pair = "pair" in daily.columns
    out = daily
    out = out.with_columns(_ema_exprs(periods, over_pair))
    for period in periods:
        ema_col = ema_period_column(period)
        dev_col = ema_dev_column(period)
        prev_col = ema_dev_prev_column(period)
        out = out.with_columns(
            ((pl.col("day_close") - pl.col(ema_col)) / pl.col(ema_col) * 100.0).alias(dev_col)
        )
        shift_expr = pl.col(dev_col).shift(1)
        if over_pair:
            shift_expr = shift_expr.over("pair")
        out = out.with_columns(shift_expr.alias(prev_col))
    return out


def build_ema_work_frame(daily: pl.DataFrame, periods: tuple[int, ...]) -> pl.DataFrame:
    if "day_close" not in daily.columns:
        return daily
    if periods and ema_dev_prev_column(periods[0]) in daily.columns:
        return daily
    if "pair" in daily.columns:
        return attach_ema_columns(daily.sort(["pair", "day_utc"]), periods)
    sorted_df = daily.sort("day_utc") if "day_utc" in daily.columns else daily
    return attach_ema_columns(sorted_df, periods)


@dataclass(frozen=True)
class EmaDeviationSummary:
    period: int
    total_days: int
    kept_days: int
    excluded_days: int
    mean_dev_pct: float
    median_dev_pct: float
    mean_abs_dev_pct: float
    std_dev_pct: float
    min_dev_pct: float
    max_dev_pct: float


def _summary_from_dev(dev: np.ndarray, period: int) -> EmaDeviationSummary | None:
    valid = dev[np.isfinite(dev)]
    if valid.size == 0:
        return None
    trimmed, excluded = _trim_by_percentile(pl.Series(valid), TRIM_LO_PCT, TRIM_HI_PCT)
    kept = trimmed.len()
    if kept == 0:
        return None
    arr = trimmed.to_numpy().astype(np.float64, copy=False)
    return EmaDeviationSummary(
        period=period,
        total_days=int(valid.size),
        kept_days=kept,
        excluded_days=excluded,
        mean_dev_pct=float(arr.mean()),
        median_dev_pct=float(np.median(arr)),
        mean_abs_dev_pct=float(np.abs(arr).mean()),
        std_dev_pct=float(arr.std(ddof=0)),
        min_dev_pct=float(arr.min()),
        max_dev_pct=float(arr.max()),
    )


def summarize_ema_deviation_per_pair(
    daily: pl.DataFrame,
    periods: tuple[int, ...],
) -> dict[str, dict[int, EmaDeviationSummary]]:
    work = build_ema_work_frame(daily, periods)
    if work.height == 0:
        return {}
    pair_col = "pair" if "pair" in work.columns else pl.lit("_single").alias("pair")
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    result: dict[str, dict[int, EmaDeviationSummary]] = {}
    for sub in work.partition_by("pair", maintain_order=True):
        pair_id = str(sub["pair"][0])
        for period in periods:
            dev_col = ema_dev_column(period)
            dev = sub[dev_col].to_numpy().astype(np.float64, copy=False)
            summary = _summary_from_dev(dev, period)
            if summary is None:
                continue
            result.setdefault(pair_id, {})[period] = summary
    return result


def average_deviation_summaries(
    per_pair: dict[str, dict[int, EmaDeviationSummary]],
    periods: tuple[int, ...],
) -> dict[int, EmaDeviationSummary]:
    averaged: dict[int, EmaDeviationSummary] = {}
    for period in periods:
        rows = [per_pair[p][period] for p in per_pair if period in per_pair[p]]
        if not rows:
            continue
        averaged[period] = EmaDeviationSummary(
            period=period,
            total_days=int(round(sum(r.total_days for r in rows) / len(rows))),
            kept_days=int(round(sum(r.kept_days for r in rows) / len(rows))),
            excluded_days=int(round(sum(r.excluded_days for r in rows) / len(rows))),
            mean_dev_pct=float(np.mean([r.mean_dev_pct for r in rows])),
            median_dev_pct=float(np.mean([r.median_dev_pct for r in rows])),
            mean_abs_dev_pct=float(np.mean([r.mean_abs_dev_pct for r in rows])),
            std_dev_pct=float(np.mean([r.std_dev_pct for r in rows])),
            min_dev_pct=float(np.mean([r.min_dev_pct for r in rows])),
            max_dev_pct=float(np.mean([r.max_dev_pct for r in rows])),
        )
    return averaged


@dataclass(frozen=True)
class PairEmaDevThresholds:
    t1_up: float
    t2_up: float
    t1_down: float
    t2_down: float
    near_abs: float


def _trim_dev_array(dev: np.ndarray) -> np.ndarray:
    valid = dev[np.isfinite(dev)]
    if valid.size == 0:
        return valid
    trimmed, _ = _trim_by_percentile(pl.Series(valid), TRIM_LO_PCT, TRIM_HI_PCT)
    return trimmed.to_numpy().astype(np.float64, copy=False)


def compute_pair_dev_thresholds(dev: np.ndarray) -> PairEmaDevThresholds | None:
    clean = _trim_dev_array(dev)
    if clean.size < 10:
        return None
    pos = clean[clean > 0]
    neg = clean[clean < 0]
    abs_clean = np.abs(clean)
    near_abs = float(np.quantile(abs_clean, 0.10))
    if near_abs <= 0.0:
        near_abs = 0.05
    if pos.size >= 3:
        t1_up, t2_up = (float(x) for x in np.quantile(pos, [1.0 / 3.0, 2.0 / 3.0]))
    elif pos.size > 0:
        t1_up = t2_up = float(pos.max())
    else:
        t1_up = t2_up = 0.0
    if neg.size >= 3:
        t1_down, t2_down = (float(x) for x in np.quantile(neg, [1.0 / 3.0, 2.0 / 3.0]))
    elif neg.size > 0:
        t1_down = t2_down = float(neg.min())
    else:
        t1_down = t2_down = 0.0
    if t1_down > t2_down:
        t1_down, t2_down = t2_down, t1_down
    return PairEmaDevThresholds(
        t1_up=t1_up,
        t2_up=t2_up,
        t1_down=t1_down,
        t2_down=t2_down,
        near_abs=near_abs,
    )


def build_pair_thresholds_frame(
    work: pl.DataFrame,
    prev_col: str,
) -> pl.DataFrame:
    if "pair" not in work.columns:
        dev = work[prev_col].to_numpy().astype(np.float64, copy=False)
        th = compute_pair_dev_thresholds(dev)
        if th is None:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "pair": ["_single"],
                "t1_up": [th.t1_up],
                "t2_up": [th.t2_up],
                "t1_down": [th.t1_down],
                "t2_down": [th.t2_down],
                "near_abs": [th.near_abs],
            }
        )

    def _threshold_row(df: pl.DataFrame) -> pl.DataFrame:
        pair = str(df["pair"][0])
        dev = df[prev_col].to_numpy().astype(np.float64, copy=False)
        th = compute_pair_dev_thresholds(dev)
        if th is None:
            return pl.DataFrame()
        return pl.DataFrame(
            {
                "pair": [pair],
                "t1_up": [th.t1_up],
                "t2_up": [th.t2_up],
                "t1_down": [th.t1_down],
                "t2_down": [th.t2_down],
                "near_abs": [th.near_abs],
            }
        )

    parts = work.group_by("pair", maintain_order=True).map_groups(_threshold_row)
    if parts.height == 0:
        return pl.DataFrame()
    return parts


def assign_ema_dev_buckets_vectorized(
    dev: np.ndarray,
    t1_up: np.ndarray,
    t2_up: np.ndarray,
    t1_down: np.ndarray,
    t2_down: np.ndarray,
    near_abs: np.ndarray,
) -> np.ndarray:
    buckets = np.full(dev.shape[0], -1, dtype=np.int8)
    ok = np.isfinite(dev)
    near_mask = ok & (np.abs(dev) <= near_abs)
    pos = ok & (dev > 0) & ~near_mask
    neg = ok & (dev < 0) & ~near_mask

    buckets[near_mask] = 3
    buckets[pos] = 2
    buckets[pos & (t1_up > 0) & (dev > t1_up)] = 1
    buckets[pos & (t2_up > 0) & (dev > t2_up)] = 0
    buckets[neg] = 4
    buckets[neg & (t2_down < 0) & (dev < t2_down)] = 5
    buckets[neg & (t1_down < 0) & (dev < t1_down)] = 6
    return buckets
