"""Дневной range_pct и ratio к SMA(range) — вчера → return сегодня."""

from __future__ import annotations

import polars as pl


def range_dev_prev_column(period: int) -> str:
    return f"range_dev{period}_prev"


def range_ratio_column(period: int) -> str:
    return f"range_ratio{period}"


def attach_range_columns(daily: pl.DataFrame, periods: tuple[int, ...]) -> pl.DataFrame:
    if daily.is_empty() or "day_high" not in daily.columns:
        return daily
    over_pair = "pair" in daily.columns
    out = daily.sort(["pair", "day_utc"]) if over_pair else daily.sort("day_utc")
    range_pct = (pl.col("day_high") - pl.col("day_low")) / pl.col("day_close") * 100.0
    out = out.with_columns(range_pct.alias("_range_pct"))
    for period in periods:
        ratio_col = range_ratio_column(period)
        dev_col = range_dev_prev_column(period)
        sma_expr = pl.col("_range_pct").rolling_mean(window_size=period, min_periods=period)
        if over_pair:
            sma_expr = sma_expr.over("pair")
        out = out.with_columns(sma_expr.alias("_range_sma"))
        out = out.with_columns(
            pl.when(pl.col("_range_sma") > 0)
            .then(pl.col("_range_pct") / pl.col("_range_sma"))
            .otherwise(None)
            .alias(ratio_col)
        )
        dev_expr = pl.col(ratio_col) - 1.0
        shift_expr = dev_expr.shift(1)
        if over_pair:
            shift_expr = shift_expr.over("pair")
        out = out.with_columns(shift_expr.alias(dev_col))
        out = out.drop("_range_sma")
    return out.drop("_range_pct")


def build_range_work_frame(
    daily: pl.DataFrame,
    period: int,
    periods: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    if "day_high" not in daily.columns:
        return daily
    prev_col = range_dev_prev_column(period)
    if prev_col in daily.columns:
        return daily
    use_periods = periods if periods is not None else (period,)
    return attach_range_columns(daily, use_periods)
