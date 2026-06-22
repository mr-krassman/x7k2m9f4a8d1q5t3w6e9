"""Дневной ln(volume/EMA(volume)) и колонка vol_log_rel{N}_prev (вчера → return сегодня)."""

from __future__ import annotations

import polars as pl


def vol_log_rel_prev_column(period: int) -> str:
    return f"vol_log_rel{period}_prev"


def vol_log_rel_column(period: int) -> str:
    return f"vol_log_rel{period}"


def attach_volume_columns(daily: pl.DataFrame, periods: tuple[int, ...]) -> pl.DataFrame:
    if daily.height == 0 or "day_volume" not in daily.columns:
        return daily
    over_pair = "pair" in daily.columns
    out = daily.sort(["pair", "day_utc"]) if over_pair else daily.sort("day_utc")
    for period in periods:
        ema_col = f"_ema_vol{period}"
        rel_col = vol_log_rel_column(period)
        prev_col = vol_log_rel_prev_column(period)
        ema_expr = pl.col("day_volume").ewm_mean(span=period, adjust=False, min_periods=period)
        if over_pair:
            ema_expr = ema_expr.over("pair")
        out = out.with_columns(ema_expr.alias(ema_col))
        ratio = pl.col("day_volume") / pl.col(ema_col)
        out = out.with_columns(pl.when(ratio > 0).then(ratio.log()).otherwise(None).alias(rel_col))
        shift_expr = pl.col(rel_col).shift(1)
        if over_pair:
            shift_expr = shift_expr.over("pair")
        out = out.with_columns(shift_expr.alias(prev_col))
        out = out.drop(ema_col)
    return out


def build_volume_work_frame(
    daily: pl.DataFrame, period: int, periods: tuple[int, ...] | None = None
) -> pl.DataFrame:
    if "day_volume" not in daily.columns:
        return daily
    prev_col = vol_log_rel_prev_column(period)
    if prev_col in daily.columns:
        return daily
    use_periods = periods if periods is not None else (period,)
    return attach_volume_columns(daily, use_periods)
