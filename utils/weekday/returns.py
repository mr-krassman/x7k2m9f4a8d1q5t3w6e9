"""UTC-дневная доходность open → close."""

from __future__ import annotations

import polars as pl


def build_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    cast_cols = [
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.from_epoch("start_ms", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.truncate("1d")
        .alias("day_utc"),
    ]
    if "volume" in df.columns:
        cast_cols.append(pl.col("volume").cast(pl.Float64))

    minute = df.sort("start_ms").with_columns(cast_cols)

    agg_exprs: list[pl.Expr] = [
        pl.col("open").first().alias("day_open"),
        pl.col("high").max().alias("day_high"),
        pl.col("low").min().alias("day_low"),
        pl.col("close").last().alias("day_close"),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum().alias("day_volume"))

    daily = minute.group_by("day_utc").agg(agg_exprs)
    return daily.with_columns(
        ((pl.col("day_close") - pl.col("day_open")) / pl.col("day_open") * 100.0).alias(
            "return_pct"
        )
    )
