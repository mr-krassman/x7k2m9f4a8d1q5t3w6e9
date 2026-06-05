"""Buy & Hold бенчмарки."""

from __future__ import annotations

import polars as pl

BTC_PAIR = "btcusdt"
TRADING_WEEKDAYS = (3, 4, 5)


def _with_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    if int(wd.min()) >= 1:
        return daily.with_columns(((wd - 1) % 7).alias("weekday"))
    return daily.with_columns(wd.alias("weekday"))


def _daily_to_bh_portfolio(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("day_utc")
        .agg(
            pl.col("return_pct").mean().alias("gross_return_pct"),
            pl.col("weekday").first().alias("weekday"),
        )
        .with_columns(
            pl.lit(0.0).alias("net_return_pct"),
            pl.lit(1.0).alias("position"),
        )
        .sort("day_utc")
    )


def build_buy_hold_portfolio(daily: pl.DataFrame) -> pl.DataFrame:
    return _daily_to_bh_portfolio(_with_weekday(daily))


def build_btc_buy_hold_portfolio(daily: pl.DataFrame) -> pl.DataFrame | None:
    sub = daily.filter(pl.col("pair").str.to_lowercase() == BTC_PAIR)
    if sub.is_empty():
        return None
    return _daily_to_bh_portfolio(_with_weekday(sub))


def filter_daily_by_weekday_pairs(
    daily: pl.DataFrame,
    pairs_by_weekday: dict[int, list[str]],
) -> pl.DataFrame:
    df = _with_weekday(daily)
    trading = list(pairs_by_weekday.keys())
    keep = pl.lit(False)
    for wd, pairs in pairs_by_weekday.items():
        keep = keep | ((pl.col("weekday") == wd) & pl.col("pair").is_in(pairs))
    non_trading = ~pl.col("weekday").is_in(trading)
    return df.filter(keep | non_trading)


def build_scenario_buy_hold_portfolio(
    daily: pl.DataFrame,
    pairs_by_weekday: dict[int, list[str]],
) -> pl.DataFrame:
    filtered = filter_daily_by_weekday_pairs(daily, pairs_by_weekday)
    bh = build_buy_hold_portfolio(filtered)
    return bh.with_columns(
        pl.when(pl.col("weekday").is_in(list(TRADING_WEEKDAYS)))
        .then(pl.col("gross_return_pct"))
        .otherwise(0.0)
        .alias("gross_return_pct"),
    )
