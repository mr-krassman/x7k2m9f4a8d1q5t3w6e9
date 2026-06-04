import polars as pl

from crypto_research.stats.day_return_stats import build_daily_returns


def build_pooled_daily(klines_by_pair: dict[str, pl.DataFrame]) -> pl.DataFrame:
    frames = [
        build_daily_returns(bars).with_columns(pl.lit(pair).alias("pair"))
        for pair, bars in klines_by_pair.items()
        if bars.height > 0
    ]
    if not frames:
        raise RuntimeError("Нет минутных баров для расчёта дневных доходностей")
    return pl.concat(frames, how="vertical")


def build_weekday_daily(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.select(
        "return_pct",
        "day_utc",
        "day_open",
        "day_high",
        "day_low",
        "pair",
    ).with_columns(pl.col("day_utc").dt.weekday().alias("weekday"))
