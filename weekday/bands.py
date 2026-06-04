"""Пороги μ±50% по парам и классификация дневного return."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

TRIM_LO_PCT = 5.0
TRIM_HI_PCT = 95.0

COL_CLOSE_UP = "Закрытие в плюсе"
COL_UP_STRONG = "Рост close: сильный"
COL_UP_MODERATE = "Рост close: умеренный"
COL_UP_WEAK = "Рост close: слабый"
COL_CLOSE_DOWN = "Закрытие в минусе"
COL_DOWN_WEAK = "Падение close: слабое"
COL_DOWN_MODERATE = "Падение close: умеренное"
COL_DOWN_STRONG = "Падение close: сильное"
COL_HIGH_REACH_UP = "High ≥ порог роста"
COL_HIGH_ABOVE_UP = "High > сильный рост"
COL_LOW_REACH_DOWN = "Low ≤ порог падения"
COL_LOW_BELOW_DOWN = "Low < сильное падение"

STATS_COLS = [
    COL_CLOSE_UP,
    COL_UP_STRONG,
    COL_UP_MODERATE,
    COL_UP_WEAK,
    COL_CLOSE_DOWN,
    COL_DOWN_WEAK,
    COL_DOWN_MODERATE,
    COL_DOWN_STRONG,
    COL_HIGH_REACH_UP,
    COL_HIGH_ABOVE_UP,
    COL_LOW_REACH_DOWN,
    COL_LOW_BELOW_DOWN,
]


@dataclass(frozen=True)
class MeanBands:
    up_lo: float
    up_hi: float
    down_lo: float
    down_hi: float

    @staticmethod
    def from_group_stats(up: "GroupStats", down: "GroupStats") -> MeanBands:
        return MeanBands(
            up_lo=up.mean_lo_pct,
            up_hi=up.mean_hi_pct,
            down_lo=down.mean_lo_pct,
            down_hi=down.mean_hi_pct,
        )


@dataclass(frozen=True)
class GroupStats:
    label: str
    total_days: int
    kept_days: int
    excluded_days: int
    min_pct: float
    max_pct: float
    mean_pct: float
    median_pct: float
    mean_lo_pct: float
    mean_hi_pct: float
    count_lo: int
    count_mid: int
    count_hi: int


def _trim_by_percentile(series: pl.Series, lo: float, hi: float) -> tuple[pl.Series, int]:
    total = series.len()
    if total == 0:
        return series, 0
    p_lo = float(series.quantile(lo / 100.0))
    p_hi = float(series.quantile(hi / 100.0))
    trimmed = series.filter((series >= p_lo) & (series <= p_hi))
    return trimmed, total - trimmed.len()


def _mean_band_counts(trimmed: pl.Series, mean_pct: float) -> tuple[float, float, int, int, int]:
    lo = mean_pct * 0.5
    hi = mean_pct * 1.5
    if mean_pct > 0:
        count_lo = int((trimmed <= lo).sum())
        count_mid = int(((trimmed > lo) & (trimmed <= hi)).sum())
        count_hi = int((trimmed > hi).sum())
    else:
        count_lo = int((trimmed >= lo).sum())
        count_mid = int(((trimmed < lo) & (trimmed >= hi)).sum())
        count_hi = int((trimmed < hi).sum())
    return lo, hi, count_lo, count_mid, count_hi


def _stats_from_series(label: str, total_days: int, values: pl.Series) -> GroupStats:
    trimmed, excluded = _trim_by_percentile(values, TRIM_LO_PCT, TRIM_HI_PCT)
    kept = trimmed.len()
    if kept == 0:
        nan = float("nan")
        return GroupStats(
            label=label,
            total_days=total_days,
            kept_days=0,
            excluded_days=excluded,
            min_pct=nan,
            max_pct=nan,
            mean_pct=nan,
            median_pct=nan,
            mean_lo_pct=nan,
            mean_hi_pct=nan,
            count_lo=0,
            count_mid=0,
            count_hi=0,
        )
    mean_pct = float(trimmed.mean())
    lo, hi, count_lo, count_mid, count_hi = _mean_band_counts(trimmed, mean_pct)
    return GroupStats(
        label=label,
        total_days=total_days,
        kept_days=kept,
        excluded_days=excluded,
        min_pct=float(trimmed.min()),
        max_pct=float(trimmed.max()),
        mean_pct=mean_pct,
        median_pct=float(trimmed.median()),
        mean_lo_pct=lo,
        mean_hi_pct=hi,
        count_lo=count_lo,
        count_mid=count_mid,
        count_hi=count_hi,
    )


def compute_up_down_stats(daily: pl.DataFrame) -> tuple[GroupStats, GroupStats]:
    up = daily.filter(pl.col("return_pct") > 0)["return_pct"]
    down = daily.filter(pl.col("return_pct") < 0)["return_pct"]
    return (
        _stats_from_series("ДНИ РОСТА", up.len(), up),
        _stats_from_series("ДНИ ПАДЕНИЯ", down.len(), down),
    )


def build_pair_bands_map(daily: pl.DataFrame) -> dict[str, MeanBands]:
    if "pair" not in daily.columns:
        up, down = compute_up_down_stats(daily)
        return {"_single": MeanBands.from_group_stats(up, down)}
    bands: dict[str, MeanBands] = {}
    for pair in daily["pair"].unique().to_list():
        sub = daily.filter(pl.col("pair") == pair)
        up, down = compute_up_down_stats(sub)
        bands[str(pair)] = MeanBands.from_group_stats(up, down)
    return bands


def classify_return_pct(
    ret: float,
    bands: MeanBands,
    day_open: float | None = None,
    day_high: float | None = None,
    day_low: float | None = None,
) -> set[str]:
    tags: set[str] = set()
    if ret > 0:
        tags.add(COL_CLOSE_UP)
        if ret <= bands.up_lo:
            tags.add(COL_UP_WEAK)
        elif ret <= bands.up_hi:
            tags.add(COL_UP_MODERATE)
        else:
            tags.add(COL_UP_STRONG)
    elif ret < 0:
        tags.add(COL_CLOSE_DOWN)
        if ret >= bands.down_lo:
            tags.add(COL_DOWN_WEAK)
        elif ret >= bands.down_hi:
            tags.add(COL_DOWN_MODERATE)
        else:
            tags.add(COL_DOWN_STRONG)

    if (
        day_open is not None
        and day_high is not None
        and day_low is not None
        and day_open > 0
    ):
        up_move = (day_high - day_open) / day_open * 100.0
        down_move = (day_low - day_open) / day_open * 100.0
        if up_move >= bands.up_lo:
            tags.add(COL_HIGH_REACH_UP)
        if up_move > bands.up_hi:
            tags.add(COL_HIGH_ABOVE_UP)
        if down_move <= bands.down_lo:
            tags.add(COL_LOW_REACH_DOWN)
        if down_move < bands.down_hi:
            tags.add(COL_LOW_BELOW_DOWN)
    return tags
