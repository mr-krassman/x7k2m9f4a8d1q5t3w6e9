"""Отбор пар по train: знак Δ к BASE + повторяемость по годам (2/3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.weekday.bands import (
    COL_CLOSE_DOWN,
    COL_CLOSE_UP,
    MeanBands,
    classify_return_pct,
    STATS_COLS,
)
from crypto_research.utils.weekday.repeatability import (
    GREEN_DELTA_EPS,
    MIN_YEAR_BASE_DAYS,
    MIN_YEAR_ROW_DAYS,
    compute_cell_year_repeatability,
    years_from_frame,
)

WEEKDAY_NAMES = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
_PAIRS_PER_LINE = 11


@dataclass(frozen=True)
class WeekdayStrategySignal:
    weekday: int
    direction: str
    column: str
    label_ru: str


DAY_OF_WEEK_TRAIN_SIGNALS: tuple[WeekdayStrategySignal, ...] = (
    WeekdayStrategySignal(3, "short", COL_CLOSE_DOWN, "Чт"),
    WeekdayStrategySignal(4, "long", COL_CLOSE_UP, "Пт"),
    WeekdayStrategySignal(5, "long", COL_CLOSE_UP, "Сб"),
)


@dataclass(frozen=True)
class WeekdayPairSelection:
    signal: WeekdayStrategySignal
    sign_confirmed: list[str]
    year_confirmed: list[str]
    eligible_sign: int
    year_ratio: tuple[int, int]


def _normalize_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    wd_min = int(wd.min())
    wd_max = int(wd.max())
    if wd_min >= 1 and wd_max <= 7:
        wd = ((wd - 1) % 7).cast(pl.Int64)
    return daily.with_columns(wd.alias("weekday"))


def _build_hit_arrays(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    df = _normalize_weekday(weekday_daily)
    if "pair" not in df.columns:
        df = df.with_columns(pl.lit("_single").alias("pair"))

    returns = df["return_pct"].to_numpy().astype(np.float64, copy=False)
    opens = df["day_open"].to_numpy().astype(np.float64, copy=False)
    highs = df["day_high"].to_numpy().astype(np.float64, copy=False)
    lows = df["day_low"].to_numpy().astype(np.float64, copy=False)
    buckets = df["weekday"].to_numpy().astype(np.int8, copy=False)
    valid = np.ones(returns.shape[0], dtype=bool)
    pair_keys = np.array(df["pair"].to_list(), dtype=object)
    years = years_from_frame(df)

    hits: dict[str, np.ndarray] = {
        col: np.zeros(returns.shape[0], dtype=bool) for col in STATS_COLS
    }
    for i, ret in enumerate(returns):
        bands = pair_bands[str(pair_keys[i])]
        cls = classify_return_pct(
            float(ret),
            bands,
            float(opens[i]),
            float(highs[i]),
            float(lows[i]),
        )
        for col in STATS_COLS:
            hits[col][i] = col in cls

    return buckets, valid, pair_keys, years, hits


def _delta_sign_match(pooled_delta: float, delta: float) -> bool:
    if pooled_delta >= GREEN_DELTA_EPS:
        return delta >= GREEN_DELTA_EPS
    if pooled_delta <= -GREEN_DELTA_EPS:
        return delta <= -GREEN_DELTA_EPS
    return abs(delta) < 0.01


def _pair_delta(
    pair: str,
    row_index: int,
    valid: np.ndarray,
    buckets: np.ndarray,
    pair_keys: np.ndarray,
    hit_mask: np.ndarray,
) -> float | None:
    base_mask = valid & (pair_keys == pair)
    row_mask = base_mask & (buckets == row_index)
    if int(base_mask.sum()) < MIN_YEAR_BASE_DAYS or int(row_mask.sum()) < MIN_YEAR_ROW_DAYS:
        return None
    return float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())


def _pairs_confirming_sign(
    row_index: int,
    valid: np.ndarray,
    buckets: np.ndarray,
    pair_keys: np.ndarray,
    hit_mask: np.ndarray,
) -> list[str]:
    base_mask = valid
    row_mask = valid & (buckets == row_index)
    if not np.any(base_mask) or not np.any(row_mask):
        return []

    pooled_delta = float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())
    confirmed: list[str] = []
    for pair in np.unique(pair_keys[base_mask]):
        delta_p = _pair_delta(pair, row_index, valid, buckets, pair_keys, hit_mask)
        if delta_p is None:
            continue
        if _delta_sign_match(pooled_delta, delta_p):
            confirmed.append(str(pair))
    return sorted(confirmed)


def _year_ratio_meets_threshold(ratio_text: str, min_num: int, min_den: int) -> bool:
    if ratio_text == "n/a" or "/" not in ratio_text:
        return False
    match_s, total_s = ratio_text.split("/", 1)
    match = int(match_s)
    total = int(total_s)
    if total == 0:
        return False
    return match * min_den >= total * min_num


def _pairs_confirming_years(
    row_index: int,
    valid: np.ndarray,
    buckets: np.ndarray,
    pair_keys: np.ndarray,
    years: np.ndarray,
    hit_mask: np.ndarray,
    candidates: list[str],
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> list[str]:
    confirmed: list[str] = []
    for pair in candidates:
        scope = valid & (pair_keys == pair)
        if not np.any(scope):
            continue
        ratio = compute_cell_year_repeatability(
            years,
            buckets,
            row_index,
            valid,
            hit_mask,
            scope_mask=scope,
        )
        if _year_ratio_meets_threshold(ratio, min_year_num, min_year_den):
            confirmed.append(pair)
    return confirmed


def select_pairs_for_signal(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    signal: WeekdayStrategySignal,
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> WeekdayPairSelection:
    buckets, valid, pair_keys, years, hits = _build_hit_arrays(weekday_daily, pair_bands)
    hit_mask = hits[signal.column]
    sign_confirmed = _pairs_confirming_sign(
        signal.weekday, valid, buckets, pair_keys, hit_mask
    )
    year_confirmed = _pairs_confirming_years(
        signal.weekday,
        valid,
        buckets,
        pair_keys,
        years,
        hit_mask,
        sign_confirmed,
        min_year_num=min_year_num,
        min_year_den=min_year_den,
    )
    return WeekdayPairSelection(
        signal=signal,
        sign_confirmed=sign_confirmed,
        year_confirmed=year_confirmed,
        eligible_sign=len(sign_confirmed),
        year_ratio=(min_year_num, min_year_den),
    )


def select_day_of_week_train_pairs(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> list[WeekdayPairSelection]:
    return [
        select_pairs_for_signal(
            weekday_daily,
            pair_bands,
            signal,
            min_year_num=min_year_num,
            min_year_den=min_year_den,
        )
        for signal in DAY_OF_WEEK_TRAIN_SIGNALS
    ]


def _format_pairs_lines(pairs: list[str], prefix: str = "  ") -> list[str]:
    if not pairs:
        return [f"{prefix}—"]
    lines: list[str] = []
    for i in range(0, len(pairs), _PAIRS_PER_LINE):
        chunk = pairs[i : i + _PAIRS_PER_LINE]
        lead = f"{prefix}" if i == 0 else f"{prefix}  "
        lines.append(f"{lead}{', '.join(chunk)}")
    return lines


def build_train_pair_selection_section(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> list[str]:
    selections = select_day_of_week_train_pairs(
        weekday_daily,
        pair_bands,
        min_year_num=min_year_num,
        min_year_den=min_year_den,
    )
    lines = [
        "=== Отбор пар по train (оптимистичный сценарий day_of_week) ===",
        "",
        "Short Чт / long Пт, Сб. Пара попадает в список, если:",
        f"  1) знак Δ к BASE по колонке совпал с pooled train ([N] в таблице);",
        f"  2) тот же знак Δ повторился в ≥ {min_year_num}/{min_year_den} годов с достаточной историей.",
        "",
        f"Год учитывается при ≥ {MIN_YEAR_BASE_DAYS} дней BASE и ≥ {MIN_YEAR_ROW_DAYS} дней weekday в году.",
        "",
    ]
    for sel in selections:
        sig = sel.signal
        direction_ru = "short" if sig.direction == "short" else "long"
        lines.append(
            f"{sig.label_ru} — {direction_ru} (колонка «{sig.column}»): "
            f"знак [{sel.eligible_sign}] → {min_year_num}/{min_year_den} лет "
            f"[{len(sel.year_confirmed)}]"
        )
        lines.extend(_format_pairs_lines(sel.year_confirmed))
        lines.append("")
    return lines
