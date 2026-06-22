"""Отбор пар по train для optimistic-сценария volume_spreads (b2 HIGH × Цена росла)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import RETURN_STATS_COLS
from crypto_research.utils.volume.conditional_table import prepare_volume_condition_frame
from crypto_research.utils.volume.constants import SELECTED_VOLUME_EMA_PERIOD
from crypto_research.utils.weekday.bands import MeanBands
from crypto_research.utils.weekday.repeatability import (
    GREEN_DELTA_EPS,
    MIN_YEAR_BASE_DAYS,
    MIN_YEAR_ROW_DAYS,
    compute_cell_year_repeatability,
    years_from_frame,
)

VOLUME_BUCKET_B2 = 2
VOLUME_SIGNAL_COLUMN = RETURN_STATS_COLS[0]


@dataclass(frozen=True)
class VolumeTrainSignal:
    bucket: int
    column: str
    direction: str
    label: str


VOLUME_TRAIN_SIGNAL = VolumeTrainSignal(
    bucket=VOLUME_BUCKET_B2,
    column=VOLUME_SIGNAL_COLUMN,
    direction="long",
    label="b2",
)

VOLUME_TRAIN_SIGNALS: tuple[VolumeTrainSignal, ...] = (VOLUME_TRAIN_SIGNAL,)


@dataclass(frozen=True)
class VolumePairSelection:
    signal: VolumeTrainSignal
    sign_confirmed: list[str]
    year_confirmed: list[str]
    eligible_sign: int
    year_ratio: tuple[int, int]


def _delta_sign_match(pooled_delta: float, delta: float) -> bool:
    if pooled_delta >= GREEN_DELTA_EPS:
        return delta >= GREEN_DELTA_EPS
    if pooled_delta <= -GREEN_DELTA_EPS:
        return delta <= -GREEN_DELTA_EPS
    return abs(delta) < 0.01


def _pair_delta(
    pair: str,
    bucket: int,
    valid: np.ndarray,
    buckets: np.ndarray,
    pair_keys: np.ndarray,
    hit_mask: np.ndarray,
) -> float | None:
    base_mask = valid & (pair_keys == pair)
    row_mask = base_mask & (buckets == bucket)
    if int(base_mask.sum()) < MIN_YEAR_BASE_DAYS or int(row_mask.sum()) < MIN_YEAR_ROW_DAYS:
        return None
    return float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())


def _pairs_confirming_sign(
    bucket: int,
    valid: np.ndarray,
    buckets: np.ndarray,
    pair_keys: np.ndarray,
    hit_mask: np.ndarray,
) -> list[str]:
    base_mask = valid
    row_mask = valid & (buckets == bucket)
    if not np.any(base_mask) or not np.any(row_mask):
        return []
    pooled_delta = float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())
    confirmed: list[str] = []
    for pair in np.unique(pair_keys[base_mask]):
        delta_p = _pair_delta(pair, bucket, valid, buckets, pair_keys, hit_mask)
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
    bucket: int,
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
            bucket,
            valid,
            hit_mask,
            scope_mask=scope,
        )
        if _year_ratio_meets_threshold(ratio, min_year_num, min_year_den):
            confirmed.append(pair)
    return confirmed


def select_volume_pairs_for_signal(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    signal: VolumeTrainSignal,
    period: int = SELECTED_VOLUME_EMA_PERIOD,
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> VolumePairSelection:
    prepared = prepare_volume_condition_frame(daily, period, pair_bands)
    if prepared is None:
        return VolumePairSelection(
            signal=signal,
            sign_confirmed=[],
            year_confirmed=[],
            eligible_sign=0,
            year_ratio=(min_year_num, min_year_den),
        )
    work, buckets, valid, hit_masks = prepared
    hit_mask = hit_masks[signal.column]
    pair_keys = work["pair"].to_numpy().astype(object, copy=False)
    years = years_from_frame(work)
    sign_confirmed = _pairs_confirming_sign(
        signal.bucket, valid, buckets, pair_keys, hit_mask
    )
    year_confirmed = _pairs_confirming_years(
        signal.bucket,
        valid,
        buckets,
        pair_keys,
        years,
        hit_mask,
        sign_confirmed,
        min_year_num=min_year_num,
        min_year_den=min_year_den,
    )
    return VolumePairSelection(
        signal=signal,
        sign_confirmed=sign_confirmed,
        year_confirmed=year_confirmed,
        eligible_sign=len(sign_confirmed),
        year_ratio=(min_year_num, min_year_den),
    )


def select_volume_train_pairs(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    period: int = SELECTED_VOLUME_EMA_PERIOD,
    *,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> list[VolumePairSelection]:
    return [
        select_volume_pairs_for_signal(
            daily,
            pair_bands,
            signal,
            period,
            min_year_num=min_year_num,
            min_year_den=min_year_den,
        )
        for signal in VOLUME_TRAIN_SIGNALS
    ]


def resolve_optimistic_volume_pairs(
    data_dir,
    *,
    period: int = SELECTED_VOLUME_EMA_PERIOD,
    workers: int,
    min_year_num: int = 2,
    min_year_den: int = 3,
) -> list[str]:
    from crypto_research.utils.backtest.scenarios import OPTIMISTIC_TRAIN_FROM, OPTIMISTIC_TRAIN_TO
    from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
    from crypto_research.utils.pipeline.dates import parse_iso_utc
    from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
    from crypto_research.utils.pipeline.pair_means import compute_pair_means
    from crypto_research.utils.pipeline.paths import TEMPORAL_POOL_MAX_PAIR_START

    train_from = parse_iso_utc(OPTIMISTIC_TRAIN_FROM)
    train_to = parse_iso_utc(OPTIMISTIC_TRAIN_TO)
    max_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)
    klines = load_klines_for_period(
        data_dir,
        train_from,
        train_to,
        None,
        max_start,
        split=None,
        workers=workers,
    )
    daily = build_pooled_daily(klines)
    pair_bands = compute_pair_means(daily)
    selections = select_volume_train_pairs(
        daily,
        pair_bands,
        period,
        min_year_num=min_year_num,
        min_year_den=min_year_den,
    )
    return selections[0].year_confirmed if selections else []
