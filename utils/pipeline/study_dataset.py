"""Загрузка данных для любого исследования report_generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import datetime_to_ms, parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import (
    load_klines_for_period,
    load_pairs_klines,
    resolve_pairs,
    resolve_split_pairs,
)
from crypto_research.utils.pipeline.load_summary import log_load_summary
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.paths import (
    FULL_POOL_FROM,
    FULL_POOL_MAX_PAIR_START,
    FULL_POOL_TO,
    PAIR_UNIVERSALITY_FROM,
    PAIR_UNIVERSALITY_TO,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
)
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.weekday.bands import MeanBands


@dataclass(frozen=True)
class StudyDataset:
    daily: pl.DataFrame
    pair_bands: dict[str, MeanBands]
    pairs: list[str]


@dataclass(frozen=True)
class SummaryDatasets:
    train_daily: pl.DataFrame
    val_daily: pl.DataFrame
    train_pairs: list[str]
    val_pairs: list[str]
    temporal_train: pl.DataFrame
    temporal_val: pl.DataFrame
    temporal_pairs: list[str]
    full_daily: pl.DataFrame
    full_pairs: list[str]


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _filter_daily(
    daily: pl.DataFrame,
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
) -> pl.DataFrame:
    if daily.is_empty():
        return daily
    return daily.filter(
        pl.col("pair").is_in(pairs),
        pl.col("day_utc") >= _day_start(from_date),
        pl.col("day_utc") <= _day_start(to_date),
    )


def _resolve_summary_pair_sets(
    data_dir: Path,
    pairs: list[str] | None,
    *,
    workers: int,
) -> tuple[list[str], list[str], list[str]]:
    train_pairs = resolve_split_pairs(data_dir, "train", pairs, workers=workers)
    val_pairs = resolve_split_pairs(data_dir, "val", pairs, workers=workers)
    full_pairs = resolve_pairs(
        data_dir,
        pairs,
        parse_iso_utc(FULL_POOL_MAX_PAIR_START),
        workers=workers,
    )
    return train_pairs, val_pairs, full_pairs


def load_primary_dataset(ctx: ReportContext) -> StudyDataset:
    if ctx.from_date is None or ctx.to_date is None:
        raise ValueError("from_date и to_date обязательны для загрузки primary dataset")
    klines = load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        ctx.pairs,
        ctx.max_pair_start,
        split=ctx.split,
        workers=ctx.workers,
    )
    log_load_summary(klines)
    daily = build_pooled_daily(klines)
    pair_bands = compute_pair_means(daily)
    return StudyDataset(
        daily=daily,
        pair_bands=pair_bands,
        pairs=sorted(klines.keys()),
    )


def load_summary_datasets(ctx: ReportContext) -> SummaryDatasets:
    """Одна загрузка klines → daily; срезы train/val/temporal — фильтры в памяти."""
    train_pairs, val_pairs, full_pairs = _resolve_summary_pair_sets(
        ctx.data_dir,
        ctx.pairs,
        workers=ctx.workers,
    )
    all_pairs = sorted(set(train_pairs) | set(val_pairs) | set(full_pairs))

    from_date = parse_iso_utc(PAIR_UNIVERSALITY_FROM)
    to_date = parse_iso_utc(PAIR_UNIVERSALITY_TO)
    klines = load_pairs_klines(
        ctx.data_dir,
        all_pairs,
        from_ms=datetime_to_ms(from_date),
        to_ms=datetime_to_ms(to_date),
        workers=ctx.workers,
    )
    log_load_summary(klines)
    daily_all = build_pooled_daily(klines)

    universality_from = parse_iso_utc(PAIR_UNIVERSALITY_FROM)
    universality_to = parse_iso_utc(PAIR_UNIVERSALITY_TO)
    temporal_train_from = parse_iso_utc(TEMPORAL_TRAIN_FROM)
    temporal_train_to = parse_iso_utc(TEMPORAL_TRAIN_TO)
    temporal_val_from = parse_iso_utc(TEMPORAL_VAL_FROM)
    temporal_val_to = parse_iso_utc(TEMPORAL_VAL_TO)
    full_from = parse_iso_utc(FULL_POOL_FROM)
    full_to = parse_iso_utc(FULL_POOL_TO)

    return SummaryDatasets(
        train_daily=_filter_daily(
            daily_all, train_pairs, universality_from, universality_to
        ),
        val_daily=_filter_daily(
            daily_all, val_pairs, universality_from, universality_to
        ),
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        temporal_train=_filter_daily(
            daily_all, full_pairs, temporal_train_from, temporal_train_to
        ),
        temporal_val=_filter_daily(
            daily_all, full_pairs, temporal_val_from, temporal_val_to
        ),
        temporal_pairs=full_pairs,
        full_daily=_filter_daily(daily_all, full_pairs, full_from, full_to),
        full_pairs=full_pairs,
    )
