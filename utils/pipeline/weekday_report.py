"""Контекст и загрузка данных для отчёта по дням недели."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import (
    _DEFAULT_WORKERS,
    load_klines_for_period,
)
from crypto_research.utils.pipeline.paths import TRAIN_MAX_PAIR_START, VAL_MAX_PAIR_START


@dataclass(frozen=True)
class WeekdayReportContext:
    data_dir: Path
    from_date: datetime
    to_date: datetime
    split: str | None
    max_pair_start: datetime | None
    pairs: list[str] | None
    workers: int
    summary: bool = False


def build_weekday_report_context(args) -> WeekdayReportContext:
    split, max_pair_start = _split_and_max_pair_start(args)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    return WeekdayReportContext(
        data_dir=args.data_dir.expanduser().resolve(),
        from_date=parse_iso_utc(args.from_date),
        to_date=parse_iso_utc(args.to_date),
        split=split,
        max_pair_start=max_pair_start,
        pairs=args.pairs,
        workers=workers,
        summary=bool(args.summary),
    )


def _split_and_max_pair_start(args) -> tuple[str | None, datetime | None]:
    if args.summary:
        return None, None
    if args.train:
        return "train", parse_iso_utc(TRAIN_MAX_PAIR_START)
    if args.val:
        return "val", parse_iso_utc(VAL_MAX_PAIR_START)
    if args.max_pair_start:
        return None, parse_iso_utc(args.max_pair_start)
    return None, None


def load_report_klines(ctx: WeekdayReportContext) -> dict[str, pl.DataFrame]:
    return load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        ctx.pairs,
        ctx.max_pair_start,
        split=ctx.split,
        workers=ctx.workers,
    )


def load_train_val_dailies(
    ctx: WeekdayReportContext,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[str]]:
    train_klines = load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        ctx.pairs,
        max_pair_start=None,
        split="train",
        workers=ctx.workers,
    )
    val_klines = load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        ctx.pairs,
        max_pair_start=None,
        split="val",
        workers=ctx.workers,
    )
    return (
        build_pooled_daily(train_klines),
        build_pooled_daily(val_klines),
        sorted(train_klines.keys()),
        sorted(val_klines.keys()),
    )
