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
from crypto_research.utils.pipeline.paths import (
    FULL_POOL_FROM,
    FULL_POOL_MAX_PAIR_START,
    FULL_POOL_TO,
    PAIR_UNIVERSALITY_FROM,
    PAIR_UNIVERSALITY_TO,
    TEMPORAL_POOL_MAX_PAIR_START,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
    TRAIN_MAX_PAIR_START,
    VAL_MAX_PAIR_START,
)


@dataclass(frozen=True)
class WeekdayReportContext:
    data_dir: Path
    from_date: datetime | None
    to_date: datetime | None
    split: str | None
    max_pair_start: datetime | None
    pairs: list[str] | None
    workers: int
    summary: bool = False
    highlight_weekdays: frozenset[int] = frozenset()
    main_plot_only: bool = False


def _parse_highlight_weekdays_arg(tokens: list[str] | None) -> frozenset[int]:
    from argparse import ArgumentTypeError
    from crypto_research.utils.pipeline.weekday_plot_options import parse_highlight_weekdays

    try:
        return parse_highlight_weekdays(tokens)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


def build_weekday_report_context(args) -> WeekdayReportContext:
    split, max_pair_start = _split_and_max_pair_start(args)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    from_date = parse_iso_utc(args.from_date) if args.from_date else None
    to_date = parse_iso_utc(args.to_date) if args.to_date else None
    return WeekdayReportContext(
        data_dir=args.data_dir.expanduser().resolve(),
        from_date=from_date,
        to_date=to_date,
        split=split,
        max_pair_start=max_pair_start,
        pairs=args.pairs,
        workers=workers,
        summary=bool(args.summary),
        highlight_weekdays=_parse_highlight_weekdays_arg(args.highlight_weekdays),
        main_plot_only=bool(args.main_plot_only),
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
    if ctx.from_date is None or ctx.to_date is None:
        raise ValueError("from_date и to_date обязательны вне режима --summary")
    return load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        ctx.pairs,
        ctx.max_pair_start,
        split=ctx.split,
        workers=ctx.workers,
    )


def load_pair_universality_dailies(
    ctx: WeekdayReportContext,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], list[str]]:
    from_date = parse_iso_utc(PAIR_UNIVERSALITY_FROM)
    to_date = parse_iso_utc(PAIR_UNIVERSALITY_TO)
    train_klines = load_klines_for_period(
        ctx.data_dir,
        from_date,
        to_date,
        ctx.pairs,
        max_pair_start=None,
        split="train",
        workers=ctx.workers,
    )
    val_klines = load_klines_for_period(
        ctx.data_dir,
        from_date,
        to_date,
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


def load_temporal_stability_dailies(
    ctx: WeekdayReportContext,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    max_pair_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)
    train_from = parse_iso_utc(TEMPORAL_TRAIN_FROM)
    train_to = parse_iso_utc(TEMPORAL_TRAIN_TO)
    val_from = parse_iso_utc(TEMPORAL_VAL_FROM)
    val_to = parse_iso_utc(TEMPORAL_VAL_TO)

    train_klines = load_klines_for_period(
        ctx.data_dir,
        train_from,
        train_to,
        ctx.pairs,
        max_pair_start,
        split=None,
        workers=ctx.workers,
    )
    val_klines = load_klines_for_period(
        ctx.data_dir,
        val_from,
        val_to,
        ctx.pairs,
        max_pair_start,
        split=None,
        workers=ctx.workers,
    )
    pairs = sorted(train_klines.keys())
    return (
        build_pooled_daily(train_klines),
        build_pooled_daily(val_klines),
        pairs,
    )


def load_full_pool_daily(
    ctx: WeekdayReportContext,
) -> tuple[pl.DataFrame, list[str]]:
    from_date = parse_iso_utc(FULL_POOL_FROM)
    to_date = parse_iso_utc(FULL_POOL_TO)
    max_pair_start = parse_iso_utc(FULL_POOL_MAX_PAIR_START)
    klines = load_klines_for_period(
        ctx.data_dir,
        from_date,
        to_date,
        ctx.pairs,
        max_pair_start,
        split=None,
        workers=ctx.workers,
    )
    return build_pooled_daily(klines), sorted(klines.keys())
