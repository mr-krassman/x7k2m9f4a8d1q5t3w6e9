"""Единый контекст оркестратора report_generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from crypto_research.utils.ema_spreads.constants import (
    DEFAULT_EMA_PERIODS,
    DEFAULT_SCREEN_EMA_PERIODS,
)
from crypto_research.utils.pipeline.study_ids import STUDY_EMA_PERIOD_SCREEN
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import _DEFAULT_WORKERS
from crypto_research.utils.pipeline.paths import TRAIN_MAX_PAIR_START, VAL_MAX_PAIR_START


@dataclass(frozen=True)
class ReportContext:
    study: str
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
    select_pairs_by_train: bool = False
    ema_periods: tuple[int, ...] = DEFAULT_EMA_PERIODS


def parse_ema_periods(
    tokens: list[int] | None,
    *,
    default: tuple[int, ...] = DEFAULT_EMA_PERIODS,
) -> tuple[int, ...]:
    if not tokens:
        return default
    periods = tuple(sorted({int(t) for t in tokens}))
    if any(p < 2 for p in periods):
        raise ValueError("Период EMA должен быть ≥ 2")
    return periods


def _parse_highlight_weekdays_arg(tokens: list[str] | None) -> frozenset[int]:
    from argparse import ArgumentTypeError

    from crypto_research.utils.pipeline.weekday_plot_options import parse_highlight_weekdays

    try:
        return parse_highlight_weekdays(tokens)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


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


def build_report_context(args) -> ReportContext:
    from crypto_research.utils.pipeline.studies import STUDY_HANDLERS

    split, max_pair_start = _split_and_max_pair_start(args)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    from_date = parse_iso_utc(args.from_date) if args.from_date else None
    to_date = parse_iso_utc(args.to_date) if args.to_date else None
    handler = STUDY_HANDLERS[args.study]
    from_date, to_date = handler.resolve_dates(from_date, to_date)
    max_pair_start = handler.resolve_max_pair_start(
        max_pair_start,
        summary=bool(args.summary),
    )
    return ReportContext(
        study=args.study,
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
        select_pairs_by_train=bool(args.select_pairs_by_train),
        ema_periods=parse_ema_periods(
            args.ema_periods,
            default=(
                DEFAULT_SCREEN_EMA_PERIODS
                if args.study == STUDY_EMA_PERIOD_SCREEN
                else DEFAULT_EMA_PERIODS
            ),
        ),
    )
