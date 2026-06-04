#!/usr/bin/env python3
"""Оркестратор отчётов crypto_research."""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.cli import parse_report_args
from crypto_research.utils.daily_pool import build_pooled_daily
from crypto_research.utils.dates import parse_iso_utc
from crypto_research.utils.load_pairs import _DEFAULT_WORKERS, load_klines_for_period
from crypto_research.utils.load_summary import log_load_summary
from crypto_research.utils.pair_means import compute_pair_means
from crypto_research.utils.weekday_effects import run_weekday_effects


def run(args) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    from_date = parse_iso_utc(args.from_date)
    to_date = parse_iso_utc(args.to_date)
    max_pair_start = parse_iso_utc(args.max_pair_start) if args.max_pair_start else None

    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    klines = load_klines_for_period(
        data_dir, from_date, to_date, args.pairs, max_pair_start, workers=workers
    )
    log_load_summary(klines)

    daily = build_pooled_daily(klines)
    pair_bands = compute_pair_means(daily)

    run_weekday_effects(
        daily,
        pair_bands,
        pairs=sorted(klines.keys()),
        from_date=from_date,
        to_date=to_date,
        max_pair_start=max_pair_start,
    )


def main() -> int:
    run(parse_report_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
