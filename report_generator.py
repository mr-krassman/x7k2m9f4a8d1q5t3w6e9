#!/usr/bin/env python3
"""Оркестратор отчётов crypto_research."""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.pipeline.cli import parse_report_args
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.load_summary import log_load_summary
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.weekday_effects import run_weekday_effects
from crypto_research.utils.pipeline.weekday_report import (
    build_weekday_report_context,
    load_full_pool_daily,
    load_pair_universality_dailies,
    load_report_klines,
    load_temporal_stability_dailies,
)
from crypto_research.utils.pipeline.weekday_train_val_summary import run_summary_report


def run(args) -> None:
    ctx = build_weekday_report_context(args)
    if ctx.summary:
        train_daily, val_daily, train_pairs, val_pairs = load_pair_universality_dailies(ctx)
        temporal_train, temporal_val, temporal_pairs = load_temporal_stability_dailies(ctx)
        full_daily, full_pairs = load_full_pool_daily(ctx)
        run_summary_report(
            ctx,
            train_daily,
            val_daily,
            train_pairs,
            val_pairs,
            temporal_train,
            temporal_val,
            temporal_pairs,
            full_daily,
            full_pairs,
        )
        return

    klines = load_report_klines(ctx)
    log_load_summary(klines)
    daily = build_pooled_daily(klines)
    pair_bands = compute_pair_means(daily)
    run_weekday_effects(
        daily,
        pair_bands,
        pairs=sorted(klines.keys()),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        max_pair_start=ctx.max_pair_start,
        split=ctx.split,
        highlight_weekdays=ctx.highlight_weekdays,
        main_plot_only=ctx.main_plot_only,
        select_pairs_by_train=ctx.select_pairs_by_train,
    )


def main() -> int:
    run(parse_report_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
