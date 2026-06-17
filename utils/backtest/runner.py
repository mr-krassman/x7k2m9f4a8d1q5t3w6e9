"""Запуск бэктестов по имени стратегии."""

from __future__ import annotations

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.strategies.registry import get_strategy_handler
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
from crypto_research.utils.pipeline.load_summary import log_load_summary


def _load_full_pool_daily(ctx: BacktestContext):
    from crypto_research.utils.pipeline.dates import parse_iso_utc
    from crypto_research.utils.pipeline.paths import FULL_POOL_MAX_PAIR_START

    max_start = parse_iso_utc(FULL_POOL_MAX_PAIR_START)
    klines = load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
        ctx.to_date,
        None,
        max_start,
        split=None,
        workers=ctx.workers,
    )
    return build_pooled_daily(klines), sorted(klines.keys())


def run_backtest(ctx: BacktestContext) -> BacktestResult:
    handler = get_strategy_handler(
        ctx.strategy,
        algo_spec=ctx.algo_spec,
        ml_spec=ctx.ml_spec,
    )
    prepare = handler.prepare(ctx)

    pair_filter = prepare.pair_filter if prepare.pair_filter is not None else ctx.pairs
    max_pair_start = prepare.max_pair_start or ctx.max_pair_start
    load_from = prepare.load_from_date or ctx.from_date

    klines = load_klines_for_period(
        ctx.data_dir,
        load_from,
        ctx.to_date,
        pair_filter,
        max_pair_start,
        split=None,
        workers=ctx.workers,
    )
    log_load_summary(klines)
    daily = build_pooled_daily(klines)
    pairs = sorted(klines.keys())

    daily_benchmark_49, benchmark_pairs = _load_full_pool_daily(ctx)

    return handler.run(
        ctx,
        daily,
        pairs,
        daily_benchmark_49,
        len(benchmark_pairs),
        prepare,
    )
