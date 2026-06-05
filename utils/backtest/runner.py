"""Запуск бэктестов по имени стратегии."""

from __future__ import annotations

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_OPTIMISTIC,
    resolve_optimistic_pairs_by_weekday,
    union_pairs,
)
from crypto_research.utils.backtest.strategies.day_of_week import (
    DayOfWeekBacktestContext,
    run_day_of_week_backtest,
)
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
from crypto_research.utils.pipeline.load_summary import log_load_summary
from crypto_research.utils.pipeline.paths import FULL_POOL_MAX_PAIR_START, TEMPORAL_POOL_MAX_PAIR_START


def _load_full_pool_daily(ctx: BacktestContext):
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
    pairs_by_weekday: dict[int, list[str]] | None = None
    pair_filter = ctx.pairs

    if ctx.scenario == SCENARIO_OPTIMISTIC:
        pairs_by_weekday = resolve_optimistic_pairs_by_weekday(
            ctx.data_dir,
            workers=ctx.workers,
        )
        pair_filter = union_pairs(pairs_by_weekday)

    max_pair_start = ctx.max_pair_start
    if ctx.scenario == SCENARIO_OPTIMISTIC:
        max_pair_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)

    klines = load_klines_for_period(
        ctx.data_dir,
        ctx.from_date,
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

    if ctx.strategy == "day_of_week":
        dow_ctx = DayOfWeekBacktestContext(
            data_dir=ctx.data_dir,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            pairs=pairs,
            workers=ctx.workers,
            fee=ctx.fee,
            scenario=ctx.scenario,
            pairs_by_weekday=pairs_by_weekday,
            daily_benchmark_49=daily_benchmark_49,
            n_benchmark_pairs=len(benchmark_pairs),
        )
        return run_day_of_week_backtest(daily, pairs, dow_ctx)

    raise ValueError(f"Неизвестная стратегия: {ctx.strategy}")
