"""Адаптер стратегии day_of_week для backtester."""

from __future__ import annotations

import argparse

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_OPTIMISTIC,
    resolve_optimistic_pairs_by_weekday,
    union_pairs,
)
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.day_of_week import (
    DayOfWeekBacktestContext,
    STRATEGY_NAME,
    run_day_of_week_backtest,
)
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import TEMPORAL_POOL_MAX_PAIR_START


class DayOfWeekStrategyHandler(StrategyHandler):
    name = STRATEGY_NAME

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        return

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        if ctx.scenario != SCENARIO_OPTIMISTIC:
            return StrategyPrepareResult()
        pairs_by_weekday = resolve_optimistic_pairs_by_weekday(
            ctx.data_dir,
            workers=ctx.workers,
        )
        return StrategyPrepareResult(
            pair_filter=union_pairs(pairs_by_weekday),
            max_pair_start=parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START),
            extras={"pairs_by_weekday": pairs_by_weekday},
        )

    def run(
        self,
        ctx: BacktestContext,
        daily,
        pairs: list[str],
        daily_benchmark_49,
        n_benchmark_pairs: int,
        prepare: StrategyPrepareResult,
    ) -> BacktestResult:
        pairs_by_weekday = prepare.extras.get("pairs_by_weekday")
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
            n_benchmark_pairs=n_benchmark_pairs,
        )
        return run_day_of_week_backtest(daily, pairs, dow_ctx)
