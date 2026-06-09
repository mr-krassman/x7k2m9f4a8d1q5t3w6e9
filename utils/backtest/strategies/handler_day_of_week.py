"""Адаптер стратегии day_of_week для backtester."""

from __future__ import annotations

import argparse

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.day_of_week import (
    DayOfWeekBacktestContext,
    STRATEGY_NAME,
    run_day_of_week_backtest,
)


class DayOfWeekStrategyHandler(StrategyHandler):
    name = STRATEGY_NAME

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        return

    def run(
        self,
        ctx: BacktestContext,
        daily,
        pairs: list[str],
        daily_benchmark_49,
        n_benchmark_pairs: int,
        pairs_by_weekday: dict[int, list[str]] | None,
    ) -> BacktestResult:
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
