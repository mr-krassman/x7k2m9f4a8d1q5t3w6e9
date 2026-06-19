"""Адаптер стратегии price_sequences для backtester."""

from __future__ import annotations

import argparse
from datetime import timedelta

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.scenarios import SCENARIO_OPTIMISTIC
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.backtest.strategies.price_sequences import (
    PriceSequencesBacktestContext,
    STRATEGY_NAME,
    WARMUP_CALENDAR_DAYS,
    run_price_sequences_backtest,
)
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import TEMPORAL_POOL_MAX_PAIR_START
from crypto_research.utils.price_sequences.pair_selection import resolve_optimistic_price_sequences_pairs


class PriceSequencesStrategyHandler(StrategyHandler):
    name = STRATEGY_NAME

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        return

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        max_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)
        extras: dict = {}
        pair_filter = None
        if ctx.scenario == SCENARIO_OPTIMISTIC:
            optimistic = resolve_optimistic_price_sequences_pairs(
                ctx.data_dir,
                workers=ctx.workers,
            )
            pair_filter = optimistic.union
            extras["pairs_by_segment"] = optimistic.as_segment_dict()
            extras["selected_pairs"] = pair_filter

        load_from = ctx.from_date - timedelta(days=WARMUP_CALENDAR_DAYS)
        return StrategyPrepareResult(
            pair_filter=pair_filter,
            max_pair_start=max_start if ctx.scenario == SCENARIO_OPTIMISTIC else None,
            load_from_date=load_from,
            extras=extras,
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
        ps_ctx = PriceSequencesBacktestContext(
            data_dir=ctx.data_dir,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            pairs=pairs,
            workers=ctx.workers,
            fee=ctx.fee,
            scenario=ctx.scenario,
            selected_pairs=prepare.extras.get("selected_pairs"),
            pairs_by_segment=prepare.extras.get("pairs_by_segment"),
            daily_benchmark_49=daily_benchmark_49,
            n_benchmark_pairs=n_benchmark_pairs,
        )
        return run_price_sequences_backtest(daily, pairs, ps_ctx)
