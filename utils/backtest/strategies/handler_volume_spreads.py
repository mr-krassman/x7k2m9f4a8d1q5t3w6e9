"""Адаптер стратегии volume_spreads для backtester."""

from __future__ import annotations

import argparse
from datetime import timedelta

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.scenarios import (
    OPTIMISTIC_TRAIN_FROM,
    OPTIMISTIC_TRAIN_TO,
    SCENARIO_OPTIMISTIC,
)
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.backtest.strategies.volume_spreads import (
    STRATEGY_NAME,
    WARMUP_CALENDAR_DAYS,
    VolumeSpreadsBacktestContext,
    compute_frozen_volume_thresholds,
    run_volume_spreads_backtest,
)
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
from crypto_research.utils.pipeline.paths import TEMPORAL_POOL_MAX_PAIR_START
from crypto_research.utils.volume.constants import SELECTED_VOLUME_EMA_PERIOD
from crypto_research.utils.volume.pair_selection import resolve_optimistic_volume_pairs


class VolumeSpreadsStrategyHandler(StrategyHandler):
    name = STRATEGY_NAME

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        period = getattr(args, "vol_period", SELECTED_VOLUME_EMA_PERIOD)
        if period != SELECTED_VOLUME_EMA_PERIOD:
            parser.error(
                f"Бэктест volume_spreads поддерживает только EMA(volume, {SELECTED_VOLUME_EMA_PERIOD}), "
                f"получено: {period}"
            )

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        train_from = parse_iso_utc(OPTIMISTIC_TRAIN_FROM)
        train_to = parse_iso_utc(OPTIMISTIC_TRAIN_TO)
        max_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)
        period = SELECTED_VOLUME_EMA_PERIOD

        train_klines = load_klines_for_period(
            ctx.data_dir,
            train_from,
            train_to,
            None,
            max_start,
            split=None,
            workers=ctx.workers,
        )
        train_daily = build_pooled_daily(train_klines)
        frozen_thresholds = compute_frozen_volume_thresholds(train_daily, period)

        extras: dict = {
            "frozen_thresholds": frozen_thresholds,
            "vol_period": period,
        }
        pair_filter = None
        if ctx.scenario == SCENARIO_OPTIMISTIC:
            selected = resolve_optimistic_volume_pairs(
                ctx.data_dir,
                period=period,
                workers=ctx.workers,
            )
            pair_filter = selected
            extras["selected_pairs"] = selected

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
        vol_ctx = VolumeSpreadsBacktestContext(
            data_dir=ctx.data_dir,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            pairs=pairs,
            workers=ctx.workers,
            fee=ctx.fee,
            scenario=ctx.scenario,
            vol_period=prepare.extras.get("vol_period", SELECTED_VOLUME_EMA_PERIOD),
            frozen_thresholds=prepare.extras.get("frozen_thresholds"),
            selected_pairs=prepare.extras.get("selected_pairs"),
            daily_benchmark_49=daily_benchmark_49,
            n_benchmark_pairs=n_benchmark_pairs,
        )
        return run_volume_spreads_backtest(daily, pairs, vol_ctx)
