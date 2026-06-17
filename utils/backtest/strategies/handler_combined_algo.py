"""Combined rule-based bundle (day_of_week + ema_spreads, optimistic)."""

from __future__ import annotations

import argparse
from datetime import timedelta

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.scenarios import (
    OPTIMISTIC_TRAIN_FROM,
    OPTIMISTIC_TRAIN_TO,
    SCENARIO_OPTIMISTIC,
    resolve_optimistic_pairs_by_weekday,
    union_pairs,
)
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.combined_algo import (
    CombinedAlgoBacktestContext,
    run_combined_algo_backtest,
)
from crypto_research.utils.backtest.strategies.ema_spreads import (
    WARMUP_CALENDAR_DAYS,
    compute_frozen_thresholds,
)
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.pair_selection import resolve_optimistic_ema_pairs
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
from crypto_research.utils.pipeline.paths import TEMPORAL_POOL_MAX_PAIR_START


class CombinedAlgoStrategyHandler(StrategyHandler):
    name = "combined_algo"

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if getattr(args, "algo_spec", None) is None:
            parser.error("algo_spec не задан для combined rule-based")
        if args.scenario != SCENARIO_OPTIMISTIC:
            parser.error(
                "Combined rule-based поддерживает только --scenario optimistic "
                f"(получено: {args.scenario})"
            )

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        pairs_by_weekday = resolve_optimistic_pairs_by_weekday(
            ctx.data_dir,
            workers=ctx.workers,
        )
        ema_selected = resolve_optimistic_ema_pairs(
            ctx.data_dir,
            period=SELECTED_EMA_PERIOD,
            workers=ctx.workers,
        )
        train_from = parse_iso_utc(OPTIMISTIC_TRAIN_FROM)
        train_to = parse_iso_utc(OPTIMISTIC_TRAIN_TO)
        max_start = parse_iso_utc(TEMPORAL_POOL_MAX_PAIR_START)
        train_klines = load_klines_for_period(
            ctx.data_dir,
            train_from,
            train_to,
            None,
            max_start,
            split=None,
            workers=ctx.workers,
        )
        frozen_thresholds = compute_frozen_thresholds(
            build_pooled_daily(train_klines),
            SELECTED_EMA_PERIOD,
        )
        pair_filter = sorted(set(union_pairs(pairs_by_weekday)) | set(ema_selected))
        load_from = ctx.from_date - timedelta(days=WARMUP_CALENDAR_DAYS)
        return StrategyPrepareResult(
            pair_filter=pair_filter,
            max_pair_start=max_start,
            load_from_date=load_from,
            extras={
                "pairs_by_weekday": pairs_by_weekday,
                "ema_selected_pairs": ema_selected,
                "frozen_thresholds": frozen_thresholds,
                "ema_period": SELECTED_EMA_PERIOD,
            },
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
        algo_ctx = CombinedAlgoBacktestContext(
            data_dir=ctx.data_dir,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            pairs=pairs,
            workers=ctx.workers,
            fee=ctx.fee,
            algo_spec=ctx.algo_spec,
            pairs_by_weekday=prepare.extras.get("pairs_by_weekday"),
            ema_selected_pairs=prepare.extras.get("ema_selected_pairs"),
            frozen_thresholds=prepare.extras.get("frozen_thresholds"),
            ema_period=prepare.extras.get("ema_period", SELECTED_EMA_PERIOD),
            daily_benchmark_49=daily_benchmark_49,
            n_benchmark_pairs=n_benchmark_pairs,
            strategy_name=ctx.strategy,
            bundle_id=ctx.bundle_id,
            bundle_kind=ctx.bundle_kind or "algo",
        )
        return run_combined_algo_backtest(daily, pairs, algo_ctx)
