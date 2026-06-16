"""Адаптер ML-стратегии day_of_week_ml для backtester."""

from __future__ import annotations

import argparse
import json
import pickle

from crypto_research.utils.backtest.context import BacktestContext
from crypto_research.utils.backtest.report import BacktestResult
from crypto_research.utils.backtest.strategies.base import StrategyHandler
from crypto_research.utils.backtest.strategies.day_of_week_ml import (
    DayOfWeekMlBacktestContext,
    DayOfWeekMlPolicy,
    STRATEGY_NAME,
    run_day_of_week_ml_backtest,
)
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import (
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_TO,
    weekday_ml_model_bundle_path,
    weekday_ml_policy_path,
)


class DayOfWeekMlStrategyHandler(StrategyHandler):
    name = STRATEGY_NAME

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if args.ml_policy_path is None:
            policy_default = weekday_ml_policy_path(
                49,
                parse_iso_utc(TEMPORAL_TRAIN_FROM),
                parse_iso_utc(TEMPORAL_VAL_TO),
            )
            args.ml_policy_path = policy_default
        if args.ml_model_path is None:
            model_default = weekday_ml_model_bundle_path(
                49,
                parse_iso_utc(TEMPORAL_TRAIN_FROM),
                parse_iso_utc(TEMPORAL_TRAIN_TO),
            )
            args.ml_model_path = model_default

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        with ctx.ml_policy_path.open("r", encoding="utf-8") as f:  # type: ignore[attr-defined]
            policy_payload = json.load(f)
        with ctx.ml_model_path.open("rb") as f:  # type: ignore[attr-defined]
            model_bundle = pickle.load(f)

        pairs_by_weekday = {
            int(k): sorted(v)
            for k, v in policy_payload["selected_pairs_by_weekday"].items()
        }
        selected_union = sorted({p for vals in pairs_by_weekday.values() for p in vals})
        policy = DayOfWeekMlPolicy(
            t_long=float(policy_payload["thresholds"]["t_long"]),
            t_short=float(policy_payload["thresholds"]["t_short"]),
            pairs_by_weekday=pairs_by_weekday,
        )
        return StrategyPrepareResult(
            pair_filter=selected_union,
            extras={"policy": policy, "model_bundle": model_bundle},
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
        ml_ctx = DayOfWeekMlBacktestContext(
            data_dir=ctx.data_dir,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            pairs=pairs,
            workers=ctx.workers,
            fee=ctx.fee,
            scenario=ctx.scenario,
            policy=prepare.extras.get("policy"),
            model_bundle=prepare.extras.get("model_bundle"),
            daily_benchmark_49=daily_benchmark_49,
            n_benchmark_pairs=n_benchmark_pairs,
        )
        return run_day_of_week_ml_backtest(daily, pairs, ml_ctx)

