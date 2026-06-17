"""Адаптер ML-стратегий (single-study и combined bundle) для backtester."""

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
    run_day_of_week_ml_backtest,
)
from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult
from crypto_research.utils.ml.registry import (
    BUNDLE_ID_TO_STUDIES,
    is_combined_bundle_id,
    is_ml_study_id,
    resolve_ml_study,
)
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import (
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_TO,
    ml_model_bundle_path,
    ml_policy_path,
)


class MlStrategyHandler(StrategyHandler):
    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        spec = getattr(args, "ml_spec", None)
        if spec is None:
            if is_combined_bundle_id(args.strategy):
                spec = resolve_ml_study(BUNDLE_ID_TO_STUDIES[args.strategy])
            elif is_ml_study_id(args.strategy):
                spec = resolve_ml_study((args.strategy,))
            else:
                parser.error(f"Не ML-стратегия: {args.strategy}")
        args.ml_spec = spec
        n_pairs = 49
        train_from = parse_iso_utc(TEMPORAL_TRAIN_FROM)
        train_to = parse_iso_utc(TEMPORAL_TRAIN_TO)
        test_to = parse_iso_utc(TEMPORAL_VAL_TO)
        if args.ml_policy_path is None:
            args.ml_policy_path = ml_policy_path(spec, n_pairs, train_from, test_to)
        if args.ml_model_path is None:
            args.ml_model_path = ml_model_bundle_path(spec, n_pairs, train_from, train_to)
        args.ml_output_study = spec.output_study
        args.bundle_id = spec.bundle_id
        args.bundle_kind = "ml" if spec.bundle_id else None

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        spec = ctx.ml_spec
        if spec is None:
            raise RuntimeError("ml_spec не задан в контексте бэктеста")

        with ctx.ml_policy_path.open("r", encoding="utf-8") as f:  # type: ignore[attr-defined]
            policy_payload = json.load(f)
        with ctx.ml_model_path.open("rb") as f:  # type: ignore[attr-defined]
            model_bundle = pickle.load(f)

        use_global_policy = (
            spec.policy_mode == "global"
            and "selected_pairs_global" in policy_payload
            and "global_thresholds" in policy_payload
        )
        if use_global_policy:
            selected_global = sorted(str(p) for p in policy_payload["selected_pairs_global"])
            policy = DayOfWeekMlPolicy(
                t_long=float(policy_payload["global_thresholds"]["t_long"]),
                t_short=float(policy_payload["global_thresholds"]["t_short"]),
                pairs_by_weekday={wd: selected_global for wd in range(7)},
                allowed_pairs=selected_global,
            )
            pair_filter = selected_global
        else:
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
            pair_filter = selected_union
        return StrategyPrepareResult(
            pair_filter=pair_filter,
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
            strategy_name=ctx.strategy,
            ml_output_study=ctx.ml_output_study,
            bundle_id=ctx.bundle_id,
            bundle_kind=ctx.bundle_kind,
        )
        return run_day_of_week_ml_backtest(daily, pairs, ml_ctx)
