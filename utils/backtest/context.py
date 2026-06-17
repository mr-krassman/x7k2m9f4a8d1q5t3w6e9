"""Контекст бэктеста."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from crypto_research.utils.backtest.fees import DEFAULT_FEE, FeeSchedule
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_CONSERVATIVE,
    SCENARIO_MAXIMAL,
    SCENARIO_OPTIMISTIC,
    VAL_FROM,
    VAL_TO,
    normalize_scenario,
)
from crypto_research.utils.backtest.bundle_registry import AlgoBundleSpec, CombineMode
from crypto_research.utils.ml.registry import MlStudySpec
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import _DEFAULT_WORKERS
from crypto_research.utils.pipeline.paths import (
    FULL_POOL_FROM,
    FULL_POOL_MAX_PAIR_START,
    FULL_POOL_TO,
)


@dataclass(frozen=True)
class BacktestContext:
    strategy: str
    scenario: str
    data_dir: Path
    from_date: datetime
    to_date: datetime
    max_pair_start: datetime
    pairs: list[str] | None
    workers: int
    ml_policy_path: Path | None = None
    ml_model_path: Path | None = None
    ml_output_study: str | None = None
    ml_spec: MlStudySpec | None = None
    algo_spec: AlgoBundleSpec | None = None
    bundle_id: str | None = None
    bundle_kind: str | None = None
    combine_mode: CombineMode | None = None
    fee: FeeSchedule = DEFAULT_FEE


def build_backtest_context(args) -> BacktestContext:
    scenario = normalize_scenario(args.scenario)
    if getattr(args, "ml_spec", None) is not None:
        from_date = parse_iso_utc(args.from_date or VAL_FROM)
        to_date = parse_iso_utc(args.to_date or VAL_TO)
    elif getattr(args, "algo_spec", None) is not None:
        from_date = parse_iso_utc(args.from_date or VAL_FROM)
        to_date = parse_iso_utc(args.to_date or VAL_TO)
    elif scenario in (SCENARIO_OPTIMISTIC, SCENARIO_CONSERVATIVE):
        from_date = parse_iso_utc(args.from_date or VAL_FROM)
        to_date = parse_iso_utc(args.to_date or VAL_TO)
    else:
        from_date = parse_iso_utc(args.from_date or FULL_POOL_FROM)
        to_date = parse_iso_utc(args.to_date or FULL_POOL_TO)
    max_pair_start = parse_iso_utc(args.max_pair_start or FULL_POOL_MAX_PAIR_START)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    return BacktestContext(
        strategy=args.strategy,
        scenario=scenario,
        data_dir=args.data_dir.expanduser().resolve(),
        from_date=from_date,
        to_date=to_date,
        max_pair_start=max_pair_start,
        pairs=args.pairs,
        workers=workers,
        ml_policy_path=getattr(args, "ml_policy_path", None),
        ml_model_path=getattr(args, "ml_model_path", None),
        ml_output_study=getattr(args, "ml_output_study", None),
        ml_spec=getattr(args, "ml_spec", None),
        algo_spec=getattr(args, "algo_spec", None),
        bundle_id=getattr(args, "bundle_id", None),
        bundle_kind=getattr(args, "bundle_kind", None),
        combine_mode=getattr(args, "combine_mode", None),
    )
