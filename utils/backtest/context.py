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
    fee: FeeSchedule = DEFAULT_FEE


def build_backtest_context(args) -> BacktestContext:
    scenario = normalize_scenario(args.scenario)
    if scenario in (SCENARIO_OPTIMISTIC, SCENARIO_CONSERVATIVE):
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
    )
