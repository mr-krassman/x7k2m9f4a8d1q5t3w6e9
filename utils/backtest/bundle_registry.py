"""Реестр combined bundle и пути combined/dow_ema_sp/backtest/{ml,algo}."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from crypto_research.utils.ml.registry import (
    RULE_BASED_STRATEGIES,
    RULE_BASED_TO_ML_STUDY,
    MlStudySpec,
    bundle_id_for_ml_studies,
    canonical_ml_studies,
    is_ml_study_id,
    resolve_ml_study,
)

BundleKind = Literal["ml", "algo"]
CombineMode = Literal["and", "or"]

ALGO_STUDY_ORDER: tuple[str, ...] = (
    "day_of_week",
    "ema_spreads",
    "rsi_spreads",
    "price_sequences",
    "volume_spreads",
)

COMBINE_MODE_OR: CombineMode = "or"
COMBINE_MODE_AND: CombineMode = "and"


@dataclass(frozen=True)
class AlgoBundleEntry:
    bundle_id: str
    studies: tuple[str, ...]


@dataclass(frozen=True)
class AlgoBundleSpec:
    bundle_id: str
    studies: tuple[str, ...]
    combine_mode: CombineMode


@dataclass(frozen=True)
class ParsedBacktestStrategies:
    strategy_key: str
    ml_spec: MlStudySpec | None = None
    algo_spec: AlgoBundleSpec | None = None
    bundle_id: str | None = None
    bundle_kind: BundleKind | None = None
    combine_mode: CombineMode | None = None


ALGO_BUNDLE_REGISTRY: dict[tuple[str, ...], AlgoBundleEntry] = {
    ("day_of_week", "ema_spreads"): AlgoBundleEntry(
        bundle_id="dow_ema_sp",
        studies=("day_of_week", "ema_spreads"),
    ),
    ("day_of_week", "ema_spreads", "rsi_spreads"): AlgoBundleEntry(
        bundle_id="dow_ema_rsi_sp",
        studies=("day_of_week", "ema_spreads", "rsi_spreads"),
    ),
    ("day_of_week", "ema_spreads", "rsi_spreads", "price_sequences"): AlgoBundleEntry(
        bundle_id="dow_ema_rsi_streak",
        studies=("day_of_week", "ema_spreads", "rsi_spreads", "price_sequences"),
    ),
    (
        "day_of_week",
        "ema_spreads",
        "rsi_spreads",
        "price_sequences",
        "volume_spreads",
    ): AlgoBundleEntry(
        bundle_id="dow_ema_rsi_streak_vol",
        studies=(
            "day_of_week",
            "ema_spreads",
            "rsi_spreads",
            "price_sequences",
            "volume_spreads",
        ),
    ),
}

ALGO_BUNDLE_ID_TO_STUDIES: dict[str, tuple[str, ...]] = {
    entry.bundle_id: entry.studies for entry in ALGO_BUNDLE_REGISTRY.values()
}


def is_algo_bundle_id(name: str) -> bool:
    return name in ALGO_BUNDLE_ID_TO_STUDIES


def canonical_algo_studies(studies: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(studies))
    unknown = set(ordered) - RULE_BASED_STRATEGIES
    if unknown:
        raise ValueError(f"Неизвестные rule-based стратегии: {sorted(unknown)}")
    return tuple(study for study in ALGO_STUDY_ORDER if study in ordered)


def algo_bundle_for(studies: tuple[str, ...]) -> AlgoBundleEntry | None:
    if len(studies) <= 1:
        return None
    return ALGO_BUNDLE_REGISTRY.get(studies)


def resolve_algo_bundle(studies: tuple[str, ...]) -> AlgoBundleEntry:
    ordered = canonical_algo_studies(studies)
    registered = algo_bundle_for(ordered)
    if registered is not None:
        return registered
    ml_studies = tuple(RULE_BASED_TO_ML_STUDY[study_id] for study_id in ordered)
    bundle_id, _ = bundle_id_for_ml_studies(ml_studies)
    return AlgoBundleEntry(bundle_id=bundle_id, studies=ordered)


def combined_bundle_root(bundle_id: str) -> Path:
    from crypto_research.utils.pipeline.paths import RESEARCH_ROOT

    return RESEARCH_ROOT / "research_outputs" / "combined" / bundle_id


def bundle_ml_root(bundle_id: str) -> Path:
    return combined_bundle_root(bundle_id) / "ml"


def bundle_algo_root(bundle_id: str) -> Path:
    return combined_bundle_root(bundle_id) / "algo"


def bundle_backtest_dir(bundle_id: str, kind: BundleKind) -> Path:
    return combined_bundle_root(bundle_id) / "backtest" / kind


def bundle_backtest_plots_dir(bundle_id: str, kind: BundleKind) -> Path:
    return bundle_backtest_dir(bundle_id, kind) / "plots"


def parse_backtest_strategy_args(
    raw: list[str],
    *,
    combine_mode: CombineMode | None = None,
) -> ParsedBacktestStrategies:
    if not raw:
        raise ValueError("Укажите хотя бы одну стратегию")

    if len(raw) == 1:
        name = raw[0]
        if name in RULE_BASED_STRATEGIES:
            return ParsedBacktestStrategies(strategy_key=name)
        if is_ml_study_id(name):
            return ParsedBacktestStrategies(
                strategy_key=name,
                ml_spec=resolve_ml_study((name,)),
            )
        raise ValueError(f"Неизвестная стратегия: {name}")

    try:
        ml_names = canonical_ml_studies(raw)
        if len(ml_names) == len(raw) and len(ml_names) > 1:
            spec = resolve_ml_study(ml_names)
            return ParsedBacktestStrategies(
                strategy_key=spec.bundle_id,
                ml_spec=spec,
                bundle_id=spec.bundle_id,
                bundle_kind="ml",
            )
    except ValueError:
        pass

    algo_names = canonical_algo_studies(raw)
    if len(algo_names) != len(raw):
        raise ValueError(
            "Смешивание rule-based, ML и неизвестных стратегий в одном запуске не поддерживается; "
            f"получено: {raw}"
        )
    if len(algo_names) > 1:
        bundle = resolve_algo_bundle(algo_names)
        if combine_mode is None:
            raise ValueError(
                f"Для combined rule-based ({' '.join(algo_names)}) укажите --mode and или --mode or"
            )
        return ParsedBacktestStrategies(
            strategy_key=bundle.bundle_id,
            algo_spec=AlgoBundleSpec(
                bundle_id=bundle.bundle_id,
                studies=bundle.studies,
                combine_mode=combine_mode,
            ),
            bundle_id=bundle.bundle_id,
            bundle_kind="algo",
            combine_mode=combine_mode,
        )

    raise ValueError(f"Неизвестная стратегия: {raw[0]}")
