"""Сценарии бэктеста: maximal, conservative, optimistic."""

from __future__ import annotations

from datetime import datetime

from crypto_research.utils.pipeline.daily_pool import build_pooled_daily, build_weekday_daily
from crypto_research.utils.pipeline.load_pairs import load_klines_for_period
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.paths import (
    TEMPORAL_POOL_MAX_PAIR_START,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
)
from crypto_research.utils.weekday.pair_selection import select_day_of_week_train_pairs

SCENARIO_MAXIMAL = "maximal"
SCENARIO_CONSERVATIVE = "conservative"
SCENARIO_OPTIMISTIC = "optimistic"

# Устаревшее имя CLI (обратная совместимость)
SCENARIO_MAXIMAL_VAL = SCENARIO_CONSERVATIVE

SCENARIO_ALIASES: dict[str, str] = {
    "maximal_val": SCENARIO_CONSERVATIVE,
}

SCENARIO_LABEL_RU: dict[str, str] = {
    SCENARIO_CONSERVATIVE: "консервативный",
    SCENARIO_OPTIMISTIC: "оптимистичный",
    SCENARIO_MAXIMAL: "maximal",
}

SCENARIO_REPORT_HEADER: dict[str, str] = {
    SCENARIO_CONSERVATIVE: (
        "Сценарий: консервативный (val, все 49 пар — baseline для сравнения с оптимистичным)"
    ),
    SCENARIO_OPTIMISTIC: (
        "Сценарий: оптимистичный (val, train-отбор пар — свой набор на Чт / Пт / Сб)"
    ),
}

EMA_SCENARIO_REPORT_HEADER: dict[str, str] = {
    SCENARIO_CONSERVATIVE: (
        "Сценарий: консервативный (val, все 49 пар, b6 long, пороги frozen train)"
    ),
    SCENARIO_OPTIMISTIC: (
        "Сценарий: оптимистичный (val, train-отбор пар по b6 × «Цена росла»)"
    ),
}

VAL_FROM = TEMPORAL_VAL_FROM
VAL_TO = TEMPORAL_VAL_TO
OPTIMISTIC_TRAIN_FROM = TEMPORAL_TRAIN_FROM
OPTIMISTIC_TRAIN_TO = TEMPORAL_TRAIN_TO
OPTIMISTIC_MAX_PAIR_START = TEMPORAL_POOL_MAX_PAIR_START


def normalize_scenario(name: str) -> str:
    return SCENARIO_ALIASES.get(name, name)


def scenario_label_ru(name: str) -> str | None:
    return SCENARIO_LABEL_RU.get(normalize_scenario(name))


def resolve_optimistic_pairs_by_weekday(
    data_dir,
    *,
    workers: int,
) -> dict[int, list[str]]:
    train_from = datetime.fromisoformat(OPTIMISTIC_TRAIN_FROM)
    train_to = datetime.fromisoformat(OPTIMISTIC_TRAIN_TO)
    max_start = datetime.fromisoformat(OPTIMISTIC_MAX_PAIR_START)
    klines = load_klines_for_period(
        data_dir,
        train_from,
        train_to,
        None,
        max_start,
        split=None,
        workers=workers,
    )
    daily = build_pooled_daily(klines)
    bands = compute_pair_means(daily)
    weekday_daily = build_weekday_daily(daily)
    selections = select_day_of_week_train_pairs(weekday_daily, bands)
    return {sel.signal.weekday: sel.year_confirmed for sel in selections}


def union_pairs(pairs_by_weekday: dict[int, list[str]]) -> list[str]:
    out: set[str] = set()
    for pairs in pairs_by_weekday.values():
        out.update(pairs)
    return sorted(out)
