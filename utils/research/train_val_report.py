"""Общие блоки сводного train/val отчёта (универсальность + устойчивость + полный пул)."""

from __future__ import annotations

from crypto_research.utils.pipeline.paths import (
    FULL_POOL_FROM,
    FULL_POOL_MAX_PAIR_START,
    FULL_POOL_TO,
    PAIR_UNIVERSALITY_FROM,
    PAIR_UNIVERSALITY_TO,
    TEMPORAL_POOL_MAX_PAIR_START,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
    TRAIN_MAX_PAIR_START,
    VAL_MAX_PAIR_START,
)
from crypto_research.utils.pipeline.weekday_effects import format_pairs_lines


def format_pairs_block(label: str, pairs: list[str]) -> list[str]:
    lines = format_pairs_lines(pairs)
    if not lines:
        return [f"{label}: —"]
    lines[0] = lines[0].replace("Пары: ", f"{label}: ", 1)
    return lines


def pair_universality_intro(
    train_pairs: list[str],
    val_pairs: list[str],
) -> list[str]:
    return [
        "=== Универсальность среди пар ===",
        "",
        f"Период (UTC): {PAIR_UNIVERSALITY_FROM} .. {PAIR_UNIVERSALITY_TO}",
        f"Train: {len(train_pairs)} пар (первая свеча ≤ {TRAIN_MAX_PAIR_START})",
        f"Val: {len(val_pairs)} пар (пул ≤ {VAL_MAX_PAIR_START}, вне train-cohort)",
        "Пары train и val не пересекаются.",
        "",
        *format_pairs_block("Train", train_pairs),
        "",
        *format_pairs_block("Val", val_pairs),
        "",
    ]


def temporal_stability_intro(pairs: list[str]) -> list[str]:
    return [
        "=== Устойчивость во времени ===",
        "",
        f"Пул: {len(pairs)} пар (первая свеча ≤ {TEMPORAL_POOL_MAX_PAIR_START})",
        f"Train-период (UTC): {TEMPORAL_TRAIN_FROM} .. {TEMPORAL_TRAIN_TO}",
        f"Val-период (UTC): {TEMPORAL_VAL_FROM} .. {TEMPORAL_VAL_TO}",
        "Одни и те же пары в обоих периодах.",
        "",
        *format_pairs_block("Пары", pairs),
        "",
    ]


def full_pool_intro(pairs: list[str]) -> list[str]:
    return [
        "=== Полный пул (все пары, весь период) ===",
        "",
        f"Пар: {len(pairs)} (первая свеча ≤ {FULL_POOL_MAX_PAIR_START})",
        f"Период (UTC): {FULL_POOL_FROM} .. {FULL_POOL_TO}",
        "",
        *format_pairs_block("Пары", pairs),
        "",
    ]
