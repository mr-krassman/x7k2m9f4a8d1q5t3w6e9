"""Сводный отчёт: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.logger import get_logger
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
    weekday_summary_log_path,
)
from crypto_research.utils.pipeline.weekday_effects import format_pairs_lines
from crypto_research.utils.pipeline.weekday_report import WeekdayReportContext
from crypto_research.utils.weekday.summary import (
    ConfirmationMode,
    WeekdaySummaryRow,
    _intersect_status,
    compute_weekday_pooled_summary,
    compute_weekday_summary,
    format_pooled_summary_table,
    format_summary_table,
)

log = get_logger("weekday_train_val_summary")


def _format_pairs_block(label: str, pairs: list[str]) -> list[str]:
    lines = format_pairs_lines(pairs)
    if not lines:
        return [f"{label}: —"]
    lines[0] = lines[0].replace("Пары: ", f"{label}: ", 1)
    return lines


def _status_by_weekday(
    pair_rows: list[WeekdaySummaryRow],
    temporal_rows: list[WeekdaySummaryRow],
) -> dict[int, str]:
    return {
        pair_row.weekday: _intersect_status(pair_row.status, temporal_row.status)
        for pair_row, temporal_row in zip(pair_rows, temporal_rows, strict=True)
    }


def _pair_universality_section(
    train_pairs: list[str],
    val_pairs: list[str],
    pair_rows: list[WeekdaySummaryRow],
) -> list[str]:
    lines = [
        "=== Универсальность среди пар ===",
        "",
        f"Период (UTC): {PAIR_UNIVERSALITY_FROM} .. {PAIR_UNIVERSALITY_TO}",
        f"Train: {len(train_pairs)} пар (первая свеча ≤ {TRAIN_MAX_PAIR_START})",
        f"Val: {len(val_pairs)} пар (пул ≤ {VAL_MAX_PAIR_START}, вне train-cohort)",
        "Пары train и val не пересекаются.",
        "",
    ]
    lines.extend(_format_pairs_block("Train", train_pairs))
    lines.append("")
    lines.extend(_format_pairs_block("Val", val_pairs))
    lines.append("")
    lines.extend(
        format_summary_table(
            pair_rows,
            title="=== Сводная таблица: универсальность среди пар ===",
            val_confirm_hint=(
                "Подтверждение на val: val-пары с тем же знаком среднего return, "
                f"что pooled train; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    return lines


def _temporal_stability_section(
    pairs: list[str],
    temporal_rows: list[WeekdaySummaryRow],
) -> list[str]:
    lines = [
        "=== Устойчивость во времени ===",
        "",
        f"Пул: {len(pairs)} пар (первая свеча ≤ {TEMPORAL_POOL_MAX_PAIR_START})",
        f"Train-период (UTC): {TEMPORAL_TRAIN_FROM} .. {TEMPORAL_TRAIN_TO}",
        f"Val-период (UTC): {TEMPORAL_VAL_FROM} .. {TEMPORAL_VAL_TO}",
        "Одни и те же пары в обоих периодах.",
        "",
    ]
    lines.extend(_format_pairs_block("Пары", pairs))
    lines.append("")
    lines.extend(
        format_summary_table(
            temporal_rows,
            title="=== Сводная таблица: устойчивость во времени ===",
            val_confirm_hint=(
                "Подтверждение на val: пары с тем же знаком среднего return "
                "во 2-м периоде, что в train-периоде; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    return lines


def _full_pool_section(
    pairs: list[str],
    full_daily: pl.DataFrame,
    status_by_weekday: dict[int, str],
) -> list[str]:
    lines = [
        "=== Полный пул (все пары, весь период) ===",
        "",
        f"Пар: {len(pairs)} (первая свеча ≤ {FULL_POOL_MAX_PAIR_START})",
        f"Период (UTC): {FULL_POOL_FROM} .. {FULL_POOL_TO}",
        "",
    ]
    lines.extend(_format_pairs_block("Пары", pairs))
    lines.append("")
    rows = compute_weekday_pooled_summary(full_daily, status_by_weekday=status_by_weekday)
    lines.extend(format_pooled_summary_table(rows))
    return lines


def assemble_summary_report(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    train_pairs: list[str],
    val_pairs: list[str],
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
    temporal_pairs: list[str],
    full_daily: pl.DataFrame,
    full_pairs: list[str],
) -> str:
    pair_rows = compute_weekday_summary(
        train_daily,
        val_daily,
        confirmation_mode=ConfirmationMode.COHORT,
    )
    temporal_rows = compute_weekday_summary(
        temporal_train_daily,
        temporal_val_daily,
        confirmation_mode=ConfirmationMode.PER_PAIR,
    )
    status_by_weekday = _status_by_weekday(pair_rows, temporal_rows)

    parts = [
        "=== Отчёт: сводка дней недели ===",
        "",
        "Три блока (см. docs/journal.md):",
        "  1. Универсальность среди пар — train/val когорты на одном периоде",
        "  2. Устойчивость во времени — один пул пар, два периода",
        "  3. Полный пул — все пары, весь период, итог = пересечение статусов 1 и 2",
        "",
    ]
    parts.extend(_pair_universality_section(train_pairs, val_pairs, pair_rows))
    parts.extend(_temporal_stability_section(temporal_pairs, temporal_rows))
    parts.extend(_full_pool_section(full_pairs, full_daily, status_by_weekday))
    return "\n".join(parts)


def save_summary_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[3] Сводка сохранена: %s", path)
    return path


def run_summary_report(
    ctx: WeekdayReportContext,
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    train_pairs: list[str],
    val_pairs: list[str],
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
    temporal_pairs: list[str],
    full_daily: pl.DataFrame,
    full_pairs: list[str],
) -> Path:
    text = assemble_summary_report(
        train_daily,
        val_daily,
        train_pairs,
        val_pairs,
        temporal_train_daily,
        temporal_val_daily,
        temporal_pairs,
        full_daily,
        full_pairs,
    )
    return save_summary_report(text, weekday_summary_log_path())
