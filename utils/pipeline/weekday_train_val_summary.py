"""Сводный отчёт: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import (
    PAIR_UNIVERSALITY_FROM,
    PAIR_UNIVERSALITY_TO,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
    weekday_check_plot_path,
    weekday_summary_log_path,
)
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.weekday_plots import save_weekday_train_val_nav_comparison
from crypto_research.utils.research.signal_validation import ConfirmationMode, intersect_status
from crypto_research.utils.research.train_val_report import (
    full_pool_intro,
    pair_universality_intro,
    temporal_stability_intro,
)
from crypto_research.utils.weekday.summary import (
    WeekdaySummaryRow,
    compute_weekday_pooled_summary,
    compute_weekday_summary,
    format_pooled_summary_table,
    format_summary_table,
)

log = get_logger("weekday_train_val_summary")


def _status_by_weekday(
    pair_rows: list[WeekdaySummaryRow],
    temporal_rows: list[WeekdaySummaryRow],
) -> dict[int, str]:
    return {
        pair_row.weekday: intersect_status(pair_row.status, temporal_row.status)
        for pair_row, temporal_row in zip(pair_rows, temporal_rows, strict=True)
    }


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
    parts.extend(pair_universality_intro(train_pairs, val_pairs))
    parts.extend(
        format_summary_table(
            pair_rows,
            title="=== Сводная таблица: универсальность среди пар ===",
            val_confirm_hint=(
                "Подтверждение на val: val-пары с тем же знаком среднего return, "
                "что pooled train; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    parts.extend(temporal_stability_intro(temporal_pairs))
    parts.extend(
        format_summary_table(
            temporal_rows,
            title="=== Сводная таблица: устойчивость во времени ===",
            val_confirm_hint=(
                "Подтверждение на val: пары с тем же знаком среднего return "
                "во 2-м периоде, что в train-периоде; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    parts.extend(full_pool_intro(full_pairs))
    parts.extend(
        format_pooled_summary_table(
            compute_weekday_pooled_summary(full_daily, status_by_weekday=status_by_weekday)
        )
    )
    return "\n".join(parts)


def save_summary_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[3] Сводка сохранена: %s", path)
    return path


def _save_check_plots(
    train_daily: pl.DataFrame,
    train_pairs: list[str],
    val_daily: pl.DataFrame,
    val_pairs: list[str],
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
    temporal_pairs: list[str],
) -> tuple[Path, Path]:
    pair_univ_from = parse_iso_utc(PAIR_UNIVERSALITY_FROM)
    pair_univ_to = parse_iso_utc(PAIR_UNIVERSALITY_TO)
    temporal_train_from = parse_iso_utc(TEMPORAL_TRAIN_FROM)
    temporal_train_to = parse_iso_utc(TEMPORAL_TRAIN_TO)
    temporal_val_from = parse_iso_utc(TEMPORAL_VAL_FROM)
    temporal_val_to = parse_iso_utc(TEMPORAL_VAL_TO)

    plot_pair_univ = weekday_check_plot_path("pair_universality")
    save_weekday_train_val_nav_comparison(
        train_daily,
        train_pairs,
        pair_univ_from,
        pair_univ_to,
        f"Train ({len(train_pairs)} pairs)",
        val_daily,
        val_pairs,
        pair_univ_from,
        pair_univ_to,
        f"Val ({len(val_pairs)} pairs)",
        plot_pair_univ,
        suptitle="Проверка 1 — универсальность среди пар (cross-asset)",
    )

    plot_temporal = weekday_check_plot_path("temporal_stability")
    save_weekday_train_val_nav_comparison(
        temporal_train_daily,
        temporal_pairs,
        temporal_train_from,
        temporal_train_to,
        f"Train period ({len(temporal_pairs)} pairs)",
        temporal_val_daily,
        temporal_pairs,
        temporal_val_from,
        temporal_val_to,
        f"Val period ({len(temporal_pairs)} pairs)",
        plot_temporal,
        suptitle="Проверка 2 — устойчивость во времени (out-of-time)",
    )
    return plot_pair_univ, plot_temporal


def run_summary_report(
    ctx: ReportContext,
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
    plot_pair_univ, plot_temporal = _save_check_plots(
        train_daily,
        train_pairs,
        val_daily,
        val_pairs,
        temporal_train_daily,
        temporal_val_daily,
        temporal_pairs,
    )
    lines = text.splitlines()
    lines.extend([
        "",
        "=== Графики ===",
        f"Проверка 1 (train/val): {plot_pair_univ}",
        f"Проверка 2 (train/val): {plot_temporal}",
        "",
    ])
    return save_summary_report("\n".join(lines), weekday_summary_log_path())
