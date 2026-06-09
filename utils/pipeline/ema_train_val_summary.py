"""Сводный отчёт ema_spreads: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.summary import (
    EmaSignalSummaryRow,
    compute_ema_signal_pooled_summary,
    compute_ema_signal_summary,
    format_pooled_signal_table,
    format_signal_summary_table,
    status_by_signal_key,
)
from crypto_research.utils.ema_spreads.summary_plots import (
    save_confirmation_chart,
    save_signal_delta_chart,
)
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.paths import ema_summary_log_path, ema_summary_plot_path
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.research.signal_validation import ConfirmationMode
from crypto_research.utils.research.train_val_report import (
    full_pool_intro,
    pair_universality_intro,
    temporal_stability_intro,
)

log = get_logger("ema_train_val_summary")


def _compute_summary_rows(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
    *,
    ema_period: int,
):
    train_bands = compute_pair_means(train_daily)
    val_bands = compute_pair_means(val_daily)
    temporal_train_bands = compute_pair_means(temporal_train_daily)
    temporal_val_bands = compute_pair_means(temporal_val_daily)
    periods = (ema_period,)
    pair_rows = compute_ema_signal_summary(
        train_daily,
        val_daily,
        train_bands,
        val_bands,
        period=ema_period,
        periods=periods,
        confirmation_mode=ConfirmationMode.COHORT,
    )
    temporal_rows = compute_ema_signal_summary(
        temporal_train_daily,
        temporal_val_daily,
        temporal_train_bands,
        temporal_val_bands,
        period=ema_period,
        periods=periods,
        confirmation_mode=ConfirmationMode.PER_PAIR,
    )
    return pair_rows, temporal_rows


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
    pair_rows: list[EmaSignalSummaryRow],
    temporal_rows: list[EmaSignalSummaryRow],
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> str:
    periods = (ema_period,)
    full_bands = compute_pair_means(full_daily)
    status_map = status_by_signal_key(pair_rows, temporal_rows)

    parts = [
        f"=== Отчёт: сводка EMA({ema_period}) — подтверждение сигналов ===",
        "",
        "Три блока (как day_of_week):",
        "  1. Универсальность среди пар — train/val когорты на одном периоде",
        "  2. Устойчивость во времени — один пул пар, два периода",
        "  3. Полный пул — все пары, весь период, итог = пересечение статусов 1 и 2",
        "",
    ]
    parts.extend(pair_universality_intro(train_pairs, val_pairs))
    parts.extend(
        format_signal_summary_table(
            pair_rows,
            title="=== Сводная таблица: универсальность среди пар ===",
            val_confirm_hint=(
                "Подтверждение на val: val-пары с тем же знаком Δ, "
                "что pooled train; ✅ ≥ 60%, иначе ❌."
            ),
            ema_period=ema_period,
        )
    )
    parts.extend(temporal_stability_intro(temporal_pairs))
    parts.extend(
        format_signal_summary_table(
            temporal_rows,
            title="=== Сводная таблица: устойчивость во времени ===",
            val_confirm_hint=(
                "Подтверждение на val: пары с тем же знаком Δ "
                "во 2-м периоде, что в train-периоде; ✅ ≥ 60%, иначе ❌."
            ),
            ema_period=ema_period,
        )
    )
    parts.extend(full_pool_intro(full_pairs))
    pooled_rows = compute_ema_signal_pooled_summary(
        full_daily,
        full_bands,
        status_by_key=status_map,
        period=ema_period,
        periods=periods,
    )
    parts.extend(format_pooled_signal_table(pooled_rows, ema_period=ema_period))
    return "\n".join(parts)


def save_summary_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("Сводка EMA сохранена: %s", path)
    return path


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
    ema_period = ctx.ema_periods[0] if ctx.ema_periods else SELECTED_EMA_PERIOD
    pair_rows, temporal_rows = _compute_summary_rows(
        train_daily,
        val_daily,
        temporal_train_daily,
        temporal_val_daily,
        ema_period=ema_period,
    )
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
        pair_rows,
        temporal_rows,
        ema_period=ema_period,
    )

    plot_delta = ema_summary_plot_path(ema_period, "signal_delta")
    plot_confirm = ema_summary_plot_path(ema_period, "confirmation")
    save_signal_delta_chart(pair_rows, plot_delta, ema_period=ema_period)
    save_confirmation_chart(pair_rows, temporal_rows, plot_confirm, ema_period=ema_period)

    path = ema_summary_log_path(ema_period)

    lines = text.splitlines()
    lines.extend([
        "",
        "=== Графики ===",
        f"Δ train vs val: {plot_delta}",
        f"Подтверждение на val: {plot_confirm}",
        "",
    ])
    save_summary_report("\n".join(lines), path)
    return path
