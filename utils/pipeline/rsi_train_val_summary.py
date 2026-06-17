"""Сводный отчёт rsi_spreads: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.paths import rsi_summary_log_path, rsi_summary_plot_path
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.research.signal_validation import ConfirmationMode
from crypto_research.utils.research.spread_summary_plots import save_check_signal_delta_chart
from crypto_research.utils.research.train_val_report import (
    full_pool_intro,
    pair_universality_intro,
    temporal_stability_intro,
)
from crypto_research.utils.rsi.constants import SELECTED_RSI_PERIOD
from crypto_research.utils.rsi.summary import (
    RSI_SIGNAL_SPEC,
    compute_rsi_signal_pooled_summary,
    compute_rsi_signal_summary,
    format_rsi_pooled_signal_table,
    format_rsi_signal_summary_table,
    status_by_signal_key,
)

log = get_logger("rsi_train_val_summary")


def _compute_summary_rows(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
    *,
    rsi_period: int,
):
    train_bands = compute_pair_means(train_daily)
    val_bands = compute_pair_means(val_daily)
    temporal_train_bands = compute_pair_means(temporal_train_daily)
    temporal_val_bands = compute_pair_means(temporal_val_daily)
    pair_rows = compute_rsi_signal_summary(
        train_daily,
        val_daily,
        train_bands,
        val_bands,
        period=rsi_period,
        confirmation_mode=ConfirmationMode.COHORT,
    )
    temporal_rows = compute_rsi_signal_summary(
        temporal_train_daily,
        temporal_val_daily,
        temporal_train_bands,
        temporal_val_bands,
        period=rsi_period,
        confirmation_mode=ConfirmationMode.PER_PAIR,
    )
    return pair_rows, temporal_rows


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
    rsi_period = ctx.rsi_periods[0] if ctx.rsi_periods else SELECTED_RSI_PERIOD
    pair_rows, temporal_rows = _compute_summary_rows(
        train_daily,
        val_daily,
        temporal_train_daily,
        temporal_val_daily,
        rsi_period=rsi_period,
    )
    full_bands = compute_pair_means(full_daily)
    status_map = status_by_signal_key(pair_rows, temporal_rows)
    indicator = RSI_SIGNAL_SPEC.indicator_label(rsi_period)

    parts = [
        f"=== Отчёт: сводка RSI({rsi_period}) — подтверждение сигналов ===",
        "",
        "Три блока (как day_of_week / ema_spreads):",
        "  1. Универсальность среди пар — train/val когорты на одном периоде",
        "  2. Устойчивость во времени — один пул пар, два периода",
        "  3. Полный пул — все пары, весь период, итог = пересечение статусов 1 и 2",
        "",
    ]
    parts.extend(pair_universality_intro(train_pairs, val_pairs))
    parts.extend(
        format_rsi_signal_summary_table(
            pair_rows,
            title="=== Сводная таблица: универсальность среди пар ===",
            val_confirm_hint=(
                "Подтверждение на val: val-пары с тем же знаком Δ, "
                "что pooled train; ✅ ≥ 60%, иначе ❌."
            ),
            rsi_period=rsi_period,
        )
    )
    parts.extend(temporal_stability_intro(temporal_pairs))
    parts.extend(
        format_rsi_signal_summary_table(
            temporal_rows,
            title="=== Сводная таблица: устойчивость во времени ===",
            val_confirm_hint=(
                "Подтверждение на val: пары с тем же знаком Δ "
                "во 2-м периоде, что в train-периоде; ✅ ≥ 60%, иначе ❌."
            ),
            rsi_period=rsi_period,
        )
    )
    parts.extend(full_pool_intro(full_pairs))
    pooled_rows = compute_rsi_signal_pooled_summary(
        full_daily,
        full_bands,
        status_by_key=status_map,
        period=rsi_period,
    )
    parts.extend(format_rsi_pooled_signal_table(pooled_rows, rsi_period=rsi_period))

    plot_check1 = rsi_summary_plot_path(rsi_period, "check_pair_universality_delta")
    plot_check2 = rsi_summary_plot_path(rsi_period, "check_temporal_stability_delta")
    save_check_signal_delta_chart(
        pair_rows,
        plot_check1,
        title=f"RSI({rsi_period}) — проверка 1: универсальность среди пар",
        subtitle="зелёная полоса = статус «значим» (24 train / 25 val, 2022–2026)",
    )
    save_check_signal_delta_chart(
        temporal_rows,
        plot_check2,
        title=f"RSI({rsi_period}) — проверка 2: устойчивость во времени",
        subtitle="зелёная полоса = статус «значим» (49 пар, train 2022–2024-04 / val 2024-04–2026)",
    )

    path = rsi_summary_log_path(rsi_period)
    lines = parts + [
        "",
        "=== Графики ===",
        f"Проверка 1 — Δ train vs val: {plot_check1}",
        f"Проверка 2 — Δ train vs val: {plot_check2}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Сводка RSI сохранена: %s", path)
    return path
