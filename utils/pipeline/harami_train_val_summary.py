"""Сводный отчёт harami: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.candlestick_patterns.harami.audit_log import save_harami_audit_log
from crypto_research.utils.candlestick_patterns.harami.constants import PATTERN_NOTE
from crypto_research.utils.candlestick_patterns.harami.summary import (
    compute_harami_signal_pooled_summary,
    compute_harami_signal_summary,
    format_harami_pooled_signal_table,
    format_harami_signal_summary_table,
    status_by_signal_key,
)
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.pair_means import compute_pair_means
from crypto_research.utils.pipeline.paths import (
    harami_signals_audit_log_path,
    harami_summary_log_path,
    harami_summary_plot_path,
)
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.research.signal_validation import ConfirmationMode
from crypto_research.utils.research.spread_summary_plots import save_check_signal_delta_chart
from crypto_research.utils.research.train_val_report import (
    full_pool_intro,
    pair_universality_intro,
    temporal_stability_intro,
)

log = get_logger("harami_train_val_summary")


def _compute_summary_rows(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    temporal_train_daily: pl.DataFrame,
    temporal_val_daily: pl.DataFrame,
):
    train_bands = compute_pair_means(train_daily)
    val_bands = compute_pair_means(val_daily)
    temporal_train_bands = compute_pair_means(temporal_train_daily)
    temporal_val_bands = compute_pair_means(temporal_val_daily)
    pair_rows = compute_harami_signal_summary(
        train_daily,
        val_daily,
        train_bands,
        val_bands,
        confirmation_mode=ConfirmationMode.COHORT,
    )
    temporal_rows = compute_harami_signal_summary(
        temporal_train_daily,
        temporal_val_daily,
        temporal_train_bands,
        temporal_val_bands,
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
    pair_rows,
    temporal_rows,
) -> str:
    full_bands = compute_pair_means(full_daily)
    status_map = status_by_signal_key(pair_rows, temporal_rows)

    parts = [
        "=== Отчёт: сводка Harami — подтверждение сигналов ===",
        "",
        PATTERN_NOTE,
        "",
        "Три блока (как day_of_week / ema / rsi):",
        "  1. Универсальность среди пар — train/val когорты на одном периоде",
        "  2. Устойчивость во времени — один пул пар, два периода",
        "  3. Полный пул — все пары, весь период, итог = пересечение статусов 1 и 2",
        "",
    ]
    parts.extend(pair_universality_intro(train_pairs, val_pairs))
    parts.extend(
        format_harami_signal_summary_table(
            pair_rows,
            title="=== Сводная таблица: универсальность среди пар ===",
            val_confirm_hint=(
                "Подтверждение на val: val-пары с тем же знаком Δ, "
                "что pooled train; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    parts.extend(temporal_stability_intro(temporal_pairs))
    parts.extend(
        format_harami_signal_summary_table(
            temporal_rows,
            title="=== Сводная таблица: устойчивость во времени ===",
            val_confirm_hint=(
                "Подтверждение на val: пары с тем же знаком Δ "
                "во 2-м периоде, что в train-периоде; ✅ ≥ 60%, иначе ❌."
            ),
        )
    )
    parts.extend(full_pool_intro(full_pairs))
    pooled_rows = compute_harami_signal_pooled_summary(
        full_daily,
        full_bands,
        status_by_key=status_map,
    )
    parts.extend(format_harami_pooled_signal_table(pooled_rows))
    return "\n".join(parts)


def save_summary_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[harami] сводка сохранена: %s", path)
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
    del ctx
    pair_rows, temporal_rows = _compute_summary_rows(
        train_daily,
        val_daily,
        temporal_train_daily,
        temporal_val_daily,
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
    )

    plot_check1 = harami_summary_plot_path("check_pair_universality_delta")
    plot_check2 = harami_summary_plot_path("check_temporal_stability_delta")
    save_check_signal_delta_chart(
        pair_rows,
        plot_check1,
        title="harami — проверка 1: универсальность среди пар",
        subtitle="зелёная полоса = статус «значим» (train/val когорты, 2022–2026)",
    )
    save_check_signal_delta_chart(
        temporal_rows,
        plot_check2,
        title="harami — проверка 2: устойчивость во времени",
        subtitle="зелёная полоса = статус «значим» (49 пар, train 2022–2024-04 / val 2024-04–2026)",
    )

    save_harami_audit_log(full_daily, harami_signals_audit_log_path())

    lines = text.splitlines()
    audit_path = harami_signals_audit_log_path()
    lines.extend([
        "",
        "=== Аудит сигналов (OHLC t−1, t, t+1) ===",
        f"Полный лог событий: {audit_path}",
        "",
        "=== Графики ===",
        f"Проверка 1 — Δ train vs val: {plot_check1}",
        f"Проверка 2 — Δ train vs val: {plot_check2}",
        "",
    ])
    return save_summary_report("\n".join(lines), harami_summary_log_path())
