"""Сводка train→val для сигналов EMA (бакет × колонка)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import (
    EMA_SCENARIO_ROWS,
    N_EMA_SCENARIOS,
    SCREEN_MIN_POOLED_DELTA_PP,
    SCREEN_STATS_COLS,
    SELECTED_EMA_PERIOD,
)
from crypto_research.utils.ema_spreads.signal_metrics import compute_all_cell_deltas_pp
from crypto_research.utils.research.signal_validation import (
    ALPHA,
    ConfirmationMode,
    bonferroni_alpha,
    count_confirm,
    fmt_delta_pp,
    fmt_p,
    intersect_status,
    permutation_p,
    signal_status,
    val_confirm_text,
)
from crypto_research.utils.weekday.bands import MeanBands

BONFERRONI_N = N_EMA_SCENARIOS * len(SCREEN_STATS_COLS)
BONF_ALPHA = bonferroni_alpha(BONFERRONI_N)


@dataclass(frozen=True)
class EmaSignalKey:
    bucket: int
    column: str

    @property
    def label(self) -> str:
        return f"b{self.bucket} {EMA_SCENARIO_ROWS[self.bucket]} × {self.column}"


@dataclass(frozen=True)
class EmaSignalSummaryRow:
    key: EmaSignalKey
    train_delta_pp: float
    val_delta_pp: float | None
    p_value: float
    val_agree: int | None
    val_total: int
    status: str

    @property
    def delta_val_train_pp(self) -> float | None:
        if self.val_delta_pp is None:
            return None
        if self.train_delta_pp != self.train_delta_pp or self.val_delta_pp != self.val_delta_pp:
            return None
        return self.val_delta_pp - self.train_delta_pp


def _signal_keys() -> list[EmaSignalKey]:
    return [
        EmaSignalKey(bucket=b, column=col)
        for b in range(N_EMA_SCENARIOS)
        for col in SCREEN_STATS_COLS
    ]


def _summary_rows_from_deltas(
    train_cells: dict[tuple[int, str], tuple[float, dict[str, float]]],
    val_cells: dict[tuple[int, str], tuple[float, dict[str, float]]] | None,
    *,
    confirmation_mode: ConfirmationMode,
) -> list[EmaSignalSummaryRow]:
    rows: list[EmaSignalSummaryRow] = []
    for key in _signal_keys():
        train_delta, train_pair = train_cells.get((key.bucket, key.column), (float("nan"), {}))
        p_value = permutation_p(np.array(list(train_pair.values()), dtype=np.float64))

        val_delta: float | None = None
        if val_cells is not None:
            val_delta, val_pair = val_cells.get((key.bucket, key.column), (float("nan"), {}))
            val_agree, val_total = count_confirm(
                confirmation_mode,
                train_delta,
                train_pair,
                val_pair,
            )
        else:
            val_agree, val_total = None, 0

        rows.append(
            EmaSignalSummaryRow(
                key=key,
                train_delta_pp=train_delta,
                val_delta_pp=val_delta,
                p_value=p_value,
                val_agree=val_agree,
                val_total=val_total,
                status=signal_status(
                    p_value,
                    val_agree,
                    val_total,
                    bonferroni_n=BONFERRONI_N,
                    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
                    train_effect_pp=train_delta,
                ),
            )
        )
    return rows


def compute_ema_signal_summary(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame | None,
    train_bands: dict[str, MeanBands],
    val_bands: dict[str, MeanBands] | None,
    *,
    period: int = SELECTED_EMA_PERIOD,
    periods: tuple[int, ...] = (SELECTED_EMA_PERIOD,),
    confirmation_mode: ConfirmationMode = ConfirmationMode.COHORT,
) -> list[EmaSignalSummaryRow]:
    train_cells = compute_all_cell_deltas_pp(
        train_daily, period, train_bands, periods, columns=SCREEN_STATS_COLS
    )
    val_cells = None
    if val_daily is not None and val_bands is not None:
        val_cells = compute_all_cell_deltas_pp(
            val_daily, period, val_bands, periods, columns=SCREEN_STATS_COLS
        )
    return _summary_rows_from_deltas(
        train_cells, val_cells, confirmation_mode=confirmation_mode
    )


@dataclass(frozen=True)
class EmaSignalPooledRow:
    key: EmaSignalKey
    delta_pp: float
    status: str


def compute_ema_signal_pooled_summary(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    status_by_key: dict[tuple[int, str], str],
    period: int = SELECTED_EMA_PERIOD,
    periods: tuple[int, ...] = (SELECTED_EMA_PERIOD,),
) -> list[EmaSignalPooledRow]:
    cells = compute_all_cell_deltas_pp(
        daily, period, pair_bands, periods, columns=SCREEN_STATS_COLS
    )
    return [
        EmaSignalPooledRow(
            key=key,
            delta_pp=cells.get((key.bucket, key.column), (float("nan"), {}))[0],
            status=status_by_key.get((key.bucket, key.column), "не значим"),
        )
        for key in _signal_keys()
    ]


def format_signal_summary_table(
    rows: list[EmaSignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> list[str]:
    headers = (
        "Сигнал",
        f"Δ train, п.п.",
        f"Δ val, п.п.",
        "Δ (val−train)",
        "p-value (train)",
        "Подтверждение на val",
        "Статус",
    )
    body = [
        (
            row.key.label,
            fmt_delta_pp(row.train_delta_pp),
            fmt_delta_pp(row.val_delta_pp),
            fmt_delta_pp(row.delta_val_train_pp),
            fmt_p(row.p_value),
            val_confirm_text(row.val_agree, row.val_total),
            row.status,
        )
        for row in rows
    ]
    col_w = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(cell))

    header = " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
    sep = "-" * len(header)
    lines = [
        title,
        "",
        f"EMA({ema_period}). Δ — отклонение доли колонки от BASE (п.п.), как в блоке Δ ema_spreads.",
        f"Материальность на train: |Δ| ≥ {SCREEN_MIN_POOLED_DELTA_PP:g} п.п.",
        "p-value (train) — permutation по Δ каждой train-пары (sign-flip), H₀: mean=0.",
        f"Поправка Бонферрони: α={ALPHA}, порог «значим» p < {BONF_ALPHA:.4f} ({ALPHA}/{BONFERRONI_N}).",
        val_confirm_hint,
        "",
        header,
        sep,
    ]
    for cells in body:
        lines.append(" | ".join(f"{cell:<{col_w[i]}}" for i, cell in enumerate(cells)))
    lines.append("")
    return lines


def format_pooled_signal_table(
    rows: list[EmaSignalPooledRow],
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> list[str]:
    headers = ("Сигнал", "Δ, п.п.", "Статус")
    body = [
        (row.key.label, fmt_delta_pp(row.delta_pp), row.status)
        for row in rows
    ]
    col_w = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(cell))
    header = " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
    sep = "-" * len(header)
    lines = [
        "=== Сводная таблица: полный пул (все пары, весь период) ===",
        "",
        f"EMA({ema_period}). Δ — pooled отклонение доли колонки от BASE на полном периоде.",
        "Статус: пересечение статусов из «универсальность среди пар» и «устойчивость во времени».",
        "",
        header,
        sep,
    ]
    for cells in body:
        lines.append(" | ".join(f"{cell:<{col_w[i]}}" for i, cell in enumerate(cells)))
    lines.append("")
    return lines


def status_by_signal_key(
    pair_rows: list[EmaSignalSummaryRow],
    temporal_rows: list[EmaSignalSummaryRow],
) -> dict[tuple[int, str], str]:
    return {
        (pair_row.key.bucket, pair_row.key.column): intersect_status(
            pair_row.status, temporal_row.status
        )
        for pair_row, temporal_row in zip(pair_rows, temporal_rows, strict=True)
    }
