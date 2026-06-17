"""Сводка train→val для сигналов spread-исследований (бакет × колонка)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import SCREEN_STATS_COLS
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


@dataclass(frozen=True)
class SpreadSignalSpec:
    n_buckets: int
    min_effect_pp: float
    bucket_label: Callable[[int], str]
    indicator_label: Callable[[int], str]

    @property
    def bonferroni_n(self) -> int:
        return self.n_buckets * len(SCREEN_STATS_COLS)

    @property
    def bonf_alpha(self) -> float:
        return bonferroni_alpha(self.bonferroni_n)

    def signal_label(self, bucket: int, column: str) -> str:
        return f"b{bucket} {self.bucket_label(bucket)} × {column}"


@dataclass(frozen=True)
class SignalKey:
    bucket: int
    column: str
    label: str


@dataclass(frozen=True)
class SignalSummaryRow:
    key: SignalKey
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


@dataclass(frozen=True)
class SignalPooledRow:
    key: SignalKey
    delta_pp: float
    status: str


def _signal_keys(spec: SpreadSignalSpec) -> list[SignalKey]:
    return [
        SignalKey(
            bucket=b,
            column=col,
            label=spec.signal_label(b, col),
        )
        for b in range(spec.n_buckets)
        for col in SCREEN_STATS_COLS
    ]


def _summary_rows_from_deltas(
    spec: SpreadSignalSpec,
    train_cells: dict[tuple[int, str], tuple[float, dict[str, float]]],
    val_cells: dict[tuple[int, str], tuple[float, dict[str, float]]] | None,
    *,
    confirmation_mode: ConfirmationMode,
) -> list[SignalSummaryRow]:
    rows: list[SignalSummaryRow] = []
    for key in _signal_keys(spec):
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
            SignalSummaryRow(
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
                    bonferroni_n=spec.bonferroni_n,
                    min_effect_pp=spec.min_effect_pp,
                    train_effect_pp=train_delta,
                ),
            )
        )
    return rows


def compute_signal_summary(
    spec: SpreadSignalSpec,
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame | None,
    train_bands: dict[str, MeanBands],
    val_bands: dict[str, MeanBands] | None,
    *,
    period: int,
    compute_deltas: Callable[..., dict[tuple[int, str], tuple[float, dict[str, float]]]],
    compute_kwargs: dict | None = None,
    confirmation_mode: ConfirmationMode = ConfirmationMode.COHORT,
) -> list[SignalSummaryRow]:
    kwargs = {"period": period, **(compute_kwargs or {})}
    train_cells = compute_deltas(train_daily, train_bands, **kwargs)
    val_cells = None
    if val_daily is not None and val_bands is not None:
        val_cells = compute_deltas(val_daily, val_bands, **kwargs)
    return _summary_rows_from_deltas(
        spec, train_cells, val_cells, confirmation_mode=confirmation_mode
    )


def compute_signal_pooled_summary(
    spec: SpreadSignalSpec,
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    status_by_key: dict[tuple[int, str], str],
    period: int,
    compute_deltas: Callable[..., dict[tuple[int, str], tuple[float, dict[str, float]]]],
    compute_kwargs: dict | None = None,
) -> list[SignalPooledRow]:
    kwargs = {"period": period, **(compute_kwargs or {})}
    cells = compute_deltas(daily, pair_bands, **kwargs)
    return [
        SignalPooledRow(
            key=key,
            delta_pp=cells.get((key.bucket, key.column), (float("nan"), {}))[0],
            status=status_by_key.get((key.bucket, key.column), "не значим"),
        )
        for key in _signal_keys(spec)
    ]


def format_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    indicator: str,
    min_effect_pp: float,
    bonferroni_n: int,
    bonf_alpha: float,
) -> list[str]:
    headers = (
        "Сигнал",
        "Δ train, п.п.",
        "Δ val, п.п.",
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
        f"{indicator}. Δ — отклонение доли колонки от BASE (п.п.), как в блоке Δ spread.",
        f"Материальность на train: |Δ| ≥ {min_effect_pp:g} п.п.",
        "p-value (train) — permutation по Δ каждой train-пары (sign-flip), H₀: mean=0.",
        f"Поправка Бонферрони: α={ALPHA}, порог «значим» p < {bonf_alpha:.4f} ({ALPHA}/{bonferroni_n}).",
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
    rows: list[SignalPooledRow],
    *,
    indicator: str,
) -> list[str]:
    headers = ("Сигнал", "Δ, п.п.", "Статус")
    body = [(row.key.label, fmt_delta_pp(row.delta_pp), row.status) for row in rows]
    col_w = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(cell))
    header = " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
    sep = "-" * len(header)
    lines = [
        "=== Сводная таблица: полный пул (все пары, весь период) ===",
        "",
        f"{indicator}. Δ — pooled отклонение доли колонки от BASE на полном периоде.",
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
    pair_rows: list[SignalSummaryRow],
    temporal_rows: list[SignalSummaryRow],
) -> dict[tuple[int, str], str]:
    return {
        (pair_row.key.bucket, pair_row.key.column): intersect_status(
            pair_row.status, temporal_row.status
        )
        for pair_row, temporal_row in zip(pair_rows, temporal_rows, strict=True)
    }
