"""Сводка train→val для сигналов price_sequences."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import SCREEN_MIN_POOLED_DELTA_PP
from crypto_research.utils.price_sequences.constants import SCENARIO_ROWS
from crypto_research.utils.price_sequences.signal_metrics import compute_price_sequence_deltas
from crypto_research.utils.research.spread_summary import (
    SignalPooledRow,
    SignalSummaryRow,
    SpreadSignalSpec,
    compute_signal_pooled_summary,
    compute_signal_summary,
    format_pooled_signal_table,
    format_signal_summary_table,
    status_by_signal_key,
)

PRICE_SEQUENCE_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=len(SCENARIO_ROWS),
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: SCENARIO_ROWS[b],
    indicator_label=lambda _p: "price_sequences",
)


def compute_price_sequence_signal_summary(*args, period: int = 0, **kwargs) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        PRICE_SEQUENCE_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_price_sequence_deltas,
        **kwargs,
    )


def compute_price_sequence_signal_pooled_summary(
    *args, period: int = 0, **kwargs
) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        PRICE_SEQUENCE_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_price_sequence_deltas,
        **kwargs,
    )


def format_price_sequence_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
) -> list[str]:
    spec = PRICE_SEQUENCE_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(0),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_price_sequence_pooled_signal_table(rows: list[SignalPooledRow]) -> list[str]:
    return format_pooled_signal_table(rows, indicator=PRICE_SEQUENCE_SIGNAL_SPEC.indicator_label(0))
