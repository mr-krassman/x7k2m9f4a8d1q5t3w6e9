"""Сводка train→val для сигналов RSI."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import SCREEN_MIN_POOLED_DELTA_PP
from crypto_research.utils.research.spread_summary import (
    SpreadSignalSpec,
    SignalPooledRow,
    SignalSummaryRow,
    compute_signal_pooled_summary,
    compute_signal_summary,
    format_pooled_signal_table,
    format_signal_summary_table,
    status_by_signal_key,
)
from crypto_research.utils.rsi.constants import N_RSI_QUANTILES, SELECTED_RSI_PERIOD
from crypto_research.utils.rsi.signal_metrics import compute_rsi_deltas

RSI_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=N_RSI_QUANTILES,
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: f"Q{b + 1}",
    indicator_label=lambda p: f"RSI({p})",
)


def compute_rsi_signal_summary(*args, period: int = SELECTED_RSI_PERIOD, **kwargs) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        RSI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_rsi_deltas,
        **kwargs,
    )


def compute_rsi_signal_pooled_summary(*args, period: int = SELECTED_RSI_PERIOD, **kwargs) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        RSI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_rsi_deltas,
        **kwargs,
    )


def format_rsi_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    rsi_period: int = SELECTED_RSI_PERIOD,
) -> list[str]:
    spec = RSI_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(rsi_period),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_rsi_pooled_signal_table(
    rows: list[SignalPooledRow],
    *,
    rsi_period: int = SELECTED_RSI_PERIOD,
) -> list[str]:
    return format_pooled_signal_table(
        rows,
        indicator=RSI_SIGNAL_SPEC.indicator_label(rsi_period),
    )
