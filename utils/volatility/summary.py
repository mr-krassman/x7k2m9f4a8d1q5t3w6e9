"""Сводка train→val для сигналов volatility."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import SCREEN_MIN_POOLED_DELTA_PP
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
from crypto_research.utils.volatility.constants import (
    N_VOLATILITY_BUCKETS,
    SELECTED_RANGE_SMA_PERIOD,
    VOLATILITY_SCENARIO_ROWS,
)
from crypto_research.utils.volatility.signal_metrics import compute_volatility_deltas

VOLATILITY_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=N_VOLATILITY_BUCKETS,
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: VOLATILITY_SCENARIO_ROWS[b],
    indicator_label=lambda p: f"SMA(range,{p})",
)


def compute_volatility_signal_summary(
    *args, period: int = SELECTED_RANGE_SMA_PERIOD, **kwargs
) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        VOLATILITY_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_volatility_deltas,
        **kwargs,
    )


def compute_volatility_signal_pooled_summary(
    *args, period: int = SELECTED_RANGE_SMA_PERIOD, **kwargs
) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        VOLATILITY_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_volatility_deltas,
        **kwargs,
    )


def format_volatility_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    sma_period: int = SELECTED_RANGE_SMA_PERIOD,
) -> list[str]:
    spec = VOLATILITY_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(sma_period),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_volatility_pooled_signal_table(
    rows: list[SignalPooledRow],
    *,
    sma_period: int = SELECTED_RANGE_SMA_PERIOD,
) -> list[str]:
    return format_pooled_signal_table(
        rows,
        indicator=VOLATILITY_SIGNAL_SPEC.indicator_label(sma_period),
    )
