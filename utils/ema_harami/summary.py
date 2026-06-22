"""Сводка train→val для ema_harami."""

from __future__ import annotations

from crypto_research.utils.ema_harami.constants import N_EMA_HARAMI_SCENARIOS, SCENARIO_ROWS, SELECTED_EMA_PERIOD
from crypto_research.utils.ema_harami.signal_metrics import compute_ema_harami_deltas
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

EMA_HARAMI_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=N_EMA_HARAMI_SCENARIOS,
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: SCENARIO_ROWS[b],
    indicator_label=lambda p: f"ema_harami EMA({p})",
)


def compute_ema_harami_signal_summary(
    *args,
    period: int = SELECTED_EMA_PERIOD,
    **kwargs,
) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        EMA_HARAMI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_ema_harami_deltas,
        compute_kwargs={"periods": (period,)},
        **kwargs,
    )


def compute_ema_harami_signal_pooled_summary(
    *args,
    period: int = SELECTED_EMA_PERIOD,
    **kwargs,
) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        EMA_HARAMI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_ema_harami_deltas,
        compute_kwargs={"periods": (period,)},
        **kwargs,
    )


def format_ema_harami_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> list[str]:
    spec = EMA_HARAMI_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(ema_period),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_ema_harami_pooled_signal_table(
    rows: list[SignalPooledRow],
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> list[str]:
    return format_pooled_signal_table(
        rows, indicator=EMA_HARAMI_SIGNAL_SPEC.indicator_label(ema_period)
    )
