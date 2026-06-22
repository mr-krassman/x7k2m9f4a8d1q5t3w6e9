"""Сводка train→val для сигналов volume."""

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
from crypto_research.utils.volume.constants import (
    N_VOLUME_BUCKETS,
    SELECTED_VOLUME_EMA_PERIOD,
    VOLUME_BUCKET_LABELS,
)
from crypto_research.utils.volume.signal_metrics import compute_volume_deltas

VOLUME_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=N_VOLUME_BUCKETS,
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: VOLUME_BUCKET_LABELS[b],
    indicator_label=lambda p: f"EMA(volume,{p})",
)


def compute_volume_signal_summary(
    *args, period: int = SELECTED_VOLUME_EMA_PERIOD, **kwargs
) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        VOLUME_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_volume_deltas,
        **kwargs,
    )


def compute_volume_signal_pooled_summary(
    *args, period: int = SELECTED_VOLUME_EMA_PERIOD, **kwargs
) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        VOLUME_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_volume_deltas,
        **kwargs,
    )


def format_volume_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
    vol_period: int = SELECTED_VOLUME_EMA_PERIOD,
) -> list[str]:
    spec = VOLUME_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(vol_period),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_volume_pooled_signal_table(
    rows: list[SignalPooledRow],
    *,
    vol_period: int = SELECTED_VOLUME_EMA_PERIOD,
) -> list[str]:
    return format_pooled_signal_table(
        rows,
        indicator=VOLUME_SIGNAL_SPEC.indicator_label(vol_period),
    )
