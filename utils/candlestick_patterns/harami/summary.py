"""Сводка train→val для сигналов harami."""

from __future__ import annotations

from crypto_research.utils.candlestick_patterns.harami.constants import SCENARIO_ROWS
from crypto_research.utils.candlestick_patterns.harami.signal_metrics import compute_harami_deltas
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

HARAMI_SIGNAL_SPEC = SpreadSignalSpec(
    n_buckets=len(SCENARIO_ROWS),
    min_effect_pp=SCREEN_MIN_POOLED_DELTA_PP,
    bucket_label=lambda b: SCENARIO_ROWS[b],
    indicator_label=lambda _p: "harami",
)


def compute_harami_signal_summary(*args, period: int = 0, **kwargs) -> list[SignalSummaryRow]:
    return compute_signal_summary(
        HARAMI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_harami_deltas,
        **kwargs,
    )


def compute_harami_signal_pooled_summary(*args, period: int = 0, **kwargs) -> list[SignalPooledRow]:
    return compute_signal_pooled_summary(
        HARAMI_SIGNAL_SPEC,
        *args,
        period=period,
        compute_deltas=compute_harami_deltas,
        **kwargs,
    )


def format_harami_signal_summary_table(
    rows: list[SignalSummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
) -> list[str]:
    spec = HARAMI_SIGNAL_SPEC
    return format_signal_summary_table(
        rows,
        title=title,
        val_confirm_hint=val_confirm_hint,
        indicator=spec.indicator_label(0),
        min_effect_pp=spec.min_effect_pp,
        bonferroni_n=spec.bonferroni_n,
        bonf_alpha=spec.bonf_alpha,
    )


def format_harami_pooled_signal_table(rows: list[SignalPooledRow]) -> list[str]:
    return format_pooled_signal_table(rows, indicator=HARAMI_SIGNAL_SPEC.indicator_label(0))
