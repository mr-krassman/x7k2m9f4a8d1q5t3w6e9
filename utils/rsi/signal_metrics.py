"""Δ к BASE (п.п.) по ячейке бакет×колонка RSI."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.ema_spreads.constants import RETURN_STATS_COLS, SCREEN_STATS_COLS
from crypto_research.utils.research.spread_signal_metrics import compute_all_cell_deltas_pp as _compute
from crypto_research.utils.rsi.conditional_table import prepare_rsi_condition_frame
from crypto_research.utils.rsi.constants import N_RSI_QUANTILES
from crypto_research.utils.weekday.bands import MeanBands


def compute_all_cell_deltas_pp(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    columns: tuple[str, ...] = SCREEN_STATS_COLS,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    return _compute(
        daily,
        period,
        pair_bands,
        n_buckets=N_RSI_QUANTILES,
        columns=columns,
        prepare_frame=prepare_rsi_condition_frame,
    )


def compute_rsi_deltas(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    period: int,
    columns: tuple[str, ...] = SCREEN_STATS_COLS,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    return compute_all_cell_deltas_pp(daily, period, pair_bands, columns=columns)
