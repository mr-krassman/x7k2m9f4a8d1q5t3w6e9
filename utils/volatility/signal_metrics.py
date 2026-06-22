"""Δ к BASE (п.п.) по ячейке бакет×колонка volatility."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.ema_spreads.constants import SCREEN_STATS_COLS
from crypto_research.utils.research.spread_signal_metrics import compute_all_cell_deltas_pp as _compute
from crypto_research.utils.volatility.conditional_table import prepare_volatility_condition_frame
from crypto_research.utils.volatility.constants import N_VOLATILITY_BUCKETS
from crypto_research.utils.weekday.bands import MeanBands


def compute_volatility_deltas(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    period: int,
    columns: tuple[str, ...] = SCREEN_STATS_COLS,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    return _compute(
        daily,
        period,
        pair_bands,
        n_buckets=N_VOLATILITY_BUCKETS,
        columns=columns,
        prepare_frame=prepare_volatility_condition_frame,
        prepare_kwargs={"periods": (period,)},
    )
