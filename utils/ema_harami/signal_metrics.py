"""Δ к BASE (п.п.) по ячейке EMA×Harami×колонка."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.ema_harami.condition_frame import prepare_ema_harami_frame
from crypto_research.utils.ema_harami.constants import N_EMA_HARAMI_SCENARIOS
from crypto_research.utils.ema_spreads.constants import RETURN_STATS_COLS, SCREEN_STATS_COLS, SELECTED_EMA_PERIOD
from crypto_research.utils.research.spread_signal_metrics import compute_all_cell_deltas_pp as _compute
from crypto_research.utils.weekday.bands import MeanBands


def compute_ema_harami_deltas(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    period: int = SELECTED_EMA_PERIOD,
    columns: tuple[str, ...] = SCREEN_STATS_COLS,
    periods: tuple[int, ...] | None = None,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    ema_periods = periods or (period,)
    return _compute(
        daily,
        period,
        pair_bands,
        n_buckets=N_EMA_HARAMI_SCENARIOS,
        columns=columns,
        prepare_frame=prepare_ema_harami_frame,
        prepare_kwargs={"periods": ema_periods},
    )
