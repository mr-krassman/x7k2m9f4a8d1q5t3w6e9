"""Δ к BASE (п.п.) по ячейке паттерн×колонка."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.candlestick_patterns.harami.constants import SCENARIO_ROWS
from crypto_research.utils.candlestick_patterns.harami.condition_frame import prepare_harami_frame
from crypto_research.utils.ema_spreads.constants import SCREEN_STATS_COLS
from crypto_research.utils.research.spread_signal_metrics import compute_all_cell_deltas_pp as _compute
from crypto_research.utils.weekday.bands import MeanBands

N_SCENARIO_ROWS = len(SCENARIO_ROWS)


def compute_harami_deltas(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    period: int = 0,
    columns: tuple[str, ...] = SCREEN_STATS_COLS,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    return _compute(
        daily,
        period,
        pair_bands,
        n_buckets=N_SCENARIO_ROWS,
        columns=columns,
        prepare_frame=prepare_harami_frame,
    )
