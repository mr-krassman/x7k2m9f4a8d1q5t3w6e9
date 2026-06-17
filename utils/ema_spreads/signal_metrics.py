"""Δ к BASE (п.п.) по ячейке бакет×колонка — pooled и по парам."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.ema_spreads.conditional_table import prepare_ema_condition_frame
from crypto_research.utils.ema_spreads.constants import N_EMA_SCENARIOS, RETURN_STATS_COLS
from crypto_research.utils.research.spread_signal_metrics import compute_all_cell_deltas_pp as _compute
from crypto_research.utils.weekday.bands import MeanBands


def compute_all_cell_deltas_pp(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
    columns: tuple[str, ...] = RETURN_STATS_COLS,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    return _compute(
        daily,
        period,
        pair_bands,
        n_buckets=N_EMA_SCENARIOS,
        columns=columns,
        prepare_frame=prepare_ema_condition_frame,
        prepare_kwargs={"periods": periods},
    )


def compute_cell_delta_pp(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
    bucket: int,
    column: str,
) -> tuple[float, dict[str, float]]:
    cells = compute_all_cell_deltas_pp(
        daily, period, pair_bands, periods, columns=(column,)
    )
    return cells.get((bucket, column), (float("nan"), {}))
