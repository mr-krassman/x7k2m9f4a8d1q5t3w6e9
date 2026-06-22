"""Метрики стабильности периода SMA(range)."""

from __future__ import annotations

import polars as pl

from crypto_research.utils.pipeline.period_screen_core import (
    MaterialCell,
    PeriodStabilityMetrics,
    YearlyMaterialAgreement,
    compute_period_stability as _compute_period_stability,
    material_cells_for_period as _material_cells_for_period,
    rank_period_stabilities,
    yearly_material_agreement_rates as _yearly_material_agreement_rates,
)
from crypto_research.utils.volatility.conditional_table import (
    prepare_volatility_condition_frame,
    volatility_bucket_labels,
)
from crypto_research.utils.volatility.constants import N_VOLATILITY_BUCKETS, SCREEN_MIN_POOLED_DELTA_PP
from crypto_research.utils.weekday.bands import MeanBands

_SCREEN_KW = {
    "prepare_frame": prepare_volatility_condition_frame,
    "n_buckets": N_VOLATILITY_BUCKETS,
    "min_delta_pp": SCREEN_MIN_POOLED_DELTA_PP,
}


def compute_period_stability(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> PeriodStabilityMetrics | None:
    return _compute_period_stability(daily, period, pair_bands, **_SCREEN_KW)


def material_cells_for_period(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> tuple[list[MaterialCell], list[str]]:
    cells = _material_cells_for_period(daily, period, pair_bands, **_SCREEN_KW)
    return cells, volatility_bucket_labels()


def yearly_material_agreement_rates(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> list[YearlyMaterialAgreement]:
    return _yearly_material_agreement_rates(daily, period, pair_bands, **_SCREEN_KW)
