"""Векторная классификация дневного return для колонок таблицы ema_spreads."""

from __future__ import annotations

import numpy as np

from crypto_research.utils.ema_spreads.constants import RETURN_STATS_COLS
from crypto_research.utils.research.return_classification import (
    EMA_COLUMN_PERMUTATION,
    build_return_hit_matrix as _build_return_hit_matrix,
    hit_masks_from_matrix as _hit_masks_from_matrix,
)
from crypto_research.utils.weekday.bands import MeanBands

N_RETURN_TAGS = len(RETURN_STATS_COLS)


def build_return_hit_matrix(
    returns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    pair_keys: np.ndarray,
    pair_bands: dict[str, MeanBands],
) -> np.ndarray:
    """Матрица (n, 12) bool — попадание в каждую колонку RETURN_STATS_COLS."""
    return _build_return_hit_matrix(
        returns,
        opens,
        highs,
        lows,
        pair_keys,
        pair_bands,
        column_permutation=EMA_COLUMN_PERMUTATION,
    )


def hit_masks_from_matrix(hit_matrix: np.ndarray) -> dict[str, np.ndarray]:
    return _hit_masks_from_matrix(hit_matrix, RETURN_STATS_COLS)
