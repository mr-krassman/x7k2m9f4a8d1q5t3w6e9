"""Классификация дневного return по μ±50% (общая математика для всех исследований)."""

from __future__ import annotations

import numpy as np

from crypto_research.utils.weekday.bands import MeanBands

N_RETURN_TAGS = 12

# Порядок колонок ema_spreads (RETURN_STATS_COLS) отличается от weekday (STATS_COLS) для индексов 1–3.
EMA_COLUMN_PERMUTATION: tuple[int, ...] = (0, 3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11)


def classify_return_tag_indices(
    ret: float,
    bands: MeanBands,
    day_open: float | None = None,
    day_high: float | None = None,
    day_low: float | None = None,
) -> set[int]:
    """Индексы тегов в порядке weekday STATS_COLS (0=close up, 1=strong, 2=moderate, 3=weak, …)."""
    tags: set[int] = set()
    if ret > 0:
        tags.add(0)
        if ret <= bands.up_lo:
            tags.add(3)
        elif ret <= bands.up_hi:
            tags.add(2)
        else:
            tags.add(1)
    elif ret < 0:
        tags.add(4)
        if ret >= bands.down_lo:
            tags.add(5)
        elif ret >= bands.down_hi:
            tags.add(6)
        else:
            tags.add(7)

    if (
        day_open is not None
        and day_high is not None
        and day_low is not None
        and day_open > 0
    ):
        up_move = (day_high - day_open) / day_open * 100.0
        down_move = (day_low - day_open) / day_open * 100.0
        if up_move >= bands.up_lo:
            tags.add(8)
        if up_move > bands.up_hi:
            tags.add(9)
        if down_move <= bands.down_lo:
            tags.add(10)
        if down_move < bands.down_hi:
            tags.add(11)
    return tags


def classify_return_pct(
    ret: float,
    bands: MeanBands,
    day_open: float | None = None,
    day_high: float | None = None,
    day_low: float | None = None,
    *,
    columns: tuple[str, ...],
) -> set[str]:
    return {columns[i] for i in classify_return_tag_indices(ret, bands, day_open, day_high, day_low)}


def bands_to_arrays(
    pair_bands: dict[str, MeanBands],
    pair_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    default = next(iter(pair_bands.values()))
    up_lo = np.empty(pair_keys.shape[0], dtype=np.float64)
    up_hi = np.empty(pair_keys.shape[0], dtype=np.float64)
    down_lo = np.empty(pair_keys.shape[0], dtype=np.float64)
    down_hi = np.empty(pair_keys.shape[0], dtype=np.float64)
    cache: dict[str, MeanBands] = {}
    for i, key in enumerate(pair_keys):
        pair = str(key)
        if pair not in cache:
            cache[pair] = pair_bands.get(pair, default)
        bands = cache[pair]
        up_lo[i] = bands.up_lo
        up_hi[i] = bands.up_hi
        down_lo[i] = bands.down_lo
        down_hi[i] = bands.down_hi
    return up_lo, up_hi, down_lo, down_hi


def build_return_hit_matrix_canonical(
    returns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    pair_keys: np.ndarray,
    pair_bands: dict[str, MeanBands],
) -> np.ndarray:
    """Матрица (n, 12) bool в порядке weekday STATS_COLS."""
    n = returns.shape[0]
    hits = np.zeros((n, N_RETURN_TAGS), dtype=bool)
    up_lo, up_hi, down_lo, down_hi = bands_to_arrays(pair_bands, pair_keys)

    pos = returns > 0
    neg = returns < 0
    hits[:, 0] = pos
    hits[:, 1] = pos & (returns > up_hi)
    hits[:, 2] = pos & (returns > up_lo) & (returns <= up_hi)
    hits[:, 3] = pos & (returns <= up_lo)
    hits[:, 4] = neg
    hits[:, 5] = neg & (returns >= down_lo)
    hits[:, 6] = neg & (returns < down_lo) & (returns >= down_hi)
    hits[:, 7] = neg & (returns < down_hi)

    valid_ohlc = (opens > 0) & np.isfinite(opens) & np.isfinite(highs) & np.isfinite(lows)
    up_move = np.where(valid_ohlc, (highs - opens) / opens * 100.0, np.nan)
    down_move = np.where(valid_ohlc, (lows - opens) / opens * 100.0, np.nan)
    hits[:, 8] = valid_ohlc & (up_move >= up_lo)
    hits[:, 9] = valid_ohlc & (up_move > up_hi)
    hits[:, 10] = valid_ohlc & (down_move <= down_lo)
    hits[:, 11] = valid_ohlc & (down_move < down_hi)
    return hits


def build_return_hit_matrix(
    returns: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    pair_keys: np.ndarray,
    pair_bands: dict[str, MeanBands],
    *,
    column_permutation: tuple[int, ...] | None = None,
) -> np.ndarray:
    hits = build_return_hit_matrix_canonical(
        returns, opens, highs, lows, pair_keys, pair_bands
    )
    if column_permutation is not None:
        hits = hits[:, column_permutation]
    return hits


def hit_masks_from_matrix(hit_matrix: np.ndarray, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {col: hit_matrix[:, i] for i, col in enumerate(columns)}
