"""Δ к BASE (п.п.) по ячейке бакет×колонка — pooled и по парам."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from crypto_research.utils.weekday.bands import MeanBands


def _delta_pp(hit_mask: np.ndarray, row_mask: np.ndarray, base_mask: np.ndarray) -> float:
    if not np.any(base_mask) or not np.any(row_mask):
        return float("nan")
    return float((hit_mask[row_mask].mean() - hit_mask[base_mask].mean()) * 100.0)


def compute_all_cell_deltas_pp(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    *,
    n_buckets: int,
    columns: tuple[str, ...],
    prepare_frame: Callable[..., tuple | None],
    prepare_kwargs: dict | None = None,
) -> dict[tuple[int, str], tuple[float, dict[str, float]]]:
    prepared = prepare_frame(daily, period, pair_bands, **(prepare_kwargs or {}))
    if prepared is None:
        return {}
    work, buckets, valid, hit_masks, *_rest = prepared
    pair_keys = work["pair"].to_numpy().astype(object, copy=False)
    base_mask = valid
    pairs = [str(p) for p in np.unique(pair_keys[valid])]

    out: dict[tuple[int, str], tuple[float, dict[str, float]]] = {}
    for bucket in range(n_buckets):
        row_mask = valid & (buckets == bucket)
        for column in columns:
            if column not in hit_masks:
                continue
            hit = hit_masks[column]
            pooled = _delta_pp(hit, row_mask, base_mask)
            per_pair: dict[str, float] = {}
            for pair in pairs:
                p_base = valid & (pair_keys == pair)
                p_row = p_base & (buckets == bucket)
                if not p_row.any() or not p_base.any():
                    continue
                per_pair[pair] = _delta_pp(hit, p_row, p_base)
            out[(bucket, column)] = (pooled, per_pair)
    return out
