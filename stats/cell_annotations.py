"""Годы (X/Y) и [пары] для ячеек таблиц weekday."""

from __future__ import annotations

import numpy as np

from crypto_research.stats.year_repeatability import (
    GREEN_DELTA_EPS,
    MIN_YEAR_BASE_DAYS,
    MIN_YEAR_ROW_DAYS,
    compute_cell_month_repeatability,
    compute_cell_year_repeatability,
)


def _compute_cell_pair_support_scoped(
    pair_keys: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    *,
    min_base_days: int = MIN_YEAR_BASE_DAYS,
    min_row_days: int = MIN_YEAR_ROW_DAYS,
) -> int | None:
    base_mask = valid
    row_mask = valid & (buckets == row_index)
    if not np.any(base_mask) or not np.any(row_mask):
        return None

    pooled_delta = float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())
    support = 0
    eligible = 0
    for pair in np.unique(pair_keys[base_mask]):
        p_base = base_mask & (pair_keys == pair)
        p_row = row_mask & (pair_keys == pair)
        if int(p_base.sum()) < min_base_days or int(p_row.sum()) < min_row_days:
            continue
        delta_p = float(hit_mask[p_row].mean() - hit_mask[p_base].mean())
        eligible += 1
        if pooled_delta >= GREEN_DELTA_EPS:
            if delta_p >= GREEN_DELTA_EPS:
                support += 1
        elif pooled_delta <= -GREEN_DELTA_EPS:
            if delta_p <= -GREEN_DELTA_EPS:
                support += 1
        elif abs(delta_p) < 0.01:
            support += 1
    if eligible == 0:
        return None
    return support


def cell_year_month_pair_annotations(
    years: np.ndarray,
    months: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    pair_keys: np.ndarray | None,
) -> tuple[str, str, int | None, int | None]:
    y_rep = compute_cell_year_repeatability(
        years, buckets, row_index, valid, hit_mask
    )
    m_rep = compute_cell_month_repeatability(
        months, buckets, row_index, valid, hit_mask
    )
    p_sup: int | None = None
    if pair_keys is not None:
        p_sup = _compute_cell_pair_support_scoped(
            pair_keys, buckets, row_index, valid, hit_mask
        )
    return y_rep, m_rep, p_sup, None


def annotations_for_columns(
    years: np.ndarray,
    months: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_masks: dict[str, np.ndarray],
    pair_keys: np.ndarray | None,
    columns: list[str],
) -> tuple[list[str], list[str], list[int | None], list[int | None]]:
    year_reps: list[str] = []
    month_reps: list[str] = []
    pair_supports: list[int | None] = []
    pair_validates: list[int | None] = []
    for col in columns:
        y, m, ps, pv = cell_year_month_pair_annotations(
            years,
            months,
            buckets,
            row_index,
            valid,
            hit_masks[col],
            pair_keys,
        )
        year_reps.append(y)
        month_reps.append(m)
        pair_supports.append(ps)
        pair_validates.append(pv)
    return year_reps, month_reps, pair_supports, pair_validates
