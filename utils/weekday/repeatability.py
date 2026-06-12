"""Повторяемость (X/Y) и поддержка [пары] в ячейках таблицы."""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_YEAR_BASE_DAYS = 40
MIN_YEAR_ROW_DAYS = 15
MIN_QUARTER_BASE_DAYS = 20
MIN_QUARTER_ROW_DAYS = 8
MIN_MONTH_BASE_DAYS = 10
MIN_MONTH_ROW_DAYS = 5
GREEN_DELTA_EPS = 0.001


def years_from_frame(work: pl.DataFrame) -> np.ndarray:
    if "day_utc" not in work.columns or work.height == 0:
        return np.full(work.height, -1, dtype=np.int32)
    return (
        work.with_columns(pl.col("day_utc").dt.year().alias("_yr"))["_yr"]
        .to_numpy()
        .astype(np.int32, copy=False)
    )


def quarters_from_frame(work: pl.DataFrame) -> np.ndarray:
    if "day_utc" not in work.columns or work.height == 0:
        return np.full(work.height, -1, dtype=np.int32)
    yq = work.with_columns(
        (
            pl.col("day_utc").dt.year() * 10 + pl.col("day_utc").dt.quarter()
        ).alias("_yq")
    )["_yq"]
    return yq.to_numpy().astype(np.int32, copy=False)


def months_from_frame(work: pl.DataFrame) -> np.ndarray:
    if "day_utc" not in work.columns or work.height == 0:
        return np.full(work.height, -1, dtype=np.int32)
    ym = work.with_columns(
        (pl.col("day_utc").dt.year() * 100 + pl.col("day_utc").dt.month()).alias("_ym")
    )["_ym"]
    return ym.to_numpy().astype(np.int32, copy=False)


def _compute_cell_repeatability_for_groups(
    groups: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    min_base_days: int,
    min_row_days: int,
    scope_mask: np.ndarray | None = None,
    group_year_min: int | None = None,
    group_year_max: int | None = None,
) -> str:
    if scope_mask is not None:
        valid = valid & scope_mask
    base_mask = valid
    row_mask = valid & (buckets == row_index)
    if not np.any(base_mask) or not np.any(row_mask):
        return "n/a"

    pooled_delta = float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())
    match = 0
    total = 0
    for group in np.unique(groups[base_mask]):
        if group < 0:
            continue
        if group_year_min is not None and group < group_year_min:
            continue
        if group_year_max is not None and group > group_year_max:
            continue
        g_base = base_mask & (groups == group)
        g_row = row_mask & (groups == group)
        n_base = int(g_base.sum())
        n_row = int(g_row.sum())
        if n_base < min_base_days or n_row < min_row_days:
            continue
        delta_g = float(hit_mask[g_row].mean() - hit_mask[g_base].mean())
        total += 1
        if pooled_delta >= GREEN_DELTA_EPS:
            if delta_g >= GREEN_DELTA_EPS:
                match += 1
        elif pooled_delta <= -GREEN_DELTA_EPS:
            if delta_g <= -GREEN_DELTA_EPS:
                match += 1
        elif abs(delta_g) < 0.01:
            match += 1
    if total == 0:
        return "n/a"
    return f"{match}/{total}"


def compute_cell_year_repeatability(
    years: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    scope_mask: np.ndarray | None = None,
    group_year_min: int | None = None,
    group_year_max: int | None = None,
) -> str:
    return _compute_cell_repeatability_for_groups(
        groups=years,
        buckets=buckets,
        row_index=row_index,
        valid=valid,
        hit_mask=hit_mask,
        min_base_days=MIN_YEAR_BASE_DAYS,
        min_row_days=MIN_YEAR_ROW_DAYS,
        scope_mask=scope_mask,
        group_year_min=group_year_min,
        group_year_max=group_year_max,
    )


def compute_cell_quarter_repeatability(
    quarters: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    scope_mask: np.ndarray | None = None,
) -> str:
    return _compute_cell_repeatability_for_groups(
        groups=quarters,
        buckets=buckets,
        row_index=row_index,
        valid=valid,
        hit_mask=hit_mask,
        min_base_days=MIN_QUARTER_BASE_DAYS,
        min_row_days=MIN_QUARTER_ROW_DAYS,
        scope_mask=scope_mask,
    )


def compute_cell_month_repeatability(
    months: np.ndarray,
    buckets: np.ndarray,
    row_index: int,
    valid: np.ndarray,
    hit_mask: np.ndarray,
    scope_mask: np.ndarray | None = None,
) -> str:
    return _compute_cell_repeatability_for_groups(
        groups=months,
        buckets=buckets,
        row_index=row_index,
        valid=valid,
        hit_mask=hit_mask,
        min_base_days=MIN_MONTH_BASE_DAYS,
        min_row_days=MIN_MONTH_ROW_DAYS,
        scope_mask=scope_mask,
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
