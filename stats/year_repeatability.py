"""Повторяемость сигнала по календарным годам (доля vs BASE)."""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_YEAR_BASE_DAYS = 40
MIN_YEAR_ROW_DAYS = 15
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
