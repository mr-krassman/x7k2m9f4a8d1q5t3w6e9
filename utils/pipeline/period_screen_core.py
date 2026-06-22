"""Общая логика скрининга периода (EMA / RSI / volume EMA)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import SCREEN_STATS_COLS
from crypto_research.utils.weekday.bands import MeanBands
from crypto_research.utils.weekday.repeatability import (
    GREEN_DELTA_EPS,
    MIN_YEAR_BASE_DAYS,
    MIN_YEAR_ROW_DAYS,
    _compute_cell_pair_support_scoped,
    compute_cell_quarter_repeatability,
    quarters_from_frame,
    years_from_frame,
)

PrepareFrameFn = Callable[
    [pl.DataFrame, int, dict[str, MeanBands]],
    tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None,
]


@dataclass(frozen=True)
class MaterialCell:
    bucket: int
    column: str
    delta_pp: float
    quarters_match: int
    quarters_total: int
    pairs_match: int | None
    pairs_eligible: int | None

    @property
    def quarters_rate(self) -> float | None:
        if self.quarters_total == 0:
            return None
        return self.quarters_match / self.quarters_total

    @property
    def pairs_rate(self) -> float | None:
        if not self.pairs_eligible:
            return None
        return (self.pairs_match or 0) / self.pairs_eligible


@dataclass(frozen=True)
class PeriodStabilityMetrics:
    period: int
    significant_cell_count: int
    avg_abs_delta_pp: float
    avg_quarters_rate: float
    avg_pairs_rate: float
    stability_index_pct: float
    rank: int = 0

    @property
    def avg_quarters_label(self) -> str:
        return f"{self.avg_quarters_rate:.0%}"

    @property
    def avg_pairs_label(self) -> str:
        return f"{self.avg_pairs_rate:.0%}"

    @property
    def avg_abs_delta_label(self) -> str:
        return f"{self.avg_abs_delta_pp:.1f}"


@dataclass(frozen=True)
class YearlyMaterialAgreement:
    year: int
    agreement_rate: float


def _pair_eligible_count(
    pair_keys: np.ndarray,
    buckets: np.ndarray,
    bucket: int,
    valid: np.ndarray,
) -> int | None:
    row_mask = valid & (buckets == bucket)
    if not np.any(valid) or not np.any(row_mask):
        return None
    eligible = 0
    for pair in np.unique(pair_keys[valid]):
        p_base = valid & (pair_keys == pair)
        p_row = p_base & (buckets == bucket)
        if (
            int(p_base.sum()) < MIN_YEAR_BASE_DAYS
            or int(p_row.sum()) < MIN_YEAR_ROW_DAYS
        ):
            continue
        eligible += 1
    return eligible if eligible > 0 else None


def _parse_match_rep(label: str) -> tuple[int, int]:
    if label == "n/a" or "/" not in label:
        return 0, 0
    left, right = label.split("/", maxsplit=1)
    return int(left), int(right)


def _pooled_delta_pp(
    hit_mask: np.ndarray,
    buckets: np.ndarray,
    valid: np.ndarray,
    bucket: int,
) -> float:
    base_mask = valid
    row_mask = valid & (buckets == bucket)
    if not np.any(base_mask) or not np.any(row_mask):
        return float("nan")
    return (float(hit_mask[row_mask].mean()) - float(hit_mask[base_mask].mean())) * 100.0


def _year_sign_agrees(
    hit_mask: np.ndarray,
    years: np.ndarray,
    buckets: np.ndarray,
    valid: np.ndarray,
    bucket: int,
    year: int,
    pooled_delta: float,
) -> bool | None:
    y_mask = years == year
    base_mask = valid & y_mask
    row_mask = valid & y_mask & (buckets == bucket)
    if (
        int(base_mask.sum()) < MIN_YEAR_BASE_DAYS
        or int(row_mask.sum()) < MIN_YEAR_ROW_DAYS
    ):
        return None
    delta_g = float(hit_mask[row_mask].mean() - hit_mask[base_mask].mean())
    if pooled_delta >= GREEN_DELTA_EPS:
        return delta_g >= GREEN_DELTA_EPS
    if pooled_delta <= -GREEN_DELTA_EPS:
        return delta_g <= -GREEN_DELTA_EPS
    return abs(delta_g) < 0.01


def material_cells_for_period(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    *,
    prepare_frame: PrepareFrameFn,
    n_buckets: int,
    min_delta_pp: float,
) -> list[MaterialCell]:
    prepared = prepare_frame(daily, period, pair_bands)
    if prepared is None:
        return []
    work, buckets, valid, hit_masks = prepared
    quarters = quarters_from_frame(work)
    pair_keys = work["pair"].to_numpy().astype(object, copy=False)

    cells: list[MaterialCell] = []
    for bucket in range(n_buckets):
        for col in SCREEN_STATS_COLS:
            delta_pp = _pooled_delta_pp(hit_masks[col], buckets, valid, bucket)
            if not np.isfinite(delta_pp) or abs(delta_pp) < min_delta_pp:
                continue
            q_label = compute_cell_quarter_repeatability(
                quarters, buckets, bucket, valid, hit_masks[col]
            )
            q_match, q_total = _parse_match_rep(q_label)
            cells.append(
                MaterialCell(
                    bucket=bucket,
                    column=col,
                    delta_pp=delta_pp,
                    quarters_match=q_match,
                    quarters_total=q_total,
                    pairs_match=_compute_cell_pair_support_scoped(
                        pair_keys, buckets, bucket, valid, hit_masks[col]
                    ),
                    pairs_eligible=_pair_eligible_count(
                        pair_keys, buckets, bucket, valid
                    ),
                )
            )
    return cells


def compute_period_stability(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    *,
    prepare_frame: PrepareFrameFn,
    n_buckets: int,
    min_delta_pp: float,
) -> PeriodStabilityMetrics | None:
    cells = material_cells_for_period(
        daily,
        period,
        pair_bands,
        prepare_frame=prepare_frame,
        n_buckets=n_buckets,
        min_delta_pp=min_delta_pp,
    )
    if not cells:
        return None
    quarter_rates = [r for c in cells if (r := c.quarters_rate) is not None]
    pair_rates = [r for c in cells if (r := c.pairs_rate) is not None]
    return PeriodStabilityMetrics(
        period=period,
        significant_cell_count=len(cells),
        avg_abs_delta_pp=float(np.mean([abs(c.delta_pp) for c in cells])),
        avg_quarters_rate=float(np.mean(quarter_rates)) if quarter_rates else 0.0,
        avg_pairs_rate=float(np.mean(pair_rates)) if pair_rates else 0.0,
        stability_index_pct=0.0,
    )


def rank_period_stabilities(
    metrics: list[PeriodStabilityMetrics],
) -> list[PeriodStabilityMetrics]:
    max_abs_delta = max(m.avg_abs_delta_pp for m in metrics)
    max_cells = max(m.significant_cell_count for m in metrics)

    def _index(m: PeriodStabilityMetrics) -> float:
        return 0.25 * (
            (m.avg_abs_delta_pp / max_abs_delta * 100.0 if max_abs_delta > 0 else 0.0)
            + m.avg_quarters_rate * 100.0
            + m.avg_pairs_rate * 100.0
            + (
                m.significant_cell_count / max_cells * 100.0
                if max_cells > 0
                else 0.0
            )
        )

    scored = [
        PeriodStabilityMetrics(
            period=m.period,
            significant_cell_count=m.significant_cell_count,
            avg_abs_delta_pp=m.avg_abs_delta_pp,
            avg_quarters_rate=m.avg_quarters_rate,
            avg_pairs_rate=m.avg_pairs_rate,
            stability_index_pct=_index(m),
        )
        for m in metrics
    ]
    ordered = sorted(
        scored,
        key=lambda m: (
            -m.stability_index_pct,
            -m.avg_pairs_rate,
            -m.avg_quarters_rate,
            -m.avg_abs_delta_pp,
            -m.significant_cell_count,
            m.period,
        ),
    )
    return [
        PeriodStabilityMetrics(
            period=m.period,
            significant_cell_count=m.significant_cell_count,
            avg_abs_delta_pp=m.avg_abs_delta_pp,
            avg_quarters_rate=m.avg_quarters_rate,
            avg_pairs_rate=m.avg_pairs_rate,
            stability_index_pct=m.stability_index_pct,
            rank=i + 1,
        )
        for i, m in enumerate(ordered)
    ]


def yearly_material_agreement_rates(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    *,
    prepare_frame: PrepareFrameFn,
    n_buckets: int,
    min_delta_pp: float,
) -> list[YearlyMaterialAgreement]:
    cells = material_cells_for_period(
        daily,
        period,
        pair_bands,
        prepare_frame=prepare_frame,
        n_buckets=n_buckets,
        min_delta_pp=min_delta_pp,
    )
    if not cells:
        return []
    prepared = prepare_frame(daily, period, pair_bands)
    if prepared is None:
        return []
    _work, buckets, valid, hit_masks = prepared
    years = years_from_frame(_work)

    pooled: list[tuple[int, str, float]] = []
    for cell in cells:
        row_mask = valid & (buckets == cell.bucket)
        if not row_mask.any():
            continue
        hit = hit_masks[cell.column]
        pooled.append(
            (
                cell.bucket,
                cell.column,
                float(hit[row_mask].mean() - hit[valid].mean()),
            )
        )
    if not pooled:
        return []

    out: list[YearlyMaterialAgreement] = []
    for year in sorted(np.unique(years[valid])):
        if year < 0:
            continue
        flags: list[bool] = []
        for bucket, col, pooled_delta in pooled:
            agrees = _year_sign_agrees(
                hit_masks[col], years, buckets, valid, bucket, int(year), pooled_delta
            )
            if agrees is not None:
                flags.append(agrees)
        if flags:
            out.append(YearlyMaterialAgreement(int(year), float(np.mean(flags))))
    return out
