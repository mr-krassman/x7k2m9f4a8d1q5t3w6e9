"""Таблицы EMA(period) вчера → return сегодня (векторно)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import (
    EMA_BUCKET_THRESHOLDS_NOTE,
    EMA_SCENARIO_ROWS,
    N_EMA_SCENARIOS,
    REPEATABILITY_NOTE,
    RETURN_STATS_COLS,
    ROW_TITLE,
)
from crypto_research.utils.ema_spreads.ema import (
    assign_ema_dev_buckets_vectorized,
    build_ema_work_frame,
    build_pair_thresholds_frame,
    ema_dev_prev_column,
)
from crypto_research.utils.ema_spreads.return_tags import (
    build_return_hit_matrix,
    hit_masks_from_matrix,
)
from crypto_research.utils.ema_spreads.table_format import (
    annotate_row_values,
    delta_row_values,
    pct_row_values,
    scenario_row_label,
    table_header,
    table_row,
    table_separator,
)
from crypto_research.utils.weekday.bands import MeanBands
from crypto_research.utils.weekday.repeatability import months_from_frame, years_from_frame


def prepare_ema_condition_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    work = build_ema_work_frame(daily, periods)
    prev_col = ema_dev_prev_column(period)
    if work.height == 0 or prev_col not in work.columns:
        return None

    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    thresholds = build_pair_thresholds_frame(work, prev_col)
    if thresholds.height == 0:
        return None

    merged = work.join(thresholds, on="pair", how="inner")
    dev = merged[prev_col].to_numpy().astype(np.float64, copy=False)
    buckets = assign_ema_dev_buckets_vectorized(
        dev,
        merged["t1_up"].to_numpy(),
        merged["t2_up"].to_numpy(),
        merged["t1_down"].to_numpy(),
        merged["t2_down"].to_numpy(),
        merged["near_abs"].to_numpy(),
    )
    valid = buckets >= 0
    if not valid.any():
        return None

    pair_arr = merged["pair"].to_numpy().astype(object, copy=False)
    hit_matrix = build_return_hit_matrix(
        merged["return_pct"].to_numpy().astype(np.float64, copy=False),
        merged["day_open"].to_numpy().astype(np.float64, copy=False),
        merged["day_high"].to_numpy().astype(np.float64, copy=False),
        merged["day_low"].to_numpy().astype(np.float64, copy=False),
        pair_arr,
        pair_bands,
    )
    return merged, buckets, valid, hit_masks_from_matrix(hit_matrix)


def build_ema_period_table_lines(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
    periods: tuple[int, ...],
) -> list[str]:
    prepared = prepare_ema_condition_frame(daily, period, pair_bands, periods)
    if prepared is None:
        return [f"=== EMA({period}): недостаточно данных ===", ""]

    work, buckets, valid, hit_masks = prepared
    returns = work["return_pct"].to_numpy().astype(np.float64, copy=False)
    pair_arr = work["pair"].to_numpy().astype(object, copy=False)
    years = years_from_frame(work)
    months = months_from_frame(work)

    row_labels: list[str] = []
    row_totals: list[int] = []
    counts: list[dict[str, int]] = []

    for b in range(N_EMA_SCENARIOS):
        mask = valid & (buckets == b)
        n = int(mask.sum())
        row_totals.append(n)
        if n > 0:
            bucket_hits = np.stack([hit_masks[col][mask] for col in RETURN_STATS_COLS], axis=1)
            row_counts = {
                col: int(bucket_hits[:, i].sum()) for i, col in enumerate(RETURN_STATS_COLS)
            }
            mean_ret = float(returns[mask].mean())
        else:
            row_counts = {col: 0 for col in RETURN_STATS_COLS}
            mean_ret = float("nan")
        counts.append(row_counts)
        row_labels.append(scenario_row_label(EMA_SCENARIO_ROWS[b], n, mean_ret))

    total_events = int(valid.sum())
    valid_hits = np.stack([hit_masks[col][valid] for col in RETURN_STATS_COLS], axis=1)
    base_pct = {
        col: float(valid_hits[:, i].mean() * 100.0) for i, col in enumerate(RETURN_STATS_COLS)
    }

    row_w = max(len(ROW_TITLE), max(len(x) for x in row_labels))
    header = table_header(ROW_TITLE, row_w)
    sep = table_separator(header)

    lines: list[str] = []
    lines.append(f"=== EMA({period}) вчера → return сегодня, % ===")
    lines.append(
        "dev вчера = (close−EMA)/EMA×100; бакеты b0–b6 по порогам t1⁺/t2⁺/t1⁻/t2⁻/near "
        "своей пары (в % — индивидуально, в подписи строк не выводятся). "
        f"{EMA_BUCKET_THRESHOLDS_NOTE} Колонки mean — пороги доходности своей пары."
    )
    lines.append(REPEATABILITY_NOTE)
    lines.append(header)
    lines.append(sep)
    for b in range(N_EMA_SCENARIOS):
        vals = annotate_row_values(
            pct_row_values(counts[b], row_totals[b]),
            years,
            months,
            buckets,
            b,
            valid,
            hit_masks,
            pair_arr,
        )
        lines.append(table_row(row_labels[b], vals, row_w))
    lines.append(sep)
    lines.append(
        table_row(
            f"BASE без условий (n={total_events})",
            [f"{base_pct[col]:.1f}" for col in RETURN_STATS_COLS],
            row_w,
        )
    )
    lines.append("")
    lines.append(f"=== Δ к BASE: EMA({period}) (п.п.) ===")
    lines.append(header)
    lines.append(sep)
    for b in range(N_EMA_SCENARIOS):
        if row_totals[b] == 0 or total_events == 0:
            vals = ["n/a"] * len(RETURN_STATS_COLS)
        else:
            vals = delta_row_values(counts[b], row_totals[b], base_pct)
        vals = annotate_row_values(
            vals, years, months, buckets, b, valid, hit_masks, pair_arr
        )
        lines.append(table_row(row_labels[b], vals, row_w))
    lines.append("")
    return lines
