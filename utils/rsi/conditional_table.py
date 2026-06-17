"""RSI(N) вчера → return сегодня (векторно)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.constants import (
    REPEATABILITY_NOTE,
    RETURN_STATS_COLS,
    ROW_TITLE,
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
from crypto_research.utils.rsi.constants import N_RSI_QUANTILES, RSI_BUCKET_THRESHOLDS_NOTE
from crypto_research.utils.rsi.rsi import (
    assign_rsi_buckets,
    build_rsi_work_frame,
    quantile_edges,
    rsi_bucket_label,
    rsi_prev_column,
)
from crypto_research.utils.weekday.bands import MeanBands
from crypto_research.utils.weekday.repeatability import months_from_frame, years_from_frame


def prepare_rsi_condition_frame(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
    work = build_rsi_work_frame(daily, period)
    prev_col = rsi_prev_column(period)
    if work.height == 0 or prev_col not in work.columns:
        return None

    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    rsi_prev = work[prev_col].to_numpy().astype(np.float64, copy=False)
    edges = quantile_edges(rsi_prev)
    buckets = assign_rsi_buckets(rsi_prev, edges)
    valid = buckets >= 0
    if not valid.any():
        return None

    pair_arr = work["pair"].to_numpy().astype(object, copy=False)
    hit_matrix = build_return_hit_matrix(
        work["return_pct"].to_numpy().astype(np.float64, copy=False),
        work["day_open"].to_numpy().astype(np.float64, copy=False),
        work["day_high"].to_numpy().astype(np.float64, copy=False),
        work["day_low"].to_numpy().astype(np.float64, copy=False),
        pair_arr,
        pair_bands,
    )
    return work, buckets, valid, hit_masks_from_matrix(hit_matrix), edges


def rsi_bucket_labels(edges: np.ndarray) -> list[str]:
    return [rsi_bucket_label(b, edges) for b in range(N_RSI_QUANTILES)]


def _rsi_row_label(bucket: int, edges: np.ndarray, n: int, mean_ret: float) -> str:
    base = rsi_bucket_label(bucket, edges)
    return scenario_row_label(base, n, mean_ret)


def build_rsi_period_table_lines(
    daily: pl.DataFrame,
    period: int,
    pair_bands: dict[str, MeanBands],
) -> list[str]:
    prepared = prepare_rsi_condition_frame(daily, period, pair_bands)
    if prepared is None:
        return [f"=== RSI({period}): недостаточно данных ===", ""]

    work, buckets, valid, hit_masks, edges = prepared
    returns = work["return_pct"].to_numpy().astype(np.float64, copy=False)
    pair_arr = work["pair"].to_numpy().astype(object, copy=False)
    years = years_from_frame(work)
    months = months_from_frame(work)

    row_labels: list[str] = []
    row_totals: list[int] = []
    counts: list[dict[str, int]] = []

    for b in range(N_RSI_QUANTILES):
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
        row_labels.append(_rsi_row_label(b, edges, n, mean_ret))

    total_events = int(valid.sum())
    valid_hits = np.stack([hit_masks[col][valid] for col in RETURN_STATS_COLS], axis=1)
    base_pct = {
        col: float(valid_hits[:, i].mean() * 100.0) for i, col in enumerate(RETURN_STATS_COLS)
    }

    row_w = max(len(ROW_TITLE), max(len(x) for x in row_labels))
    header = table_header(ROW_TITLE, row_w)
    sep = table_separator(header)

    lines: list[str] = []
    lines.append(f"=== RSI({period}) вчера → return сегодня, % ===")
    lines.append(
        f"RSI Wilder({period}) по дневному close; 6 квантилей по RSI вчера. "
        f"{RSI_BUCKET_THRESHOLDS_NOTE} Колонки mean — пороги доходности своей пары."
    )
    lines.append(REPEATABILITY_NOTE)
    lines.append(header)
    lines.append(sep)
    for b in range(N_RSI_QUANTILES):
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
    lines.append(f"=== Δ к BASE: RSI({period}) (п.п.) ===")
    lines.append(header)
    lines.append(sep)
    for b in range(N_RSI_QUANTILES):
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
