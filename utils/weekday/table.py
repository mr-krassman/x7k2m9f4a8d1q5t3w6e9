"""Текстовые таблицы weekday (% и Δ к BASE)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.weekday.bands import (
    STATS_COLS,
    MeanBands,
    classify_return_pct,
)
from crypto_research.utils.weekday.repeatability import (
    annotations_for_columns,
    months_from_frame,
    years_from_frame,
)


def _table_widths(
    row_title: str,
    row_labels: list[str],
    col_headers: list[str],
    table_rows: list[tuple[str, list[str]]],
) -> tuple[int, list[int]]:
    row_w = len(row_title)
    for label in row_labels:
        row_w = max(row_w, len(label))
    col_ws = [len(col) for col in col_headers]
    for _label, cells in table_rows:
        for i, cell in enumerate(cells):
            if i < len(col_ws):
                col_ws[i] = max(col_ws[i], len(cell))
    return row_w, col_ws


def _stats_header_sized(row_title: str, col_headers: list[str], row_w: int, col_ws: list[int]) -> str:
    return f"{row_title:<{row_w}} | " + " | ".join(
        f"{col:>{w}}" for col, w in zip(col_headers, col_ws, strict=True)
    )


def _stats_separator_sized(row_title: str, col_headers: list[str], row_w: int, col_ws: list[int]) -> str:
    return "-" * len(_stats_header_sized(row_title, col_headers, row_w, col_ws))


def _stats_row_sized(row_label: str, values: list[str], row_w: int, col_ws: list[int]) -> str:
    return f"{row_label:<{row_w}} | " + " | ".join(
        v.rjust(w) for v, w in zip(values, col_ws, strict=True)
    )


def _pct_values(counts_row: dict[str, int], total: int) -> list[str]:
    if total == 0:
        return ["n/a"] * len(STATS_COLS)
    return [f"{counts_row[col] * 100.0 / total:.1f}" for col in STATS_COLS]


def _delta_values(counts_row: dict[str, int], total: int, base_pct: dict[str, float]) -> list[str]:
    if total == 0:
        return ["n/a"] * len(STATS_COLS)
    return [
        f"{(counts_row[col] * 100.0 / total) - base_pct[col]:+.1f}" for col in STATS_COLS
    ]


def _join_values_with_repeatability(
    values: list[str],
    year_reps: list[str],
    month_reps: list[str],
    pair_supports: list[int | None],
    pair_validates: list[int | None] | None = None,
) -> list[str]:
    out: list[str] = []
    if pair_validates is None:
        pair_validates = [None] * len(values)
    for v, y_rep, _m_rep, p_sup, p_val in zip(
        values, year_reps, month_reps, pair_supports, pair_validates, strict=True
    ):
        if v == "n/a":
            out.append(v)
        else:
            y_txt = y_rep if y_rep != "n/a" else "n/a"
            p_txt = str(p_sup) if p_sup is not None else "n/a"
            if p_val is not None:
                out.append(f"{v} ({y_txt}) [{p_txt}] [{p_val}]")
            else:
                out.append(f"{v} ({y_txt}) [{p_txt}]")
    return out


def _fmt_pct(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:+.1f}"


def build_weekday_table(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    *,
    auto_width: bool = True,
    table_intro: bool = True,
) -> list[str]:
    del auto_width  # всегда auto_width через _table_widths

    wd_min = int(weekday_daily["weekday"].min())
    wd_max = int(weekday_daily["weekday"].max())
    if wd_min >= 1 and wd_max <= 7:
        weekday_daily = weekday_daily.with_columns(
            (((pl.col("weekday") - 1) % 7).cast(pl.Int64)).alias("weekday")
        )

    weekday_names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
    rows_order = [0, 1, 2, 3, 4, 5, 6]
    row_totals = {d: 0 for d in rows_order}
    row_mean = {d: float("nan") for d in rows_order}
    counts = {d: {col: 0 for col in STATS_COLS} for d in rows_order}

    if "pair" not in weekday_daily.columns:
        weekday_daily = weekday_daily.with_columns(pl.lit("_single").alias("pair"))

    wd_returns = weekday_daily["return_pct"].to_numpy().astype(np.float64, copy=False)
    wd_opens = weekday_daily["day_open"].to_numpy().astype(np.float64, copy=False)
    wd_highs = weekday_daily["day_high"].to_numpy().astype(np.float64, copy=False)
    wd_lows = weekday_daily["day_low"].to_numpy().astype(np.float64, copy=False)
    wd_years = years_from_frame(weekday_daily)
    wd_months = months_from_frame(weekday_daily)
    wd_buckets = weekday_daily["weekday"].to_numpy().astype(np.int8, copy=False)
    wd_valid = np.ones(wd_returns.shape[0], dtype=bool)
    wd_pairs = weekday_daily["pair"].to_list()
    wd_pair_arr = np.array(wd_pairs, dtype=object)
    wd_hits: dict[str, np.ndarray] = {
        col: np.zeros(wd_returns.shape[0], dtype=bool) for col in STATS_COLS
    }
    for i, ret in enumerate(wd_returns):
        bands = pair_bands[str(wd_pairs[i])]
        cls = classify_return_pct(
            float(ret),
            bands,
            float(wd_opens[i]),
            float(wd_highs[i]),
            float(wd_lows[i]),
        )
        for col in STATS_COLS:
            wd_hits[col][i] = col in cls

    grouped = (
        weekday_daily.group_by("weekday")
        .agg(
            pl.count().alias("n"),
            pl.col("return_pct").mean().alias("mean_ret"),
            pl.col("return_pct").alias("rets"),
            pl.col("pair").alias("pairs"),
            pl.col("day_open").alias("opens"),
            pl.col("day_high").alias("highs"),
            pl.col("day_low").alias("lows"),
        )
        .to_dicts()
    )
    for row in grouped:
        d = int(row["weekday"])
        if d not in row_totals:
            continue
        row_totals[d] = int(row["n"])
        row_mean[d] = float(row["mean_ret"])
        rets = row["rets"]
        pairs = row["pairs"]
        opens = row["opens"]
        highs = row["highs"]
        lows = row["lows"]
        for ret, pair, opn, hi, lo in zip(rets, pairs, opens, highs, lows, strict=True):
            bands = pair_bands[str(pair)]
            for col in classify_return_pct(float(ret), bands, float(opn), float(hi), float(lo)):
                counts[d][col] += 1

    total_events = sum(row_totals.values())
    base_pct = {
        col: (sum(counts[d][col] for d in rows_order) * 100.0 / total_events)
        if total_events > 0
        else float("nan")
        for col in STATS_COLS
    }

    def weekday_row_label(d: int) -> str:
        mean_ret = row_mean[d]
        if mean_ret != mean_ret:
            trend = "n/a"
        elif mean_ret > 0:
            trend = f"ср: росла {_fmt_pct(mean_ret)}%"
        elif mean_ret < 0:
            trend = f"ср: падала {_fmt_pct(mean_ret)}%"
        else:
            trend = "ср: 0.0%"
        return f"{weekday_names[d]} ({trend}, n={row_totals[d]})"

    base_label = f"BASE без условий (n={total_events})"
    row_title = "День недели"

    pct_rows: list[tuple[str, list[str]]] = []
    delta_rows: list[tuple[str, list[str]]] = []
    row_labels: list[str] = []

    for d in rows_order:
        total = row_totals[d]
        label = weekday_row_label(d)
        row_labels.append(label)
        reps, month_reps, pair_supports, _pair_validates = annotations_for_columns(
            wd_years,
            wd_months,
            wd_buckets,
            d,
            wd_valid,
            wd_hits,
            wd_pair_arr,
            STATS_COLS,
        )
        pct_rows.append(
            (
                label,
                _join_values_with_repeatability(
                    _pct_values(counts[d], total),
                    reps,
                    month_reps,
                    pair_supports,
                ),
            )
        )
        if total_events == 0:
            delta_vals = ["n/a"] * len(STATS_COLS)
        else:
            delta_vals = _delta_values(counts[d], total, base_pct)
        delta_rows.append(
            (
                label,
                _join_values_with_repeatability(
                    delta_vals, reps, month_reps, pair_supports
                ),
            )
        )

    if total_events > 0:
        row_labels.append(base_label)
        pct_rows.append((base_label, [f"{base_pct[col]:.1f}" for col in STATS_COLS]))

    all_rows = pct_rows + delta_rows
    row_w, col_ws = _table_widths(row_title, row_labels, STATS_COLS, all_rows)
    header = _stats_header_sized(row_title, STATS_COLS, row_w, col_ws)
    sep = _stats_separator_sized(row_title, STATS_COLS, row_w, col_ws)

    def emit_row(label: str, cells: list[str]) -> str:
        return _stats_row_sized(label, cells, row_w, col_ws)

    lines: list[str] = []
    lines.append("=== Таблица по дням недели, % ===")
    if table_intro:
        lines.append(
            "Строка = день недели UTC; в скобках — средний return дня (close→open) и n дней выборки. "
            "Пороги μ±50% в колонках — отдельно по каждой паре (после обрезки 5–95%)."
        )
    lines.append(header)
    lines.append(sep)
    for label, cells in pct_rows:
        lines.append(emit_row(label, cells))
    lines.append("")
    lines.append("=== Δ к BASE по дням недели (п.п.) ===")
    lines.append(header)
    lines.append(sep)
    for label, cells in delta_rows:
        lines.append(emit_row(label, cells))
    lines.append("")
    return lines
