"""Таблица прогностической силы серий роста/падения."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.price_sequences.constants import MAX_STREAK_DAYS, SCENARIO_ROWS
from crypto_research.utils.weekday.bands import STATS_COLS, MeanBands, classify_return_pct
from crypto_research.utils.weekday.repeatability import (
    annotations_for_columns,
    months_from_frame,
    years_from_frame,
)
from crypto_research.utils.weekday.table import (
    _delta_values,
    _join_values_with_repeatability,
    _pct_values,
    _stats_header_sized,
    _stats_row_sized,
    _stats_separator_sized,
    _table_widths,
)


def _row_bucket(day_before_sign: int, streak_len: int) -> str | None:
    if day_before_sign not in (-1, 1) or streak_len <= 0:
        return None
    bucket = min(streak_len, MAX_STREAK_DAYS)
    if day_before_sign == -1:
        return f"После {bucket}д падения"
    return f"После {bucket}д роста"


def _process_pair_streaks(
    sub: pl.DataFrame,
    returns: list[float],
    pair_key: str,
    bands: MeanBands,
    row_totals: dict[str, int],
    counts: dict[str, dict[str, int]],
    event_years: list[int],
    event_months: list[int],
    event_pairs: list[str],
    event_buckets: list[int],
    event_hits: dict[str, list[bool]],
) -> None:
    years = years_from_frame(sub)
    months = months_from_frame(sub)
    opens = sub["day_open"].to_numpy().astype(np.float64, copy=False)
    highs = sub["day_high"].to_numpy().astype(np.float64, copy=False)
    lows = sub["day_low"].to_numpy().astype(np.float64, copy=False)
    row_to_idx = {name: i for i, name in enumerate(SCENARIO_ROWS)}
    for i in range(1, len(returns)):
        prev = returns[i - 1]
        if prev == 0:
            continue
        sign = 1 if prev > 0 else -1
        streak = 1
        j = i - 2
        while j >= 0:
            rj = returns[j]
            if rj == 0:
                break
            same_sign = (rj > 0 and sign == 1) or (rj < 0 and sign == -1)
            if not same_sign:
                break
            streak += 1
            j -= 1
        row_name = _row_bucket(sign, streak)
        if row_name is None:
            continue
        ret = float(returns[i])
        row_totals[row_name] += 1
        for col in classify_return_pct(
            ret,
            bands,
            float(opens[i]),
            float(highs[i]),
            float(lows[i]),
        ):
            counts[row_name][col] += 1
        event_years.append(int(years[i]))
        event_months.append(int(months[i]))
        event_pairs.append(pair_key)
        event_buckets.append(row_to_idx[row_name])
        cls = set(
            classify_return_pct(
                ret,
                bands,
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
            )
        )
        for col in STATS_COLS:
            event_hits[col].append(col in cls)


def build_price_sequence_table(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
) -> list[str]:
    row_totals = {row: 0 for row in SCENARIO_ROWS}
    counts = {row: {col: 0 for col in STATS_COLS} for row in SCENARIO_ROWS}
    event_years: list[int] = []
    event_months: list[int] = []
    event_pairs: list[str] = []
    event_buckets: list[int] = []
    event_hits: dict[str, list[bool]] = {col: [] for col in STATS_COLS}

    if "pair" in daily.columns:
        for pair in daily["pair"].unique().to_list():
            sub = daily.filter(pl.col("pair") == pair).sort("day_utc")
            returns = [float(v) for v in sub["return_pct"].to_list()]
            _process_pair_streaks(
                sub,
                returns,
                str(pair),
                pair_bands[str(pair)],
                row_totals,
                counts,
                event_years,
                event_months,
                event_pairs,
                event_buckets,
                event_hits,
            )
    else:
        sub = daily.sort("day_utc") if "day_utc" in daily.columns else daily
        returns = [float(v) for v in sub["return_pct"].to_list()]
        bands = next(iter(pair_bands.values()))
        _process_pair_streaks(
            sub,
            returns,
            "_single",
            bands,
            row_totals,
            counts,
            event_years,
            event_months,
            event_pairs,
            event_buckets,
            event_hits,
        )

    years_arr = np.array(event_years, dtype=np.int32)
    months_arr = np.array(event_months, dtype=np.int32)
    pairs_arr = np.array(event_pairs, dtype=object)
    buckets_arr = np.array(event_buckets, dtype=np.int8)
    valid_events = np.ones(years_arr.shape[0], dtype=bool)
    hit_arr = {col: np.array(event_hits[col], dtype=bool) for col in STATS_COLS}

    total_events = sum(row_totals.values())
    base_pct = {
        col: (sum(counts[row][col] for row in SCENARIO_ROWS) * 100.0 / total_events)
        if total_events > 0
        else float("nan")
        for col in STATS_COLS
    }

    row_title = "Сценарий"
    pct_rows: list[tuple[str, list[str]]] = []
    delta_rows: list[tuple[str, list[str]]] = []
    row_labels: list[str] = []

    for row_idx, row in enumerate(SCENARIO_ROWS):
        total = row_totals[row]
        label = f"{row} (n={total})"
        row_labels.append(label)
        reps, month_reps, pair_supports, _ = annotations_for_columns(
            years_arr,
            months_arr,
            buckets_arr,
            row_idx,
            valid_events,
            hit_arr,
            pairs_arr,
            STATS_COLS,
        )
        pct_rows.append(
            (
                label,
                _join_values_with_repeatability(
                    _pct_values(counts[row], total),
                    reps,
                    month_reps,
                    pair_supports,
                ),
            )
        )
        if total_events == 0:
            delta_vals = ["n/a"] * len(STATS_COLS)
        else:
            delta_vals = _delta_values(counts[row], total, base_pct)
        delta_rows.append(
            (
                label,
                _join_values_with_repeatability(delta_vals, reps, month_reps, pair_supports),
            )
        )

    base_label = f"BASE без условий (n={total_events})"
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
    lines.append("=== Таблица условных вероятностей, % ===")
    lines.append(
        "Условие по строке: до текущего дня шла серия роста/падения. "
        f"Для {MAX_STREAK_DAYS}д используется бакет >={MAX_STREAK_DAYS}."
    )
    lines.append(
        "Колонки: пороги μ×0.5 / μ×1.5 от среднего **своей пары** "
        "(mean роста/падения считается отдельно по каждому символу)."
    )
    lines.append(header)
    lines.append(sep)
    for label, cells in pct_rows:
        lines.append(emit_row(label, cells))
    lines.append("")
    lines.append("=== Δ к BASE (п.п.) ===")
    lines.append(header)
    lines.append(sep)
    for label, cells in delta_rows:
        lines.append(emit_row(label, cells))
    lines.append("")
    lines.append("Примечание: n/a = в истории не было таких строковых условий.")
    lines.append("BASE = безусловная вероятность по всем дням, где определён сценарий строки.")
    lines.append("")
    return lines
