"""Дневная доходность UTC и таблицы weekday."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.stats.cell_annotations import annotations_for_columns
from crypto_research.stats.year_repeatability import months_from_frame, years_from_frame

TRIM_LO_PCT = 5.0
TRIM_HI_PCT = 95.0

COL_CLOSE_UP = "Закрытие в плюсе"
COL_UP_STRONG = "Рост close: сильный"
COL_UP_MODERATE = "Рост close: умеренный"
COL_UP_WEAK = "Рост close: слабый"
COL_CLOSE_DOWN = "Закрытие в минусе"
COL_DOWN_WEAK = "Падение close: слабое"
COL_DOWN_MODERATE = "Падение close: умеренное"
COL_DOWN_STRONG = "Падение close: сильное"
COL_HIGH_REACH_UP = "High ≥ порог роста"
COL_HIGH_ABOVE_UP = "High > сильный рост"
COL_LOW_REACH_DOWN = "Low ≤ порог падения"
COL_LOW_BELOW_DOWN = "Low < сильное падение"

STATS_COLS = [
    COL_CLOSE_UP,
    COL_UP_STRONG,
    COL_UP_MODERATE,
    COL_UP_WEAK,
    COL_CLOSE_DOWN,
    COL_DOWN_WEAK,
    COL_DOWN_MODERATE,
    COL_DOWN_STRONG,
    COL_HIGH_REACH_UP,
    COL_HIGH_ABOVE_UP,
    COL_LOW_REACH_DOWN,
    COL_LOW_BELOW_DOWN,
]


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


@dataclass(frozen=True)
class MeanBands:
    up_lo: float
    up_hi: float
    down_lo: float
    down_hi: float

    @staticmethod
    def from_group_stats(up: "GroupStats", down: "GroupStats") -> MeanBands:
        return MeanBands(
            up_lo=up.mean_lo_pct,
            up_hi=up.mean_hi_pct,
            down_lo=down.mean_lo_pct,
            down_hi=down.mean_hi_pct,
        )


@dataclass(frozen=True)
class GroupStats:
    label: str
    total_days: int
    kept_days: int
    excluded_days: int
    min_pct: float
    max_pct: float
    mean_pct: float
    median_pct: float
    mean_lo_pct: float
    mean_hi_pct: float
    count_lo: int
    count_mid: int
    count_hi: int


def build_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    cast_cols = [
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.from_epoch("start_ms", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.truncate("1d")
        .alias("day_utc"),
    ]
    if "volume" in df.columns:
        cast_cols.append(pl.col("volume").cast(pl.Float64))

    minute = df.sort("start_ms").with_columns(cast_cols)

    agg_exprs: list[pl.Expr] = [
        pl.col("open").first().alias("day_open"),
        pl.col("high").max().alias("day_high"),
        pl.col("low").min().alias("day_low"),
        pl.col("close").last().alias("day_close"),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum().alias("day_volume"))

    daily = minute.group_by("day_utc").agg(agg_exprs)
    return daily.with_columns(
        ((pl.col("day_close") - pl.col("day_open")) / pl.col("day_open") * 100.0).alias(
            "return_pct"
        )
    )


def _trim_by_percentile(series: pl.Series, lo: float, hi: float) -> tuple[pl.Series, int]:
    total = series.len()
    if total == 0:
        return series, 0
    p_lo = float(series.quantile(lo / 100.0))
    p_hi = float(series.quantile(hi / 100.0))
    trimmed = series.filter((series >= p_lo) & (series <= p_hi))
    return trimmed, total - trimmed.len()


def _mean_band_counts(trimmed: pl.Series, mean_pct: float) -> tuple[float, float, int, int, int]:
    lo = mean_pct * 0.5
    hi = mean_pct * 1.5
    if mean_pct > 0:
        count_lo = int((trimmed <= lo).sum())
        count_mid = int(((trimmed > lo) & (trimmed <= hi)).sum())
        count_hi = int((trimmed > hi).sum())
    else:
        count_lo = int((trimmed >= lo).sum())
        count_mid = int(((trimmed < lo) & (trimmed >= hi)).sum())
        count_hi = int((trimmed < hi).sum())
    return lo, hi, count_lo, count_mid, count_hi


def _stats_from_series(label: str, total_days: int, values: pl.Series) -> GroupStats:
    trimmed, excluded = _trim_by_percentile(values, TRIM_LO_PCT, TRIM_HI_PCT)
    kept = trimmed.len()
    if kept == 0:
        nan = float("nan")
        return GroupStats(
            label=label,
            total_days=total_days,
            kept_days=0,
            excluded_days=excluded,
            min_pct=nan,
            max_pct=nan,
            mean_pct=nan,
            median_pct=nan,
            mean_lo_pct=nan,
            mean_hi_pct=nan,
            count_lo=0,
            count_mid=0,
            count_hi=0,
        )
    mean_pct = float(trimmed.mean())
    lo, hi, count_lo, count_mid, count_hi = _mean_band_counts(trimmed, mean_pct)
    return GroupStats(
        label=label,
        total_days=total_days,
        kept_days=kept,
        excluded_days=excluded,
        min_pct=float(trimmed.min()),
        max_pct=float(trimmed.max()),
        mean_pct=mean_pct,
        median_pct=float(trimmed.median()),
        mean_lo_pct=lo,
        mean_hi_pct=hi,
        count_lo=count_lo,
        count_mid=count_mid,
        count_hi=count_hi,
    )


def compute_up_down_stats(daily: pl.DataFrame) -> tuple[GroupStats, GroupStats]:
    up = daily.filter(pl.col("return_pct") > 0)["return_pct"]
    down = daily.filter(pl.col("return_pct") < 0)["return_pct"]
    return (
        _stats_from_series("ДНИ РОСТА", up.len(), up),
        _stats_from_series("ДНИ ПАДЕНИЯ", down.len(), down),
    )


def build_pair_bands_map(daily: pl.DataFrame) -> dict[str, MeanBands]:
    if "pair" not in daily.columns:
        up, down = compute_up_down_stats(daily)
        return {"_single": MeanBands.from_group_stats(up, down)}
    bands: dict[str, MeanBands] = {}
    for pair in daily["pair"].unique().to_list():
        sub = daily.filter(pl.col("pair") == pair)
        up, down = compute_up_down_stats(sub)
        bands[str(pair)] = MeanBands.from_group_stats(up, down)
    return bands


def classify_return_pct(
    ret: float,
    bands: MeanBands,
    day_open: float | None = None,
    day_high: float | None = None,
    day_low: float | None = None,
) -> set[str]:
    tags: set[str] = set()
    if ret > 0:
        tags.add(COL_CLOSE_UP)
        if ret <= bands.up_lo:
            tags.add(COL_UP_WEAK)
        elif ret <= bands.up_hi:
            tags.add(COL_UP_MODERATE)
        else:
            tags.add(COL_UP_STRONG)
    elif ret < 0:
        tags.add(COL_CLOSE_DOWN)
        if ret >= bands.down_lo:
            tags.add(COL_DOWN_WEAK)
        elif ret >= bands.down_hi:
            tags.add(COL_DOWN_MODERATE)
        else:
            tags.add(COL_DOWN_STRONG)

    if (
        day_open is not None
        and day_high is not None
        and day_low is not None
        and day_open > 0
    ):
        up_move = (day_high - day_open) / day_open * 100.0
        down_move = (day_low - day_open) / day_open * 100.0
        if up_move >= bands.up_lo:
            tags.add(COL_HIGH_REACH_UP)
        if up_move > bands.up_hi:
            tags.add(COL_HIGH_ABOVE_UP)
        if down_move <= bands.down_lo:
            tags.add(COL_LOW_REACH_DOWN)
        if down_move < bands.down_hi:
            tags.add(COL_LOW_BELOW_DOWN)
    return tags


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
