"""Форматирование строк таблиц ema_spreads."""

from __future__ import annotations

from crypto_research.utils.ema_spreads.constants import COL_WIDTH, RETURN_STATS_COLS
from crypto_research.utils.weekday.repeatability import annotations_for_columns


def _pct_values(counts_row: dict[str, int], total: int) -> list[str]:
    if total == 0:
        return ["n/a"] * len(RETURN_STATS_COLS)
    return [f"{counts_row[col] * 100.0 / total:.1f}" for col in RETURN_STATS_COLS]


def _delta_values(counts_row: dict[str, int], total: int, base_pct: dict[str, float]) -> list[str]:
    if total == 0:
        return ["n/a"] * len(RETURN_STATS_COLS)
    return [
        f"{(counts_row[col] * 100.0 / total) - base_pct[col]:+.1f}"
        for col in RETURN_STATS_COLS
    ]


def _join_values_with_repeatability(
    values: list[str],
    year_reps: list[str],
    month_reps: list[str],
    pair_supports: list[int | None],
) -> list[str]:
    out: list[str] = []
    for v, y_rep, _m_rep, p_sup in zip(
        values, year_reps, month_reps, pair_supports, strict=True
    ):
        if v == "n/a":
            out.append(v)
        else:
            y_txt = y_rep if y_rep != "n/a" else "n/a"
            p_txt = str(p_sup) if p_sup is not None else "n/a"
            out.append(f"{v} ({y_txt}) [{p_txt}]")
    return out


def table_header(row_title: str, row_w: int) -> str:
    return f"{row_title:<{row_w}} | " + " | ".join(
        f"{col:>{COL_WIDTH}}" for col in RETURN_STATS_COLS
    )


def table_separator(header: str) -> str:
    return "-" * len(header)


def table_row(label: str, values: list[str], row_w: int) -> str:
    return f"{label:<{row_w}} | " + " | ".join(v.rjust(COL_WIDTH) for v in values)


def annotate_row_values(
    values: list[str],
    years,
    months,
    buckets,
    row_index: int,
    valid,
    hit_masks: dict[str, object],
    pair_keys,
) -> list[str]:
    reps, month_reps, pair_supports, _ = annotations_for_columns(
        years,
        months,
        buckets,
        row_index,
        valid,
        hit_masks,
        pair_keys,
        list(RETURN_STATS_COLS),
    )
    return _join_values_with_repeatability(values, reps, month_reps, pair_supports)


def scenario_row_label(name: str, n: int, mean_ret: float) -> str:
    if mean_ret != mean_ret:
        trend = "ср n/a"
    else:
        trend = f"ср ret {mean_ret:+.1f}%"
    return f"{name} ({trend}, n={n})"


def pct_row_values(counts: dict[str, int], total: int) -> list[str]:
    return _pct_values(counts, total)


def delta_row_values(counts: dict[str, int], total: int, base_pct: dict[str, float]) -> list[str]:
    return _delta_values(counts, total, base_pct)
