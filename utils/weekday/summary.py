"""Итоговая таблица train → val: p-value по дням недели и подтверждение на val-парах."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

WEEKDAY_NAMES: tuple[str, ...] = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
WEEKDAYS: tuple[int, ...] = tuple(range(7))
BONFERRONI_N = 7
ALPHA = 0.05
BONF_ALPHA = ALPHA / BONFERRONI_N
VAL_CONFIRM_RATIO = 0.80
_PERM_N = 20_000
_PERM_RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class WeekdaySummaryRow:
    weekday: int
    name: str
    train_mean_return_pct: float
    val_mean_return_pct: float | None
    p_value: float
    val_agree: int | None
    val_total: int
    status: str

    @property
    def delta_val_train_pp(self) -> float | None:
        if self.val_mean_return_pct is None:
            return None
        if self.train_mean_return_pct != self.train_mean_return_pct:
            return None
        if self.val_mean_return_pct != self.val_mean_return_pct:
            return None
        return self.val_mean_return_pct - self.train_mean_return_pct


def _normalize_weekday(df: pl.DataFrame) -> pl.DataFrame:
    wd_min = int(df["weekday"].min())
    wd_max = int(df["weekday"].max())
    if wd_min >= 1 and wd_max <= 7:
        return df.with_columns(
            (((pl.col("weekday") - 1) % 7).cast(pl.Int64)).alias("weekday")
        )
    return df


def _with_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    cols = {"return_pct", "day_utc", "pair"}
    missing = cols - set(daily.columns)
    if missing:
        raise ValueError(f"daily missing columns: {sorted(missing)}")
    return _normalize_weekday(
        daily.select("return_pct", "day_utc", "pair").with_columns(
            pl.col("day_utc").dt.weekday().alias("weekday")
        )
    )


def _per_pair_mean(df: pl.DataFrame, weekday: int) -> dict[str, float]:
    sub = df.filter(pl.col("weekday") == weekday)
    if sub.is_empty():
        return {}
    out: dict[str, float] = {}
    for row in sub.group_by("pair").agg(pl.col("return_pct").mean().alias("mean")).iter_rows(
        named=True
    ):
        out[str(row["pair"])] = float(row["mean"])
    return out


def _pooled_mean_return(df: pl.DataFrame, weekday: int) -> float:
    sub = df.filter(pl.col("weekday") == weekday)
    if sub.is_empty():
        return float("nan")
    return float(sub["return_pct"].mean())


def _permutation_p(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    obs = float(arr.mean())
    if obs == 0.0:
        return 1.0
    signs = _PERM_RNG.choice([-1.0, 1.0], size=(_PERM_N, n))
    perm_means = (signs * arr).mean(axis=1)
    return float(np.mean(np.abs(perm_means) >= abs(obs)))


def _fmt_p(p: float) -> str:
    if p != p:
        return "n/a"
    if p >= 0.05:
        return ">0.05"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return "<0.01"
    return f"{p:.3f}"


def _fmt_effect(mean_ret: float | None) -> str:
    if mean_ret is None:
        return "—"
    if mean_ret != mean_ret:
        return "n/a"
    return f"{mean_ret:+.2f}%"


def _fmt_delta_pp(delta: float | None) -> str:
    if delta is None or delta != delta:
        return "—"
    return f"{delta:+.2f} п.п."


def _status(p_value: float, val_agree: int | None, val_total: int) -> str:
    if p_value != p_value or p_value >= 0.05:
        return "не значим"
    if val_total == 0 or val_agree is None:
        return "требует проверки"
    ratio = val_agree / val_total
    if p_value < BONF_ALPHA and ratio >= VAL_CONFIRM_RATIO:
        return "значим"
    return "требует проверки"


def _val_confirm_text(val_agree: int | None, val_total: int) -> str:
    if val_total == 0 or val_agree is None:
        return "—"
    mark = "✅" if val_agree / val_total >= VAL_CONFIRM_RATIO else "❓"
    return f"{val_agree}/{val_total} {mark}"


def _count_return_confirm(
    train_mean: float,
    val_pair_means: dict[str, float],
) -> tuple[int | None, int]:
    if not val_pair_means:
        return None, 0
    if train_mean != train_mean or train_mean == 0:
        return None, len(val_pair_means)
    sign = 1 if train_mean > 0 else -1
    agree = sum(
        1
        for mean in val_pair_means.values()
        if (mean > 0 and sign > 0) or (mean < 0 and sign < 0)
    )
    return agree, len(val_pair_means)


def compute_weekday_summary(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame | None,
) -> list[WeekdaySummaryRow]:
    train = _with_weekday(train_daily)
    val = _with_weekday(val_daily) if val_daily is not None else None

    rows: list[WeekdaySummaryRow] = []
    for wd in WEEKDAYS:
        pair_means = _per_pair_mean(train, wd)
        train_mean = _pooled_mean_return(train, wd)
        p_value = _permutation_p(np.array(list(pair_means.values()), dtype=np.float64))

        val_mean: float | None = None
        if val is not None:
            val_mean = _pooled_mean_return(val, wd)
            val_agree, val_total = _count_return_confirm(
                train_mean,
                _per_pair_mean(val, wd),
            )
        else:
            val_agree, val_total = None, 0

        rows.append(
            WeekdaySummaryRow(
                weekday=wd,
                name=WEEKDAY_NAMES[wd],
                train_mean_return_pct=train_mean,
                val_mean_return_pct=val_mean,
                p_value=p_value,
                val_agree=val_agree,
                val_total=val_total,
                status=_status(p_value, val_agree, val_total),
            )
        )
    return rows


def format_summary_table(rows: list[WeekdaySummaryRow]) -> list[str]:
    headers = (
        "День",
        "Эффект (train), %",
        "Эффект (val), %",
        "Δ (val−train)",
        "p-value (train)",
        "Подтверждение на val",
        "Статус",
    )
    body = [
        (
            row.name,
            _fmt_effect(row.train_mean_return_pct),
            _fmt_effect(row.val_mean_return_pct),
            _fmt_delta_pp(row.delta_val_train_pp),
            _fmt_p(row.p_value),
            _val_confirm_text(row.val_agree, row.val_total),
            row.status,
        )
        for row in rows
    ]
    col_w = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(cell))

    header = " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
    sep = "-" * len(header)
    lines = [
        "=== Итоговая таблица ===",
        "",
        "«Эффект (train), %» / «Эффект (val), %» — pooled средний intraday return "
        "(close−open)/open×100 по train / val (все пары когорты × все дни с этим weekday).",
        "«Δ (val−train)» — разница val и train в процентных пунктах (п.п.); "
        "отрицательное при том же знаке = эффект на val слабее по модулю.",
        "p-value (train) — permutation по средним return каждой train-пары (sign-flip), H₀: mean=0.",
        f"Поправка Бонферрони: α={ALPHA}, порог «значим» p < {BONF_ALPHA:.4f} ({ALPHA}/{BONFERRONI_N}).",
        f"Подтверждение на val: число пар с тем же знаком среднего return, что train; "
        f"✅ ≥ {VAL_CONFIRM_RATIO:.0%}, иначе ❓.",
        "",
        header,
        sep,
    ]
    for cells in body:
        lines.append(
            " | ".join(f"{cell:<{col_w[i]}}" for i, cell in enumerate(cells))
        )
    lines.append("")
    return lines
