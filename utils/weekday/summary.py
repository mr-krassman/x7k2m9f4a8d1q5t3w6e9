"""Сводные таблицы train → val: универсальность среди пар и устойчивость во времени."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from crypto_research.utils.research.signal_validation import (
    ALPHA,
    VAL_CONFIRM_RATIO,
    ConfirmationMode,
    bonferroni_alpha,
    count_confirm,
    fmt_p,
    intersect_status,
    permutation_p,
    signal_status,
    val_confirm_text,
)

WEEKDAY_NAMES: tuple[str, ...] = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
WEEKDAYS: tuple[int, ...] = tuple(range(7))
BONFERRONI_N = 7
BONF_ALPHA = bonferroni_alpha(BONFERRONI_N)


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


def compute_weekday_summary(
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame | None,
    *,
    confirmation_mode: ConfirmationMode = ConfirmationMode.COHORT,
) -> list[WeekdaySummaryRow]:
    train = _with_weekday(train_daily)
    val = _with_weekday(val_daily) if val_daily is not None else None

    rows: list[WeekdaySummaryRow] = []
    for wd in WEEKDAYS:
        train_pair_means = _per_pair_mean(train, wd)
        train_mean = _pooled_mean_return(train, wd)
        p_value = permutation_p(np.array(list(train_pair_means.values()), dtype=np.float64))

        val_mean: float | None = None
        if val is not None:
            val_pair_means = _per_pair_mean(val, wd)
            val_mean = _pooled_mean_return(val, wd)
            val_agree, val_total = count_confirm(
                confirmation_mode,
                train_mean,
                train_pair_means,
                val_pair_means,
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
                status=signal_status(
                    p_value,
                    val_agree,
                    val_total,
                    bonferroni_n=BONFERRONI_N,
                ),
            )
        )
    return rows


def format_summary_table(
    rows: list[WeekdaySummaryRow],
    *,
    title: str,
    val_confirm_hint: str,
) -> list[str]:
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
            fmt_p(row.p_value),
            val_confirm_text(row.val_agree, row.val_total),
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
        title,
        "",
        "«Эффект (train), %» / «Эффект (val), %» — pooled средний intraday return "
        "(close−open)/open×100 по train / val.",
        "«Δ (val−train)» — разница val и train в процентных пунктах (п.п.).",
        "p-value (train) — permutation по средним return каждой train-единицы (sign-flip), H₀: mean=0.",
        f"Поправка Бонферрони: α={ALPHA}, порог «значим» p < {BONF_ALPHA:.4f} ({ALPHA}/{BONFERRONI_N}).",
        val_confirm_hint,
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


@dataclass(frozen=True)
class WeekdayPooledRow:
    weekday: int
    name: str
    mean_pct: float
    median_pct: float
    volatility_pct: float
    day_agree: int
    day_total: int
    status: str


def _count_days_by_sign(mean_ret: float, returns: np.ndarray) -> tuple[int, int]:
    total = int(returns.size)
    if total == 0 or mean_ret != mean_ret or mean_ret == 0:
        return 0, total
    if mean_ret > 0:
        agree = int(np.sum(returns > 0))
    else:
        agree = int(np.sum(returns < 0))
    return agree, total


def _day_sign_share_text(day_agree: int, day_total: int) -> str:
    if day_total == 0:
        return "—"
    return f"{day_agree / day_total * 100:.1f}%"


def _fmt_volatility(std_pct: float) -> str:
    if std_pct != std_pct:
        return "n/a"
    return f"{std_pct:.2f}%"


def compute_weekday_pooled_summary(
    daily: pl.DataFrame,
    *,
    status_by_weekday: dict[int, str],
) -> list[WeekdayPooledRow]:
    frame = _with_weekday(daily)
    rows: list[WeekdayPooledRow] = []
    for wd in WEEKDAYS:
        sub = frame.filter(pl.col("weekday") == wd)
        returns = sub["return_pct"].to_numpy().astype(np.float64, copy=False)
        mean_ret = float(returns.mean()) if returns.size else float("nan")
        median_ret = float(np.median(returns)) if returns.size else float("nan")
        vol_ret = float(returns.std(ddof=1)) if returns.size > 1 else float("nan")
        day_agree, day_total = _count_days_by_sign(mean_ret, returns)
        rows.append(
            WeekdayPooledRow(
                weekday=wd,
                name=WEEKDAY_NAMES[wd],
                mean_pct=mean_ret,
                median_pct=median_ret,
                volatility_pct=vol_ret,
                day_agree=day_agree,
                day_total=day_total,
                status=status_by_weekday.get(wd, "не значим"),
            )
        )
    return rows


def format_pooled_summary_table(rows: list[WeekdayPooledRow]) -> list[str]:
    headers = (
        "День",
        "Среднее, %",
        "Медиана, %",
        "Волатильность, %",
        "Доля дней по знаку",
        "Кол-во наблюдений",
        "Статус",
    )
    body = [
        (
            row.name,
            _fmt_effect(row.mean_pct),
            _fmt_effect(row.median_pct),
            _fmt_volatility(row.volatility_pct),
            _day_sign_share_text(row.day_agree, row.day_total),
            str(row.day_total),
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
        "=== Сводная таблица: полный пул (все пары, весь период) ===",
        "",
        "Среднее / медиана / волатильность — intraday return (close−open)/open×100 по всем дням weekday.",
        "Волатильность — стандартное отклонение return по дням выборки.",
        "Доля дней по знаку — % дней, закрывшихся в сторону среднего (рост при mean>0, падение при mean<0).",
        "Статус: пересечение статусов из таблиц «универсальность среди пар» и «устойчивость во времени».",
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
