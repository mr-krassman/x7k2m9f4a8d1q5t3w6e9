"""Расширенная аналитика портфеля."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from crypto_research.utils.backtest.metrics import (
    INITIAL_NAV,
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    compute_metrics,
    equity_curve_simple,
    max_drawdown_pct,
)

WEEKDAY_NAMES = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


@dataclass(frozen=True)
class DayExtreme:
    date: datetime
    weekday: int
    return_pct: float


@dataclass(frozen=True)
class DrawdownStats:
    max_drawdown_pct: float
    trough_date: datetime | None
    peak_before_date: datetime | None
    recovery_date: datetime | None
    recovery_days: int | None
    longest_underwater_days: int
    pct_time_in_drawdown: float


@dataclass(frozen=True)
class PortfolioAnalytics:
    metrics: PerformanceMetrics
    cagr_pct: float
    avg_all_days_pct: float
    avg_trading_day_pct: float
    trade_std_pct: float
    trade_median_pct: float
    skewness: float
    kurtosis: float
    var_1_pct: float
    var_5_pct: float
    cvar_1_pct: float
    cvar_5_pct: float
    drawdown: DrawdownStats
    sharpe_per_exposure: float
    best_day: DayExtreme | None
    worst_day: DayExtreme | None


def compute_cagr_pct(total_return_pct: float, n_obs: int) -> float:
    years = n_obs / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    nav_end = INITIAL_NAV + total_return_pct
    if nav_end <= 0:
        return float("nan")
    return ((nav_end / INITIAL_NAV) ** (1.0 / years) - 1.0) * 100.0


def historical_var_pct(values: np.ndarray, quantile_pct: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, quantile_pct))


def historical_cvar_pct(values: np.ndarray, quantile_pct: float) -> float:
    if values.size == 0:
        return float("nan")
    var = float(np.percentile(values, quantile_pct))
    tail = values[values <= var]
    if tail.size == 0:
        return var
    return float(tail.mean())


def _distribution_stats(values: np.ndarray) -> tuple[float, float]:
    if values.size < 3:
        return 0.0, 0.0
    return float(_skew(values)), float(_kurtosis_excess(values))


def _skew(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def compute_underwater_metrics(nav: np.ndarray) -> tuple[int, float]:
    """Longest peak→recovery span and share of days below running peak."""
    n = nav.size
    if n == 0:
        return 0, 0.0
    run_peak = np.maximum.accumulate(nav)
    pct = float(np.sum(nav < run_peak - 1e-9)) / n * 100.0

    longest = 0
    i = 0
    while i < n:
        is_ath = i == 0 or nav[i] > run_peak[i - 1] + 1e-9
        if not is_ath:
            i += 1
            continue
        level = nav[i]
        rec = next((j for j in range(i + 1, n) if nav[j] >= level - 1e-9), None)
        if rec is not None:
            longest = max(longest, rec - i)
            i = rec
        else:
            longest = max(longest, (n - 1) - i)
            break
    return longest, pct


def compute_drawdown_stats(dates: np.ndarray, returns_pct: np.ndarray) -> DrawdownStats:
    if returns_pct.size == 0:
        return DrawdownStats(0.0, None, None, None, None, 0, 0.0)
    nav = equity_curve_simple(returns_pct)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak * 100.0
    trough_idx = int(np.argmin(dd))
    max_dd = float(dd[trough_idx])
    peak_idx = int(np.argmax(nav[: trough_idx + 1])) if trough_idx >= 0 else 0

    recovery_idx: int | None = None
    peak_level = nav[peak_idx]
    for i in range(trough_idx + 1, nav.size):
        if nav[i] >= peak_level:
            recovery_idx = i
            break

    def _date(idx: int) -> datetime:
        dt = dates[idx]
        if hasattr(dt, "to_pydatetime"):
            return dt.to_pydatetime()
        return dt

    recovery_days = recovery_idx - trough_idx if recovery_idx is not None else None
    longest_uw, pct_uw = compute_underwater_metrics(nav)
    return DrawdownStats(
        max_drawdown_pct=max_dd,
        trough_date=_date(trough_idx),
        peak_before_date=_date(peak_idx),
        recovery_date=_date(recovery_idx) if recovery_idx is not None else None,
        recovery_days=recovery_days,
        longest_underwater_days=longest_uw,
        pct_time_in_drawdown=pct_uw,
    )


def _day_extreme(dates: np.ndarray, weekdays: np.ndarray, returns: np.ndarray, best: bool) -> DayExtreme | None:
    if returns.size == 0:
        return None
    idx = int(np.argmax(returns) if best else np.argmin(returns))
    dt = dates[idx]
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    return DayExtreme(
        date=dt,
        weekday=int(weekdays[idx]),
        return_pct=float(returns[idx]),
    )


def build_portfolio_analytics(
    portfolio: pl.DataFrame,
    column: str,
    *,
    trading_mask: np.ndarray | None = None,
) -> PortfolioAnalytics:
    dates = portfolio["day_utc"].to_numpy()
    returns = portfolio[column].to_numpy()
    weekdays = portfolio["weekday"].to_numpy()

    if trading_mask is None:
        trade = returns != 0.0
    else:
        trade = np.asarray(trading_mask, dtype=bool)

    metrics = compute_metrics(returns, trading_mask=trade)
    traded = returns[trade]
    exposure_frac = metrics.exposure_pct / 100.0
    sharpe_exp = metrics.sharpe / np.sqrt(exposure_frac) if exposure_frac > 0 else 0.0
    skew, kurt = _distribution_stats(traded)

    return PortfolioAnalytics(
        metrics=metrics,
        cagr_pct=compute_cagr_pct(metrics.total_return_pct, metrics.n_obs),
        avg_all_days_pct=float(returns.mean()) if returns.size else 0.0,
        avg_trading_day_pct=float(traded.mean()) if traded.size else 0.0,
        trade_std_pct=float(traded.std(ddof=1)) if traded.size > 1 else 0.0,
        trade_median_pct=float(np.median(traded)) if traded.size else 0.0,
        skewness=skew,
        kurtosis=kurt,
        var_1_pct=historical_var_pct(traded, 1.0),
        var_5_pct=historical_var_pct(traded, 5.0),
        cvar_1_pct=historical_cvar_pct(traded, 1.0),
        cvar_5_pct=historical_cvar_pct(traded, 5.0),
        drawdown=compute_drawdown_stats(dates, returns),
        sharpe_per_exposure=sharpe_exp,
        best_day=_day_extreme(dates, weekdays, returns, best=True),
        worst_day=_day_extreme(dates, weekdays, returns, best=False),
    )


def information_ratio(strategy: np.ndarray, benchmark: np.ndarray) -> float:
    diff = strategy - benchmark
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return 0.0
    std = float(diff.std(ddof=1))
    if std == 0:
        return 0.0
    ann = float(diff.mean()) * TRADING_DAYS_PER_YEAR
    return ann / (std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _pivot_weekday_column(pivot: pl.DataFrame, wd: int) -> str | None:
    for key in (wd, str(wd)):
        if key in pivot.columns:
            return str(key)
    return None


def weekday_correlation_matrix(
    portfolio: pl.DataFrame,
    column: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    frame = portfolio.filter(pl.col("position") != 0).with_columns(
        pl.col("day_utc").dt.week().alias("iso_week")
    )
    pivot = frame.pivot(on="weekday", index="iso_week", values=column, aggregate_function="first")
    full = np.full((7, 7), np.nan)
    available = [wd for wd in range(7) if _pivot_weekday_column(pivot, wd) is not None]
    if len(available) < 2:
        np.fill_diagonal(full, 1.0)
        return WEEKDAY_NAMES, full

    for wi in available:
        for wj in available:
            col_i = _pivot_weekday_column(pivot, wi)
            col_j = _pivot_weekday_column(pivot, wj)
            assert col_i is not None and col_j is not None
            a = pivot[col_i].to_numpy().astype(np.float64)
            b = pivot[col_j].to_numpy().astype(np.float64)
            mask = np.isfinite(a) & np.isfinite(b)
            if wi == wj:
                full[wi, wj] = 1.0
            elif mask.sum() >= 3:
                full[wi, wj] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return WEEKDAY_NAMES, full


def drawdown_series(returns_pct: np.ndarray) -> np.ndarray:
    nav = equity_curve_simple(returns_pct)
    peak = np.maximum.accumulate(nav)
    return (nav - peak) / peak * 100.0


def _weekday_portfolio_view(
    portfolio: pl.DataFrame,
    wd: int,
    column: str,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Полный календарь val: доходность только в целевой weekday, иначе 0%."""
    returns = portfolio[column].to_numpy()
    weekdays = portfolio["weekday"].to_numpy()
    position = portfolio["position"].to_numpy()
    masked = np.where(weekdays == wd, returns, 0.0)
    trade = (weekdays == wd) & (position != 0)
    view = portfolio.with_columns(pl.Series(column, masked))
    return view, trade


def analytics_by_weekday(
    portfolio: pl.DataFrame,
    column: str,
) -> dict[int, PortfolioAnalytics]:
    by_wd: dict[int, PortfolioAnalytics] = {}
    for wd in range(7):
        view, trade = _weekday_portfolio_view(portfolio, wd, column)
        by_wd[wd] = build_portfolio_analytics(view, column, trading_mask=trade)
    return by_wd
