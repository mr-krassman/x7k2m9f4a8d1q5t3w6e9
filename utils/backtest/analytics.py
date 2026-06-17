"""Расширенная аналитика портфеля."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

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
    n_trades: int = 0


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


def _active_trade_returns(pair_returns: pl.DataFrame, column: str) -> np.ndarray:
    active = pair_returns.filter(pl.col("position") != 0)
    if active.is_empty():
        return np.array([], dtype=np.float64)
    rets = active[column].to_numpy().astype(np.float64, copy=False)
    return rets[np.isfinite(rets)]


def trade_win_rate_pct(pair_returns: pl.DataFrame, column: str) -> float:
    """Доля прибыльных сделок: строки pair_returns с position != 0."""
    rets = _active_trade_returns(pair_returns, column)
    if rets.size == 0:
        return 0.0
    return float(np.sum(rets > 0) / rets.size * 100.0)


def trade_profit_factor(trade_returns: np.ndarray) -> float:
    if trade_returns.size == 0:
        return 0.0
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0
    return gross_win / abs(gross_loss) if gross_loss < 0 else 0.0


def build_strategy_portfolio_analytics(
    portfolio: pl.DataFrame,
    pair_returns: pl.DataFrame,
    *,
    trading_mask: np.ndarray,
) -> tuple[PortfolioAnalytics, PortfolioAnalytics, PortfolioAnalytics]:
    """Сводка портфеля (net / gross / net maker) с win rate по сделкам."""
    kw = {"trading_mask": trading_mask, "pair_returns": pair_returns}
    return (
        build_portfolio_analytics(portfolio, "net_return_pct", **kw),
        build_portfolio_analytics(portfolio, "gross_return_pct", **kw),
        build_portfolio_analytics(portfolio, "net_maker_return_pct", **kw),
    )


def build_portfolio_analytics(
    portfolio: pl.DataFrame,
    column: str,
    *,
    trading_mask: np.ndarray | None = None,
    pair_returns: pl.DataFrame | None = None,
) -> PortfolioAnalytics:
    dates = portfolio["day_utc"].to_numpy()
    returns = portfolio[column].to_numpy()
    weekdays = portfolio["weekday"].to_numpy()

    if trading_mask is None:
        trade = returns != 0.0
    else:
        trade = np.asarray(trading_mask, dtype=bool)

    metrics = compute_metrics(returns, trading_mask=trade)
    trade_rets = _active_trade_returns(pair_returns, column) if pair_returns is not None else None
    if trade_rets is not None and trade_rets.size > 0:
        metrics = replace(
            metrics,
            win_rate_pct=trade_win_rate_pct(pair_returns, column),
            profit_factor=trade_profit_factor(trade_rets),
        )
        per_trade = trade_rets
        n_trades = int(trade_rets.size)
    else:
        per_trade = returns[trade]
        n_trades = 0

    exposure_frac = metrics.exposure_pct / 100.0
    sharpe_exp = metrics.sharpe / np.sqrt(exposure_frac) if exposure_frac > 0 else 0.0
    skew, kurt = _distribution_stats(per_trade)

    return PortfolioAnalytics(
        metrics=metrics,
        cagr_pct=compute_cagr_pct(metrics.total_return_pct, metrics.n_obs),
        avg_all_days_pct=float(returns.mean()) if returns.size else 0.0,
        avg_trading_day_pct=float(per_trade.mean()) if per_trade.size else 0.0,
        trade_std_pct=float(per_trade.std(ddof=1)) if per_trade.size > 1 else 0.0,
        trade_median_pct=float(np.median(per_trade)) if per_trade.size else 0.0,
        skewness=skew,
        kurtosis=kurt,
        var_1_pct=historical_var_pct(per_trade, 1.0),
        var_5_pct=historical_var_pct(per_trade, 5.0),
        cvar_1_pct=historical_cvar_pct(per_trade, 1.0),
        cvar_5_pct=historical_cvar_pct(per_trade, 5.0),
        drawdown=compute_drawdown_stats(dates, returns),
        sharpe_per_exposure=sharpe_exp,
        best_day=_day_extreme(dates, weekdays, returns, best=True),
        worst_day=_day_extreme(dates, weekdays, returns, best=False),
        n_trades=n_trades,
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
    *,
    pair_returns: pl.DataFrame | None = None,
) -> dict[int, PortfolioAnalytics]:
    by_wd: dict[int, PortfolioAnalytics] = {}
    for wd in range(7):
        view, trade = _weekday_portfolio_view(portfolio, wd, column)
        pr = pair_returns.filter(pl.col("weekday") == wd) if pair_returns is not None else None
        by_wd[wd] = build_portfolio_analytics(
            view, column, trading_mask=trade, pair_returns=pr
        )
    return by_wd


def peak_eligible_pairs_per_day(
    pairs_by_weekday: dict[int, list[str]] | None,
    n_pairs: int,
    *,
    trading_weekdays: tuple[int, ...] | None = None,
) -> int:
    """Макс. число пар, допущенных в один день (для веса позиции 1/peak)."""
    if pairs_by_weekday:
        weekdays = trading_weekdays if trading_weekdays is not None else tuple(pairs_by_weekday)
        counts = [len(pairs_by_weekday.get(wd, ())) for wd in weekdays]
        counts = [c for c in counts if c > 0]
        if counts:
            return max(counts)
    return max(n_pairs, 1)


PositionWeightMode = Literal["peak", "active_daily"]
POSITION_WEIGHT_PEAK = "peak"
POSITION_WEIGHT_ACTIVE_DAILY = "active_daily"

# Единственное место переключения режима веса позиций во всех бэктестах.
# peak — фиксированно 1/peak_pairs; active_daily — 1/число активных позиций в день.
BACKTEST_POSITION_WEIGHT_MODE: PositionWeightMode = POSITION_WEIGHT_PEAK


def build_portfolio_daily_weighted(
    pair_returns: pl.DataFrame,
    peak_pairs: int | None = None,
    *,
    weight_mode: PositionWeightMode | None = None,
    first_cols: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Портфель: вес позиции задаётся weight_mode (по умолчанию BACKTEST_POSITION_WEIGHT_MODE).

    - active_daily: 1 / число активных позиций в этот день (10 позиций → 0.1 на каждую).
    - peak: фиксированно 1 / peak_pairs на каждую активную позицию.
    """
    mode = weight_mode if weight_mode is not None else BACKTEST_POSITION_WEIGHT_MODE
    if mode == POSITION_WEIGHT_PEAK:
        if peak_pairs is None or peak_pairs < 1:
            raise ValueError(f"peak_pairs must be >= 1 for weight_mode='peak', got {peak_pairs}")
    if "weekday" in pair_returns.columns:
        calendar = pair_returns.group_by("day_utc").agg(pl.col("weekday").first())
    else:
        calendar = pair_returns.group_by("day_utc").agg(
            ((pl.col("day_utc").dt.weekday() - 1) % 7).first().alias("weekday")
        )
    active = pair_returns.filter(pl.col("position") != 0)
    if mode == POSITION_WEIGHT_ACTIVE_DAILY:
        active = active.with_columns(
            (1.0 / pl.len().over("day_utc")).alias("_weight")
        )
    else:
        active = active.with_columns(pl.lit(1.0 / peak_pairs).alias("_weight"))
    agg_exprs: list[pl.Expr] = [
        (pl.col("gross_return_pct") * pl.col("_weight")).sum().alias("gross_return_pct"),
        (pl.col("net_return_pct") * pl.col("_weight")).sum().alias("net_return_pct"),
        (pl.col("net_maker_return_pct") * pl.col("_weight")).sum().alias("net_maker_return_pct"),
        pl.lit(1.0).alias("position"),
    ]
    for col in first_cols:
        agg_exprs.append(pl.col(col).first().alias(col))
    active_daily = active.group_by("day_utc").agg(*agg_exprs)
    out = (
        calendar.join(active_daily, on="day_utc", how="left")
        .with_columns(
            pl.col("gross_return_pct").fill_null(0.0),
            pl.col("net_return_pct").fill_null(0.0),
            pl.col("net_maker_return_pct").fill_null(0.0),
            pl.col("position").fill_null(0.0),
        )
        .sort("day_utc")
    )
    for col in first_cols:
        if col in out.columns:
            out = out.with_columns(pl.col(col).fill_null(""))
    return out


def build_portfolio_daily_peak_weighted(
    pair_returns: pl.DataFrame,
    peak_pairs: int,
    *,
    first_cols: tuple[str, ...] = (),
) -> pl.DataFrame:
    return build_portfolio_daily_weighted(
        pair_returns,
        peak_pairs,
        weight_mode=POSITION_WEIGHT_PEAK,
        first_cols=first_cols,
    )


def avg_active_long_short_pairs_by_weekday(
    pair_returns: pl.DataFrame,
) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
    """Пары: среднее число активных в день; Long/Short: общее число сделок по weekday."""
    daily = (
        pair_returns.group_by("day_utc", "weekday")
        .agg(
            (pl.col("position") > 0).sum().alias("n_long"),
            (pl.col("position") < 0).sum().alias("n_short"),
        )
        .with_columns((pl.col("n_long") + pl.col("n_short")).alias("n_active"))
        .filter(pl.col("n_active") > 0)
    )
    total: dict[int, float] = {}
    long_: dict[int, int] = {}
    short: dict[int, int] = {}
    for wd in range(7):
        sub = daily.filter(pl.col("weekday") == wd)
        if sub.height == 0:
            total[wd] = 0.0
            long_[wd] = 0
            short[wd] = 0
        else:
            total[wd] = float(sub["n_active"].mean())
            long_[wd] = int(sub["n_long"].sum())
            short[wd] = int(sub["n_short"].sum())
    return total, long_, short


def avg_active_pairs_by_weekday(pair_returns: pl.DataFrame) -> dict[int, float]:
    total, _, _ = avg_active_long_short_pairs_by_weekday(pair_returns)
    return total
