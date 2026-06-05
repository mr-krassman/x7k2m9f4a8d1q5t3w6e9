"""Нормированные метрики бэктеста (crypto fund style)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 365.25
INITIAL_NAV = 100.0


@dataclass(frozen=True)
class PerformanceMetrics:
    n_obs: int
    n_trading: int
    total_return_pct: float
    ann_return_pct: float
    volatility_ann_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    win_rate_pct: float
    profit_factor: float
    avg_return_pct: float
    best_pct: float
    worst_pct: float
    exposure_pct: float


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        n_obs=0,
        n_trading=0,
        total_return_pct=0.0,
        ann_return_pct=0.0,
        volatility_ann_pct=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown_pct=0.0,
        calmar=0.0,
        win_rate_pct=0.0,
        profit_factor=0.0,
        avg_return_pct=0.0,
        best_pct=0.0,
        worst_pct=0.0,
        exposure_pct=0.0,
    )


def equity_curve_simple(returns_pct: np.ndarray, initial: float = INITIAL_NAV) -> np.ndarray:
    if returns_pct.size == 0:
        return np.array([initial], dtype=np.float64)
    return initial + np.cumsum(returns_pct.astype(np.float64))


def max_drawdown_pct(nav: np.ndarray) -> float:
    if nav.size < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak * 100.0
    return float(dd.min())


def compute_metrics(
    returns_pct: np.ndarray,
    *,
    trading_mask: np.ndarray | None = None,
) -> PerformanceMetrics:
    arr = np.asarray(returns_pct, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return _empty_metrics()

    if trading_mask is not None:
        trade = np.asarray(trading_mask, dtype=bool)[:n]
        if trade.size != n:
            trade = np.zeros(n, dtype=bool)
    else:
        trade = arr != 0.0

    n_trading = int(np.sum(trade))
    total = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    years = n / TRADING_DAYS_PER_YEAR
    ann_return = total / years if years > 0 else 0.0
    vol_ann = std * np.sqrt(TRADING_DAYS_PER_YEAR)

    downside = arr[arr < 0]
    down_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    down_ann = down_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = ann_return / vol_ann if vol_ann > 0 else 0.0
    sortino = ann_return / down_ann if down_ann > 0 else 0.0

    nav = equity_curve_simple(arr)
    mdd = max_drawdown_pct(nav)
    calmar = ann_return / abs(mdd) if mdd < 0 else 0.0

    traded = arr[trade]
    wins = traded[traded > 0]
    losses = traded[traded < 0]
    win_rate = float(wins.size / n_trading * 100.0) if n_trading > 0 else 0.0
    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0
    pf = gross_win / abs(gross_loss) if gross_loss < 0 else 0.0

    exposure = n_trading / n * 100.0 if n > 0 else 0.0

    return PerformanceMetrics(
        n_obs=n,
        n_trading=n_trading,
        total_return_pct=total,
        ann_return_pct=ann_return,
        volatility_ann_pct=vol_ann,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=mdd,
        calmar=calmar,
        win_rate_pct=win_rate,
        profit_factor=pf,
        avg_return_pct=mean,
        best_pct=float(arr.max()),
        worst_pct=float(arr.min()),
        exposure_pct=exposure,
    )
