"""Формат отчёта бэктеста."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from crypto_research.utils.backtest.analytics import PortfolioAnalytics, WEEKDAY_NAMES
from crypto_research.utils.backtest.fees import FeeSchedule
from crypto_research.utils.backtest.scenarios import SCENARIO_REPORT_HEADER

RISK_FREE_NOTE = "Sharpe / Sortino: безрисковая ставка = 0% (стандарт для крипто)."
_PAIRS_PER_LINE = 11


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    strategy_description: str
    from_date: datetime
    to_date: datetime
    pairs: list[str]
    fee: FeeSchedule
    portfolio_net: PortfolioAnalytics
    portfolio_gross: PortfolioAnalytics
    portfolio_net_maker: PortfolioAnalytics
    benchmark: PortfolioAnalytics
    benchmark_btc: PortfolioAnalytics | None
    information_ratio_net: float
    information_ratio_net_maker: float
    by_weekday_net: dict[int, PortfolioAnalytics]
    by_weekday_gross: dict[int, PortfolioAnalytics]
    weekday_corr: tuple[tuple[str, ...], np.ndarray]
    trading_weekdays: tuple[int, ...] = (3, 4, 5)
    scenario: str = "maximal"
    pairs_by_weekday: dict[int, list[str]] | None = None
    n_benchmark_pairs: int = 49


def _fmt_pct(value: float, digits: int = 2) -> str:
    if value != value:
        return "n/a"
    return f"{value:+.{digits}f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_day_extreme(label: str, day) -> str:
    if day is None:
        return f"  {label}: n/a"
    dt = _as_date(day.date)
    date_s = dt.date().isoformat() if dt is not None else "n/a"
    return f"  {label}: {_fmt_pct(day.return_pct)} — {date_s} ({WEEKDAY_NAMES[day.weekday]})"


def _as_date(dt) -> datetime | None:
    if dt is None:
        return None
    if isinstance(dt, np.datetime64):
        return datetime.utcfromtimestamp(int(dt.astype("datetime64[s]").astype(int)))
    if hasattr(dt, "to_pydatetime"):
        return dt.to_pydatetime()
    return dt


def _recovery_text(dd) -> str:
    if dd.recovery_days is None:
        return "не восстановился к концу периода"
    rec = _as_date(dd.recovery_date)
    if rec is None:
        return f"{dd.recovery_days} календарных дней"
    return f"{dd.recovery_days} календарных дней (до {rec.date().isoformat()})"


def _analytics_block(title: str, a: PortfolioAnalytics, *, compact: bool = False) -> list[str]:
    m = a.metrics
    lines = [
        title,
        "",
        f"  NAV start / end (simple, base=100): 100.00 → {100.0 + m.total_return_pct:.2f}",
        f"  Total Return:              {_fmt_pct(m.total_return_pct)}",
        f"  Ann. Return (linear):      {_fmt_pct(m.ann_return_pct)}",
        f"  CAGR (compounded):         {_fmt_pct(a.cagr_pct)}",
        f"  Volatility (ann.):         {_fmt_pct(m.volatility_ann_pct)}",
        f"  Sharpe:                    {_fmt_num(m.sharpe)}  ({RISK_FREE_NOTE})",
        f"  Sortino:                   {_fmt_num(m.sortino)}",
        f"  Sharpe / sqrt(exposure):   {_fmt_num(a.sharpe_per_exposure)}",
        f"  Max Drawdown:              {_fmt_pct(m.max_drawdown_pct)}",
        f"  Recovery from max DD:      {_recovery_text(a.drawdown)}",
        f"  Longest underwater:        {a.drawdown.longest_underwater_days} календарных дней",
        f"  % time in drawdown:        {_fmt_num(a.drawdown.pct_time_in_drawdown, 1)}%",
        f"  Calmar:                    {_fmt_num(m.calmar)}",
        f"  Win Rate (trades):         {_fmt_pct(m.win_rate_pct, 1)}",
        f"  Profit Factor:             {_fmt_num(m.profit_factor)}",
        f"  Avg return (all days):     {_fmt_pct(a.avg_all_days_pct, 3)}",
        f"  Avg return (trading days): {_fmt_pct(a.avg_trading_day_pct, 3)}",
        f"  Trade return σ:            {_fmt_pct(a.trade_std_pct, 3)}",
        f"  Trade return median:       {_fmt_pct(a.trade_median_pct, 3)}",
    ]
    if not compact:
        lines.extend(
            [
                _fmt_day_extreme("Best day", a.best_day),
                _fmt_day_extreme("Worst day", a.worst_day),
                f"  Exposure:                  {_fmt_pct(m.exposure_pct, 1)} "
                f"({m.n_trading}/{m.n_obs} days in market)",
            ]
        )
    lines.append("")
    return lines


def _distribution_row(label: str, gross_val: float, net_val: float, *, pct: bool = False) -> str:
    if pct:
        g = _fmt_pct(gross_val, 1)
        n = _fmt_pct(net_val, 1)
    else:
        g = _fmt_num(gross_val, 2)
        n = _fmt_num(net_val, 2)
    return f"{label:<12} {g:>10} {n:>10}"


def _distribution_block(gross: PortfolioAnalytics, net: PortfolioAnalytics) -> list[str]:
    return [
        "=== Статистика распределения (только торговые дни) ===",
        "",
        f"{'':12} {'Gross':>10} {'Net':>10}",
        _distribution_row("Skewness:", gross.skewness, net.skewness),
        _distribution_row("Kurtosis:", gross.kurtosis, net.kurtosis),
        _distribution_row("1% VaR:", gross.var_1_pct, net.var_1_pct, pct=True),
        _distribution_row("5% VaR:", gross.var_5_pct, net.var_5_pct, pct=True),
        _distribution_row("1% CVaR:", gross.cvar_1_pct, net.cvar_1_pct, pct=True),
        _distribution_row("5% CVaR:", gross.cvar_5_pct, net.cvar_5_pct, pct=True),
        "",
        "VaR — перцентиль худших дней; CVaR (Expected Shortfall) — средний убыток в хвосте ≤ VaR.",
        "",
    ]


def _skew_kurtosis_weekday_table(
    by_wd: dict[int, PortfolioAnalytics],
    trading_weekdays: tuple[int, ...],
) -> list[str]:
    lines = [
        "=== Skewness & Kurtosis by weekday (net) ===",
        "",
        f"{'':6} {'Skewness':>10} {'Kurtosis':>10}",
        "-" * 28,
    ]
    for wd in trading_weekdays:
        a = by_wd[wd]
        lines.append(
            f"{WEEKDAY_NAMES[wd]:4}   {_fmt_num(a.skewness):>10} {_fmt_num(a.kurtosis):>10}"
        )
    lines.append("")
    return lines


def _benchmark_btc_block(btc: PortfolioAnalytics | None) -> list[str]:
    if btc is None:
        return [
            "=== Benchmark (BTC Buy & Hold, gross) ===",
            "",
            "  btcusdt: данные недоступны в выборке.",
            "",
        ]
    return _analytics_block("=== Benchmark (BTC Buy & Hold, gross) ===", btc)


def _fee_commentary(result: BacktestResult) -> list[str]:
    fee = result.fee
    net = result.portfolio_net.metrics
    maker = result.portfolio_net_maker.metrics
    return [
        "=== Комментарий к комиссиям ===",
        "",
        f"Net-метрики по умолчанию консервативны: taker {fee.taker_pct:.3f}% × 2 = "
        f"{fee.round_trip_taker_pct:.3f}% round-trip на каждый торговый день.",
        f"При лимитных ордерах (maker {fee.maker_pct:.3f}% × 2 = "
        f"{fee.round_trip_maker_pct:.3f}% round-trip) net total ≈ "
        f"{maker.total_return_pct:+.2f}% — ближе к gross ({result.portfolio_gross.metrics.total_return_pct:+.2f}%).",
        "",
    ]


def _exposure_commentary(result: BacktestResult) -> list[str]:
    exp = result.portfolio_net.metrics.exposure_pct
    return [
        "=== Exposure и сравнение с бенчмарком ===",
        "",
        f"Exposure стратегии: {exp:.1f}%. Капитал в рынке только в Чт/Пт/Сб.",
        "Прямое сравнение Sharpe с Buy & Hold (100% exposure) требует осторожности.",
        f"Information Ratio (net taker vs B&H {result.n_benchmark_pairs} пар gross): "
        f"{_fmt_num(result.information_ratio_net)}.",
        f"Information Ratio (net maker vs B&H {result.n_benchmark_pairs} пар gross): "
        f"{_fmt_num(result.information_ratio_net_maker)}.",
        f"Sharpe / sqrt(exposure) (net taker): {_fmt_num(result.portfolio_net.sharpe_per_exposure)}.",
        "",
    ]


def _corr_cell(val: float) -> str:
    return f"{val:6.2f}" if np.isfinite(val) else "   n/a"


def _corr_block(
    labels: tuple[str, ...],
    corr: np.ndarray,
    trading_weekdays: tuple[int, ...],
) -> list[str]:
    sub_labels = [labels[wd] for wd in trading_weekdays]
    sub = corr[np.ix_(trading_weekdays, trading_weekdays)]
    lines = [
        "=== Корреляция дневных доходностей (ISO-week aligned) ===",
        "",
        "Только торговые дни. Корреляция net maker-доходности портфеля внутри одной календарной недели.",
        "",
    ]
    header = "     | " + " | ".join(f"{c:>6}" for c in sub_labels)
    sep = "-" * len(header)
    lines.extend([header, sep])
    for i, row_name in enumerate(sub_labels):
        cells = [_corr_cell(sub[i, j]) for j in range(len(sub_labels))]
        lines.append(f"{row_name:4} | " + " | ".join(cells))
    lines.append("")
    return lines


def _weekday_table(
    title: str,
    by_wd: dict[int, PortfolioAnalytics],
    trading_weekdays: tuple[int, ...],
) -> list[str]:
    lines = [title, ""]
    header = (
        "День | Trades | Total Ret | Avg trade | Median | σ trade | Sharpe | MaxDD"
    )
    lines.extend([header, "-" * len(header)])
    for wd in trading_weekdays:
        a = by_wd[wd]
        m = a.metrics
        lines.append(
            f"{WEEKDAY_NAMES[wd]:4} | {m.n_trading:6} | "
            f"{_fmt_pct(m.total_return_pct):>9} | "
            f"{_fmt_pct(a.avg_trading_day_pct, 3):>9} | "
            f"{_fmt_pct(a.trade_median_pct, 3):>6} | "
            f"{_fmt_pct(a.trade_std_pct, 3):>7} | "
            f"{_fmt_num(m.sharpe):>6} | {_fmt_pct(m.max_drawdown_pct):>5}"
        )
    lines.append("")
    return lines


def _pairs_by_weekday_block(pairs_by_weekday: dict[int, list[str]]) -> list[str]:
    lines = [
        "=== Пары по торговым дням (train-отбор) ===",
        "",
    ]
    for wd in (3, 4, 5):
        pairs = pairs_by_weekday.get(wd, [])
        direction = "short" if wd == 3 else "long"
        lines.append(f"{WEEKDAY_NAMES[wd]} — {direction}: {len(pairs)} пар")
        for i in range(0, len(pairs), _PAIRS_PER_LINE):
            chunk = pairs[i : i + _PAIRS_PER_LINE]
            prefix = "  " if i == 0 else "    "
            lines.append(f"{prefix}{', '.join(chunk)}")
        lines.append("")
    return lines


def _benchmark_bh_title(n_pairs: int) -> str:
    return f"=== Benchmark (Buy & Hold, equal weight {n_pairs} pairs, gross) ==="


def format_backtest_report(result: BacktestResult) -> str:
    lines = [
        f"=== Backtest: {result.strategy} ===",
        "",
        result.strategy_description,
        "",
    ]
    header = SCENARIO_REPORT_HEADER.get(result.scenario)
    if header:
        lines.append(header)
    lines.extend([
        f"Период (UTC): {result.from_date:%Y-%m-%d} .. {result.to_date:%Y-%m-%d}",
        f"Пар (union): {len(result.pairs)}",
        f"Комиссии (net): {result.fee.name} — maker {_fmt_pct(result.fee.maker_pct, 3)}, "
        f"taker {_fmt_pct(result.fee.taker_pct, 3)}, "
        f"round-trip taker {_fmt_pct(result.fee.round_trip_taker_pct, 3)}",
        "",
        "Метрики нормированы: NAV=100, простая накопленная доходность, без привязки к депозиту.",
        "Gross — до комиссий; Net — после taker round-trip; Net (maker) — сценарий лимитных ордеров.",
        "",
    ])
    if result.pairs_by_weekday:
        lines.extend(_pairs_by_weekday_block(result.pairs_by_weekday))
    lines.extend(_analytics_block("=== Portfolio (net of fees, taker) ===", result.portfolio_net))
    lines.extend(_analytics_block("=== Portfolio (gross, no fees) ===", result.portfolio_gross))
    lines.extend(
        _analytics_block("=== Portfolio (net, maker scenario) ===", result.portfolio_net_maker)
    )
    lines.extend(
        _analytics_block(
            _benchmark_bh_title(result.n_benchmark_pairs),
            result.benchmark,
        )
    )
    lines.extend(_benchmark_btc_block(result.benchmark_btc))
    lines.extend(_exposure_commentary(result))
    lines.extend(_fee_commentary(result))
    lines.extend(
        _weekday_table("=== By weekday (net) ===", result.by_weekday_net, result.trading_weekdays)
    )
    lines.extend(
        _weekday_table("=== By weekday (gross) ===", result.by_weekday_gross, result.trading_weekdays)
    )
    lines.extend(_skew_kurtosis_weekday_table(result.by_weekday_net, result.trading_weekdays))
    lines.extend(_distribution_block(result.portfolio_gross, result.portfolio_net))
    lines.extend(_corr_block(*result.weekday_corr, result.trading_weekdays))
    return "\n".join(lines)


def save_backtest_report(result: BacktestResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_backtest_report(result), encoding="utf-8")
    return path
