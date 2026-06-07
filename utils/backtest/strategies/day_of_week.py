"""Стратегия day_of_week: long Пт/Сб, short Чт, flat остальные дни."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import (
    PortfolioAnalytics,
    build_portfolio_analytics,
    information_ratio,
    weekday_correlation_matrix,
)
from crypto_research.utils.backtest.benchmark import (
    build_btc_buy_hold_portfolio,
    build_buy_hold_portfolio,
    filter_daily_by_weekday_pairs,
)
from crypto_research.utils.backtest.fees import DEFAULT_FEE, FeeSchedule
from crypto_research.utils.backtest.paths import (
    backtest_drawdown_plot_path,
    backtest_equity_plot_path,
    backtest_report_path,
    backtest_returns_hist_plot_path,
    backtest_weekday_corr_plot_path,
)
from crypto_research.utils.backtest.plots import (
    save_drawdown_plot,
    save_equity_curve_plot,
    save_returns_histogram_plot,
    save_weekday_corr_plot,
)
from crypto_research.utils.backtest.report import BacktestResult, save_backtest_report
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_CONSERVATIVE,
    SCENARIO_OPTIMISTIC,
    scenario_label_ru,
)
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("strategy_day_of_week")

STRATEGY_NAME = "day_of_week"
TRADING_WEEKDAYS: tuple[int, ...] = (3, 4, 5)

STRATEGY_DESCRIPTION_MAXIMAL = (
    "Intraday UTC, равный вес 49 пар, без реинвестирования. Сигналы — исследование day_of_week.\n"
    "  Пятница, суббота: long на open → close. Доходность = (close−open)/open×100%.\n"
    "  Четверг: short на open → close. Доходность short = (open−close)/open×100% "
    "(эквивалент −(close−open)/open).\n"
    "  Остальные дни: flat."
)

STRATEGY_DESCRIPTION_OPTIMISTIC = (
    "Оптимистичный сценарий: val-период, без реинвестирования. Пары отобраны на train "
    "(знак Δ к BASE + ≥2/3 лет); свой набор на каждый торговый день.\n"
    "  Пятница, суббота: long на open → close. Доходность = (close−open)/open×100%.\n"
    "  Четверг: short на open → close. Доходность short = (open−close)/open×100% "
    "(эквивалент −(close−open)/open).\n"
    "  Остальные дни: flat."
)


STRATEGY_DESCRIPTION_CONSERVATIVE = (
    "Консервативный сценарий: val-период, все 49 пар, без реинвестирования. "
    "Baseline для сравнения с оптимистичным.\n"
    "  Пятница, суббота: long на open → close. Доходность = (close−open)/open×100%.\n"
    "  Четверг: short на open → close. Доходность short = (open−close)/open×100% "
    "(эквивалент −(close−open)/open).\n"
    "  Остальные дни: flat."
)


@dataclass(frozen=True)
class DayOfWeekBacktestContext:
    data_dir: object
    from_date: datetime
    to_date: datetime
    pairs: list[str] | None
    workers: int
    fee: FeeSchedule = DEFAULT_FEE
    scenario: str = "maximal"
    pairs_by_weekday: dict[int, list[str]] | None = None
    daily_benchmark_49: pl.DataFrame | None = None
    n_benchmark_pairs: int = 49


def _normalize_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    wd_min = int(wd.min())
    wd_max = int(wd.max())
    if wd_min >= 1 and wd_max <= 7:
        wd = ((wd - 1) % 7).cast(pl.Int64)
    return daily.with_columns(wd.alias("weekday"))


def _position_expr() -> pl.Expr:
    return (
        pl.when(pl.col("weekday") == 3)
        .then(-1.0)
        .when(pl.col("weekday").is_in([4, 5]))
        .then(1.0)
        .otherwise(0.0)
        .alias("position")
    )


def build_pair_returns(
    daily: pl.DataFrame,
    fee: FeeSchedule,
    *,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> pl.DataFrame:
    df = _normalize_weekday(daily)
    if pairs_by_weekday:
        df = filter_daily_by_weekday_pairs(df, pairs_by_weekday)
    df = df.with_columns(_position_expr())
    return df.with_columns(
        (pl.col("position") * pl.col("return_pct")).alias("gross_return_pct"),
    ).with_columns(
        pl.when(pl.col("position") != 0)
        .then(pl.col("gross_return_pct") - fee.round_trip_taker_pct)
        .otherwise(0.0)
        .alias("net_return_pct"),
        pl.when(pl.col("position") != 0)
        .then(pl.col("gross_return_pct") - fee.round_trip_maker_pct)
        .otherwise(0.0)
        .alias("net_maker_return_pct"),
    )


def build_portfolio_daily(pair_returns: pl.DataFrame) -> pl.DataFrame:
    return (
        pair_returns.group_by("day_utc")
        .agg(
            pl.col("gross_return_pct").mean().alias("gross_return_pct"),
            pl.col("net_return_pct").mean().alias("net_return_pct"),
            pl.col("net_maker_return_pct").mean().alias("net_maker_return_pct"),
            pl.col("weekday").first().alias("weekday"),
            pl.col("position").first().alias("position"),
        )
        .sort("day_utc")
    )


def _analytics_by_weekday(
    portfolio: pl.DataFrame,
    column: str,
) -> dict[int, PortfolioAnalytics]:
    out: dict[int, PortfolioAnalytics] = {}
    for wd in range(7):
        sub = portfolio.filter(pl.col("weekday") == wd)
        trade = sub["position"].to_numpy() != 0
        out[wd] = build_portfolio_analytics(sub, column, trading_mask=trade)
    return out


def _path_kwargs(ctx: DayOfWeekBacktestContext) -> dict:
    return {
        "scenario": ctx.scenario if ctx.scenario == SCENARIO_OPTIMISTIC else None,
        "pairs_by_weekday": ctx.pairs_by_weekday,
    }


def _aligned_benchmark_returns(
    portfolio: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    column: str = "net_return_pct",
) -> tuple[np.ndarray, np.ndarray]:
    merged = portfolio.select("day_utc", column).join(
        benchmark_df.select("day_utc", pl.col("gross_return_pct").alias("bh")),
        on="day_utc",
        how="inner",
    )
    return merged[column].to_numpy(), merged["bh"].to_numpy()


def _strategy_description(ctx: DayOfWeekBacktestContext) -> str:
    if ctx.scenario == SCENARIO_OPTIMISTIC:
        return STRATEGY_DESCRIPTION_OPTIMISTIC
    if ctx.scenario == SCENARIO_CONSERVATIVE:
        return STRATEGY_DESCRIPTION_CONSERVATIVE
    return STRATEGY_DESCRIPTION_MAXIMAL


def run_day_of_week_backtest(
    daily: pl.DataFrame,
    pairs: list[str],
    ctx: DayOfWeekBacktestContext,
) -> BacktestResult:
    pair_returns = build_pair_returns(
        daily, ctx.fee, pairs_by_weekday=ctx.pairs_by_weekday
    )
    portfolio = build_portfolio_daily(pair_returns)
    bh_daily = ctx.daily_benchmark_49 if ctx.daily_benchmark_49 is not None else daily
    benchmark_df = build_buy_hold_portfolio(bh_daily)
    btc_df = build_btc_buy_hold_portfolio(bh_daily)
    active = portfolio["position"].to_numpy() != 0

    portfolio_net = build_portfolio_analytics(
        portfolio, "net_return_pct", trading_mask=active
    )
    portfolio_gross = build_portfolio_analytics(
        portfolio, "gross_return_pct", trading_mask=active
    )
    portfolio_net_maker = build_portfolio_analytics(
        portfolio, "net_maker_return_pct", trading_mask=active
    )
    benchmark = build_portfolio_analytics(
        benchmark_df,
        "gross_return_pct",
        trading_mask=np.ones(benchmark_df.height, dtype=bool),
    )
    benchmark_btc = None
    if btc_df is not None:
        benchmark_btc = build_portfolio_analytics(
            btc_df,
            "gross_return_pct",
            trading_mask=np.ones(btc_df.height, dtype=bool),
        )
    strat_ret, bh_ret = _aligned_benchmark_returns(portfolio, benchmark_df)
    ir = information_ratio(strat_ret, bh_ret)
    strat_maker, _ = _aligned_benchmark_returns(
        portfolio, benchmark_df, column="net_maker_return_pct"
    )
    ir_maker = information_ratio(strat_maker, bh_ret)
    corr = weekday_correlation_matrix(portfolio, "net_maker_return_pct")

    result = BacktestResult(
        strategy=STRATEGY_NAME,
        strategy_description=_strategy_description(ctx),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        pairs=pairs,
        fee=ctx.fee,
        portfolio_net=portfolio_net,
        portfolio_gross=portfolio_gross,
        portfolio_net_maker=portfolio_net_maker,
        benchmark=benchmark,
        benchmark_btc=benchmark_btc,
        information_ratio_net=ir,
        information_ratio_net_maker=ir_maker,
        by_weekday_net=_analytics_by_weekday(portfolio, "net_return_pct"),
        by_weekday_gross=_analytics_by_weekday(portfolio, "gross_return_pct"),
        weekday_corr=corr,
        trading_weekdays=TRADING_WEEKDAYS,
        scenario=ctx.scenario,
        pairs_by_weekday=ctx.pairs_by_weekday,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
    )

    path_kw = _path_kwargs(ctx)
    tag_args = (STRATEGY_NAME, len(pairs), ctx.from_date, ctx.to_date)
    scenario_labels = {
        SCENARIO_CONSERVATIVE: scenario_label_ru(SCENARIO_CONSERVATIVE),
        SCENARIO_OPTIMISTIC: scenario_label_ru(SCENARIO_OPTIMISTIC),
    }
    save_backtest_report(result, backtest_report_path(*tag_args, **path_kw))
    save_equity_curve_plot(
        portfolio,
        benchmark_df,
        btc=btc_df,
        trading_weekdays=TRADING_WEEKDAYS,
        strategy=STRATEGY_NAME,
        scenario_label=scenario_labels.get(ctx.scenario),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        n_pairs=len(pairs),
        path=backtest_equity_plot_path(*tag_args, **path_kw),
    )
    save_drawdown_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        trading_weekdays=TRADING_WEEKDAYS,
        scenario_label=scenario_labels.get(ctx.scenario),
        path=backtest_drawdown_plot_path(*tag_args, **path_kw),
    )
    save_returns_histogram_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        path=backtest_returns_hist_plot_path(*tag_args, **path_kw),
    )
    save_weekday_corr_plot(
        *corr,
        trading_weekdays=TRADING_WEEKDAYS,
        strategy=STRATEGY_NAME,
        path=backtest_weekday_corr_plot_path(*tag_args, **path_kw),
    )

    log.info("[backtest] report: %s", backtest_report_path(*tag_args, **path_kw))
    _log_summary(result)
    return result


def _log_summary(result: BacktestResult) -> None:
    m = result.portfolio_net.metrics
    b = result.benchmark.metrics
    log.info(
        "[backtest] %s net=%+.2f%% vs B&H=%+.2f%% sharpe=%.2f IR_taker=%.2f IR_maker=%.2f",
        result.strategy,
        m.total_return_pct,
        b.total_return_pct,
        m.sharpe,
        result.information_ratio_net,
        result.information_ratio_net_maker,
    )
