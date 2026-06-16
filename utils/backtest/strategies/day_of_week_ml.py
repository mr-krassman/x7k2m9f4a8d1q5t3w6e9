"""ML-стратегия day_of_week: frozen model + frozen policy (thresholds + pairs_by_weekday)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from crypto_research.utils.backtest.analytics import (
    PortfolioAnalytics,
    analytics_by_weekday,
    avg_active_long_short_pairs_by_weekday,
    build_portfolio_analytics,
    build_portfolio_daily_peak_weighted,
    information_ratio,
    peak_eligible_pairs_per_day,
    weekday_correlation_matrix,
)
from crypto_research.utils.backtest.benchmark import build_btc_buy_hold_portfolio, build_buy_hold_portfolio
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
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("strategy_day_of_week_ml")

STRATEGY_NAME = "day_of_week_ml"
STRATEGY_DESCRIPTION = (
    "ML сценарий: frozen LightGBM (weekday+pair) + frozen policy из train-CPCV.\n"
    "  Пороговые правила: long если P(up)>=t_long, short если P(up)<=t_short, иначе flat.\n"
    "  Отбор (weekday,pair) из policy.json; применяется без дообучения на holdout test."
)


@dataclass(frozen=True)
class DayOfWeekMlPolicy:
    t_long: float
    t_short: float
    pairs_by_weekday: dict[int, list[str]]


@dataclass(frozen=True)
class DayOfWeekMlBacktestContext:
    data_dir: object
    from_date: datetime
    to_date: datetime
    pairs: list[str] | None
    workers: int
    fee: FeeSchedule = DEFAULT_FEE
    scenario: str | None = None
    policy: DayOfWeekMlPolicy | None = None
    model_bundle: dict | None = None
    daily_benchmark_49: pl.DataFrame | None = None
    n_benchmark_pairs: int = 49


def _normalize_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    wd_min = int(wd.min())
    wd_max = int(wd.max())
    if wd_min >= 1 and wd_max <= 7:
        wd = ((wd - 1) % 7).cast(pl.Int64)
    return daily.with_columns(wd.alias("weekday"))


def _predict_probabilities(df: pl.DataFrame, model_bundle: dict) -> np.ndarray:
    pair_classes = list(model_bundle["pair_encoder_classes"])
    weekday_classes = [int(x) for x in model_bundle["weekday_encoder_classes"]]
    pair_map = {p: i for i, p in enumerate(pair_classes)}
    wd_map = {wd: i for i, wd in enumerate(weekday_classes)}
    pair_id = np.array([pair_map[p] for p in df["pair"].to_list()], dtype=np.int32)
    weekday_enc = np.array([wd_map[int(w)] for w in df["weekday"].to_list()], dtype=np.int32)
    x = pd.DataFrame(
        {
            "weekday_enc": pd.Series(weekday_enc.astype(str), dtype="category"),
            "pair_id": pd.Series(pair_id.astype(str), dtype="category"),
        }
    )
    model = model_bundle["model"]
    return model.predict_proba(x)[:, 1]


def build_pair_returns_ml(
    daily: pl.DataFrame,
    fee: FeeSchedule,
    *,
    policy: DayOfWeekMlPolicy,
    model_bundle: dict,
) -> pl.DataFrame:
    df = _normalize_weekday(daily)
    y_prob = _predict_probabilities(df, model_bundle)
    allowed = {
        (wd, pair)
        for wd, pairs in policy.pairs_by_weekday.items()
        for pair in pairs
    }
    allowed_mask = np.array([(int(w), p) in allowed for w, p in zip(df["weekday"].to_list(), df["pair"].to_list())])
    pos = np.zeros(df.height, dtype=np.float64)
    pos[(y_prob >= policy.t_long) & allowed_mask] = 1.0
    pos[(y_prob <= policy.t_short) & allowed_mask] = -1.0

    return df.with_columns(
        pl.Series("y_prob", y_prob),
        pl.Series("position", pos),
    ).with_columns(
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


def _analytics_by_weekday(
    portfolio: pl.DataFrame,
    column: str,
) -> dict[int, PortfolioAnalytics]:
    return analytics_by_weekday(portfolio, column)


def run_day_of_week_ml_backtest(
    daily: pl.DataFrame,
    pairs: list[str],
    ctx: DayOfWeekMlBacktestContext,
) -> BacktestResult:
    if ctx.policy is None or ctx.model_bundle is None:
        raise RuntimeError("policy/model_bundle не переданы в контекст day_of_week_ml")
    pair_returns = build_pair_returns_ml(
        daily,
        ctx.fee,
        policy=ctx.policy,
        model_bundle=ctx.model_bundle,
    )
    peak_pairs = peak_eligible_pairs_per_day(ctx.policy.pairs_by_weekday, len(pairs))
    portfolio = build_portfolio_daily_peak_weighted(pair_returns, peak_pairs)
    bh_daily = ctx.daily_benchmark_49 if ctx.daily_benchmark_49 is not None else daily
    benchmark_df = build_buy_hold_portfolio(bh_daily)
    btc_df = build_btc_buy_hold_portfolio(bh_daily)
    active = portfolio["position"].to_numpy() != 0

    portfolio_net = build_portfolio_analytics(portfolio, "net_return_pct", trading_mask=active)
    portfolio_gross = build_portfolio_analytics(portfolio, "gross_return_pct", trading_mask=active)
    portfolio_net_maker = build_portfolio_analytics(portfolio, "net_maker_return_pct", trading_mask=active)
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
    merged = portfolio.select("day_utc", "net_return_pct", "net_maker_return_pct").join(
        benchmark_df.select("day_utc", pl.col("gross_return_pct").alias("bh")),
        on="day_utc",
        how="inner",
    )
    ir = information_ratio(merged["net_return_pct"].to_numpy(), merged["bh"].to_numpy())
    ir_maker = information_ratio(merged["net_maker_return_pct"].to_numpy(), merged["bh"].to_numpy())
    corr = weekday_correlation_matrix(portfolio, "net_maker_return_pct")
    trading_weekdays = tuple(sorted([wd for wd, pairs_wd in ctx.policy.pairs_by_weekday.items() if pairs_wd]))
    active_pairs, long_pairs, short_pairs = avg_active_long_short_pairs_by_weekday(pair_returns)

    result = BacktestResult(
        strategy=STRATEGY_NAME,
        strategy_description=STRATEGY_DESCRIPTION,
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
        trading_weekdays=trading_weekdays,
        scenario=ctx.scenario,
        pairs_by_weekday=ctx.policy.pairs_by_weekday,
        avg_active_pairs_by_weekday=active_pairs,
        avg_long_pairs_by_weekday=long_pairs,
        avg_short_pairs_by_weekday=short_pairs,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
    )

    tag_args = (STRATEGY_NAME, len(pairs), ctx.from_date, ctx.to_date)
    path_kw: dict = {}
    save_backtest_report(result, backtest_report_path(*tag_args, **path_kw))
    save_equity_curve_plot(
        portfolio,
        benchmark_df,
        btc=btc_df,
        trading_weekdays=trading_weekdays,
        strategy=STRATEGY_NAME,
        scenario_label=None,
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        n_pairs=len(pairs),
        path=backtest_equity_plot_path(*tag_args, **path_kw),
    )
    save_drawdown_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        trading_weekdays=trading_weekdays,
        scenario_label=None,
        path=backtest_drawdown_plot_path(*tag_args, **path_kw),
    )
    save_returns_histogram_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        path=backtest_returns_hist_plot_path(*tag_args, **path_kw),
    )
    save_weekday_corr_plot(
        *corr,
        trading_weekdays=trading_weekdays,
        strategy=STRATEGY_NAME,
        path=backtest_weekday_corr_plot_path(*tag_args, **path_kw),
    )
    log.info("[backtest] report: %s", backtest_report_path(*tag_args, **path_kw))
    return result

