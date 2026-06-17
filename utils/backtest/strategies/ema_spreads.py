"""Стратегия ema_spreads: long intraday при вчерашнем b6 (EMA dev < t1⁻)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import (
    PortfolioAnalytics,
    analytics_by_weekday,
    avg_active_long_short_pairs_by_weekday,
    build_strategy_portfolio_analytics,
    build_portfolio_analytics,
    build_portfolio_daily_weighted,
    information_ratio,
    weekday_correlation_matrix,
)
from crypto_research.utils.backtest.benchmark import (
    BTC_PAIR,
    benchmark_pair_returns,
    build_btc_buy_hold_portfolio,
    build_buy_hold_portfolio,
)
from crypto_research.utils.backtest.fees import DEFAULT_FEE, FeeSchedule
from crypto_research.utils.backtest.paths import (
    backtest_drawdown_plot_path,
    backtest_equity_plot_path,
    backtest_report_path,
    backtest_returns_hist_plot_path,
)
from crypto_research.utils.backtest.plots import (
    save_drawdown_plot,
    save_equity_curve_plot,
    save_returns_histogram_plot,
)
from crypto_research.utils.backtest.report import BacktestResult, save_backtest_report
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_CONSERVATIVE,
    SCENARIO_OPTIMISTIC,
    scenario_label_ru,
)
from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.ema import (
    assign_ema_dev_buckets_vectorized,
    build_ema_work_frame,
    build_pair_thresholds_frame,
    ema_dev_prev_column,
)
from crypto_research.utils.ema_spreads.pair_selection import EMA_BUCKET_B6, EMA_TRAIN_SIGNAL
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("strategy_ema_spreads")

STRATEGY_NAME = "ema_spreads"
TRADING_SEGMENTS: tuple[str, ...] = ("b6",)
WARMUP_CALENDAR_DAYS = 45

STRATEGY_DESCRIPTION_MAXIMAL = (
    "Intraday UTC, равный вес пар, без реинвестирования. Сигнал — исследование ema_spreads.\n"
    f"  EMA({SELECTED_EMA_PERIOD}): если вчера dev попал в b6 (dev < t1⁻) → long open→close.\n"
    "  Пороги b6 заморожены на train (2022-01-01 – 2024-04-01).\n"
    "  Остальные дни: flat."
)

STRATEGY_DESCRIPTION_CONSERVATIVE = (
    f"Консервативный сценарий: val-период, все 49 пар, EMA({SELECTED_EMA_PERIOD}), "
    "long при вчерашнем b6. Пороги b6 заморожены на train.\n"
    "  Доходность long = (close−open)/open×100%.\n"
    "  Остальные дни: flat."
)

STRATEGY_DESCRIPTION_OPTIMISTIC = (
    f"Оптимистичный сценарий: val, train-отбор пар по b6 × «Цена росла» "
    f"(знак Δ + ≥2/3 лет). EMA({SELECTED_EMA_PERIOD}), пороги b6 frozen train.\n"
    "  Long open→close при вчерашнем b6.\n"
    "  Остальные дни: flat."
)


@dataclass(frozen=True)
class EmaSpreadsBacktestContext:
    data_dir: object
    from_date: datetime
    to_date: datetime
    pairs: list[str]
    workers: int
    fee: FeeSchedule = DEFAULT_FEE
    scenario: str = "maximal"
    ema_period: int = SELECTED_EMA_PERIOD
    frozen_thresholds: pl.DataFrame | None = None
    selected_pairs: list[str] | None = None
    daily_benchmark_49: pl.DataFrame | None = None
    n_benchmark_pairs: int = 49


def compute_frozen_thresholds(daily_train: pl.DataFrame, period: int) -> pl.DataFrame:
    work = build_ema_work_frame(daily_train, (period,))
    prev_col = ema_dev_prev_column(period)
    return build_pair_thresholds_frame(work, prev_col)


def build_pair_returns(
    daily: pl.DataFrame,
    fee: FeeSchedule,
    *,
    period: int,
    frozen_thresholds: pl.DataFrame,
    pair_filter: list[str] | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> pl.DataFrame:
    work = build_ema_work_frame(daily, (period,))
    prev_col = ema_dev_prev_column(period)
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))
    merged = work.join(frozen_thresholds, on="pair", how="inner")
    if pair_filter:
        merged = merged.filter(pl.col("pair").is_in(pair_filter))

    dev = merged[prev_col].to_numpy().astype(np.float64, copy=False)
    buckets = assign_ema_dev_buckets_vectorized(
        dev,
        merged["t1_up"].to_numpy(),
        merged["t2_up"].to_numpy(),
        merged["t1_down"].to_numpy(),
        merged["t2_down"].to_numpy(),
        merged["near_abs"].to_numpy(),
    )
    position = np.where(buckets == EMA_BUCKET_B6, 1.0, 0.0)
    df = merged.with_columns(
        pl.Series("ema_bucket_prev", buckets),
        pl.Series("position", position),
    )
    if from_date is not None:
        df = df.filter(pl.col("day_utc") >= from_date)
    if to_date is not None:
        df = df.filter(pl.col("day_utc") <= to_date)
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
        pl.lit("b6").alias("segment"),
    )


def _analytics_by_segment(
    portfolio: pl.DataFrame,
    pair_returns: pl.DataFrame,
    column: str,
) -> dict[str, PortfolioAnalytics]:
    active = portfolio.filter(pl.col("position") != 0)
    return {
        "b6": build_portfolio_analytics(
            active,
            column,
            trading_mask=np.ones(active.height, dtype=bool) if active.height else None,
            pair_returns=pair_returns,
        )
    }


def _path_kwargs(ctx: EmaSpreadsBacktestContext) -> dict:
    return {
        "scenario": ctx.scenario if ctx.scenario == SCENARIO_OPTIMISTIC else None,
        "selected_pairs": ctx.selected_pairs,
        "optimistic_segment": EMA_TRAIN_SIGNAL.label,
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


def _strategy_description(ctx: EmaSpreadsBacktestContext) -> str:
    if ctx.scenario == SCENARIO_OPTIMISTIC:
        return STRATEGY_DESCRIPTION_OPTIMISTIC
    if ctx.scenario == SCENARIO_CONSERVATIVE:
        return STRATEGY_DESCRIPTION_CONSERVATIVE
    return STRATEGY_DESCRIPTION_MAXIMAL


def run_ema_spreads_backtest(
    daily: pl.DataFrame,
    pairs: list[str],
    ctx: EmaSpreadsBacktestContext,
) -> BacktestResult:
    if ctx.frozen_thresholds is None or ctx.frozen_thresholds.is_empty():
        raise RuntimeError("frozen_thresholds не заданы для ema_spreads backtest")

    pair_filter = ctx.selected_pairs if ctx.selected_pairs is not None else None
    pair_returns = build_pair_returns(
        daily,
        ctx.fee,
        period=ctx.ema_period,
        frozen_thresholds=ctx.frozen_thresholds,
        pair_filter=pair_filter,
        from_date=ctx.from_date,
        to_date=ctx.to_date,
    )
    peak_pairs = len(ctx.selected_pairs) if ctx.selected_pairs else len(pairs)
    portfolio = build_portfolio_daily_weighted(
        pair_returns,
        peak_pairs,
        first_cols=("segment",),
    )
    bh_daily = ctx.daily_benchmark_49 if ctx.daily_benchmark_49 is not None else daily
    benchmark_df = build_buy_hold_portfolio(bh_daily)
    btc_df = build_btc_buy_hold_portfolio(bh_daily)
    active = portfolio["position"].to_numpy() != 0
    bh_pairs = benchmark_pair_returns(bh_daily)

    portfolio_net, portfolio_gross, portfolio_net_maker = build_strategy_portfolio_analytics(
        portfolio, pair_returns, trading_mask=active
    )
    benchmark = build_portfolio_analytics(
        benchmark_df,
        "gross_return_pct",
        trading_mask=np.ones(benchmark_df.height, dtype=bool),
        pair_returns=bh_pairs,
    )
    benchmark_btc = None
    if btc_df is not None:
        btc_sub = bh_daily.filter(pl.col("pair").str.to_lowercase() == BTC_PAIR)
        benchmark_btc = build_portfolio_analytics(
            btc_df,
            "gross_return_pct",
            trading_mask=np.ones(btc_df.height, dtype=bool),
            pair_returns=benchmark_pair_returns(btc_sub) if not btc_sub.is_empty() else None,
        )
    strat_ret, bh_ret = _aligned_benchmark_returns(portfolio, benchmark_df)
    ir = information_ratio(strat_ret, bh_ret)
    strat_maker, _ = _aligned_benchmark_returns(
        portfolio, benchmark_df, column="net_maker_return_pct"
    )
    ir_maker = information_ratio(strat_maker, bh_ret)

    corr = weekday_correlation_matrix(portfolio, "net_maker_return_pct")
    pair_returns_wd = pair_returns.with_columns(
        ((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday")
    )
    active_pairs, long_pairs, short_pairs = avg_active_long_short_pairs_by_weekday(pair_returns_wd)
    result = BacktestResult(
        strategy=STRATEGY_NAME,
        strategy_description=_strategy_description(ctx),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        pairs=pairs if pair_filter is None else pair_filter,
        fee=ctx.fee,
        portfolio_net=portfolio_net,
        portfolio_gross=portfolio_gross,
        portfolio_net_maker=portfolio_net_maker,
        benchmark=benchmark,
        benchmark_btc=benchmark_btc,
        information_ratio_net=ir,
        information_ratio_net_maker=ir_maker,
        by_weekday_net=analytics_by_weekday(portfolio, "net_return_pct", pair_returns=pair_returns_wd),
        by_weekday_net_maker=analytics_by_weekday(
            portfolio, "net_maker_return_pct", pair_returns=pair_returns_wd
        ),
        by_weekday_gross=analytics_by_weekday(portfolio, "gross_return_pct", pair_returns=pair_returns_wd),
        weekday_corr=corr,
        trading_weekdays=tuple(range(7)),
        scenario=ctx.scenario,
        avg_active_pairs_by_weekday=active_pairs,
        avg_long_pairs_by_weekday=long_pairs,
        avg_short_pairs_by_weekday=short_pairs,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
        exposure_note=(
            f"Капитал в рынке только в дни с сигналом {EMA_TRAIN_SIGNAL.label} "
            f"(вчера dev < t1⁻, EMA({ctx.ema_period}))."
        ),
        by_segment_net=_analytics_by_segment(portfolio, pair_returns_wd, "net_maker_return_pct"),
        by_segment_gross=_analytics_by_segment(portfolio, pair_returns_wd, "gross_return_pct"),
        trading_segments=TRADING_SEGMENTS,
        selected_pairs=ctx.selected_pairs,
        plot_layout="simple",
    )

    path_kw = _path_kwargs(ctx)
    tag_args = (STRATEGY_NAME, len(result.pairs), ctx.from_date, ctx.to_date)
    scenario_labels = {
        SCENARIO_CONSERVATIVE: scenario_label_ru(SCENARIO_CONSERVATIVE),
        SCENARIO_OPTIMISTIC: scenario_label_ru(SCENARIO_OPTIMISTIC),
    }
    save_backtest_report(result, backtest_report_path(*tag_args, **path_kw))
    save_equity_curve_plot(
        portfolio,
        benchmark_df,
        btc=btc_df,
        trading_weekdays=(),
        strategy=STRATEGY_NAME,
        scenario_label=scenario_labels.get(ctx.scenario),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        n_pairs=len(result.pairs),
        path=backtest_equity_plot_path(*tag_args, **path_kw),
        layout="simple",
    )
    save_drawdown_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        trading_weekdays=(),
        scenario_label=scenario_labels.get(ctx.scenario),
        path=backtest_drawdown_plot_path(*tag_args, **path_kw),
        layout="simple",
    )
    save_returns_histogram_plot(
        portfolio,
        strategy=STRATEGY_NAME,
        path=backtest_returns_hist_plot_path(*tag_args, **path_kw),
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
