"""Стратегия price_sequences: short после 3д роста, long после 3д+ падения."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import (
    PortfolioAnalytics,
    analytics_by_weekday,
    avg_active_long_short_pairs_by_weekday,
    weekday_total_ret_by_side,
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
    ALL_WEEKDAYS,
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
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.price_sequences.pair_selection import PS_TRAIN_SIGNALS
from crypto_research.utils.price_sequences.streak import (
    STREAK_SIGNED_PREV_COL,
    attach_streak_signed_prev,
)

log = get_logger("strategy_price_sequences")

STRATEGY_NAME = "price_sequences"
TRADING_SEGMENTS: tuple[str, ...] = tuple(s.label for s in PS_TRAIN_SIGNALS)
WARMUP_CALENDAR_DAYS = 7

STRATEGY_DESCRIPTION_MAXIMAL = (
    "Intraday UTC, равный вес пар, без реинвестирования. Сигнал — исследование price_sequences.\n"
    "  Short: вчера завершилась серия ровно 3д роста (close>open).\n"
    "  Long: вчера завершилась серия ≥3д падения (close<open).\n"
    "  Остальные дни: flat."
)

STRATEGY_DESCRIPTION_CONSERVATIVE = (
    "Консервативный сценарий: val-период, все 49 пар.\n"
    "  Short после 3д роста; long после 3д+ падения.\n"
    "  Остальные дни: flat."
)

STRATEGY_DESCRIPTION_OPTIMISTIC = (
    "Оптимистичный сценарий: val, отдельный train-отбор пар для short (3d_up) и long (3d_down) "
    "× «Цена росла» (знак Δ + ≥2/3 лет).\n"
    "  Short open→close только на short-парах; long — только на long-парах.\n"
    "  Остальные дни: flat."
)


@dataclass(frozen=True)
class PriceSequencesBacktestContext:
    data_dir: object
    from_date: datetime
    to_date: datetime
    pairs: list[str]
    workers: int
    fee: FeeSchedule = DEFAULT_FEE
    scenario: str = "maximal"
    pairs_by_segment: dict[str, list[str]] | None = None
    selected_pairs: list[str] | None = None
    daily_benchmark_49: pl.DataFrame | None = None
    n_benchmark_pairs: int = 49


def _position_from_signed(signed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    short_up = signed == 3.0
    long_down = signed <= -3.0
    position = np.where(long_down, 1.0, np.where(short_up, -1.0, 0.0))
    segment = np.where(long_down, "3d_down_long", np.where(short_up, "3d_up_short", ""))
    return position, segment


def _apply_pairs_by_segment(df: pl.DataFrame, pairs_by_segment: dict[str, list[str]]) -> pl.DataFrame:
    short_pairs = pairs_by_segment.get("3d_up_short", [])
    long_pairs = pairs_by_segment.get("3d_down_long", [])
    return df.with_columns(
        pl.when(pl.col("segment") == "3d_up_short")
        .then(
            pl.when(pl.col("pair").is_in(short_pairs))
            .then(pl.col("position"))
            .otherwise(0.0)
        )
        .when(pl.col("segment") == "3d_down_long")
        .then(
            pl.when(pl.col("pair").is_in(long_pairs))
            .then(pl.col("position"))
            .otherwise(0.0)
        )
        .otherwise(pl.col("position"))
        .alias("position"),
    ).with_columns(
        pl.when(pl.col("position") == 0)
        .then(pl.lit(""))
        .otherwise(pl.col("segment"))
        .alias("segment"),
    )


def build_pair_returns(
    daily: pl.DataFrame,
    fee: FeeSchedule,
    *,
    pairs_by_segment: dict[str, list[str]] | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> pl.DataFrame:
    work = attach_streak_signed_prev(daily)
    signed = work[STREAK_SIGNED_PREV_COL].to_numpy().astype(np.float64, copy=False)
    position, segment = _position_from_signed(signed)
    df = work.with_columns(
        pl.Series("position", position),
        pl.Series("segment", segment),
    )
    if pairs_by_segment is not None:
        df = _apply_pairs_by_segment(df, pairs_by_segment)
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
    )


def _analytics_by_segment(
    portfolio: pl.DataFrame,
    pair_returns: pl.DataFrame,
    column: str,
    peak_pairs: int,
) -> dict[str, PortfolioAnalytics]:
    out: dict[str, PortfolioAnalytics] = {}
    for seg in TRADING_SEGMENTS:
        sub_pr = pair_returns.filter(pl.col("segment") == seg)
        if sub_pr.is_empty():
            continue
        sub_port = build_portfolio_daily_weighted(
            sub_pr,
            peak_pairs,
            first_cols=("segment",),
        )
        active = sub_port.filter(pl.col("position") != 0)
        if active.is_empty():
            continue
        out[seg] = build_portfolio_analytics(
            active,
            column,
            trading_mask=np.ones(active.height, dtype=bool),
            pair_returns=sub_pr,
        )
    return out


def _peak_active_pairs(pair_returns: pl.DataFrame) -> int:
    active = pair_returns.filter(pl.col("position") != 0)
    if active.is_empty():
        return 1
    by_day = active.group_by("day_utc").agg(pl.len().alias("n"))
    return max(int(by_day["n"].max()), 1)


def _path_kwargs(ctx: PriceSequencesBacktestContext) -> dict:
    kw: dict = {
        "scenario": ctx.scenario if ctx.scenario == SCENARIO_OPTIMISTIC else None,
        "selected_pairs": ctx.selected_pairs,
    }
    if ctx.pairs_by_segment is not None:
        short_n = len(ctx.pairs_by_segment.get("3d_up_short", ()))
        long_n = len(ctx.pairs_by_segment.get("3d_down_long", ()))
        kw["optimistic_segment"] = f"sh{short_n}_lg{long_n}"
    return kw


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


def _strategy_description(ctx: PriceSequencesBacktestContext) -> str:
    if ctx.scenario == SCENARIO_OPTIMISTIC:
        return STRATEGY_DESCRIPTION_OPTIMISTIC
    if ctx.scenario == SCENARIO_CONSERVATIVE:
        return STRATEGY_DESCRIPTION_CONSERVATIVE
    return STRATEGY_DESCRIPTION_MAXIMAL


def run_price_sequences_backtest(
    daily: pl.DataFrame,
    pairs: list[str],
    ctx: PriceSequencesBacktestContext,
) -> BacktestResult:
    pair_returns = build_pair_returns(
        daily,
        ctx.fee,
        pairs_by_segment=ctx.pairs_by_segment,
        from_date=ctx.from_date,
        to_date=ctx.to_date,
    )
    peak_pairs = (
        _peak_active_pairs(pair_returns)
        if ctx.pairs_by_segment is not None
        else len(pairs)
    )
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
    side_totals = weekday_total_ret_by_side(pair_returns_wd, peak_pairs)
    result = BacktestResult(
        strategy=STRATEGY_NAME,
        strategy_description=_strategy_description(ctx),
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        pairs=ctx.selected_pairs if ctx.selected_pairs is not None else pairs,
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
        weekday_total_ret_by_side=side_totals,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
        exposure_note=(
            "Капитал в рынке в дни с сигналом серии "
            "(3d_up_short при 3д роста, 3d_down_long при ≥3д падения)."
        ),
        by_segment_net=_analytics_by_segment(
            portfolio, pair_returns_wd, "net_maker_return_pct", peak_pairs
        ),
        by_segment_gross=_analytics_by_segment(
            portfolio, pair_returns_wd, "gross_return_pct", peak_pairs
        ),
        trading_segments=TRADING_SEGMENTS,
        selected_pairs=ctx.selected_pairs,
        selected_pairs_by_segment=ctx.pairs_by_segment,
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
        trading_weekdays=ALL_WEEKDAYS,
        scenario_label=scenario_labels.get(ctx.scenario),
        path=backtest_drawdown_plot_path(*tag_args, **path_kw),
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
