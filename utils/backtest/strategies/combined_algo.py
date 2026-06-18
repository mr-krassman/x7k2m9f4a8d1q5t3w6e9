"""Combined rule-based backtest: несколько стратегий (optimistic), mode and|or."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import (
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
from crypto_research.utils.backtest.bundle_registry import AlgoBundleSpec, CombineMode
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
from crypto_research.utils.backtest.scenarios import SCENARIO_OPTIMISTIC
from crypto_research.utils.backtest.strategies.day_of_week import build_pair_returns as build_dow_pair_returns
from crypto_research.utils.backtest.strategies.ema_spreads import build_pair_returns as build_ema_pair_returns
from crypto_research.utils.backtest.strategies.rsi_spreads import build_pair_returns as build_rsi_pair_returns
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("strategy_combined_algo")

STRATEGY_DESCRIPTION = (
    "Combined rule-based (optimistic): сигналы нескольких стратегий с train-отбором пар.\n"
    "  mode=or — все сделки складываются (каждый сигнал — отдельная позиция).\n"
    "  mode=and — сделка только при совпадении знака позиции у всех сигналов на (день, пара)."
)


@dataclass(frozen=True)
class CombinedAlgoBacktestContext:
    data_dir: object
    from_date: datetime
    to_date: datetime
    pairs: list[str]
    workers: int
    fee: FeeSchedule = DEFAULT_FEE
    algo_spec: AlgoBundleSpec | None = None
    pairs_by_weekday: dict[int, list[str]] | None = None
    ema_selected_pairs: list[str] | None = None
    rsi_selected_pairs: list[str] | None = None
    frozen_thresholds: pl.DataFrame | None = None
    frozen_edges: np.ndarray | None = None
    ema_period: int = 9
    rsi_period: int = 9
    daily_benchmark_49: pl.DataFrame | None = None
    n_benchmark_pairs: int = 49
    strategy_name: str = "dow_ema_sp"
    bundle_id: str | None = None
    bundle_kind: str = "algo"


def _with_weekday(df: pl.DataFrame) -> pl.DataFrame:
    if "weekday" in df.columns:
        return df
    return df.with_columns(
        ((pl.col("day_utc").dt.weekday() - 1) % 7).cast(pl.Int64).alias("weekday")
    )


def _positions_frame(df: pl.DataFrame) -> pl.DataFrame:
    return _with_weekday(df).select(
        "day_utc",
        "pair",
        "weekday",
        "return_pct",
        pl.col("position").alias("position"),
    )


def _active_signals(df: pl.DataFrame) -> pl.DataFrame:
    return _positions_frame(df).filter(pl.col("position") != 0)


def _combine_or(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat([_active_signals(f) for f in frames], how="vertical_relaxed")


def _combine_and(frames: list[pl.DataFrame]) -> pl.DataFrame:
    keys = ["day_utc", "pair"]
    merged = _positions_frame(frames[0])
    pos_cols = ["position"]

    for i, frame in enumerate(frames[1:], start=1):
        pf = _positions_frame(frame).rename(
            {
                "position": f"pos_{i}",
                "weekday": f"wd_{i}",
                "return_pct": f"ret_{i}",
            }
        )
        merged = merged.join(pf, on=keys, how="full", coalesce=True)
        pos_cols.append(f"pos_{i}")

    pos_arrays = [merged[c].fill_null(0.0).to_numpy() for c in pos_cols]
    all_nonzero = np.ones(pos_arrays[0].shape[0], dtype=bool)
    for arr in pos_arrays:
        all_nonzero &= arr != 0
    ref_sign = np.sign(pos_arrays[0])
    same = all_nonzero.copy()
    for arr in pos_arrays[1:]:
        same &= np.sign(arr) == ref_sign
    combined = np.where(same, pos_arrays[0], 0.0)

    weekday_expr = pl.col("weekday")
    return_expr = pl.col("return_pct")
    for i in range(1, len(frames)):
        weekday_expr = weekday_expr.fill_null(pl.col(f"wd_{i}"))
        return_expr = return_expr.fill_null(pl.col(f"ret_{i}"))

    return merged.with_columns(
        weekday_expr.cast(pl.Int64).alias("weekday"),
        return_expr.alias("return_pct"),
        pl.Series("position", combined),
    ).select("day_utc", "pair", "weekday", "return_pct", "position")


def combine_positions(frames: list[pl.DataFrame], mode: CombineMode) -> pl.DataFrame:
    if not frames:
        raise ValueError("combine_positions: пустой список стратегий")
    if mode == "or":
        return _combine_or(frames)
    return _combine_and(frames)


def apply_fees(df: pl.DataFrame, fee: FeeSchedule) -> pl.DataFrame:
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


def peak_active_pairs(pair_returns: pl.DataFrame) -> int:
    active = pair_returns.filter(pl.col("position") != 0)
    if active.is_empty():
        return 1
    by_day = active.group_by("day_utc").agg(pl.len().alias("n"))
    return max(int(by_day["n"].max()), 1)


def _path_kwargs(ctx: CombinedAlgoBacktestContext) -> dict:
    return {
        "scenario": SCENARIO_OPTIMISTIC,
        "bundle_id": ctx.bundle_id,
        "bundle_kind": ctx.bundle_kind,
        "combine_mode": ctx.algo_spec.combine_mode if ctx.algo_spec else None,
    }


def _build_study_returns(daily: pl.DataFrame, ctx: CombinedAlgoBacktestContext) -> list[pl.DataFrame]:
    if ctx.algo_spec is None:
        raise RuntimeError("algo_spec не задан")
    out: list[pl.DataFrame] = []
    for study in ctx.algo_spec.studies:
        if study == "day_of_week":
            out.append(
                build_dow_pair_returns(daily, ctx.fee, pairs_by_weekday=ctx.pairs_by_weekday)
            )
        elif study == "ema_spreads":
            if ctx.frozen_thresholds is None:
                raise RuntimeError("frozen_thresholds не заданы для ema_spreads")
            out.append(
                build_ema_pair_returns(
                    daily,
                    ctx.fee,
                    period=ctx.ema_period,
                    frozen_thresholds=ctx.frozen_thresholds,
                    pair_filter=ctx.ema_selected_pairs,
                    from_date=ctx.from_date,
                    to_date=ctx.to_date,
                )
            )
        elif study == "rsi_spreads":
            if ctx.frozen_edges is None:
                raise RuntimeError("frozen_edges не заданы для rsi_spreads")
            out.append(
                build_rsi_pair_returns(
                    daily,
                    ctx.fee,
                    period=ctx.rsi_period,
                    frozen_edges=ctx.frozen_edges,
                    pair_filter=ctx.rsi_selected_pairs,
                    from_date=ctx.from_date,
                    to_date=ctx.to_date,
                )
            )
        else:
            raise ValueError(f"Неизвестная стратегия в combined algo: {study}")
    return out


def run_combined_algo_backtest(
    daily: pl.DataFrame,
    pairs: list[str],
    ctx: CombinedAlgoBacktestContext,
) -> BacktestResult:
    if ctx.algo_spec is None:
        raise RuntimeError("algo_spec не задан")

    mode = ctx.algo_spec.combine_mode
    study_frames = _build_study_returns(daily, ctx)
    combined_pos = combine_positions(study_frames, mode)
    pair_returns = apply_fees(combined_pos, ctx.fee)
    peak_pairs = peak_active_pairs(pair_returns)
    portfolio = build_portfolio_daily_weighted(pair_returns, peak_pairs)

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

    merged = portfolio.select("day_utc", "net_return_pct", "net_maker_return_pct").join(
        benchmark_df.select("day_utc", pl.col("gross_return_pct").alias("bh")),
        on="day_utc",
        how="inner",
    )
    ir = information_ratio(merged["net_return_pct"].to_numpy(), merged["bh"].to_numpy())
    ir_maker = information_ratio(merged["net_maker_return_pct"].to_numpy(), merged["bh"].to_numpy())
    corr = weekday_correlation_matrix(portfolio, "net_maker_return_pct")
    active_pairs, long_pairs, short_pairs = avg_active_long_short_pairs_by_weekday(pair_returns)
    trading_weekdays = tuple(range(7))
    studies_label = "+".join(s.upper() for s in ctx.algo_spec.studies)

    desc = (
        f"{STRATEGY_DESCRIPTION}\n"
        f"  Bundle: {ctx.algo_spec.bundle_id}; studies: {', '.join(ctx.algo_spec.studies)}; mode={mode}."
    )
    selected = sorted(
        set(ctx.ema_selected_pairs or []) | set(ctx.rsi_selected_pairs or [])
    ) or None
    result = BacktestResult(
        strategy=ctx.strategy_name,
        strategy_description=desc,
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
        by_weekday_net=analytics_by_weekday(portfolio, "net_return_pct", pair_returns=pair_returns),
        by_weekday_net_maker=analytics_by_weekday(
            portfolio, "net_maker_return_pct", pair_returns=pair_returns
        ),
        by_weekday_gross=analytics_by_weekday(portfolio, "gross_return_pct", pair_returns=pair_returns),
        weekday_corr=corr,
        trading_weekdays=trading_weekdays,
        scenario=SCENARIO_OPTIMISTIC,
        pairs_by_weekday=ctx.pairs_by_weekday,
        avg_active_pairs_by_weekday=active_pairs,
        avg_long_pairs_by_weekday=long_pairs,
        avg_short_pairs_by_weekday=short_pairs,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
        selected_pairs=selected,
        exposure_note=(
            f"Combined algo mode={mode}; peak_slots={peak_pairs} "
            f"(макс. активных сигналов в день; or — {studies_label} складываются)."
        ),
    )

    path_kw = _path_kwargs(ctx)
    tag_args = (ctx.strategy_name, len(pairs), ctx.from_date, ctx.to_date)
    save_backtest_report(result, backtest_report_path(*tag_args, **path_kw))
    save_equity_curve_plot(
        portfolio,
        benchmark_df,
        btc=btc_df,
        trading_weekdays=trading_weekdays,
        strategy=ctx.strategy_name,
        scenario_label=f"mode={mode}",
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        n_pairs=len(pairs),
        path=backtest_equity_plot_path(*tag_args, **path_kw),
    )
    save_drawdown_plot(
        portfolio,
        strategy=ctx.strategy_name,
        trading_weekdays=trading_weekdays,
        scenario_label=f"mode={mode}",
        path=backtest_drawdown_plot_path(*tag_args, **path_kw),
    )
    save_returns_histogram_plot(
        portfolio,
        strategy=ctx.strategy_name,
        path=backtest_returns_hist_plot_path(*tag_args, **path_kw),
    )
    save_weekday_corr_plot(
        *corr,
        trading_weekdays=trading_weekdays,
        strategy=ctx.strategy_name,
        path=backtest_weekday_corr_plot_path(*tag_args, **path_kw),
    )
    log.info("[combined_algo] report saved for bundle=%s mode=%s", ctx.bundle_id, mode)
    return result
