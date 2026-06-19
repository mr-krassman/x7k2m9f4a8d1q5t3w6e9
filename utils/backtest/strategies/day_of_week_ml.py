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
    weekday_total_ret_by_side,
    build_strategy_portfolio_analytics,
    build_portfolio_analytics,
    build_portfolio_daily_weighted,
    peak_eligible_pairs_per_day,
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
    backtest_weekday_corr_plot_path,
)
from crypto_research.utils.backtest.plots import (
    save_drawdown_plot,
    save_equity_curve_plot,
    save_returns_histogram_plot,
    save_weekday_corr_plot,
)
from crypto_research.utils.backtest.report import BacktestResult, save_backtest_report
from crypto_research.utils.ml.numeric_features import (
    NUMERIC_FEATURE_SPECS,
    attach_normalized_features,
    bounds_map_from_bundle,
    needs_return_pct,
)
from crypto_research.utils.ml.registry import FEATURE_PAIR_ID, FEATURE_WEEKDAY_ENC
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("strategy_day_of_week_ml")

STRATEGY_NAME = "day_of_week_ml"
STRATEGY_DESCRIPTION = (
    "ML сценарий: frozen LightGBM (feature-columns из model_bundle) + frozen policy из train-CPCV.\n"
    "  Пороговые правила: long если P(up)>=t_long, short если P(up)<=t_short, иначе flat.\n"
    "  Отбор (weekday,pair) из policy.json; применяется без дообучения на holdout test."
)

COMBINED_ML_DESCRIPTION = (
    "Combined ML: frozen LightGBM с фичами нескольких исследований + frozen policy из train-CPCV.\n"
    "  Пороговые правила: long если P(up)>=t_long, short если P(up)<=t_short, иначе flat.\n"
    "  Отбор пар по weekday-policy; модель и policy из combined bundle."
)


def _strategy_description(strategy_name: str) -> str:
    from crypto_research.utils.ml.registry import is_combined_bundle_id

    if is_combined_bundle_id(strategy_name):
        return COMBINED_ML_DESCRIPTION
    return STRATEGY_DESCRIPTION


@dataclass(frozen=True)
class DayOfWeekMlPolicy:
    t_long: float
    t_short: float
    pairs_by_weekday: dict[int, list[str]]
    allowed_pairs: list[str] | None = None


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
    strategy_name: str = STRATEGY_NAME
    ml_output_study: str | None = None
    bundle_id: str | None = None
    bundle_kind: str | None = None


def _normalize_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    wd_min = int(wd.min())
    wd_max = int(wd.max())
    if wd_min >= 1 and wd_max <= 7:
        wd = ((wd - 1) % 7).cast(pl.Int64)
    return daily.with_columns(wd.alias("weekday"))


def _attach_numeric_features(df: pl.DataFrame, model_bundle: dict) -> pl.DataFrame:
    feature_columns = list(model_bundle["feature_columns"])
    if not any(c in NUMERIC_FEATURE_SPECS for c in feature_columns):
        return df
    base_cols = ["day_utc", "pair", "day_close"]
    if needs_return_pct(feature_columns):
        base_cols.append("return_pct")
    work = (
        df.select([c for c in base_cols if c in df.columns])
        .with_row_count("row_idx")
        .sort(["pair", "day_utc"])
    )
    bounds_map = bounds_map_from_bundle(model_bundle, feature_columns)
    enriched = attach_normalized_features(work, feature_columns, bounds_map)
    return enriched.sort("row_idx")


def _build_model_features(df: pl.DataFrame, model_bundle: dict) -> pd.DataFrame:
    feature_columns: list[str] = list(model_bundle["feature_columns"])
    numeric_frame = _attach_numeric_features(df, model_bundle)
    frame: dict[str, pd.Series] = {}
    if FEATURE_PAIR_ID in feature_columns:
        pair_classes = list(model_bundle["pair_encoder_classes"])
        pair_map = {p: i for i, p in enumerate(pair_classes)}
        pair_id = np.array([pair_map[p] for p in df["pair"].to_list()], dtype=np.int32)
        frame[FEATURE_PAIR_ID] = pd.Series(pair_id.astype(str), dtype="category")
    if FEATURE_WEEKDAY_ENC in feature_columns:
        weekday_classes = [int(x) for x in model_bundle["weekday_encoder_classes"]]
        wd_map = {wd: i for i, wd in enumerate(weekday_classes)}
        weekday_enc = np.array([wd_map[int(w)] for w in df["weekday"].to_list()], dtype=np.int32)
        frame[FEATURE_WEEKDAY_ENC] = pd.Series(weekday_enc.astype(str), dtype="category")
    for column in feature_columns:
        if column in NUMERIC_FEATURE_SPECS:
            frame[column] = pd.Series(
                numeric_frame[column].to_numpy().astype(np.float64, copy=False),
                dtype=np.float64,
            )
    x = pd.DataFrame(frame)
    missing = [c for c in feature_columns if c not in x.columns]
    if missing:
        raise RuntimeError(f"Не удалось собрать признаки для модели, отсутствуют: {missing}")
    return x[feature_columns]


def _predict_probabilities(df: pl.DataFrame, model_bundle: dict) -> np.ndarray:
    x = _build_model_features(df, model_bundle)
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
    if policy.allowed_pairs is not None:
        allowed_set = set(policy.allowed_pairs)
        allowed_mask = np.array([p in allowed_set for p in df["pair"].to_list()])
    else:
        allowed = {
            (wd, pair)
            for wd, pairs in policy.pairs_by_weekday.items()
            for pair in pairs
        }
        allowed_mask = np.array(
            [(int(w), p) in allowed for w, p in zip(df["weekday"].to_list(), df["pair"].to_list())]
        )
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
    pair_returns: pl.DataFrame,
) -> dict[int, PortfolioAnalytics]:
    return analytics_by_weekday(portfolio, column, pair_returns=pair_returns)


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
    if ctx.policy.allowed_pairs is not None:
        peak_pairs = max(len(ctx.policy.allowed_pairs), 1)
        trading_weekdays = tuple(range(7))
    else:
        peak_pairs = peak_eligible_pairs_per_day(ctx.policy.pairs_by_weekday, len(pairs))
        trading_weekdays = tuple(sorted([wd for wd, pairs_wd in ctx.policy.pairs_by_weekday.items() if pairs_wd]))
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
    side_totals = weekday_total_ret_by_side(pair_returns, peak_pairs)

    result = BacktestResult(
        strategy=ctx.strategy_name,
        strategy_description=_strategy_description(ctx.strategy_name),
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
        by_weekday_net=_analytics_by_weekday(portfolio, "net_return_pct", pair_returns),
        by_weekday_net_maker=_analytics_by_weekday(portfolio, "net_maker_return_pct", pair_returns),
        by_weekday_gross=_analytics_by_weekday(portfolio, "gross_return_pct", pair_returns),
        weekday_corr=corr,
        trading_weekdays=trading_weekdays,
        scenario=ctx.scenario,
        pairs_by_weekday=None if ctx.policy.allowed_pairs is not None else ctx.policy.pairs_by_weekday,
        avg_active_pairs_by_weekday=active_pairs,
        avg_long_pairs_by_weekday=long_pairs,
        avg_short_pairs_by_weekday=short_pairs,
        weekday_total_ret_by_side=side_totals,
        n_benchmark_pairs=ctx.n_benchmark_pairs,
        selected_pairs=ctx.policy.allowed_pairs,
        exposure_note=(
            "Капитал в рынке определяется global-пулом разрешённых пар (без weekday-гейта)."
            if ctx.policy.allowed_pairs is not None
            else "Капитал в рынке определяется weekday-policy (разные списки по дням недели)."
        ),
    )

    tag_args = (ctx.strategy_name, len(pairs), ctx.from_date, ctx.to_date)
    path_kw: dict = {}
    if ctx.bundle_id and ctx.bundle_kind:
        path_kw["bundle_id"] = ctx.bundle_id
        path_kw["bundle_kind"] = ctx.bundle_kind
    elif ctx.ml_output_study is not None:
        path_kw["output_study"] = ctx.ml_output_study
    save_backtest_report(result, backtest_report_path(*tag_args, **path_kw))
    save_equity_curve_plot(
        portfolio,
        benchmark_df,
        btc=btc_df,
        trading_weekdays=trading_weekdays,
        strategy=ctx.strategy_name,
        scenario_label=None,
        from_date=ctx.from_date,
        to_date=ctx.to_date,
        n_pairs=len(pairs),
        path=backtest_equity_plot_path(*tag_args, **path_kw),
    )
    save_drawdown_plot(
        portfolio,
        strategy=ctx.strategy_name,
        trading_weekdays=trading_weekdays,
        scenario_label=None,
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
    log.info("[backtest] report: %s", backtest_report_path(*tag_args, **path_kw))
    return result

