import os
from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.study_ids import (
    STUDY_COMBINED,
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_PERIOD_SCREEN,
    STUDY_EMA_SPREADS,
    STUDY_EMA_HARAMI,
    STUDY_HARAMI,
    STUDY_VOLATILITY_PERIOD_SCREEN,
    STUDY_VOLATILITY_SPREADS,
    STUDY_RSI_PERIOD_SCREEN,
    STUDY_RSI_SPREADS,
    STUDY_VOLUME_EMA_PERIOD_SCREEN,
    STUDY_VOLUME_SPREADS,
    STUDY_PRICE_SEQUENCES,
)

RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
_JSONL_GLOB = "*_klines_1m.jsonl"
_LEGACY_DATA_DIR = RESEARCH_ROOT.parent / "load_data_from_bybit" / "data"

_STUDY_STATS_REL: dict[str, Path] = {
    STUDY_DAY_OF_WEEK: Path("day_of_week") / "statistics",
    STUDY_EMA_SPREADS: Path("ema") / "ema_spreads" / "statistics",
    STUDY_EMA_PERIOD_SCREEN: Path("ema") / "ema_spreads" / "statistics",
    STUDY_EMA_HARAMI: Path("ema") / "ema_harami" / "statistics",
    STUDY_RSI_PERIOD_SCREEN: Path("rsi") / "rsi_quantiles" / "statistics",
    STUDY_RSI_SPREADS: Path("rsi") / "rsi_quantiles" / "statistics",
    STUDY_VOLUME_EMA_PERIOD_SCREEN: Path("volume") / "volume_regimes" / "statistics",
    STUDY_VOLUME_SPREADS: Path("volume") / "volume_regimes" / "statistics",
    STUDY_PRICE_SEQUENCES: Path("price_sequences") / "statistics",
    STUDY_HARAMI: Path("candlestick") / "harami" / "statistics",
    STUDY_VOLATILITY_PERIOD_SCREEN: Path("volatility") / "volatility_spreads" / "statistics",
    STUDY_VOLATILITY_SPREADS: Path("volatility") / "volatility_spreads" / "statistics",
}


def resolve_default_data_dir() -> Path:
    if os.environ.get("CRYPTO_DATA_DIR"):
        return Path(os.environ["CRYPTO_DATA_DIR"]).expanduser().resolve()
    local = RESEARCH_ROOT / "data"
    if any(local.glob(_JSONL_GLOB)):
        return local
    if _LEGACY_DATA_DIR.is_dir() and any(_LEGACY_DATA_DIR.glob(_JSONL_GLOB)):
        return _LEGACY_DATA_DIR
    return local


DEFAULT_DATA_DIR = resolve_default_data_dir()

TRAIN_MAX_PAIR_START = "2022-01-01"
VAL_MAX_PAIR_START = "2023-01-01"

# Универсальность среди пар: train 24 / val 25, один период
PAIR_UNIVERSALITY_FROM = "2022-01-01"
PAIR_UNIVERSALITY_TO = "2026-05-31"

# Устойчивость во времени: 49 пар, два периода
TEMPORAL_POOL_MAX_PAIR_START = VAL_MAX_PAIR_START
TEMPORAL_TRAIN_FROM = "2022-01-01"
TEMPORAL_TRAIN_TO = "2024-04-01"
TRAIN_EVAL_FROM = "2023-10-01"
TEMPORAL_VAL_FROM = "2024-04-01"
TEMPORAL_VAL_TO = "2026-05-31"

# Полный пул: все пары, весь период исследования
FULL_POOL_MAX_PAIR_START = VAL_MAX_PAIR_START
FULL_POOL_FROM = PAIR_UNIVERSALITY_FROM
FULL_POOL_TO = PAIR_UNIVERSALITY_TO


def study_research_root(study: str) -> Path:
    """Корень артефактов исследования в research_outputs (без statistics/backtest)."""
    if study == STUDY_COMBINED:
        return RESEARCH_ROOT / "research_outputs" / "combined"
    stats = _STUDY_STATS_REL[study]
    return RESEARCH_ROOT / "research_outputs" / stats.parent


def study_stats_dir(study: str) -> Path:
    return RESEARCH_ROOT / "research_outputs" / _STUDY_STATS_REL[study]


def study_plots_dir(study: str) -> Path:
    return study_stats_dir(study) / "plots"


def study_backtest_dir(study: str) -> Path:
    return study_research_root(study) / "backtest"


def study_backtest_plots_dir(study: str) -> Path:
    return study_backtest_dir(study) / "plots"


def study_ml_dir(study: str) -> Path:
    return study_research_root(study) / "ml"


# Обратная совместимость (те же пути, что до унификации)
WEEKDAY_STATS_DIR = study_stats_dir(STUDY_DAY_OF_WEEK)
WEEKDAY_PLOTS_DIR = study_plots_dir(STUDY_DAY_OF_WEEK)
WEEKDAY_ML_DIR = study_ml_dir(STUDY_DAY_OF_WEEK)
WEEKDAY_ML_MODELS_DIR = WEEKDAY_ML_DIR / "models"
WEEKDAY_ML_METRICS_DIR = WEEKDAY_ML_DIR / "metrics"
WEEKDAY_ML_PLOTS_DIR = WEEKDAY_ML_DIR / "plots"
WEEKDAY_ML_POLICIES_DIR = WEEKDAY_ML_DIR / "policies"
EMA_SPREADS_STATS_DIR = study_stats_dir(STUDY_EMA_SPREADS)
RSI_QUANTILES_STATS_DIR = study_stats_dir(STUDY_RSI_PERIOD_SCREEN)


def weekday_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> str:
    tag = f"{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    if split:
        tag = f"{tag}_{split}"
    return tag


def weekday_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date, split)
    return WEEKDAY_STATS_DIR / f"weekday_statistics_{tag}.log"


def price_sequences_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date, split)
    return study_stats_dir(STUDY_PRICE_SEQUENCES) / f"price_sequences_statistics_{tag}.log"


def price_sequences_summary_log_path() -> Path:
    return study_stats_dir(STUDY_PRICE_SEQUENCES) / "price_sequences_summary.log"


def price_sequences_summary_plot_path(plot_kind: str) -> Path:
    return study_plots_dir(STUDY_PRICE_SEQUENCES) / f"price_sequences_summary_{plot_kind}.png"


def harami_summary_log_path() -> Path:
    return study_stats_dir(STUDY_HARAMI) / "harami_summary.log"


def harami_summary_plot_path(plot_kind: str) -> Path:
    return study_plots_dir(STUDY_HARAMI) / f"harami_summary_{plot_kind}.png"


def harami_signals_audit_log_path() -> Path:
    return study_stats_dir(STUDY_HARAMI) / "harami_signals_audit.log"


def ema_harami_summary_log_path(ema_period: int) -> Path:
    return study_stats_dir(STUDY_EMA_HARAMI) / f"ema_harami_summary_ema{ema_period}.log"


def ema_harami_summary_plot_path(ema_period: int, plot_kind: str) -> Path:
    return study_plots_dir(STUDY_EMA_HARAMI) / f"ema_harami_summary_{plot_kind}_ema{ema_period}.png"


def ema_harami_sample_candles_plot_path(ema_period: int) -> Path:
    return study_plots_dir(STUDY_EMA_HARAMI) / f"ema_harami_sample_candles_ema{ema_period}.png"


def weekday_summary_log_path() -> Path:
    return WEEKDAY_STATS_DIR / "weekday_summary.log"


def weekday_train_val_summary_log_path(
    n_train: int,
    n_val: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    tag = f"{n_train}train_{n_val}val_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return WEEKDAY_STATS_DIR / f"weekday_train_val_summary_{tag}.log"


def weekday_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date, split)
    return WEEKDAY_PLOTS_DIR / f"dow_intraday_session_nav_{tag}.png"


def weekday_check_plot_path(check: str) -> Path:
    """Train+val NAV на одном полотне для проверки (pair_universality | temporal_stability)."""
    return WEEKDAY_PLOTS_DIR / f"dow_check_{check}_train_val_nav.png"


def weekday_ml_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> str:
    from crypto_research.utils.ml.spec import ML_STUDY_DAY_OF_WEEK, resolve_ml_study

    return ml_output_tag(resolve_ml_study([ML_STUDY_DAY_OF_WEEK]), n_pairs, from_date, to_date)


def ml_output_tag(
    spec,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> str:
    if spec.legacy_output_tag:
        return f"weekday_direction_{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return f"direction_{spec.feature_slug}_{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def ml_log_tag(spec, from_date: datetime, to_date: datetime) -> str:
    if spec.legacy_output_tag:
        return f"weekday_direction_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return f"direction_{spec.feature_slug}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def ml_dir(spec) -> Path:
    if spec.bundle_id:
        from crypto_research.utils.backtest.bundle_registry import bundle_ml_root

        return bundle_ml_root(spec.bundle_id)
    return study_research_root(spec.output_study) / spec.ml_subdir


def ml_models_dir(spec) -> Path:
    return ml_dir(spec) / "models"


def ml_metrics_dir(spec) -> Path:
    return ml_dir(spec) / "metrics"


def ml_plots_dir(spec) -> Path:
    return ml_dir(spec) / "plots"


def ml_policies_dir(spec) -> Path:
    return ml_dir(spec) / "policies"


def ml_log_path(spec, from_date: datetime, to_date: datetime) -> Path:
    return ml_dir(spec) / f"{ml_log_tag(spec, from_date, to_date)}.log"


def ml_metrics_path(spec, n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, from_date, to_date)
    return ml_metrics_dir(spec) / f"{tag}_cpcv.json"


def ml_oos_plot_path(spec, n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, from_date, to_date)
    return ml_plots_dir(spec) / f"{tag}_oos_prob.png"


def ml_oos_calibration_plot_path(spec, n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, from_date, to_date)
    return ml_plots_dir(spec) / f"{tag}_oos_calibration.png"


def ml_roc_auc_plot_path(spec, n_pairs: int, train_from: datetime, val_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, val_to)
    return ml_plots_dir(spec) / f"{tag}_roc_auc.png"


def ml_weekday_pair_summary_plot_path(spec, n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, from_date, to_date)
    return ml_plots_dir(spec) / f"{tag}_weekday_pair_summary.png"


def ml_learning_curve_plot_path(spec, n_pairs: int, train_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_learning_curve.png"


def ml_learning_curve_log_path(spec, n_pairs: int, train_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_dir(spec) / f"{tag}_learning_curve.log"


def ml_feature_predictive_plot_path(
    spec,
    n_pairs: int,
    test_from: datetime,
    test_to: datetime,
    plot_slug: str,
) -> Path:
    tag = ml_output_tag(spec, n_pairs, test_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_{plot_slug}_predictive.png"


def ml_correlation_matrix_plot_path(
    spec, n_pairs: int, test_from: datetime, test_to: datetime
) -> Path:
    tag = ml_output_tag(spec, n_pairs, test_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_correlation_matrix.png"


def ml_shape_summary_plot_path(spec, n_pairs: int, test_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, test_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_shape_summary.png"


def ml_feature_prob_dependence_plot_path(
    spec, n_pairs: int, test_from: datetime, test_to: datetime
) -> Path:
    tag = ml_output_tag(spec, n_pairs, test_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_feature_prob_dependence.png"


def ml_prob_return_dependence_plot_path(
    spec, n_pairs: int, test_from: datetime, test_to: datetime
) -> Path:
    tag = ml_output_tag(spec, n_pairs, test_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_prob_return_dependence.png"


def ml_p_up_density_split_plot_path(
    spec, n_pairs: int, train_from: datetime, train_to: datetime
) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, train_to)
    return ml_plots_dir(spec) / f"{tag}_p_up_density_split.png"


def ml_compare_prob_cdf_plot_path(
    spec, n_pairs: int, train_from: datetime, test_to: datetime
) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_plots_dir(spec) / f"{tag}_compare_prob_cdf.png"


def ml_ema_dev_predictive_plot_path(spec, n_pairs: int, test_from: datetime, test_to: datetime) -> Path:
    return ml_feature_predictive_plot_path(spec, n_pairs, test_from, test_to, "ema_dev")


def ml_model_bundle_path(spec, n_pairs: int, train_from: datetime, train_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, train_to)
    return ml_models_dir(spec) / f"{tag}_model_bundle.pkl"


def ml_train_test_metrics_path(spec, n_pairs: int, train_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_metrics_dir(spec) / f"{tag}_train_test.json"


def ml_metrics_summary_path(spec, n_pairs: int, train_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_metrics_dir(spec) / f"{tag}_metrics_summary.json"


def ml_policy_path(spec, n_pairs: int, train_from: datetime, test_to: datetime) -> Path:
    tag = ml_output_tag(spec, n_pairs, train_from, test_to)
    return ml_policies_dir(spec) / f"{tag}_policy.json"


def weekday_ml_log_path(
    from_date: datetime,
    to_date: datetime,
) -> Path:
    tag = f"weekday_direction_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return WEEKDAY_ML_DIR / f"{tag}.log"


def weekday_ml_metrics_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    return WEEKDAY_ML_METRICS_DIR / f"{weekday_ml_output_tag(n_pairs, from_date, to_date)}_cpcv.json"


def weekday_ml_oos_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    return WEEKDAY_ML_PLOTS_DIR / f"{weekday_ml_output_tag(n_pairs, from_date, to_date)}_oos_prob.png"


def weekday_ml_oos_calibration_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    return WEEKDAY_ML_PLOTS_DIR / f"{weekday_ml_output_tag(n_pairs, from_date, to_date)}_oos_calibration.png"


def weekday_ml_roc_auc_plot_path(
    n_pairs: int,
    train_from: datetime,
    val_to: datetime,
) -> Path:
    return WEEKDAY_ML_PLOTS_DIR / f"{weekday_ml_output_tag(n_pairs, train_from, val_to)}_roc_auc.png"


def weekday_ml_weekday_pair_summary_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    return WEEKDAY_ML_PLOTS_DIR / f"{weekday_ml_output_tag(n_pairs, from_date, to_date)}_weekday_pair_summary.png"


def weekday_ml_model_bundle_path(
    n_pairs: int,
    train_from: datetime,
    train_to: datetime,
) -> Path:
    return WEEKDAY_ML_MODELS_DIR / f"{weekday_ml_output_tag(n_pairs, train_from, train_to)}_model_bundle.pkl"


def weekday_ml_train_test_metrics_path(
    n_pairs: int,
    train_from: datetime,
    test_to: datetime,
) -> Path:
    from crypto_research.utils.ml.spec import ML_STUDY_DAY_OF_WEEK, resolve_ml_study

    return ml_train_test_metrics_path(
        resolve_ml_study([ML_STUDY_DAY_OF_WEEK]),
        n_pairs,
        train_from,
        test_to,
    )


def weekday_ml_policy_path(
    n_pairs: int,
    train_from: datetime,
    test_to: datetime,
) -> Path:
    from crypto_research.utils.ml.spec import ML_STUDY_DAY_OF_WEEK, resolve_ml_study

    return ml_policy_path(resolve_ml_study([ML_STUDY_DAY_OF_WEEK]), n_pairs, train_from, test_to)


def ema_spreads_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> str:
    periods_tag = "_".join(str(p) for p in periods)
    return f"{n_pairs}pairs_ema{periods_tag}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def ema_spreads_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return EMA_SPREADS_STATS_DIR / f"ema_spreads_{tag}.log"


def ema_period_screen_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return EMA_SPREADS_STATS_DIR / f"ema_period_screen_{tag}.log"


def ema_summary_log_path(ema_period: int) -> Path:
    return EMA_SPREADS_STATS_DIR / f"ema_summary_ema{ema_period}.log"


def ema_summary_plot_path(ema_period: int, plot_kind: str) -> Path:
    return study_plots_dir(STUDY_EMA_SPREADS) / f"ema_summary_{plot_kind}_ema{ema_period}.png"


def ema_period_screen_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    plot_kind: str,
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return study_plots_dir(STUDY_EMA_SPREADS) / f"ema_period_screen_{plot_kind}_{tag}.png"


def rsi_screen_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> str:
    periods_tag = "_".join(str(p) for p in periods)
    return f"{n_pairs}pairs_rsi{periods_tag}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def rsi_period_screen_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = rsi_screen_output_tag(n_pairs, from_date, to_date, periods)
    return RSI_QUANTILES_STATS_DIR / f"rsi_period_screen_{tag}.log"


def rsi_period_screen_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    plot_kind: str,
) -> Path:
    tag = rsi_screen_output_tag(n_pairs, from_date, to_date, periods)
    return study_plots_dir(STUDY_RSI_PERIOD_SCREEN) / f"rsi_period_screen_{plot_kind}_{tag}.png"


def rsi_spreads_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = rsi_screen_output_tag(n_pairs, from_date, to_date, periods)
    return RSI_QUANTILES_STATS_DIR / f"rsi_spreads_{tag}.log"


def rsi_summary_log_path(rsi_period: int) -> Path:
    return RSI_QUANTILES_STATS_DIR / f"rsi_summary_rsi{rsi_period}.log"


def rsi_summary_plot_path(rsi_period: int, plot_kind: str) -> Path:
    return study_plots_dir(STUDY_RSI_SPREADS) / f"rsi_summary_{plot_kind}_rsi{rsi_period}.png"


def volume_screen_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> str:
    periods_tag = "_".join(str(p) for p in periods)
    return f"{n_pairs}pairs_vol{periods_tag}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def volume_ema_period_screen_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = volume_screen_output_tag(n_pairs, from_date, to_date, periods)
    return study_stats_dir(STUDY_VOLUME_EMA_PERIOD_SCREEN) / f"volume_ema_period_screen_{tag}.log"


def volume_ema_period_screen_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    plot_kind: str,
) -> Path:
    tag = volume_screen_output_tag(n_pairs, from_date, to_date, periods)
    return (
        study_plots_dir(STUDY_VOLUME_EMA_PERIOD_SCREEN)
        / f"volume_ema_period_screen_{plot_kind}_{tag}.png"
    )


def volatility_screen_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> str:
    periods_tag = "_".join(str(p) for p in periods)
    return f"{n_pairs}pairs_sma{periods_tag}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def volatility_period_screen_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = volatility_screen_output_tag(n_pairs, from_date, to_date, periods)
    return study_stats_dir(STUDY_VOLATILITY_PERIOD_SCREEN) / f"volatility_period_screen_{tag}.log"


def volatility_period_screen_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    plot_kind: str,
) -> Path:
    tag = volatility_screen_output_tag(n_pairs, from_date, to_date, periods)
    return (
        study_plots_dir(STUDY_VOLATILITY_PERIOD_SCREEN)
        / f"volatility_period_screen_{plot_kind}_{tag}.png"
    )


def volatility_summary_log_path(sma_period: int) -> Path:
    return study_stats_dir(STUDY_VOLATILITY_SPREADS) / f"volatility_summary_sma{sma_period}.log"


def volatility_summary_plot_path(sma_period: int, plot_kind: str) -> Path:
    return (
        study_plots_dir(STUDY_VOLATILITY_SPREADS)
        / f"volatility_summary_{plot_kind}_sma{sma_period}.png"
    )


def volume_summary_log_path(vol_period: int) -> Path:
    return study_stats_dir(STUDY_VOLUME_SPREADS) / f"volume_summary_vol{vol_period}.log"


def volume_summary_plot_path(vol_period: int, plot_kind: str) -> Path:
    return (
        study_plots_dir(STUDY_VOLUME_SPREADS)
        / f"volume_summary_{plot_kind}_vol{vol_period}.png"
    )
