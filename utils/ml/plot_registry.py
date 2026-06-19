"""Реестр графиков ml_research: --plots выбирает, что строить для модели."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import polars as pl

from crypto_research.utils.ml.cpcv_train import CPCVTrainResult
from crypto_research.utils.ml.dataset import DirectionDataset, dataset_to_numpy
from crypto_research.utils.ml.feature_dependence_plot import save_feature_prob_dependence_plot
from crypto_research.utils.ml.feature_diagnostic_plots import save_correlation_matrix_heatmap
from crypto_research.utils.ml.feature_predictive_plot import save_metrics_over_feature_plot
from crypto_research.utils.ml.numeric_features import active_numeric_specs
from crypto_research.utils.ml.oos_paths import (
    save_oos_calibration_plot,
    save_oos_probability_plot,
    save_roc_auc_comparison_plot,
    save_weekday_pair_summary_plot,
)
from crypto_research.utils.ml.registry import MlStudySpec
from crypto_research.utils.ml.shap_plots import save_shap_summary_plot
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import (
    ml_correlation_matrix_plot_path,
    ml_feature_prob_dependence_plot_path,
    ml_feature_predictive_plot_path,
    ml_oos_calibration_plot_path,
    ml_oos_plot_path,
    ml_roc_auc_plot_path,
    ml_shape_summary_plot_path,
    ml_weekday_pair_summary_plot_path,
)

log = get_logger("ml_plot_registry")

ML_PLOT_CORRELATION_MATRIX = "correlation_matrix_heatmap"
ML_PLOT_SHAPE_SUMMARY = "shape_summary_plot"
ML_PLOT_OOS_PROB = "oos_prob"
ML_PLOT_OOS_CALIBRATION = "oos_calibration"
ML_PLOT_WEEKDAY_PAIR_SUMMARY = "weekday_pair_summary"
ML_PLOT_ROC_AUC = "roc_auc"
ML_PLOT_FEATURE_PREDICTIVE = "feature_predictive"
ML_PLOT_FEATURE_PROB_DEPENDENCE = "feature_prob_dependence"
ML_PLOT_LEARNING_CURVE = "learning_curve"

ML_PLOT_ALIASES: dict[str, str] = {
    "тепловая_карта_корреляционной_матрицы": ML_PLOT_CORRELATION_MATRIX,
    ML_PLOT_CORRELATION_MATRIX: ML_PLOT_CORRELATION_MATRIX,
    ML_PLOT_SHAPE_SUMMARY: ML_PLOT_SHAPE_SUMMARY,
    "shap_summary_plot": ML_PLOT_SHAPE_SUMMARY,
    ML_PLOT_OOS_PROB: ML_PLOT_OOS_PROB,
    ML_PLOT_OOS_CALIBRATION: ML_PLOT_OOS_CALIBRATION,
    ML_PLOT_WEEKDAY_PAIR_SUMMARY: ML_PLOT_WEEKDAY_PAIR_SUMMARY,
    ML_PLOT_ROC_AUC: ML_PLOT_ROC_AUC,
    ML_PLOT_FEATURE_PREDICTIVE: ML_PLOT_FEATURE_PREDICTIVE,
    ML_PLOT_FEATURE_PROB_DEPENDENCE: ML_PLOT_FEATURE_PROB_DEPENDENCE,
    "dependence_plot": ML_PLOT_FEATURE_PROB_DEPENDENCE,
    ML_PLOT_LEARNING_CURVE: ML_PLOT_LEARNING_CURVE,
}

DEFAULT_TRAIN_TEST_PLOTS: tuple[str, ...] = (
    ML_PLOT_OOS_PROB,
    ML_PLOT_OOS_CALIBRATION,
    ML_PLOT_WEEKDAY_PAIR_SUMMARY,
    ML_PLOT_ROC_AUC,
)

ML_PLOT_CHOICES: tuple[str, ...] = tuple(dict.fromkeys(ML_PLOT_ALIASES.values()))


@dataclass
class MlPlotContext:
    spec: MlStudySpec
    args: object
    n_pairs: int
    train_from: datetime
    train_to: datetime
    test_from: datetime
    test_to: datetime
    test_dataset: DirectionDataset
    test_oos: pl.DataFrame
    y_test: np.ndarray
    y_test_prob: np.ndarray
    train_dataset: DirectionDataset | None = None
    train_fit_oos: pl.DataFrame | None = None
    train_cpcv: CPCVTrainResult | None = None
    fit_result: object | None = None
    model: object | None = None


def resolve_plot_ids(raw: list[str] | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_TRAIN_TEST_PLOTS
    out: list[str] = []
    for item in raw:
        key = ML_PLOT_ALIASES.get(item)
        if key is None:
            known = sorted(set(ML_PLOT_ALIASES.values()) | set(ML_PLOT_ALIASES.keys()))
            raise ValueError(f"Неизвестный график: {item!r}. Допустимо: {known}")
        if key not in out:
            out.append(key)
    return tuple(out)


def _period_label(test_from: datetime, test_to: datetime) -> str:
    return f"holdout {test_from:%Y-%m-%d}..{test_to:%Y-%m-%d}"


def _plot_correlation_matrix_heatmap(ctx: MlPlotContext) -> dict[str, object]:
    path, payload = save_correlation_matrix_heatmap(
        ctx.test_dataset.frame,
        ml_correlation_matrix_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
        ctx.spec.feature_columns,
        title="Pearson correlation: model features",
        period_label=_period_label(ctx.test_from, ctx.test_to),
    )
    return {"correlation_matrix_heatmap": str(path), "feature_correlations": payload}


def _plot_shape_summary(ctx: MlPlotContext) -> dict[str, str]:
    if ctx.model is None:
        raise RuntimeError("shape_summary_plot: нет модели (нужен fit или frozen bundle)")
    x_test, _, _, _ = dataset_to_numpy(ctx.test_dataset)
    path = save_shap_summary_plot(
        ctx.model,
        x_test,
        ctx.spec.feature_columns,
        ml_shape_summary_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
        title="SHAP summary: P(up)",
        period_label=_period_label(ctx.test_from, ctx.test_to),
    )
    return {"shape_summary_plot": str(path)}


def _plot_feature_prob_dependence(ctx: MlPlotContext) -> dict[str, str]:
    path = save_feature_prob_dependence_plot(
        ctx.test_dataset.frame,
        ctx.y_test_prob,
        ctx.spec.feature_columns,
        ml_feature_prob_dependence_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
        title="Dependence: feature value vs P(up)",
        period_label=_period_label(ctx.test_from, ctx.test_to),
    )
    return {"feature_prob_dependence": str(path)}


def _plot_oos_prob(ctx: MlPlotContext) -> dict[str, str]:
    path = save_oos_probability_plot(
        ctx.test_oos,
        ml_oos_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
    )
    return {"oos_prob": str(path)}


def _plot_oos_calibration(ctx: MlPlotContext) -> dict[str, str]:
    path = save_oos_calibration_plot(
        ctx.test_oos,
        ml_oos_calibration_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
    )
    return {"oos_calibration": str(path)}


def _plot_weekday_pair_summary(ctx: MlPlotContext) -> dict[str, str]:
    out: dict[str, str] = {}
    path_test = save_weekday_pair_summary_plot(
        ctx.test_oos,
        ml_weekday_pair_summary_plot_path(ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to),
        title="Holdout test: weekday × pair metrics",
    )
    out["weekday_pair_summary"] = str(path_test)
    if ctx.train_cpcv is not None and ctx.train_cpcv.oos_predictions is not None:
        path_train = save_weekday_pair_summary_plot(
            ctx.train_cpcv.oos_predictions,
            ml_weekday_pair_summary_plot_path(ctx.spec, ctx.n_pairs, ctx.train_from, ctx.train_to),
            title="Train CPCV OOS: weekday × pair metrics",
        )
        out["weekday_pair_summary_train"] = str(path_train)
    return out


def _plot_roc_auc(ctx: MlPlotContext) -> dict[str, str]:
    if ctx.train_cpcv is None or ctx.train_cpcv.oos_paths is None:
        raise RuntimeError("roc_auc: нет train CPCV OOS (нужен полный train/test пайплайн)")
    train_oos = ctx.train_cpcv.oos_paths
    path = save_roc_auc_comparison_plot(
        train_oos[:, 2].astype(np.int8),
        train_oos[:, 1],
        ctx.y_test,
        ctx.y_test_prob,
        ml_roc_auc_plot_path(ctx.spec, ctx.n_pairs, ctx.train_from, ctx.test_to),
    )
    return {"roc_auc": str(path)}


def _plot_feature_predictive(ctx: MlPlotContext) -> dict[str, str]:
    if ctx.train_fit_oos is None:
        raise RuntimeError("feature_predictive: нет train-fit OOS")
    specs = active_numeric_specs(ctx.spec.feature_columns)
    if not specs:
        log.warning("[ml] feature_predictive: нет numeric-фич")
        return {}
    out: dict[str, str] = {}
    for ns in specs:
        path = save_metrics_over_feature_plot(
            ctx.train_fit_oos,
            ctx.test_oos,
            ml_feature_predictive_plot_path(
                ctx.spec, ctx.n_pairs, ctx.test_from, ctx.test_to, ns.plot_slug
            ),
            predictive_feature=ns.column,
            predictive_plot_slug=ns.plot_slug,
            train_title=f"Train {ctx.args.train_from}..{ctx.args.train_to}",
            val_title=f"Val {ctx.args.test_from}..{ctx.args.test_to}",
        )
        out[f"feature_predictive_{ns.plot_slug}"] = str(path)
    return out


def _plot_learning_curve(ctx: MlPlotContext) -> dict[str, str]:
    if ctx.fit_result is None or getattr(ctx.fit_result, "learning_curve_plot_path", None) is None:
        log.warning("[ml] learning_curve: нет learning curve (нужен --early-stopping)")
        return {}
    return {"learning_curve": str(ctx.fit_result.learning_curve_plot_path)}


_PLOT_HANDLERS: dict[str, Callable[[MlPlotContext], dict[str, object]]] = {
    ML_PLOT_CORRELATION_MATRIX: _plot_correlation_matrix_heatmap,
    ML_PLOT_SHAPE_SUMMARY: _plot_shape_summary,
    ML_PLOT_FEATURE_PROB_DEPENDENCE: _plot_feature_prob_dependence,
    ML_PLOT_OOS_PROB: _plot_oos_prob,
    ML_PLOT_OOS_CALIBRATION: _plot_oos_calibration,
    ML_PLOT_WEEKDAY_PAIR_SUMMARY: _plot_weekday_pair_summary,
    ML_PLOT_ROC_AUC: _plot_roc_auc,
    ML_PLOT_FEATURE_PREDICTIVE: _plot_feature_predictive,
    ML_PLOT_LEARNING_CURVE: _plot_learning_curve,
}


def run_selected_ml_plots(plot_ids: tuple[str, ...], ctx: MlPlotContext) -> dict[str, object]:
    paths: dict[str, object] = {}
    for plot_id in plot_ids:
        handler = _PLOT_HANDLERS.get(plot_id)
        if handler is None:
            raise ValueError(f"График не зарегистрирован: {plot_id}")
        log.info("[ml] plot: %s", plot_id)
        paths.update(handler(ctx))
    return paths
