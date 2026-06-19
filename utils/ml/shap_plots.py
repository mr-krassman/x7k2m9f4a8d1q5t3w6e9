"""SHAP summary (beeswarm) plot для интерпретации LightGBM на holdout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from crypto_research.utils.ml.plot_features import model_plot_feature_columns
from crypto_research.utils.ml.registry import FEATURE_WEEKDAY_ENC
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_shap_plots")

PLOT_DPI = 200
DEFAULT_SHAP_MAX_SAMPLES = 3000
SCATTER_SIZE_FACTOR = 0.5


def _positive_class_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3 and vals.shape[-1] >= 2:
            return vals[:, :, 1]
        return vals
    vals = np.asarray(shap_values)
    if vals.ndim == 3 and vals.shape[-1] >= 2:
        return vals[:, :, 1]
    return vals


def _weekday_enc_as_numeric(series: pd.Series) -> pd.Series:
    """Пн=0 (low) … Вс=6 (high) для цветовой шкалы SHAP."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.astype(np.float64)
    return pd.to_numeric(series, errors="coerce").astype(np.float64)


def _feature_matrix_for_color(x: pd.DataFrame) -> pd.DataFrame:
    """Числовая матрица для раскраски beeswarm (weekday_enc → 0..6)."""
    out = x.copy()
    if FEATURE_WEEKDAY_ENC in out.columns:
        out[FEATURE_WEEKDAY_ENC] = _weekday_enc_as_numeric(out[FEATURE_WEEKDAY_ENC])
    return out


def _feature_labels_with_importance(
    feature_names: list[str],
    shap_vals: np.ndarray,
) -> list[str]:
    importances = np.abs(shap_vals).mean(axis=0)
    return [f"{name} ({imp:.4f})" for name, imp in zip(feature_names, importances)]


def _scale_scatter_sizes(fig: plt.Figure, factor: float) -> None:
    from matplotlib.collections import PathCollection

    for ax in fig.axes:
        for coll in ax.collections:
            if not isinstance(coll, PathCollection):
                continue
            sizes = coll.get_sizes()
            if sizes is not None and len(sizes):
                coll.set_sizes(np.asarray(sizes, dtype=float) * factor)


def save_shap_summary_plot(
    model,
    x: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    path: Path,
    *,
    title: str = "SHAP summary",
    period_label: str = "holdout test",
    max_samples: int = DEFAULT_SHAP_MAX_SAMPLES,
    random_state: int = 42,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = model_plot_feature_columns(feature_columns)
    model_names = list(getattr(model, "feature_name_", None) or feature_names)
    if set(model_names) != set(feature_names):
        log.warning(
            "[ml] SHAP: feature_name_ модели %s != spec %s; используем модель",
            model_names,
            feature_names,
        )
        feature_names = model_names
    x_plot = x[feature_names].copy()
    n_total = int(x_plot.shape[0])
    if n_total == 0:
        log.warning("[ml] shape_summary_plot: пустая holdout-выборка")
        return path

    if n_total > max_samples:
        idx = np.random.default_rng(random_state).choice(n_total, size=max_samples, replace=False)
        x_plot = x_plot.iloc[idx]
        log.info("[ml] SHAP subsample: %s / %s rows", x_plot.shape[0], n_total)

    explainer = shap.TreeExplainer(model)
    shap_vals = _positive_class_shap_values(explainer.shap_values(x_plot))
    x_color = _feature_matrix_for_color(x_plot)
    display_names = _feature_labels_with_importance(feature_names, shap_vals)

    height = max(4.5, 0.65 * len(feature_names) + 1.5)
    plt.figure(figsize=(10, height))
    shap.summary_plot(
        shap_vals,
        x_color,
        feature_names=display_names,
        show=False,
        plot_type="dot",
        max_display=len(feature_names),
        color_bar_label="значение фичи",
    )
    fig = plt.gcf()
    _scale_scatter_sizes(fig, SCATTER_SIZE_FACTOR)
    fig.suptitle(
        f"{title}\n({period_label}, n={x_plot.shape[0]}/{n_total})",
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] SHAP summary plot saved: %s", path)
    return path
