"""Фичи модели для диагностических графиков (correlation / SHAP)."""

from __future__ import annotations

import numpy as np
import polars as pl

from crypto_research.utils.ml.registry import FEATURE_WEEKDAY_ENC


def model_plot_feature_columns(feature_columns: tuple[str, ...] | list[str]) -> list[str]:
    """Все входные фичи модели в порядке bundle (pair_id, weekday_enc, …)."""
    return list(feature_columns)


def feature_values_as_float(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column == FEATURE_WEEKDAY_ENC and "weekday" in frame.columns:
        return frame["weekday"].to_numpy().astype(np.float64)
    series = frame[column]
    if series.dtype == pl.Categorical:
        decoded = series.cast(pl.Utf8).cast(pl.Int32, strict=False)
        return decoded.to_numpy().astype(np.float64)
    if series.dtype in (pl.Utf8, pl.String):
        return series.cast(pl.Int32, strict=False).to_numpy().astype(np.float64)
    return series.to_numpy().astype(np.float64)


def feature_correlation_matrix(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> tuple[list[str], np.ndarray]:
    cols = [c for c in model_plot_feature_columns(feature_columns) if c in frame.columns]
    if not cols:
        return [], np.empty((0, 0))
    matrix = np.column_stack([feature_values_as_float(frame, c) for c in cols])
    mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[mask]
    if matrix.shape[0] < 2:
        corr = np.eye(len(cols), dtype=float)
    else:
        corr = np.corrcoef(matrix, rowvar=False)
    return cols, corr
