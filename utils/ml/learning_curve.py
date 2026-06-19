"""Early stopping на хвосте train и график binary_logloss."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from crypto_research.utils.ml.cpcv_train import _default_lgbm_params
from crypto_research.utils.ml.dataset import DirectionDataset, categorical_feature_names
from crypto_research.utils.ml.spec import CATEGORICAL_FEATURES
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_learning_curve")

PLOT_DPI = 200
FIG_W = 10.0
FIG_H = 5.0
DEFAULT_EARLY_STOPPING_ROUNDS = 50
FINAL_FIT_N_ESTIMATORS = 1000


@dataclass(frozen=True)
class EarlyStoppingFitResult:
    model: lgb.LGBMClassifier
    best_iteration: int
    n_fit_rows: int
    n_eval_rows: int
    eval_period: dict[str, str]
    learning_curve: dict[str, list[float]]
    learning_curve_plot_path: Path | None = None
    learning_curve_log_path: Path | None = None


def split_train_eval_frames(
    frame: pl.DataFrame,
    *,
    eval_from: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Основной fit: day < eval_from; eval для early stopping: day >= eval_from."""
    fit = frame.filter(pl.col("day_utc") < eval_from)
    eval_frame = frame.filter(pl.col("day_utc") >= eval_from)
    if fit.is_empty() or eval_frame.is_empty():
        raise RuntimeError(
            f"Пустой train/eval split: fit={fit.height} eval={eval_frame.height} eval_from={eval_from}"
        )
    return fit, eval_frame


def _frame_to_xy(frame: pl.DataFrame, dataset: DirectionDataset) -> tuple[pd.DataFrame, np.ndarray]:
    x = frame.select(*dataset.feature_columns).to_pandas()
    for col in dataset.feature_columns:
        if col in CATEGORICAL_FEATURES:
            x[col] = x[col].astype("category")
        else:
            x[col] = x[col].astype(np.float64)
    y = frame[dataset.target_column].to_numpy()
    return x, y


def save_learning_curve_log(
    learning_curve: dict[str, list[float]],
    path: Path,
    *,
    best_iteration: int,
    train_period: dict[str, str],
    eval_period: dict[str, str],
    n_fit_rows: int,
    n_eval_rows: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_loss = learning_curve.get("train", [])
    eval_loss = learning_curve.get("eval", [])
    lines = [
        "=== LightGBM learning curve (binary_logloss) ===",
        f"train_period: {train_period.get('from', '')} .. {train_period.get('to', '')}",
        f"eval_period: {eval_period.get('from', '')} .. {eval_period.get('to', '')}",
        f"fit_rows: {n_fit_rows}",
        f"eval_rows: {n_eval_rows}",
        f"best_iteration: {best_iteration}",
        "",
        "iteration\ttrain_logloss\teval_logloss\tbest",
    ]
    for i, (tr, ev) in enumerate(zip(train_loss, eval_loss, strict=True), start=1):
        marker = "*" if i == best_iteration else ""
        lines.append(f"{i}\t{tr:.6f}\t{ev:.6f}\t{marker}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("[ml] learning curve log saved: %s (%s iterations)", path, len(train_loss))
    return path


def save_learning_curve_plot(
    learning_curve: dict[str, list[float]],
    path: Path,
    *,
    best_iteration: int,
    title: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_loss = learning_curve.get("train", [])
    eval_loss = learning_curve.get("eval", [])
    if not train_loss or not eval_loss:
        log.warning("[ml] learning curve plot skipped: пустая история logloss")
        return path

    iterations = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(iterations, train_loss, color="#2563eb", linewidth=1.8, label="train logloss")
    ax.plot(iterations, eval_loss, color="#16a34a", linewidth=1.8, label="eval logloss (holdout test)")
    if 1 <= best_iteration <= len(eval_loss):
        ax.axvline(best_iteration, color="#64748b", linestyle="--", linewidth=1.0, label=f"best iter={best_iteration}")
    ax.set_xlabel("Итерация (дерево)")
    ax.set_ylabel("binary_logloss")
    ax.set_title(title, fontsize=11, fontweight="semibold", loc="left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] learning curve saved: %s (best_iter=%s)", path, best_iteration)
    return path


def record_learning_curve_diagnostic(
    train_dataset: DirectionDataset,
    eval_dataset: DirectionDataset,
    *,
    train_from: datetime,
    train_to: datetime,
    eval_from: datetime,
    eval_to: datetime,
    plot_path: Path,
    log_path: Path,
    lgbm_params: dict | None = None,
) -> tuple[dict[str, list[float]], Path, Path, int, int, int, dict[str, str]]:
    """Диагностический probe: fit на всём train, eval logloss на holdout test."""
    if train_dataset.frame.is_empty() or eval_dataset.frame.is_empty():
        raise RuntimeError(
            f"Пустой train/eval для learning curve: train={train_dataset.frame.height} "
            f"eval={eval_dataset.frame.height}"
        )
    x_fit, y_fit = _frame_to_xy(train_dataset.frame, train_dataset)
    x_eval, y_eval = _frame_to_xy(eval_dataset.frame, eval_dataset)
    cat_features = categorical_feature_names(train_dataset.feature_columns)

    params = _default_lgbm_params()
    if lgbm_params:
        params.update(lgbm_params)

    eval_result: dict[str, dict[str, list[float]]] = {}
    probe = lgb.LGBMClassifier(**params)
    probe.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_eval, y_eval)],
        eval_names=["train", "eval"],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=[lgb.record_evaluation(eval_result)],
    )
    curve = {
        "train": list(eval_result["train"]["binary_logloss"]),
        "eval": list(eval_result["eval"]["binary_logloss"]),
    }
    eval_loss = np.array(curve["eval"], dtype=float)
    min_eval_iter = int(np.argmin(eval_loss)) + 1
    train_period = {"from": train_from.isoformat(), "to": train_to.isoformat()}
    eval_period = {"from": eval_from.isoformat(), "to": eval_to.isoformat()}
    log.info(
        "[ml] learning curve diagnostic: fit_rows=%s eval_rows=%s n_estimators=%s min_eval_iter=%s min_eval_logloss=%.4f",
        train_dataset.frame.height,
        eval_dataset.frame.height,
        len(curve["train"]),
        min_eval_iter,
        float(eval_loss.min()),
    )
    title = (
        f"LightGBM logloss (train: {train_from:%Y-%m-%d}..{train_to:%Y-%m-%d}, "
        f"eval: {eval_from:%Y-%m-%d}..{eval_to:%Y-%m-%d})"
    )
    saved_plot = save_learning_curve_plot(
        curve,
        plot_path,
        best_iteration=min_eval_iter,
        title=title,
    )
    saved_log = save_learning_curve_log(
        curve,
        log_path,
        best_iteration=min_eval_iter,
        train_period=train_period,
        eval_period=eval_period,
        n_fit_rows=train_dataset.frame.height,
        n_eval_rows=eval_dataset.frame.height,
    )
    return (
        curve,
        saved_plot,
        saved_log,
        train_dataset.frame.height,
        eval_dataset.frame.height,
        min_eval_iter,
        eval_period,
    )


def fit_lightgbm_full_train(
    dataset: DirectionDataset,
    *,
    lgbm_params: dict | None = None,
) -> EarlyStoppingFitResult:
    """Финальный fit на всём train без early stopping (как CPCV: n_estimators из defaults)."""
    x_full, y_full = _frame_to_xy(dataset.frame, dataset)
    cat_features = categorical_feature_names(dataset.feature_columns)
    params = _default_lgbm_params()
    if lgbm_params:
        params.update(lgbm_params)
    n_estimators = int(params["n_estimators"])
    model = lgb.LGBMClassifier(**params)
    model.fit(x_full, y_full, categorical_feature=cat_features)
    log.info("[ml] full train fit: rows=%s n_estimators=%s", dataset.frame.height, n_estimators)
    return EarlyStoppingFitResult(
        model=model,
        best_iteration=n_estimators,
        n_fit_rows=dataset.frame.height,
        n_eval_rows=0,
        eval_period={},
        learning_curve={"train": [], "eval": []},
    )


def fit_lightgbm_with_holdout_early_stopping(
    train_dataset: DirectionDataset,
    eval_dataset: DirectionDataset,
    *,
    train_from: datetime,
    train_to: datetime,
    eval_from: datetime,
    eval_to: datetime,
    plot_path: Path | None = None,
    log_path: Path | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    lgbm_params: dict | None = None,
) -> EarlyStoppingFitResult:
    """Early stopping по holdout test; финальный fit на всём train с best_iteration."""
    if train_dataset.frame.is_empty() or eval_dataset.frame.is_empty():
        raise RuntimeError(
            f"Пустой train/eval для early stopping: train={train_dataset.frame.height} "
            f"eval={eval_dataset.frame.height}"
        )
    x_fit, y_fit = _frame_to_xy(train_dataset.frame, train_dataset)
    x_eval, y_eval = _frame_to_xy(eval_dataset.frame, eval_dataset)
    cat_features = categorical_feature_names(train_dataset.feature_columns)

    params = _default_lgbm_params()
    params["n_estimators"] = FINAL_FIT_N_ESTIMATORS
    if lgbm_params:
        params.update(lgbm_params)

    eval_result: dict[str, dict[str, list[float]]] = {}
    probe = lgb.LGBMClassifier(**params)
    probe.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_eval, y_eval)],
        eval_names=["train", "eval"],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=[
            lgb.record_evaluation(eval_result),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )
    best_iteration = int(probe.best_iteration_ or probe.n_estimators)
    log.info(
        "[ml] early stopping (holdout): fit_rows=%s eval_rows=%s best_iteration=%s eval_logloss=%.4f",
        train_dataset.frame.height,
        eval_dataset.frame.height,
        best_iteration,
        eval_result["eval"]["binary_logloss"][best_iteration - 1],
    )

    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    model = lgb.LGBMClassifier(**final_params)
    model.fit(x_fit, y_fit, categorical_feature=cat_features)

    curve = {
        "train": list(eval_result["train"]["binary_logloss"]),
        "eval": list(eval_result["eval"]["binary_logloss"]),
    }
    train_period = {"from": train_from.isoformat(), "to": train_to.isoformat()}
    eval_period = {"from": eval_from.isoformat(), "to": eval_to.isoformat()}
    title = (
        f"LightGBM logloss (train: {train_from:%Y-%m-%d}..{train_to:%Y-%m-%d}, "
        f"eval: {eval_from:%Y-%m-%d}..{eval_to:%Y-%m-%d})"
    )
    saved_plot = None
    saved_log = None
    if plot_path is not None:
        saved_plot = save_learning_curve_plot(
            curve,
            plot_path,
            best_iteration=best_iteration,
            title=title,
        )
    if log_path is not None:
        saved_log = save_learning_curve_log(
            curve,
            log_path,
            best_iteration=best_iteration,
            train_period=train_period,
            eval_period=eval_period,
            n_fit_rows=train_dataset.frame.height,
            n_eval_rows=eval_dataset.frame.height,
        )

    return EarlyStoppingFitResult(
        model=model,
        best_iteration=best_iteration,
        n_fit_rows=train_dataset.frame.height,
        n_eval_rows=eval_dataset.frame.height,
        eval_period=eval_period,
        learning_curve=curve,
        learning_curve_plot_path=saved_plot,
        learning_curve_log_path=saved_log,
    )


def fit_lightgbm_with_eval_curve(
    dataset: DirectionDataset,
    *,
    eval_from: datetime,
    eval_to: datetime,
    plot_path: Path | None = None,
    log_path: Path | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    lgbm_params: dict | None = None,
) -> EarlyStoppingFitResult:
    """Устаревший split внутри train; оставлен для совместимости."""
    fit_frame, eval_frame = split_train_eval_frames(dataset.frame, eval_from=eval_from)
    x_fit, y_fit = _frame_to_xy(fit_frame, dataset)
    x_eval, y_eval = _frame_to_xy(eval_frame, dataset)
    x_full, y_full = _frame_to_xy(dataset.frame, dataset)
    cat_features = categorical_feature_names(dataset.feature_columns)

    params = _default_lgbm_params()
    params["n_estimators"] = FINAL_FIT_N_ESTIMATORS
    if lgbm_params:
        params.update(lgbm_params)

    eval_result: dict[str, dict[str, list[float]]] = {}
    probe = lgb.LGBMClassifier(**params)
    probe.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_eval, y_eval)],
        eval_names=["train", "eval"],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=[
            lgb.record_evaluation(eval_result),
            lgb.early_stopping(early_stopping_rounds, verbose=False),
        ],
    )
    best_iteration = int(probe.best_iteration_ or probe.n_estimators)
    log.info(
        "[ml] early stopping: fit_rows=%s eval_rows=%s best_iteration=%s eval_logloss=%.4f",
        fit_frame.height,
        eval_frame.height,
        best_iteration,
        eval_result["eval"]["binary_logloss"][best_iteration - 1],
    )

    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    model = lgb.LGBMClassifier(**final_params)
    model.fit(x_full, y_full, categorical_feature=cat_features)

    curve = {
        "train": list(eval_result["train"]["binary_logloss"]),
        "eval": list(eval_result["eval"]["binary_logloss"]),
    }
    eval_period = {"from": eval_from.isoformat(), "to": eval_to.isoformat()}
    fit_from = fit_frame["day_utc"].min()
    fit_to = fit_frame["day_utc"].max()
    train_period = {"from": fit_from.isoformat(), "to": fit_to.isoformat()}
    saved_plot = None
    saved_log = None
    if plot_path is not None:
        saved_plot = save_learning_curve_plot(
            curve,
            plot_path,
            best_iteration=best_iteration,
            title=f"LightGBM logloss (eval: {eval_from:%Y-%m-%d} .. {eval_to:%Y-%m-%d})",
        )
    if log_path is not None:
        saved_log = save_learning_curve_log(
            curve,
            log_path,
            best_iteration=best_iteration,
            train_period=train_period,
            eval_period=eval_period,
            n_fit_rows=fit_frame.height,
            n_eval_rows=eval_frame.height,
        )

    return EarlyStoppingFitResult(
        model=model,
        best_iteration=best_iteration,
        n_fit_rows=fit_frame.height,
        n_eval_rows=eval_frame.height,
        eval_period=eval_period,
        learning_curve=curve,
        learning_curve_plot_path=saved_plot,
        learning_curve_log_path=saved_log,
    )
