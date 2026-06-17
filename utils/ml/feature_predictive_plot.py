"""График ROC AUC / accuracy vs непрерывный признак (train и val рядом)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score

from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_feature_predictive_plot")

PLOT_DPI = 200
FIG_W = 14.0
FIG_H = 5.0
BASELINE = 0.5
DEFAULT_BIN_WIDTH = 0.1
DEFAULT_MIN_N = 40
Y_PAD_FRAC = 0.05


def _metrics_for_slice(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    n = int(y_true.size)
    if n == 0:
        return {"n": 0, "roc_auc": float("nan"), "accuracy": float("nan")}
    y_pred = (y_prob >= 0.5).astype(np.int8)
    out: dict[str, float | int] = {
        "n": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def build_feature_bin_curve_metrics(
    oos: pl.DataFrame,
    feature_column: str,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
    min_n: int = DEFAULT_MIN_N,
) -> pl.DataFrame:
    """Метрики по бинам непрерывного признака (центр бина = X)."""
    required = {feature_column, "y_true", "y_prob"}
    missing = required - set(oos.columns)
    if missing:
        raise ValueError(f"oos должен содержать {sorted(required)}, нет: {sorted(missing)}")

    values = oos[feature_column].to_numpy().astype(np.float64, copy=False)
    y_true = oos["y_true"].to_numpy().astype(np.int8, copy=False)
    y_prob = oos["y_prob"].to_numpy().astype(np.float64, copy=False)
    ok = np.isfinite(values)
    values = values[ok]
    y_true = y_true[ok]
    y_prob = y_prob[ok]
    if values.size == 0:
        return pl.DataFrame(
            schema={
                "feature_center": pl.Float64,
                "roc_auc": pl.Float64,
                "accuracy": pl.Float64,
                "n": pl.Int64,
            }
        )

    lo = float(np.floor(values.min() / bin_width) * bin_width)
    hi = float(np.ceil(values.max() / bin_width) * bin_width)
    edges = np.arange(lo, hi + bin_width, bin_width)
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        m = _metrics_for_slice(y_true[mask], y_prob[mask])
        center = (left + right) / 2.0
        auc = float(m["roc_auc"])
        acc = float(m["accuracy"])
        n = int(m["n"])
        if n < min_n:
            auc = float("nan")
            acc = float("nan")
        rows.append({"feature_center": center, "roc_auc": auc, "accuracy": acc, "n": n})
    return pl.DataFrame(rows)


def _y_limits_from_metrics(*metrics_frames: pl.DataFrame) -> tuple[float, float]:
    values: list[float] = []
    for frame in metrics_frames:
        for col in ("roc_auc", "accuracy"):
            arr = frame[col].to_numpy()
            values.extend(float(v) for v in arr if np.isfinite(v))
    if not values:
        return 0.4, 0.6
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if y_min == y_max:
        pad = 0.02
    else:
        pad = (y_max - y_min) * Y_PAD_FRAC
    return y_min - pad, y_max + pad


def _plot_curve_panel(
    ax: plt.Axes,
    metrics: pl.DataFrame,
    *,
    title: str,
    x_label: str,
    y_min: float,
    y_max: float,
) -> None:
    x = metrics["feature_center"].to_numpy()
    auc = metrics["roc_auc"].to_numpy()
    acc = metrics["accuracy"].to_numpy()

    ax.axhline(BASELINE, color="#64748b", linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(x, auc, color="#16a34a", linewidth=1.8, marker="o", markersize=3.5, label="ROC AUC", zorder=3)
    ax.plot(x, acc, color="#2563eb", linewidth=1.8, marker="o", markersize=3.5, label="Accuracy", zorder=3)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Метрика")
    ax.set_title(title, fontsize=11, fontweight="semibold", loc="left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9)


def save_feature_curve_plot(
    train_oos: pl.DataFrame,
    val_oos: pl.DataFrame,
    path: Path,
    *,
    feature_column: str,
    x_label: str | None = None,
    train_title: str = "Train",
    val_title: str = "Val (holdout)",
    bin_width: float = DEFAULT_BIN_WIDTH,
    min_n: int = DEFAULT_MIN_N,
) -> Path:
    """Два графика рядом: метрики vs непрерывный признак."""
    path.parent.mkdir(parents=True, exist_ok=True)
    label = x_label or feature_column
    if train_oos.is_empty() or val_oos.is_empty():
        log.warning("[ml] feature curve plot skipped: пустой train или val (%s)", feature_column)
        return path

    train_metrics = build_feature_bin_curve_metrics(
        train_oos, feature_column, bin_width=bin_width, min_n=min_n
    )
    val_metrics = build_feature_bin_curve_metrics(
        val_oos, feature_column, bin_width=bin_width, min_n=min_n
    )
    y_min, y_max = _y_limits_from_metrics(train_metrics, val_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    _plot_curve_panel(axes[0], train_metrics, title=train_title, x_label=label, y_min=y_min, y_max=y_max)
    _plot_curve_panel(axes[1], val_metrics, title=val_title, x_label=label, y_min=y_min, y_max=y_max)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] feature curve plot saved: %s (%s)", path, feature_column)
    return path
