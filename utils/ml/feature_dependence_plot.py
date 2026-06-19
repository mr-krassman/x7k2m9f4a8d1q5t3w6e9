"""Dependence plot: X = значение фичи, Y = P(up) на holdout."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.ml.plot_features import (
    feature_values_as_float,
    model_plot_feature_columns,
)
from crypto_research.utils.ml.registry import FEATURE_PAIR_ID, FEATURE_WEEKDAY_ENC
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_feature_dependence_plot")

PLOT_DPI = 200
CONTINUOUS_BINS = 40
SCATTER_MAX_POINTS = 2500
SCATTER_ALPHA = 0.12
SCATTER_SIZE = 4
AXIS_PAD_RATIO = 0.08
MIN_Y_SPAN = 0.03
MIN_X_SPAN = 0.1
N_WEEKDAYS = 7


def _padded_limits(
    *arrays: np.ndarray,
    pad_ratio: float = AXIS_PAD_RATIO,
    min_span: float | None = None,
    clamp_lo: float | None = None,
    clamp_hi: float | None = None,
) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    if min_span is not None and span < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
        span = min_span
    pad = max(span * pad_ratio, 1e-6)
    lo -= pad
    hi += pad
    if clamp_lo is not None:
        lo = max(clamp_lo, lo)
    if clamp_hi is not None:
        hi = min(clamp_hi, hi)
    if hi <= lo:
        hi = lo + max(min_span or 1e-3, 1e-3)
    return lo, hi


def dependence_feature_columns(feature_columns: tuple[str, ...] | list[str]) -> list[str]:
    return [c for c in model_plot_feature_columns(feature_columns) if c != FEATURE_PAIR_ID]


def _binned_mean(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = float(np.nanpercentile(x, 1)), float(np.nanpercentile(x, 99))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.any():
            means[i] = float(np.mean(y[mask]))
            counts[i] = int(mask.sum())
    valid = np.isfinite(means)
    return centers[valid], means[valid], counts[valid]


def _weekday_mean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(N_WEEKDAYS, dtype=float)
    means = np.full(N_WEEKDAYS, np.nan)
    counts = np.zeros(N_WEEKDAYS, dtype=int)
    xi = x.astype(int)
    for wd in range(N_WEEKDAYS):
        mask = xi == wd
        if mask.any():
            means[wd] = float(np.mean(y[mask]))
            counts[wd] = int(mask.sum())
    return xs, means, counts


def _subsample_mask(n: int, *, max_points: int, seed: int = 42) -> np.ndarray:
    if n <= max_points:
        return np.ones(n, dtype=bool)
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def _panel_legend(ax: plt.Axes, *, mean_label: str, show_half: bool) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#64748b",
            markeredgecolor="none",
            alpha=0.55,
            markersize=4,
            label="строки holdout",
        ),
        Line2D([0], [0], color="#dc2626", linewidth=2.0, marker="o", markersize=4, label=mean_label),
    ]
    if show_half:
        handles.append(Line2D([0], [0], color="#9ca3af", linestyle="--", linewidth=1.0, label="P(up)=0.5"))
    ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.9)


def _plot_panel(ax: plt.Axes, x: np.ndarray, y_prob: np.ndarray, feature: str) -> None:
    mask = np.isfinite(x) & np.isfinite(y_prob)
    x = x[mask]
    y_prob = y_prob[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes)
        return

    scatter_m = _subsample_mask(x.size, max_points=SCATTER_MAX_POINTS)
    ax.scatter(
        x[scatter_m],
        y_prob[scatter_m],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        color="#64748b",
        edgecolors="none",
        rasterized=True,
    )

    mean_label = "среднее P(up) по бину"
    weekday_empty: list[int] = []
    if feature == FEATURE_WEEKDAY_ENC:
        bx, by, counts = _weekday_mean(x, y_prob)
        valid = np.isfinite(by)
        ax.plot(
            bx[valid],
            by[valid],
            color="#dc2626",
            linewidth=2.0,
            marker="o",
            markersize=5,
            zorder=3,
        )
        ax.set_xticks(range(N_WEEKDAYS))
        ax.set_xticklabels(WEEKDAY_NAMES, fontsize=9)
        ax.set_xlim(-0.5, N_WEEKDAYS - 0.5)
        mean_label = "среднее P(up) по дню"
        weekday_empty = [wd for wd in range(N_WEEKDAYS) if counts[wd] == 0]
    else:
        bx, by, _ = _binned_mean(x, y_prob, n_bins=CONTINUOUS_BINS)
        ax.plot(bx, by, color="#dc2626", linewidth=2.0, zorder=3)
        x_p1, x_p99 = np.nanpercentile(x, [1, 99])
        x_view = x[(x >= x_p1) & (x <= x_p99)]
        x_lo, x_hi = _padded_limits(x_view.astype(float), min_span=MIN_X_SPAN)
        ax.set_xlim(x_lo, x_hi)

    y_vals = np.concatenate([y_prob[np.isfinite(y_prob)], by[np.isfinite(by)]])
    if y_vals.size:
        y_p1, y_p99 = np.nanpercentile(y_vals, [1, 99])
        y_view = y_vals[(y_vals >= y_p1) & (y_vals <= y_p99)]
    else:
        y_view = y_vals
    y_lo, y_hi = _padded_limits(
        y_view,
        min_span=MIN_Y_SPAN,
        clamp_lo=0.0,
        clamp_hi=1.0,
    )
    ax.set_ylim(y_lo, y_hi)
    for wd in weekday_empty:
        ax.text(wd, y_lo, "—", ha="center", va="bottom", fontsize=8, color="#9ca3af")
    show_half = y_lo < 0.5 < y_hi
    if show_half:
        ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel("P(up)", fontsize=9)
    ax.grid(True, alpha=0.25)
    _panel_legend(ax, mean_label=mean_label, show_half=show_half)


def save_feature_prob_dependence_plot(
    frame: pl.DataFrame,
    y_prob: np.ndarray,
    feature_columns: tuple[str, ...] | list[str],
    path: Path,
    *,
    title: str = "Dependence: feature vs P(up)",
    period_label: str = "holdout test",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = dependence_feature_columns(feature_columns)
    if not features:
        log.warning("[ml] feature_prob_dependence: нет фич (кроме pair_id)")
        return path

    n = len(features)
    n_cols = 2
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4.2 * n_rows),
        squeeze=False,
    )

    y_prob_arr = np.asarray(y_prob, dtype=float)
    for i, feature in enumerate(features):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        x = feature_values_as_float(frame, feature)
        _plot_panel(ax, x, y_prob_arr, feature)

    for j in range(n, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(f"{title}\n({period_label}, n={frame.height})", fontweight="semibold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] feature P(up) dependence plot saved: %s (features=%s)", path, features)
    return path
