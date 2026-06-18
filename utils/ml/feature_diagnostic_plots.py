"""Диагностические графики по числовым ML-фичам (корреляция, форма распределения)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import skew

from crypto_research.utils.ml.numeric_features import active_numeric_specs
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_feature_diagnostic_plots")

PLOT_DPI = 200
HIST_BINS = 40


def _numeric_columns(feature_columns: tuple[str, ...] | list[str]) -> list[str]:
    return [ns.column for ns in active_numeric_specs(feature_columns)]


def feature_correlation_matrix(
    frame: pl.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> tuple[list[str], np.ndarray]:
    cols = [c for c in _numeric_columns(feature_columns) if c in frame.columns]
    if not cols:
        return [], np.empty((0, 0))
    matrix = np.column_stack([frame[c].to_numpy() for c in cols]).astype(float)
    mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[mask]
    if matrix.shape[0] < 2:
        corr = np.eye(len(cols), dtype=float)
    else:
        corr = np.corrcoef(matrix, rowvar=False)
    return cols, corr


def save_correlation_matrix_heatmap(
    frame: pl.DataFrame,
    path: Path,
    feature_columns: tuple[str, ...] | list[str],
    *,
    title: str,
    period_label: str = "holdout test",
) -> tuple[Path, dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols, corr = feature_correlation_matrix(frame, feature_columns)
    if not cols:
        log.warning("[ml] correlation_matrix_heatmap: нет числовых фич в датасете")
        return path, {"columns": [], "matrix": [], "n_rows": 0}

    n = len(cols)
    fig_w = max(5.0, 1.2 * n + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = float(corr[i, j])
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(f"{title}\n({period_label}, n={frame.height})", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    pairs: dict[str, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairs[f"{cols[i]}__{cols[j]}"] = float(corr[i, j])

    payload: dict[str, object] = {
        "columns": cols,
        "matrix": [[float(corr[i, j]) for j in range(n)] for i in range(n)],
        "pairwise": pairs,
        "n_rows": int(frame.height),
        "method": "pearson",
    }
    for key, val in pairs.items():
        log.info("[ml] feature corr %s: %.4f", key.replace("__", " vs "), val)
    log.info("[ml] correlation matrix heatmap saved: %s", path)
    return path, payload


def _shared_hist_xlim(arrays: list[np.ndarray], *, pad_ratio: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(1e-6, (hi - lo) * pad_ratio)
    return lo - pad, hi + pad


def save_shape_summary_plot(
    frame: pl.DataFrame,
    path: Path,
    feature_columns: tuple[str, ...] | list[str],
    *,
    title: str,
    period_label: str = "holdout test",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in _numeric_columns(feature_columns) if c in frame.columns]
    if not cols:
        log.warning("[ml] shape_summary_plot: нет числовых фич в датасете")
        return path

    arrays = [frame[c].to_numpy().astype(float) for c in cols]
    xlim = _shared_hist_xlim(arrays)
    bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)
    ymax = 0.0
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            counts, _ = np.histogram(finite, bins=bins, density=True)
            ymax = max(ymax, float(counts.max()) if counts.size else 0.0)
    ymax *= 1.08 if ymax > 0 else 1.0

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(max(6.0, 4.5 * n), 4.2), squeeze=False)
    for i, (col, arr) in enumerate(zip(cols, arrays)):
        ax = axes[0, i]
        finite = arr[np.isfinite(arr)]
        ax.hist(finite, bins=bins, density=True, alpha=0.75, color="#2563eb", edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.mean(finite)), color="#dc2626", linestyle="--", linewidth=1.0, label="mean")
        ax.axvline(float(np.median(finite)), color="#16a34a", linestyle=":", linewidth=1.0, label="median")
        sk = float(skew(finite)) if finite.size > 2 else float("nan")
        ax.set_title(
            f"{col}\nμ={np.mean(finite):.3f}  σ={np.std(finite):.3f}  skew={sk:.2f}",
            fontsize=9,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel(col, fontsize=9)
        if i == 0:
            ax.set_ylabel("плотность")
        ax.grid(True, alpha=0.25)
        if i == n - 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{title}\n({period_label}, n={frame.height})", fontweight="semibold", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] shape summary plot saved: %s", path)
    return path
