"""Диагностические графики по ML-фичам модели (корреляция)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from crypto_research.utils.ml.plot_features import feature_correlation_matrix
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_feature_diagnostic_plots")

PLOT_DPI = 200


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
        log.warning("[ml] correlation_matrix_heatmap: нет фич модели в датасете")
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
