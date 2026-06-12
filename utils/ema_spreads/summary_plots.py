"""Графики сводки подтверждения сигналов EMA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ema_spreads.summary import EmaSignalSummaryRow
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ema_summary_plots")

PLOT_DPI = 160
COLOR_SIGNIFICANT = "#2ca02c"


def _short_label(row: EmaSignalSummaryRow) -> str:
    return f"b{row.key.bucket}·{row.key.column[:6]}"


def save_check_signal_delta_chart(
    rows: list[EmaSignalSummaryRow],
    path: Path,
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
    title: str,
    subtitle: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    labels = [_short_label(r) for r in rows]
    train_vals = [r.train_delta_pp for r in rows]
    val_vals = [
        r.val_delta_pp if r.val_delta_pp == r.val_delta_pp else 0.0
        for r in rows
    ]
    y = np.arange(len(labels))
    bar_height = 0.35
    fig_h = max(2.25, len(labels) * 0.225)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(y - bar_height / 2, train_vals, height=bar_height, label="Δ train", color="#1f77b4")
    ax.barh(y + bar_height / 2, val_vals, height=bar_height, label="Δ val", color="#ff7f0e")
    for i, row in enumerate(rows):
        if row.status == "значим":
            ax.axhspan(i - 0.5, i + 0.5, color=COLOR_SIGNIFICANT, alpha=0.08, zorder=0)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Δ к BASE, п.п.")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("График Δ сигналов: %s", path)
