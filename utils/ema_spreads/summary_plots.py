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
COLOR_OTHER = "#4c72b0"


def _short_label(row: EmaSignalSummaryRow) -> str:
    return f"b{row.key.bucket}·{row.key.column[:6]}"


def save_signal_delta_chart(
    pair_rows: list[EmaSignalSummaryRow],
    path: Path,
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    material = [
        r for r in pair_rows
        if r.train_delta_pp == r.train_delta_pp
        and abs(r.train_delta_pp) >= 0.01
    ]
    if not material:
        return
    labels = [_short_label(r) for r in material]
    train_vals = [r.train_delta_pp for r in material]
    val_vals = [
        r.val_delta_pp if r.val_delta_pp == r.val_delta_pp else 0.0
        for r in material
    ]
    colors = [
        COLOR_SIGNIFICANT if r.status == "значим" else COLOR_OTHER
        for r in material
    ]
    y = np.arange(len(labels))
    height = 0.35
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(labels) * 0.45)))
    ax.barh(y - height / 2, train_vals, height=height, label="Δ train", color="#1f77b4")
    ax.barh(y + height / 2, val_vals, height=height, label="Δ val", color="#ff7f0e")
    for i, c in enumerate(colors):
        if c == COLOR_SIGNIFICANT:
            ax.axhspan(i - 0.5, i + 0.5, color=COLOR_SIGNIFICANT, alpha=0.08, zorder=0)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Δ к BASE, п.п.")
    ax.set_title(
        f"EMA({ema_period}) — Δ train vs val по сигналам (универсальность среди пар)\n"
        "зелёная полоса = итог «значим» в блоке 1",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("График Δ сигналов: %s", path)


def save_confirmation_chart(
    pair_rows: list[EmaSignalSummaryRow],
    temporal_rows: list[EmaSignalSummaryRow],
    path: Path,
    *,
    ema_period: int = SELECTED_EMA_PERIOD,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [_short_label(r) for r in pair_rows]
    pair_pct = [
        (r.val_agree or 0) / r.val_total * 100 if r.val_total else 0.0
        for r in pair_rows
    ]
    temporal_pct = [
        (r.val_agree or 0) / r.val_total * 100 if r.val_total else 0.0
        for r in temporal_rows
    ]
    y = np.arange(len(labels))
    height = 0.35
    fig, ax = plt.subplots(figsize=(11, max(5.0, len(labels) * 0.45)))
    ax.barh(y - height / 2, pair_pct, height=height, label="Пары train/val", color="#1f77b4")
    ax.barh(y + height / 2, temporal_pct, height=height, label="Время train/val", color="#ff7f0e")
    ax.axvline(60.0, color=COLOR_SIGNIFICANT, linestyle="--", linewidth=1.2, label="порог 60%")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Подтверждение на val, %")
    ax.set_title(
        f"EMA({ema_period}) — доля подтверждения сигналов на val\n"
        "блок 1: когорты пар · блок 2: те же пары, другой период",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("График подтверждения сигналов: %s", path)
