"""Графики стабильности для этапа выбора периода EMA."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from crypto_research.utils.ema_spreads.period_stability import (
    PeriodStabilityMetrics,
    yearly_material_agreement_rates,
)
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.weekday.bands import MeanBands

log = get_logger("ema_period_screen_plots")

PLOT_DPI = 160
COLOR_AGREE = "#2ca02c"


def save_stability_index_chart(
    metrics: list[PeriodStabilityMetrics],
    path: Path,
    *,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(metrics, key=lambda m: m.stability_index_pct, reverse=True)
    labels = [f"EMA({m.period})" for m in ordered]
    values = [m.stability_index_pct for m in ordered]
    colors = [COLOR_AGREE if m.rank == 1 else "#4c72b0" for m in ordered]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.55)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Индекс стабильности (%)")
    ax.set_title(
        "Выбор периода EMA — индекс: ср.|Δ|, годы, пары, значимые ячейки (по 25%)\n"
        f"«Цена росла» / «Цена падала» · {n_pairs} пар · "
        f"{from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d} (UTC)",
        fontsize=10,
    )
    for i, (val, m) in enumerate(zip(values, ordered, strict=True)):
        ax.text(
            val + 1.0,
            i,
            f"{val:.1f}%  (#{m.rank})",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("График индекса стабильности: %s", path)


def save_yearly_agreement_heatmap(
    daily: pl.DataFrame,
    periods: tuple[int, ...],
    pair_bands: dict[str, MeanBands],
    path: Path,
    *,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series: dict[int, dict[int, float]] = {}
    all_years: set[int] = set()
    for period in periods:
        rows = yearly_material_agreement_rates(daily, period, periods, pair_bands)
        series[period] = {r.year: r.agreement_rate for r in rows}
        all_years.update(series[period].keys())
    if not all_years:
        return

    years_sorted = sorted(all_years)
    period_list = list(periods)
    matrix = np.full((len(period_list), len(years_sorted)), np.nan, dtype=np.float64)
    year_index = {y: i for i, y in enumerate(years_sorted)}
    for row, period in enumerate(period_list):
        for y, rate in series[period].items():
            matrix[row, year_index[y]] = rate

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "agreement", ["#d62728", "#ffee99", "#2ca02c"]
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(years_sorted)), [str(y) for y in years_sorted])
    ax.set_yticks(np.arange(len(period_list)), [f"EMA({p})" for p in period_list])
    ax.set_xlabel("Год (UTC)")
    ax.set_ylabel("Период EMA")
    ax.set_title(
        "Доля материальных ячеек с согласованным знаком Δ в году\n"
        f"{n_pairs} пар · {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("доля ячеек")
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap согласия по годам: %s", path)
