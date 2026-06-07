"""Графики бэктеста."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES, drawdown_series
from crypto_research.utils.backtest.metrics import INITIAL_NAV, equity_curve_simple
from crypto_research.utils.plotting.date_axis import format_date_axis
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("backtest_plots")

PLOT_DPI = 200
FIG_W = 16.0
FIG_H = 7.0

COLOR_MAKER = "#dc2626"
COLOR_TAKER = "#2563eb"
COLOR_BH = "#374151"
COLOR_BTC = "#cbd5e1"
LW_MAKER = 2.4
LW_TAKER = 1.5
LW_BENCH = 1.3
LW_WEEKDAY_DD = 1.2

WEEKDAY_DD_COLORS: dict[int, str] = {
    3: "#2563eb",  # Чт
    4: "#d97706",  # Пт
    5: "#059669",  # Сб
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafbfc",
            "axes.edgecolor": "#c5cdd8",
            "font.family": "sans-serif",
        }
    )


def _returns_on_weekday_only(
    returns: np.ndarray,
    weekdays: np.ndarray,
    target_wd: int,
) -> np.ndarray:
    return np.where(weekdays == target_wd, returns, 0.0)


def _plot_equity_lines(
    ax: plt.Axes,
    dates: np.ndarray,
    nav_taker: np.ndarray,
    nav_maker: np.ndarray,
    nav_bh: np.ndarray | None = None,
    nav_btc: np.ndarray | None = None,
    *,
    show_legend: bool,
    include_benchmarks: bool = True,
    compact: bool = False,
) -> None:
    if include_benchmarks and nav_bh is not None:
        ax.plot(dates, nav_bh, color=COLOR_BH, linewidth=LW_BENCH, label="B&H gross", alpha=0.95)
    if include_benchmarks and nav_btc is not None:
        ax.plot(dates, nav_btc, color=COLOR_BTC, linewidth=LW_BENCH, label="BTC B&H", alpha=0.95)
    ax.plot(dates, nav_taker, color=COLOR_TAKER, linewidth=LW_TAKER, label="Net taker")
    ax.plot(dates, nav_maker, color=COLOR_MAKER, linewidth=LW_MAKER, label="Net maker")
    ax.axhline(INITIAL_NAV, color="#64748b", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="upper left", fontsize=9)
    format_date_axis(ax, rotate=40 if compact else 25, labelsize=8 if compact else None)


def _equity_nav_series(
    returns: np.ndarray,
    weekdays: np.ndarray,
    target_wd: int | None,
) -> np.ndarray:
    if target_wd is None:
        series = returns
    else:
        series = _returns_on_weekday_only(returns, weekdays, target_wd)
    return equity_curve_simple(series)


def save_equity_curve_plot(
    portfolio: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    btc: pl.DataFrame | None = None,
    trading_weekdays: tuple[int, ...] = (3, 4, 5),
    strategy: str,
    scenario_label: str | None = None,
    from_date: datetime,
    to_date: datetime,
    n_pairs: int,
    path: Path,
) -> Path:
    _apply_style()
    merged = (
        portfolio.select("day_utc", "weekday", "net_return_pct", "net_maker_return_pct")
        .join(
            benchmark.select("day_utc", pl.col("gross_return_pct").alias("bh_return_pct")),
            on="day_utc",
            how="left",
        )
    )
    if btc is not None:
        merged = merged.join(
            btc.select("day_utc", pl.col("gross_return_pct").alias("btc_return_pct")),
            on="day_utc",
            how="left",
        )
    else:
        merged = merged.with_columns(pl.lit(None).cast(pl.Float64).alias("btc_return_pct"))

    dates = merged["day_utc"].to_numpy()
    weekdays = merged["weekday"].to_numpy()
    net = merged["net_return_pct"].fill_null(0.0).to_numpy()
    maker = merged["net_maker_return_pct"].fill_null(0.0).to_numpy()
    bh = merged["bh_return_pct"].fill_null(0.0).to_numpy()
    btc_col = merged["btc_return_pct"]
    has_btc = btc is not None and btc_col.null_count() < btc_col.len()
    btc_arr = btc_col.fill_null(0.0).to_numpy() if has_btc else None

    nav_taker = _equity_nav_series(net, weekdays, None)
    nav_maker = _equity_nav_series(maker, weekdays, None)
    nav_bh = _equity_nav_series(bh, weekdays, None)
    nav_btc = _equity_nav_series(btc_arr, weekdays, None) if btc_arr is not None else None

    fig = plt.figure(figsize=(28.0, 18.0), dpi=PLOT_DPI)
    gs = fig.add_gridspec(2, len(trading_weekdays), height_ratios=[2.2, 1.0], hspace=0.38, wspace=0.22)
    ax_main = fig.add_subplot(gs[0, :])

    _plot_equity_lines(
        ax_main, dates, nav_taker, nav_maker, nav_bh, nav_btc, show_legend=True
    )
    ax_main.set_ylabel("NAV (simple, base=100)", fontsize=10)
    ax_main.set_xlabel("")
    title_line = f"Backtest equity — {strategy}"
    if scenario_label:
        title_line = f"{title_line} · {scenario_label}"
    ax_main.set_title(
        f"{title_line}\n"
        f"{n_pairs} pairs · {from_date:%Y-%m-%d} — {to_date:%Y-%m-%d} · equal weight · no reinvest",
        loc="left",
        fontsize=12,
        fontweight="semibold",
    )

    for col, wd in enumerate(trading_weekdays):
        ax = fig.add_subplot(gs[1, col])
        wd_taker = _equity_nav_series(net, weekdays, wd)
        wd_maker = _equity_nav_series(maker, weekdays, wd)
        _plot_equity_lines(
            ax, dates, wd_taker, wd_maker, show_legend=False, include_benchmarks=False, compact=True
        )
        ax.set_title(WEEKDAY_NAMES[wd], loc="left", fontsize=11, fontweight="semibold")
        ax.set_ylabel("NAV", fontsize=9)
        ax.set_xlabel("Date (UTC)", fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    log.info("[backtest] plot: %s", path)
    return path


def _drawdown_for_weekday(
    returns: np.ndarray,
    weekdays: np.ndarray,
    target_wd: int | None,
) -> np.ndarray:
    if target_wd is not None:
        returns = _returns_on_weekday_only(returns, weekdays, target_wd)
    return drawdown_series(returns)


def save_drawdown_plot(
    portfolio: pl.DataFrame,
    *,
    strategy: str,
    trading_weekdays: tuple[int, ...] = (3, 4, 5),
    scenario_label: str | None = None,
    path: Path,
) -> Path:
    _apply_style()
    dates = portfolio["day_utc"].to_numpy()
    weekdays = portfolio["weekday"].to_numpy()
    maker = portfolio["net_maker_return_pct"].fill_null(0.0).to_numpy()
    dd_total = _drawdown_for_weekday(maker, weekdays, None)

    fig, ax = plt.subplots(figsize=(FIG_W, 5.5), dpi=PLOT_DPI)
    for wd in trading_weekdays:
        dd_wd = _drawdown_for_weekday(maker, weekdays, wd)
        color = WEEKDAY_DD_COLORS.get(wd, "#64748b")
        ax.plot(
            dates,
            dd_wd,
            color=color,
            linewidth=LW_WEEKDAY_DD,
            label=WEEKDAY_NAMES[wd],
            alpha=0.9,
        )
    ax.plot(
        dates,
        dd_total,
        color=COLOR_MAKER,
        linewidth=LW_MAKER,
        label="Общий (maker)",
        zorder=5,
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date (UTC)")
    title_line = f"Drawdown — {strategy}"
    if scenario_label:
        title_line = f"{title_line} · {scenario_label}"
    ax.set_title(title_line, loc="left", fontweight="semibold")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=9)
    format_date_axis(ax, rotate=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info("[backtest] plot: %s", path)
    return path


def save_returns_histogram_plot(
    portfolio: pl.DataFrame,
    *,
    strategy: str,
    path: Path,
) -> Path:
    _apply_style()
    traded = portfolio.filter(pl.col("position") != 0)
    net = traded["net_return_pct"].to_numpy()
    gross = traded["gross_return_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, 5.0), dpi=PLOT_DPI)
    bins = np.linspace(min(net.min(), gross.min()) - 0.5, max(net.max(), gross.max()) + 0.5, 40)
    ax.hist(gross, bins=bins, alpha=0.45, color="#94a3b8", label="Gross (trading days)", density=True)
    ax.hist(net, bins=bins, alpha=0.55, color="#2563eb", label="Net (trading days)", density=True)
    ax.axvline(0, color="#64748b", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Daily portfolio return (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Return distribution — {strategy} (trading days only)", loc="left", fontweight="semibold")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info("[backtest] plot: %s", path)
    return path


def save_weekday_corr_plot(
    labels: tuple[str, ...],
    corr: np.ndarray,
    *,
    trading_weekdays: tuple[int, ...],
    strategy: str,
    path: Path,
) -> Path:
    _apply_style()
    sub_labels = [labels[i] for i in trading_weekdays]
    sub = corr[np.ix_(trading_weekdays, trading_weekdays)]

    fig, ax = plt.subplots(figsize=(4.0, 3.25), dpi=PLOT_DPI)
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sub_labels)))
    ax.set_yticks(range(len(sub_labels)))
    ax.set_xticklabels(sub_labels, fontsize=9)
    ax.set_yticklabels(sub_labels, fontsize=9)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            val = sub[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(
        f"Weekday return correlation — {strategy}",
        loc="left",
        fontweight="semibold",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info("[backtest] plot: %s", path)
    return path
