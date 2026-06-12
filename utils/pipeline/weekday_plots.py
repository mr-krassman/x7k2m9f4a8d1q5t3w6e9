"""Simple cumulative return index: constant daily notional, non-compounded (UTC DOW)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import polars as pl

from crypto_research.utils.plotting.date_axis import format_date_axis
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("weekday_plots")

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_LABELS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
WEEKDAY_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]
Y_LABEL = "Cumulative Simple Return (%)"
ZERO_LINE = 0.0
PLOT_DPI = 160
FIG_W_IN = 18.0
TITLE_TOP_IN = 0.10
TITLE_LINE_H_IN = 0.15
TITLE_N_LINES = 3
TITLE_GAP_PLOT_IN = 0.05
LINE_WIDTH_DEFAULT = 1.6
LINE_WIDTH_COMPACT = 1.2
LINE_WIDTH_HIGHLIGHT = 3.2
LINE_WIDTH_DIM = 0.9
AGGREGATE_COLOR = "#c41e3a"
AGGREGATE_LINE_WIDTH = 3.2
CHECK_FIG_W_IN = 20.0
CHECK_FIG_H_IN = 6.5
MAIN_PLOT_BOTTOM = 0.18
MAIN_PLOT_LEFT = 0.07
MAIN_PLOT_WIDTH = 0.91
SAVE_PAD_INCHES = 0.12


def _title_block_height_in() -> float:
    return TITLE_TOP_IN + TITLE_N_LINES * TITLE_LINE_H_IN + TITLE_GAP_PLOT_IN


def _add_figure_title(fig: plt.Figure, fig_h_in: float, n_pairs: int, period: str) -> None:
    lines = [
        "Intraday Session P&L Profile — Long at Open, Flat at Close (Gross, UTC)",
        f"Equal-weight mean · {n_pairs} pairs · Cumulative Simple Return by Weekday",
        period,
    ]
    y = 1.0 - TITLE_TOP_IN / fig_h_in
    step = TITLE_LINE_H_IN / fig_h_in
    for i, line in enumerate(lines):
        fig.text(
            0.07,
            y - i * step,
            line,
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold" if i == 0 else "normal",
        )


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafbfc",
            "axes.edgecolor": "#c5cdd8",
            "axes.labelcolor": "#1a1d21",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "#4a5568",
            "ytick.color": "#4a5568",
            "grid.color": "#e2e8f0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d1d9e6",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        }
    )


def _normalize_weekday(daily: pl.DataFrame) -> pl.DataFrame:
    wd = daily["weekday"] if "weekday" in daily.columns else daily["day_utc"].dt.weekday()
    wd_min = int(wd.min())
    wd_max = int(wd.max())
    if wd_min >= 1 and wd_max <= 7:
        wd = ((wd - 1) % 7).cast(pl.Int64)
    return daily.with_columns(wd.alias("weekday"))


def _session_returns(daily: pl.DataFrame, pair: str | None) -> pl.DataFrame:
    df = _normalize_weekday(daily)
    if pair is not None:
        df = df.filter(pl.col("pair") == pair)
    return (
        df.group_by("day_utc", "weekday")
        .agg(pl.col("return_pct").mean().alias("return_pct"))
        .sort("day_utc")
    )


def _cumulative_simple_return_pct(returns_pct: np.ndarray) -> np.ndarray:
    """Constant notional; arithmetic sum of daily simple returns, in percent (pp)."""
    return np.cumsum(returns_pct.astype(np.float64))


def _all_weekdays_curve(daily: pl.DataFrame, pair: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    df = _normalize_weekday(daily)
    if pair is not None:
        df = df.filter(pl.col("pair") == pair)
    agg = (
        df.group_by("day_utc")
        .agg(pl.col("return_pct").mean().alias("return_pct"))
        .sort("day_utc")
    )
    if agg.height == 0:
        return None
    dates = agg["day_utc"].to_numpy()
    nav = _cumulative_simple_return_pct(agg["return_pct"].to_numpy())
    return dates, nav


def _weekday_curves(session: pl.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for wd in range(7):
        sub = session.filter(pl.col("weekday") == wd).sort("day_utc")
        if sub.height == 0:
            continue
        dates = sub["day_utc"].to_numpy()
        nav = _cumulative_simple_return_pct(sub["return_pct"].to_numpy())
        out[wd] = (dates, nav)
    return out


def _format_date_axis(ax: plt.Axes, *, rotate: int = 0) -> None:
    format_date_axis(ax, rotate=rotate)

def _line_style(
    wd: int,
    *,
    compact: bool,
    highlight_weekdays: frozenset[int],
) -> tuple[float, float, int]:
    base = LINE_WIDTH_COMPACT if compact else LINE_WIDTH_DEFAULT
    if not highlight_weekdays:
        return base, 0.92, 2
    if wd in highlight_weekdays:
        return LINE_WIDTH_HIGHLIGHT, 1.0, 4
    return LINE_WIDTH_DIM, 0.35, 1


def _plot_weekday_nav(
    ax: plt.Axes,
    curves: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    compact: bool,
    xlabel: str | None = None,
    highlight_weekdays: frozenset[int] = frozenset(),
    aggregate_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    for wd in range(7):
        if wd not in curves:
            continue
        dates, nav = curves[wd]
        lw, alpha, zorder = _line_style(
            wd,
            compact=compact,
            highlight_weekdays=highlight_weekdays,
        )
        ax.plot(
            dates,
            nav,
            color=WEEKDAY_COLORS[wd],
            linewidth=lw,
            label=f"{WEEKDAY_LABELS[wd]} ({WEEKDAY_LABELS_RU[wd]})",
            alpha=alpha,
            zorder=zorder,
        )
    if aggregate_curve is not None:
        dates, nav = aggregate_curve
        ax.plot(
            dates,
            nav,
            color=AGGREGATE_COLOR,
            linewidth=AGGREGATE_LINE_WIDTH,
            label="All weekdays (Все дни)",
            alpha=1.0,
            zorder=5,
        )
    ax.axhline(ZERO_LINE, color="#94a3b8", linewidth=0.8, linestyle="--", zorder=0)
    if title:
        ax.set_title(title, loc="left", pad=4 if compact else 8)
    if not compact:
        ax.set_ylabel(Y_LABEL)
        ax.set_xlabel(xlabel or "Date (UTC)", labelpad=10)
    else:
        ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.85)
    ax.grid(True, axis="x", alpha=0.35)
    _format_date_axis(ax, rotate=40 if compact else 25)
    ax.legend(
        loc="upper left",
        fontsize=7 if compact else 8,
        ncol=2 if not compact else 1,
        framealpha=0.95,
    )


def _build_figure(
    daily: pl.DataFrame,
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    *,
    highlight_weekdays: frozenset[int] = frozenset(),
    main_plot_only: bool = False,
) -> plt.Figure:
    pair_list = sorted(pairs)
    n_pairs = len(pair_list)
    period = f"{from_date:%Y-%m-%d} — {to_date:%Y-%m-%d}"

    if main_plot_only:
        fig_h = 6.5
        fig = plt.figure(figsize=(FIG_W_IN, fig_h), dpi=PLOT_DPI)
        _add_figure_title(fig, fig_h, n_pairs, period)
        grid_top = 1.0 - _title_block_height_in() / fig_h
        ax = fig.add_axes(
            [
                MAIN_PLOT_LEFT,
                MAIN_PLOT_BOTTOM,
                MAIN_PLOT_WIDTH,
                grid_top - MAIN_PLOT_BOTTOM,
            ]
        )
        agg_curves = _weekday_curves(_session_returns(daily, None))
        _plot_weekday_nav(
            ax,
            agg_curves,
            title="",
            compact=False,
            xlabel=f"Date (UTC) — equal-weight mean across {n_pairs} pairs",
            highlight_weekdays=highlight_weekdays,
        )
        return fig

    n_rows = max(1, (len(pair_list) + 1) // 2)
    fig_h = 5.5 + n_rows * 2.4
    fig = plt.figure(figsize=(FIG_W_IN, fig_h), dpi=PLOT_DPI)
    _add_figure_title(fig, fig_h, n_pairs, period)
    grid_top = 1.0 - _title_block_height_in() / fig_h
    gs = gridspec.GridSpec(
        1 + n_rows,
        2,
        figure=fig,
        height_ratios=[2.8] + [1.0] * n_rows,
        hspace=0.442,
        wspace=0.22,
        top=grid_top,
        bottom=0.06,
        left=0.07,
        right=0.98,
    )

    agg_curves = _weekday_curves(_session_returns(daily, None))
    ax_top = fig.add_subplot(gs[0, :])
    _plot_weekday_nav(
        ax_top,
        agg_curves,
        title="",
        compact=False,
        xlabel=f"Date (UTC) — equal-weight mean across {n_pairs} pairs",
        highlight_weekdays=highlight_weekdays,
    )

    for idx, pair in enumerate(pair_list):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[1 + row, col])
        curves = _weekday_curves(_session_returns(daily, pair))
        _plot_weekday_nav(
            ax,
            curves,
            title=f"{pair.upper()} | Simple Cumulative Return",
            compact=True,
            highlight_weekdays=highlight_weekdays,
        )

    return fig


def _plot_train_val_panel(
    ax: plt.Axes,
    daily: pl.DataFrame,
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    *,
    panel_title: str,
) -> None:
    session = _session_returns(daily, None)
    curves = _weekday_curves(session)
    aggregate = _all_weekdays_curve(daily, None)
    period = f"{from_date:%Y-%m-%d} — {to_date:%Y-%m-%d}"
    _plot_weekday_nav(
        ax,
        curves,
        title=f"{panel_title}\n{period} · {len(pairs)} pairs",
        compact=False,
        xlabel="Date (UTC)",
        aggregate_curve=aggregate,
    )


def save_weekday_train_val_nav_comparison(
    train_daily: pl.DataFrame,
    train_pairs: list[str],
    train_from: datetime,
    train_to: datetime,
    train_panel_title: str,
    val_daily: pl.DataFrame,
    val_pairs: list[str],
    val_from: datetime,
    val_to: datetime,
    val_panel_title: str,
    path: Path,
    *,
    suptitle: str,
) -> Path:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(CHECK_FIG_W_IN, CHECK_FIG_H_IN), dpi=PLOT_DPI)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.98)
    _plot_train_val_panel(
        axes[0],
        train_daily,
        train_pairs,
        train_from,
        train_to,
        panel_title=train_panel_title,
    )
    _plot_train_val_panel(
        axes[1],
        val_daily,
        val_pairs,
        val_from,
        val_to,
        panel_title=val_panel_title,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    log.info("[2] Train/val NAV comparison: %s", path)
    return path


def save_weekday_nav_plots(
    daily: pl.DataFrame,
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    path: Path,
    *,
    highlight_weekdays: frozenset[int] = frozenset(),
    main_plot_only: bool = False,
) -> Path:
    _apply_plot_style()
    fig = _build_figure(
        daily,
        pairs,
        from_date,
        to_date,
        highlight_weekdays=highlight_weekdays,
        main_plot_only=main_plot_only,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {
        "dpi": PLOT_DPI,
        "facecolor": fig.get_facecolor(),
        "pad_inches": SAVE_PAD_INCHES,
    }
    if main_plot_only:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    log.info("[2] NAV plots: %s", path)
    return path

