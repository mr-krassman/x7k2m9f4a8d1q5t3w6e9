"""Dependence plot: mean return open→close vs P(up), train | test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from crypto_research.utils.ml.trading_thresholds import resolve_prob_return_threshold_pair
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_prob_return_dependence_plot")

PLOT_DPI = 200
FINE_BINS = 50
SCATTER_MAX_POINTS = 2500
SCATTER_ALPHA = 0.12
SCATTER_SIZE = 4
AXIS_PAD_RATIO = 0.08
X_TICK_STEP = 0.01
X_TICK_LABEL_FONTSIZE = 6
LONG_LINE_COLOR = "#2563eb"
SHORT_LINE_COLOR = "#dc2626"
CURVE_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
THRESHOLD_EDGE_PCT = 0.25


@dataclass(frozen=True)
class SmoothCurveSpec:
    label: str
    kind: Literal["bins", "roll", "gauss"]
    n_bins: int
    window: int = 0
    sigma: float = 0.0


THRESHOLD_SMOOTH_SPEC = SmoothCurveSpec("gauss σ=4", "gauss", FINE_BINS, sigma=4.0)


SMOOTH_CURVE_SPECS: tuple[SmoothCurveSpec, ...] = (
    # SmoothCurveSpec("bins=6", "bins", 6),
    # SmoothCurveSpec("bins=10", "bins", 10),
    # SmoothCurveSpec("bins=16", "bins", 16),
    # SmoothCurveSpec("bins=24", "bins", 24),
    SmoothCurveSpec("bins=500", "bins", 50),
    # SmoothCurveSpec("roll w=3", "roll", FINE_BINS, window=3),
    # SmoothCurveSpec("roll w=7", "roll", FINE_BINS, window=7),
    # SmoothCurveSpec("roll w=13", "roll", FINE_BINS, window=13),
    SmoothCurveSpec("gauss σ=4", "gauss", FINE_BINS, sigma=4.0),
    # SmoothCurveSpec("gauss σ=2.5", "gauss", FINE_BINS, sigma=2.5),
)


def _subsample_mask(n: int, *, max_points: int, seed: int = 42) -> np.ndarray:
    if n <= max_points:
        return np.ones(n, dtype=bool)
    idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def _binned_mean(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.any():
            means[i] = float(np.mean(y[mask]))
    valid = np.isfinite(means)
    bx = centers[valid]
    by = means[valid]
    order = np.argsort(bx)
    return bx[order], by[order]


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if y.size < window or window < 2:
        return y.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def _gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if y.size < 3 or sigma <= 0:
        return y.copy()
    radius = max(1, int(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def _smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    spec: SmoothCurveSpec,
) -> tuple[np.ndarray, np.ndarray]:
    bx, by = _binned_mean(x, y, n_bins=spec.n_bins)
    if bx.size == 0:
        return bx, by
    if spec.kind == "bins":
        return bx, by
    if spec.kind == "roll":
        return bx, _rolling_mean(by, spec.window)
    return bx, _gaussian_smooth(by, spec.sigma)


def _all_smooth_curves(
    x: np.ndarray,
    y: np.ndarray,
) -> list[tuple[SmoothCurveSpec, np.ndarray, np.ndarray]]:
    out: list[tuple[SmoothCurveSpec, np.ndarray, np.ndarray]] = []
    for spec in SMOOTH_CURVE_SPECS:
        bx, by = _smooth_curve(x, y, spec)
        if bx.size:
            out.append((spec, bx, by))
    return out


def _correlation_label(x: np.ndarray, y: np.ndarray) -> str:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return "r=—, ρ=—"
    r = float(np.corrcoef(x, y)[0, 1])
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    rho = float(np.corrcoef(xr, yr)[0, 1])
    return f"Pearson r={r:.3f}, Spearman ρ={rho:.3f}"


def _edge_thresholds(
    bx: np.ndarray,
    by: np.ndarray,
    *,
    edge_pct: float,
) -> tuple[float | None, float | None]:
    """
    t_long: минимальный P(up), с которого сглаженный return > edge_pct и не падает ниже.
    t_short: максимальный P(up), до которого сглаженный return < -edge_pct и не поднимается выше.
    """
    if bx.size == 0:
        return None, None

    t_long: float | None = None
    for i in range(bx.size):
        if np.all(by[i:] > edge_pct):
            t_long = float(bx[i])
            break

    t_short: float | None = None
    for i in range(bx.size - 1, -1, -1):
        if np.all(by[: i + 1] < -edge_pct):
            t_short = float(bx[i])
            break

    return t_long, t_short


def _train_thresholds_from_smooth_curve(
    y_prob: np.ndarray,
    return_pct: np.ndarray,
    *,
    spec: SmoothCurveSpec = THRESHOLD_SMOOTH_SPEC,
    edge_pct: float = THRESHOLD_EDGE_PCT,
) -> tuple[float | None, float | None]:
    mask = np.isfinite(return_pct) & np.isfinite(y_prob)
    x = y_prob[mask]
    y = return_pct[mask]
    if x.size == 0:
        return None, None
    bx, by = _smooth_curve(x, y, spec)
    return _edge_thresholds(bx, by, edge_pct=edge_pct)


def _aligned_prob_return_from_oos(
    frame: pl.DataFrame,
    oos: pl.DataFrame,
    *,
    return_column: str = "return_pct",
) -> tuple[np.ndarray, np.ndarray]:
    merged = oos.select("day_utc", "pair", "y_prob").join(
        frame.select("day_utc", "pair", return_column),
        on=["day_utc", "pair"],
        how="inner",
    )
    return (
        merged["y_prob"].to_numpy().astype(float),
        merged[return_column].to_numpy().astype(float),
    )


def _aligned_prob_return_from_frame(
    frame: pl.DataFrame,
    y_prob: np.ndarray,
    *,
    return_column: str = "return_pct",
) -> tuple[np.ndarray, np.ndarray]:
    if return_column not in frame.columns:
        return np.array([]), np.array([])
    ret = frame[return_column].to_numpy().astype(float)
    prob = np.asarray(y_prob, dtype=float)
    if ret.size != prob.size:
        raise ValueError(f"return ({ret.size}) != P(up) ({prob.size})")
    return prob, ret


def _curve_points(curves: list[tuple[SmoothCurveSpec, np.ndarray, np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    probs: list[np.ndarray] = []
    rets: list[np.ndarray] = []
    for _, bx, by in curves:
        probs.append(bx)
        rets.append(by)
    return probs, rets


def _shared_limits(
    curve_probs: list[np.ndarray],
    curve_returns: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    bx_all = np.concatenate([b[np.isfinite(b)] for b in curve_probs if b.size])
    by_all = np.concatenate([c[np.isfinite(c)] for c in curve_returns if c.size])

    if bx_all.size == 0:
        x_lo, x_hi = 0.0, 1.0
    else:
        x_lo, x_hi = float(bx_all.min()), float(bx_all.max())
        x_pad = max((x_hi - x_lo) * AXIS_PAD_RATIO, X_TICK_STEP)
        x_lo = max(0.0, x_lo - x_pad)
        x_hi = min(1.0, x_hi + x_pad)

    if by_all.size == 0:
        y_lo, y_hi = -0.5, 0.5
    else:
        y_lo, y_hi = float(by_all.min()), float(by_all.max())
        y_pad = max((y_hi - y_lo) * AXIS_PAD_RATIO, 1e-3)
        if y_hi - y_lo < 1e-6:
            y_pad = max(abs(y_lo) * AXIS_PAD_RATIO, 0.05)
        y_lo -= y_pad
        y_hi += y_pad
        if y_lo < 0.0 < y_hi:
            y_abs = max(abs(y_lo), abs(y_hi))
            y_lo, y_hi = -y_abs, y_abs

    return (x_lo, x_hi), (y_lo, y_hi)


def _apply_x_ticks(ax: plt.Axes, xlim: tuple[float, float]) -> None:
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_FONTSIZE)


def _draw_threshold_line(ax: plt.Axes, x: float, *, color: str) -> None:
    ax.axvline(x, color=color, linestyle="--", linewidth=1.4, zorder=4)
    ax.text(
        x,
        0.01,
        f"{x:.4f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        fontweight="semibold",
    )


def _plot_prob_vs_return_panel(
    ax: plt.Axes,
    y_prob: np.ndarray,
    return_pct: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    panel_title: str,
    scatter_label: str,
    t_long: float | None = None,
    t_short: float | None = None,
) -> str:
    mask = np.isfinite(return_pct) & np.isfinite(y_prob)
    x = y_prob[mask]
    y = return_pct[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(ylim)
        _apply_x_ticks(ax, xlim)
        ax.set_title(panel_title, fontsize=10, fontweight="semibold", loc="left")
        return "r=—, ρ=—"

    scatter_m = _subsample_mask(x.size, max_points=SCATTER_MAX_POINTS)
    ax.scatter(
        x[scatter_m],
        y[scatter_m],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        color="#64748b",
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )

    curves = _all_smooth_curves(x, y)
    for i, (spec, bx, by) in enumerate(curves):
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        ax.plot(bx, by, color=color, linewidth=1.6, alpha=0.92, zorder=2 + i, label=spec.label)

    ax.set_ylim(ylim)
    _apply_x_ticks(ax, xlim)
    if xlim[0] < 0.5 < xlim[1]:
        ax.axvline(0.5, color="#9ca3af", linestyle="--", linewidth=0.8, zorder=1)
    if ylim[0] < 0.0 < ylim[1]:
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=0.8, zorder=1)
    if ylim[0] < THRESHOLD_EDGE_PCT < ylim[1]:
        ax.axhline(THRESHOLD_EDGE_PCT, color="#93c5fd", linestyle=":", linewidth=0.8, zorder=1)
    if ylim[0] < -THRESHOLD_EDGE_PCT < ylim[1]:
        ax.axhline(-THRESHOLD_EDGE_PCT, color="#fca5a5", linestyle=":", linewidth=0.8, zorder=1)

    if t_long is not None and xlim[0] <= t_long <= xlim[1]:
        _draw_threshold_line(ax, t_long, color=LONG_LINE_COLOR)
    if t_short is not None and xlim[0] <= t_short <= xlim[1]:
        _draw_threshold_line(ax, t_short, color=SHORT_LINE_COLOR)

    ax.set_xlabel("P(up)", fontsize=9)
    ax.set_ylabel("return open→close, %", fontsize=9)
    ax.grid(True, alpha=0.25)
    corr = _correlation_label(x, y)
    ax.set_title(f"{panel_title}\n{corr}", fontsize=10, fontweight="semibold", loc="left")
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
            label=scatter_label,
        ),
        *[
            Line2D([0], [0], color=CURVE_COLORS[i % len(CURVE_COLORS)], linewidth=1.6, label=spec.label)
            for i, (spec, _, _) in enumerate(curves)
        ],
        Line2D([0], [0], color=LONG_LINE_COLOR, linestyle="--", linewidth=1.4, label=f"t_long (train, >{THRESHOLD_EDGE_PCT:g}%)"),
        Line2D([0], [0], color=SHORT_LINE_COLOR, linestyle="--", linewidth=1.4, label=f"t_short (train, <−{THRESHOLD_EDGE_PCT:g}%)"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left", framealpha=0.92, ncol=2)
    return corr


def compute_prob_return_thresholds(
    train_frame: pl.DataFrame,
    train_y_prob: np.ndarray,
    *,
    return_column: str = "return_pct",
) -> dict[str, float | str | int | None]:
    """Пороги t_long/t_short по train: сглаженная кривая return vs P(up)."""
    empty_meta = {
        "edge_pct": THRESHOLD_EDGE_PCT,
        "smooth": THRESHOLD_SMOOTH_SPEC.label,
        "n_train_rows": 0,
        "source": "train prob_return_dependence",
    }
    if return_column not in train_frame.columns or train_frame.is_empty():
        t_long, t_short = resolve_prob_return_threshold_pair(None, None)
        return {"t_long": t_long, "t_short": t_short, **empty_meta}
    train_y_prob = np.asarray(train_y_prob, dtype=float)
    if train_y_prob.size == 0:
        t_long, t_short = resolve_prob_return_threshold_pair(None, None)
        return {"t_long": t_long, "t_short": t_short, **empty_meta}
    train_prob, train_ret = _aligned_prob_return_from_frame(
        train_frame, train_y_prob, return_column=return_column
    )
    t_long, t_short = resolve_prob_return_threshold_pair(
        *_train_thresholds_from_smooth_curve(train_prob, train_ret),
    )
    return {
        "t_long": t_long,
        "t_short": t_short,
        "edge_pct": THRESHOLD_EDGE_PCT,
        "smooth": THRESHOLD_SMOOTH_SPEC.label,
        "n_train_rows": int(train_prob.size),
        "source": "train prob_return_dependence",
    }


def save_prob_return_dependence_plot(
    path: Path,
    *,
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    test_y_prob: np.ndarray,
    train_oos: pl.DataFrame | None = None,
    train_y_prob: np.ndarray | None = None,
    title: str = "return vs P(up)",
    train_label: str = "train",
    test_label: str = "test",
    return_column: str = "return_pct",
) -> tuple[Path, dict[str, float | str | int | None]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    t_fb_long, t_fb_short = resolve_prob_return_threshold_pair(None, None)
    empty_thresholds: dict[str, float | str | int | None] = {
        "t_long": t_fb_long,
        "t_short": t_fb_short,
        "edge_pct": THRESHOLD_EDGE_PCT,
        "smooth": THRESHOLD_SMOOTH_SPEC.label,
        "n_train_rows": 0,
        "source": "train prob_return_dependence",
    }
    if return_column not in test_frame.columns:
        log.warning("[ml] prob_return_dependence: нет колонки %s", return_column)
        return path, empty_thresholds

    if train_y_prob is not None and not train_frame.is_empty():
        train_prob, train_ret = _aligned_prob_return_from_frame(
            train_frame, train_y_prob, return_column=return_column
        )
    elif train_oos is not None and not train_oos.is_empty():
        train_prob, train_ret = _aligned_prob_return_from_oos(
            train_frame, train_oos, return_column=return_column
        )
    else:
        train_prob, train_ret = np.array([]), np.array([])

    test_prob, test_ret = _aligned_prob_return_from_frame(
        test_frame, test_y_prob, return_column=return_column
    )

    train_curves = _all_smooth_curves(train_prob, train_ret) if train_prob.size else []
    test_curves = _all_smooth_curves(test_prob, test_ret)
    train_probs, train_rets = _curve_points(train_curves)
    test_probs, test_rets = _curve_points(test_curves)
    xlim, ylim = _shared_limits(train_probs + test_probs, train_rets + test_rets)

    n_train = int(train_prob.size)
    n_test = int(test_prob.size)
    t_long, t_short = resolve_prob_return_threshold_pair(
        *_train_thresholds_from_smooth_curve(train_prob, train_ret),
    )
    thresholds = {
        "t_long": t_long,
        "t_short": t_short,
        "edge_pct": THRESHOLD_EDGE_PCT,
        "smooth": THRESHOLD_SMOOTH_SPEC.label,
        "n_train_rows": n_train,
        "source": "train prob_return_dependence",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    _plot_prob_vs_return_panel(
        axes[0],
        train_prob,
        train_ret,
        xlim=xlim,
        ylim=ylim,
        panel_title=train_label,
        scatter_label="строки train",
        t_long=t_long,
        t_short=t_short,
    )
    _plot_prob_vs_return_panel(
        axes[1],
        test_prob,
        test_ret,
        xlim=xlim,
        ylim=ylim,
        panel_title=test_label,
        scatter_label="строки test",
        t_long=t_long,
        t_short=t_short,
    )
    fig.suptitle(
        f"{title}\n"
        f"(train n={n_train}, test n={n_test}; "
        f"пороги train {THRESHOLD_SMOOTH_SPEC.label}, |return|>{THRESHOLD_EDGE_PCT:g}%)",
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "[ml] P(up) vs return plot saved: %s (train=%s test=%s t_long=%s t_short=%s edge=%.2f%% smooth=%s)",
        path,
        n_train,
        n_test,
        t_long,
        t_short,
        THRESHOLD_EDGE_PCT,
        THRESHOLD_SMOOTH_SPEC.label,
    )
    return path, thresholds
