"""P(up) density on train: histogram/KDE, valley or Otsu split for classification threshold."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_p_up_density_split_plot")

PLOT_DPI = 200
N_BINS = 128
KDE_SMOOTH_SIGMA = 2.5
VALLEY_SMOOTH_SIGMA = 2.0
VALLEY_MIN_PEAK_FRAC = 0.15
VALLEY_SUPPORT_QUANTILES = (0.02, 0.98)
THRESHOLD_LINE_COLOR = "#7c3aed"
OTSU_LINE_COLOR = "#a78bfa"
REF_05_COLOR = "#9ca3af"


def _gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if y.size < 3 or sigma <= 0:
        return y.copy()
    radius = max(1, int(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def _histogram_density(y_prob: np.ndarray, *, n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip(np.asarray(y_prob, dtype=float)[np.isfinite(y_prob)], 0.0, 1.0)
    if x.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, np.zeros_like(centers)
    counts, edges = np.histogram(x, bins=n_bins, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _otsu_threshold(centers: np.ndarray, density: np.ndarray) -> float:
    prob = np.asarray(density, dtype=float)
    prob = np.maximum(prob, 0.0)
    total = prob.sum()
    if total <= 0 or centers.size == 0:
        return 0.5
    prob /= total
    weight0 = np.cumsum(prob)
    weight1 = 1.0 - weight0
    mean_total = float(np.sum(prob * centers))
    mean0 = np.cumsum(prob * centers) / np.maximum(weight0, 1e-12)
    mean1 = (mean_total - np.cumsum(prob * centers)) / np.maximum(weight1, 1e-12)
    var_between = weight0 * weight1 * (mean0 - mean1) ** 2
    idx = int(np.argmax(var_between))
    return float(centers[idx])


def _local_peaks(y: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
        elif y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append(i)
    if y.size >= 2:
        if y[0] >= y[1]:
            peaks.append(0)
        if y[-1] >= y[-2] and (y.size - 1) not in peaks:
            peaks.append(y.size - 1)
    return sorted(set(peaks))


def _valley_candidate(
    centers: np.ndarray, density: np.ndarray
) -> tuple[float, int, int, np.ndarray] | None:
    if centers.size < 3:
        return None
    smoothed = _gaussian_smooth(np.maximum(density, 0.0), VALLEY_SMOOTH_SIGMA)
    peaks = _local_peaks(smoothed)
    if len(peaks) < 2:
        return None
    peaks = sorted(peaks, key=lambda i: smoothed[i], reverse=True)[:2]
    left, right = sorted(peaks)
    if right <= left:
        return None
    valley_idx = left + int(np.argmin(smoothed[left : right + 1]))
    return float(centers[valley_idx]), left, right, smoothed


def _is_valid_valley(
    valley: float,
    y_prob: np.ndarray,
    *,
    left_peak: int,
    right_peak: int,
    smoothed: np.ndarray,
) -> bool:
    x = y_prob[np.isfinite(y_prob)]
    if x.size == 0:
        return False
    q_lo, q_hi = np.quantile(x, VALLEY_SUPPORT_QUANTILES)
    if valley < float(q_lo) or valley > float(q_hi):
        return False
    peak_max = float(smoothed.max())
    if peak_max <= 0.0:
        return False
    min_peak = peak_max * VALLEY_MIN_PEAK_FRAC
    return float(smoothed[left_peak]) >= min_peak and float(smoothed[right_peak]) >= min_peak


def compute_p_up_split_threshold(y_prob: np.ndarray) -> dict[str, float | str | int | None]:
    """Порог accuracy: провал KDE между двумя пиками; иначе 0.5 или Otsu (только train P(up))."""
    x = np.asarray(y_prob, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "threshold": 0.5,
            "method": "default_0_5",
            "otsu": 0.5,
            "valley": None,
            "n_train_rows": 0,
            "source": "train P(up) density",
        }
    centers, density = _histogram_density(x)
    otsu = _otsu_threshold(centers, density)
    candidate = _valley_candidate(centers, density)
    valley: float | None = None
    if candidate is not None:
        valley_raw, left_peak, right_peak, smoothed = candidate
        if _is_valid_valley(
            valley_raw,
            x,
            left_peak=left_peak,
            right_peak=right_peak,
            smoothed=smoothed,
        ):
            valley = valley_raw
    if valley is not None:
        threshold = valley
        method = "valley"
    elif candidate is not None:
        threshold = 0.5
        method = "default_0_5"
    else:
        threshold = otsu
        method = "otsu"
    return {
        "threshold": float(threshold),
        "method": method,
        "otsu": float(otsu),
        "valley": float(valley) if valley is not None else None,
        "n_train_rows": n,
        "source": "train P(up) density",
    }


def _kde_curve(centers: np.ndarray, density: np.ndarray) -> np.ndarray:
    return _gaussian_smooth(np.maximum(density, 0.0), KDE_SMOOTH_SIGMA)


def _draw_density_panel(
    ax: plt.Axes,
    y_prob: np.ndarray,
    *,
    split: dict[str, float | str | int | None],
    panel_title: str,
    show_fit_legend: bool,
) -> None:
    centers, density = _histogram_density(y_prob)
    kde = _kde_curve(centers, density)
    threshold = float(split["threshold"])
    otsu = float(split["otsu"])
    valley = split.get("valley")
    valley_f = float(valley) if valley is not None else None

    ax.fill_between(centers, density, alpha=0.25, color="#94a3b8", linewidth=0)
    ax.plot(centers, density, color="#64748b", linewidth=0.9, alpha=0.75, label="histogram")
    ax.plot(centers, kde, color="#0f172a", linewidth=2.0, label="KDE (smoothed)")

    ax.axvline(threshold, color=THRESHOLD_LINE_COLOR, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(
        threshold,
        0.98,
        f"split {threshold:.4f}\n({split['method']})",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=THRESHOLD_LINE_COLOR,
        fontweight="semibold",
    )
    if abs(otsu - threshold) > 1e-4:
        ax.axvline(otsu, color=OTSU_LINE_COLOR, linestyle=":", linewidth=1.2, zorder=3)
    if valley_f is not None and abs(valley_f - threshold) > 1e-4:
        ax.axvline(valley_f, color="#c4b5fd", linestyle=":", linewidth=1.0, zorder=3)
    ax.axvline(0.5, color=REF_05_COLOR, linestyle="--", linewidth=0.8, zorder=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("P(up)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_title(panel_title, fontsize=10, fontweight="semibold", loc="left")

    if show_fit_legend:
        handles = [
            Line2D([0], [0], color="#64748b", linewidth=0.9, alpha=0.75, label="histogram"),
            Line2D([0], [0], color="#0f172a", linewidth=2.0, label="KDE (smoothed)"),
            Line2D([0], [0], color=THRESHOLD_LINE_COLOR, linestyle="--", linewidth=1.6, label="split threshold"),
            Line2D([0], [0], color=OTSU_LINE_COLOR, linestyle=":", linewidth=1.2, label="Otsu"),
            Line2D([0], [0], color=REF_05_COLOR, linestyle="--", linewidth=0.8, label="0.5"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.92)


def save_p_up_density_split_plot(
    path: Path,
    *,
    train_y_prob: np.ndarray,
    test_y_prob: np.ndarray | None = None,
    split: dict[str, float | str | int | None] | None = None,
    train_label: str = "train",
    test_label: str = "test",
) -> tuple[Path, dict[str, float | str | int | None]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_info = split if split is not None else compute_p_up_split_threshold(train_y_prob)

    if test_y_prob is not None and test_y_prob.size:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        _draw_density_panel(
            axes[0],
            train_y_prob,
            split=split_info,
            panel_title=f"{train_label}\n(threshold fit here)",
            show_fit_legend=True,
        )
        _draw_density_panel(
            axes[1],
            test_y_prob,
            split=split_info,
            panel_title=f"{test_label}\n(frozen train threshold)",
            show_fit_legend=False,
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        _draw_density_panel(
            ax,
            train_y_prob,
            split=split_info,
            panel_title=train_label,
            show_fit_legend=True,
        )

    method = split_info.get("method", "?")
    threshold = split_info.get("threshold")
    fig.suptitle(
        f"P(up) density split for accuracy (train fit: {method}, threshold={threshold})",
        fontweight="semibold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "[ml] P(up) density split plot saved: %s (method=%s threshold=%s otsu=%s valley=%s)",
        path,
        method,
        threshold,
        split_info.get("otsu"),
        split_info.get("valley"),
    )
    return path, split_info
