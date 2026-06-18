"""Графики сравнения holdout P(up) между frozen ML-моделями (общий масштаб осей)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.ml.model_compare import CompareModelEntry, align_probabilities, signal_rates
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_model_compare_plot")

PLOT_DPI = 200
HIST_BINS = 40
CALIBRATION_BINS = 10
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _prob_arrays(entries: list[CompareModelEntry]) -> list[np.ndarray]:
    return [e.oos["y_prob"].to_numpy() for e in entries]


def _shared_xlim(arrays: list[np.ndarray], *, pad: float = 0.02) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(0.0, mid - 0.15)
        hi = min(1.0, mid + 0.15)
    return lo, hi


def _hist_max_density(arrays: list[np.ndarray], bins: np.ndarray) -> float:
    peak = 0.0
    for arr in arrays:
        if arr.size == 0:
            continue
        counts, _ = np.histogram(arr[np.isfinite(arr)], bins=bins, density=True)
        peak = max(peak, float(counts.max()) if counts.size else 0.0)
    return peak * 1.08 if peak > 0 else 1.0


def save_prob_histogram_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _prob_arrays(entries)
    xlim = _shared_xlim(arrays)
    bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)
    ymax = _hist_max_density(arrays, bins)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, entry in enumerate(entries):
        arr = entry.oos["y_prob"].to_numpy()
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.45,
            color=_color(i),
            label=entry.model_id,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(
            entry.t_long,
            color=_color(i),
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label=f"{entry.model_id} t_long",
        )
        ax.axvline(
            entry.t_short,
            color=_color(i),
            linestyle=":",
            linewidth=1.0,
            alpha=0.85,
            label=f"{entry.model_id} t_short",
        )
    ax.axvline(0.5, color="#6b7280", linestyle="-.", linewidth=1.0, label="0.5 (pred↑)")
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("P(up)")
    ax.set_ylabel("плотность")
    ax.set_title("Holdout: распределение P(up)", fontweight="semibold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_signal_rates_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["long", "flat", "short"]
    colors_seg = ("#22c55e", "#94a3b8", "#ef4444")
    rates = [signal_rates(e.oos["y_prob"].to_numpy(), e.t_long, e.t_short) for e in entries]
    ymax = max(max(r[k] for r in rates) for k in labels) * 1.12

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(entries)), 5))
    x = np.arange(len(entries))
    width = 0.22
    for j, seg in enumerate(labels):
        vals = [r[seg] * 100 for r in rates]
        ax.bar(x + (j - 1) * width, vals, width=width, label=seg, color=colors_seg[j])
    ax.set_xticks(x)
    ax.set_xticklabels([e.model_id for e in entries], rotation=15, ha="right")
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel("% строк")
    ax.set_title("Holdout: long / flat / short (policy t_long, t_short)", fontweight="semibold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    for i, entry in enumerate(entries):
        ax.text(
            i - width,
            ymax * 0.02,
            f"L={entry.t_long:.3f}\nS={entry.t_short:.3f}",
            fontsize=7,
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_prob_delta_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = entries[0]
    others = entries[1:]
    if not others:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "нужно ≥2 модели", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    deltas: list[np.ndarray] = []
    for other in others:
        p0, p1 = align_probabilities(baseline.oos, other.oos)
        deltas.append(p1 - p0)
    all_delta = np.concatenate(deltas)
    pad = max(0.01, float(np.percentile(np.abs(all_delta), 99)) * 0.1) if all_delta.size else 0.05
    xlim = (-max(pad, float(np.abs(all_delta).max()) * 1.08), max(pad, float(np.abs(all_delta).max()) * 1.08))
    bins = np.linspace(xlim[0], xlim[1], HIST_BINS + 1)
    ymax = _hist_max_density(deltas, bins)

    n = len(others)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, (other, delta) in enumerate(zip(others, deltas)):
        ax = axes[0, i]
        ax.hist(delta, bins=bins, color=_color(i + 1), alpha=0.75, edgecolor="white")
        ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
        ax.set_xlim(xlim)
        ax.set_ylim(0.0, ymax)
        ax.set_title(f"{other.model_id} − {baseline.model_id}", fontsize=10)
        ax.set_xlabel("ΔP(up)")
        if i == 0:
            ax.set_ylabel("число строк")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Holdout: сдвиг вероятностей vs baseline", fontweight="semibold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_calibration_overlay_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1.0, label="идеал")
    for i, entry in enumerate(entries):
        y_true = entry.oos["y_true"].to_numpy()
        y_prob = entry.oos["y_prob"].to_numpy()
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=CALIBRATION_BINS, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker="o", color=_color(i), label=entry.model_id, linewidth=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mean P(up)")
    ax.set_ylabel("доля y=1")
    ax.set_title("Holdout: калибровка (overlay)", fontweight="semibold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_prob_cdf_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _prob_arrays(entries)
    xlim = _shared_xlim(arrays)

    fig, ax = plt.subplots(figsize=(10, 5))
    grid = np.linspace(xlim[0], xlim[1], 200)
    for i, entry in enumerate(entries):
        arr = np.sort(entry.oos["y_prob"].to_numpy())
        if arr.size == 0:
            continue
        cdf = np.searchsorted(arr, grid, side="right") / arr.size
        ax.plot(grid, cdf, color=_color(i), label=entry.model_id, linewidth=1.8)
        ax.axvline(
            entry.t_long,
            color=_color(i),
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            label=f"{entry.model_id} t_long",
        )
        ax.axvline(
            entry.t_short,
            color=_color(i),
            linestyle=":",
            linewidth=0.9,
            alpha=0.7,
            label=f"{entry.model_id} t_short",
        )
    ax.axvline(0.5, color="#6b7280", linestyle="-.", linewidth=1.0, label="0.5 (pred↑)")
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("P(up)")
    ax.set_ylabel("CDF")
    ax.set_title("Holdout: CDF P(up)", fontweight="semibold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_weekday_prob_compare(entries: list[CompareModelEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _prob_arrays(entries)
    ylim = _shared_xlim(arrays)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    axes_flat = axes.flatten()
    positions = np.arange(len(entries))
    width = 0.7 / max(len(entries), 1)

    for wd in range(7):
        ax = axes_flat[wd]
        data_by_model: list[np.ndarray] = []
        for entry in entries:
            sub = entry.oos.filter(pl.col("weekday") == wd)["y_prob"].to_numpy()
            data_by_model.append(sub)
        bp = ax.boxplot(
            data_by_model,
            positions=positions,
            widths=width * len(entries),
            patch_artist=True,
            showfliers=False,
        )
        for patch, i in zip(bp["boxes"], range(len(entries))):
            patch.set_facecolor(_color(i))
            patch.set_alpha(0.55)
        ax.set_title(WEEKDAY_NAMES[wd], loc="left", fontsize=10, fontweight="semibold")
        ax.set_ylim(ylim)
        ax.axhline(0.5, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([e.model_id for e in entries], rotation=30, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)

    axes_flat[7].axis("off")
    fig.suptitle("Holdout: P(up) по дням недели", fontweight="semibold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("[ml] compare plot saved: %s", path)
    return path


def save_all_compare_plots(entries: list[CompareModelEntry], plots_dir: Path) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "prob_histogram": str(save_prob_histogram_compare(entries, plots_dir / "compare_prob_histogram.png")),
        "signal_rates": str(save_signal_rates_compare(entries, plots_dir / "compare_signal_rates.png")),
        "prob_delta": str(save_prob_delta_compare(entries, plots_dir / "compare_prob_delta.png")),
        "calibration_overlay": str(
            save_calibration_overlay_compare(entries, plots_dir / "compare_calibration_overlay.png")
        ),
        "prob_cdf": str(save_prob_cdf_compare(entries, plots_dir / "compare_prob_cdf.png")),
        "weekday_prob": str(save_weekday_prob_compare(entries, plots_dir / "compare_weekday_prob.png")),
    }
    return out
