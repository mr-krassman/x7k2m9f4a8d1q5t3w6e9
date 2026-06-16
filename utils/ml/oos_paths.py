"""Сбор OOS-предсказаний CPCV и график вероятности во времени."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.plotting.date_axis import format_date_axis
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_oos_paths")

PLOT_DPI = 200
FIG_W = 16.0
FIG_H = 2.2
Y_PROB_MIN = 0.4
Y_PROB_MAX = 0.6
N_WEEKDAYS = 7
CALIBRATION_BINS = 10
CALIBRATION_FIG_W = 16.0
CALIBRATION_FIG_H = 8.0
CALIBRATION_NCOLS = 4
CALIBRATION_NROWS = 2
PROB_NCOLS = 2


def _weekday_from_day_utc() -> pl.Expr:
    """0=Пн … 6=Вс (как в dataset.weekday)."""
    return ((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday")


def _daily_mean_y_prob(oos: pl.DataFrame) -> pl.DataFrame:
    """Средняя P(up) по всем парам в каждый календарный день + метка weekday."""
    return (
        oos.with_columns(_weekday_from_day_utc())
        .group_by("day_utc", "weekday")
        .agg(pl.col("y_prob").mean().alias("y_prob"))
        .sort("day_utc")
    )


def frame_rows_at(frame: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    if indices.size == 0:
        return frame.clear()
    return (
        frame.with_row_index("__row")
        .filter(pl.col("__row").is_in(indices.tolist()))
        .drop("__row")
    )


def collect_fold_predictions(
    frame: pl.DataFrame,
    test_idx: np.ndarray,
    y_prob: np.ndarray,
    y_test: np.ndarray,
    fold_idx: int,
) -> pl.DataFrame:
    return frame_rows_at(frame, test_idx).select("day_utc", "pair", "direction_up").rename(
        {"direction_up": "y_true"}
    ).with_columns(
        pl.Series("y_prob", y_prob),
        pl.lit(fold_idx).alias("fold"),
    )


def build_oos_predictions(parts: list[pl.DataFrame]) -> pl.DataFrame:
    if not parts:
        return pl.DataFrame(
            schema={
                "day_utc": pl.Datetime("us", "UTC"),
                "pair": pl.Utf8,
                "y_prob": pl.Float64,
                "y_true": pl.Int8,
                "n_folds": pl.UInt32,
            }
        )
    merged = pl.concat(parts, how="vertical")
    return (
        merged.group_by("day_utc", "pair")
        .agg(
            pl.col("y_prob").mean().alias("y_prob"),
            pl.col("y_true").first().alias("y_true"),
            pl.col("fold").len().alias("n_folds"),
        )
        .sort("day_utc", "pair")
    )


def oos_paths_array(oos: pl.DataFrame) -> np.ndarray:
    """(n_oos, 3): time_ns, y_prob, y_true — отсортировано по day_utc, pair."""
    if oos.is_empty():
        return np.empty((0, 3), dtype=np.float64)
    times = oos["day_utc"].to_numpy().astype("datetime64[ns]").view(np.int64)
    return np.column_stack(
        [
            times.astype(np.float64),
            oos["y_prob"].to_numpy(),
            oos["y_true"].to_numpy().astype(np.float64),
        ]
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = CALIBRATION_BINS,
) -> float:
    """ECE: взвешенное |P(y=1) − mean(prob)| по равномерным бинам [0, 1]."""
    if y_true.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        weight = mask.mean()
        ece += weight * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def oos_calibration_metrics(
    oos: pl.DataFrame,
    *,
    n_bins: int = CALIBRATION_BINS,
) -> dict[str, float | dict[str, dict[str, float]]]:
    """Brier, ECE и метрики по weekday для OOS-предсказаний."""
    if oos.is_empty():
        return {"brier_score": float("nan"), "ece": float("nan"), "n_bins": n_bins, "by_weekday": {}}

    y_true = oos["y_true"].to_numpy()
    y_prob = oos["y_prob"].to_numpy()
    overall = {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "n_bins": n_bins,
    }
    by_weekday: dict[str, dict[str, float]] = {}
    oos_wd = oos.with_columns(_weekday_from_day_utc())
    for wd, name in enumerate(WEEKDAY_NAMES):
        sub = oos_wd.filter(pl.col("weekday") == wd)
        if sub.is_empty():
            continue
        yt = sub["y_true"].to_numpy()
        yp = sub["y_prob"].to_numpy()
        by_weekday[name] = {
            "brier_score": float(brier_score_loss(yt, yp)),
            "ece": expected_calibration_error(yt, yp, n_bins=n_bins),
            "n_test": int(yt.size),
        }
    return {**overall, "by_weekday": by_weekday}


def _plot_calibration_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    title: str,
    n_bins: int,
) -> None:
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#9ca3af", linewidth=1.0, label="идеал")
    if y_true.size == 0 or np.unique(y_true).size < 2:
        ax.text(0.5, 0.5, "нет данных", transform=ax.transAxes, ha="center", va="center")
    else:
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy="uniform",
        )
        ax.plot(mean_pred, frac_pos, marker="o", color="#2563eb", linewidth=1.2, markersize=4)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins) if y_true.size else float("nan")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_title(f"{title} (ECE={ece:.3f})", fontsize=10, fontweight="semibold", loc="left")


def save_oos_calibration_plot(
    oos: pl.DataFrame,
    path: Path,
    *,
    n_bins: int = CALIBRATION_BINS,
) -> Path:
    """Reliability curve: pooled OOS + отдельно по каждому weekday."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if oos.is_empty():
        log.warning("[ml] OOS calibration plot skipped: нет предсказаний")
        return path

    oos_wd = oos.with_columns(_weekday_from_day_utc())
    y_true_all = oos["y_true"].to_numpy()
    y_prob_all = oos["y_prob"].to_numpy()

    fig, axes = plt.subplots(
        CALIBRATION_NROWS,
        CALIBRATION_NCOLS,
        figsize=(CALIBRATION_FIG_W, CALIBRATION_FIG_H),
    )
    axes_flat = axes.flatten()
    panels: list[tuple[plt.Axes, np.ndarray, np.ndarray, str]] = [
        (axes_flat[0], y_true_all, y_prob_all, "Все OOS"),
    ]
    for wd, name in enumerate(WEEKDAY_NAMES):
        sub = oos_wd.filter(pl.col("weekday") == wd)
        panels.append(
            (
                axes_flat[wd + 1],
                sub["y_true"].to_numpy(),
                sub["y_prob"].to_numpy(),
                name,
            )
        )

    for ax, y_true, y_prob, title in panels:
        _plot_calibration_panel(ax, y_true, y_prob, title=title, n_bins=n_bins)
        ax.set_xlabel("mean P(up)")
        ax.set_ylabel("доля y=1")

    fig.suptitle(
        "CPCV OOS: калибровочная кривая (reliability)",
        fontsize=12,
        fontweight="semibold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    metrics = oos_calibration_metrics(oos, n_bins=n_bins)
    log.info(
        "[ml] OOS calibration plot saved: %s (brier=%.4f ece=%.4f rows=%s)",
        path,
        metrics["brier_score"],
        metrics["ece"],
        oos.height,
    )
    return path


def save_oos_probability_plot(oos: pl.DataFrame, path: Path) -> Path:
    """7 подграфиков (Пн–Вс): OOS P(direction_up) во времени, Y ∈ [0.4, 0.6]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if oos.is_empty():
        log.warning("[ml] OOS plot skipped: нет предсказаний")
        return path

    daily = _daily_mean_y_prob(oos)
    n_rows = (N_WEEKDAYS + PROB_NCOLS - 1) // PROB_NCOLS

    fig, axes = plt.subplots(
        n_rows,
        PROB_NCOLS,
        figsize=(FIG_W, FIG_H * n_rows),
        sharex=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for wd, ax in enumerate(axes_flat[:N_WEEKDAYS]):
        sub = daily.filter(pl.col("weekday") == wd)
        ax.axhline(0.5, color="#9ca3af", linewidth=1.0, linestyle="--")
        ax.set_ylim(Y_PROB_MIN, Y_PROB_MAX)
        ax.set_ylabel("P(up)", fontsize=9)
        ax.set_title(WEEKDAY_NAMES[wd], loc="left", fontsize=11, fontweight="semibold")
        if sub.is_empty():
            ax.text(0.5, 0.5, "нет данных", transform=ax.transAxes, ha="center", va="center")
            continue
        x = pd.to_datetime(sub["day_utc"].to_pandas(), utc=True).to_numpy()
        y = sub["y_prob"].to_numpy()
        ax.plot(x, y, color="#2563eb", linewidth=1.0, alpha=0.9)

    for ax in axes_flat[N_WEEKDAYS:]:
        ax.set_visible(False)

    for col in range(PROB_NCOLS):
        ax = axes_flat[(n_rows - 1) * PROB_NCOLS + col]
        if ax.get_visible():
            ax.set_xlabel("UTC")
            format_date_axis(ax, rotate=30)
    fig.suptitle(
        "CPCV OOS: средняя P(direction_up) по дням (отдельно для каждого weekday)",
        fontsize=12,
        fontweight="semibold",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info(
        "[ml] OOS plot saved: %s (weekdays=%s calendar_days=%s rows=%s)",
        path,
        N_WEEKDAYS,
        daily.height,
        oos.height,
    )
    return path


ROC_AUC_FIG_W = 7.0
ROC_AUC_FIG_H = 6.0


def save_roc_auc_comparison_plot(
    train_y_true: np.ndarray,
    train_y_prob: np.ndarray,
    val_y_true: np.ndarray,
    val_y_prob: np.ndarray,
    path: Path,
    *,
    train_label: str = "Pooled OOS (train)",
    val_label: str = "Val holdout",
) -> Path:
    """ROC-кривые pooled train OOS и val holdout на одном графике."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(ROC_AUC_FIG_W, ROC_AUC_FIG_H))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.0,
        label="случайный (AUC=0.5)",
    )

    series = (
        (train_y_true, train_y_prob, "#2563eb", train_label),
        (val_y_true, val_y_prob, "#ea580c", val_label),
    )
    for y_true, y_prob, color, label in series:
        if y_true.size == 0 or np.unique(y_true).size < 2:
            log.warning("[ml] ROC AUC plot skipped series %s: один класс", label)
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{label} (AUC={auc:.3f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        "ROC curve: Pooled OOS (train) vs Val holdout",
        fontsize=12,
        fontweight="semibold",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info("[ml] ROC AUC comparison plot saved: %s", path)
    return path


def save_weekday_pair_summary_plot(
    oos: pl.DataFrame,
    path: Path,
    *,
    title: str,
) -> Path:
    """Сводное полотно heatmap: weekday × pair для ключевых метрик."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if oos.is_empty():
        log.warning("[ml] Weekday×pair summary plot skipped: нет предсказаний")
        return path

    df = oos.with_columns(_weekday_from_day_utc())
    grouped = df.group_by("weekday", "pair").agg(
        pl.col("y_true").mean().alias("base_rate_up"),
        pl.col("y_prob").mean().alias("mean_p_up"),
        (((pl.col("y_prob") >= 0.5).cast(pl.Int8) == pl.col("y_true")).cast(pl.Float64).mean()).alias("accuracy"),
        pl.len().alias("n_obs"),
    )
    grouped = grouped.with_columns((pl.col("mean_p_up") - pl.col("base_rate_up")).alias("edge_p_up"))

    pairs = sorted(grouped["pair"].unique().to_list())
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    n_pairs = len(pairs)
    n_weekdays = N_WEEKDAYS

    metric_names = ("mean_p_up", "base_rate_up", "accuracy", "edge_p_up")
    vmins = {"mean_p_up": 0.4, "base_rate_up": 0.4, "accuracy": 0.4, "edge_p_up": -0.1}
    vmaxs = {"mean_p_up": 0.6, "base_rate_up": 0.6, "accuracy": 0.6, "edge_p_up": 0.1}
    cmaps = {"mean_p_up": "viridis", "base_rate_up": "viridis", "accuracy": "viridis", "edge_p_up": "coolwarm"}

    mats = {name: np.full((n_weekdays, n_pairs), np.nan, dtype=float) for name in metric_names}
    for row in grouped.iter_rows(named=True):
        wd = int(row["weekday"])
        pi = pair_to_idx[str(row["pair"])]
        for name in metric_names:
            mats[name][wd, pi] = float(row[name])

    fig, axes = plt.subplots(2, 2, figsize=(max(18, n_pairs * 0.33), 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, metric in zip(axes_flat, metric_names):
        im = ax.imshow(
            mats[metric],
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmaps[metric],
            vmin=vmins[metric],
            vmax=vmaxs[metric],
        )
        ax.set_title(metric, fontsize=10, loc="left", fontweight="semibold")
        ax.set_yticks(range(n_weekdays))
        ax.set_yticklabels(WEEKDAY_NAMES, fontsize=8)
        ax.set_xticks(range(n_pairs))
        ax.set_xticklabels(pairs, rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

    fig.suptitle(title, fontsize=12, fontweight="semibold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info("[ml] Weekday×pair summary plot saved: %s (pairs=%s rows=%s)", path, n_pairs, oos.height)
    return path
