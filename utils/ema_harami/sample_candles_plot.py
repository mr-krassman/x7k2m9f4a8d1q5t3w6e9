"""Полотно: 20 случайных Bullish + 20 Bearish EMA×Harami вокруг дня сигнала t+1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec

from crypto_research.utils.candlestick_patterns.harami.constants import BUCKET_BEARISH, BUCKET_BULLISH
from crypto_research.utils.candlestick_patterns.harami.detection import confirmed_harami_on_signal_day
from crypto_research.utils.ema_harami.constants import (
    BEARISH_HARAMI_LABEL,
    BULLISH_HARAMI_LABEL,
    EMA_HARAMI_HARAMI_BY_BUCKET,
)
from crypto_research.utils.ema_spreads.ema import (
    assign_ema_dev_buckets_vectorized,
    attach_ema_columns,
    build_ema_work_frame,
    build_pair_thresholds_frame,
    ema_dev_prev_column,
    ema_period_column,
)
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ema_harami_sample_candles")

PLOT_DPI = 110
HALF_WINDOW_DAYS = 10
N_PER_SIDE = 20
N_COLS = 5
SECTION_ROWS = 4
HEADER_ROW_RATIO = 0.12
RANDOM_SEED = 42

COLOR_UP = "#26a641"
COLOR_DOWN = "#ef5350"
COLOR_EMA = "#d97706"
COLOR_SIGNAL = "#2563eb"
COLOR_PATTERN = "#7c3aed"


@dataclass(frozen=True)
class EmaHaramiEvent:
    pair: str
    kind: str
    signal_idx: int
    ema_bucket: int
    day_signal: str


def _harami_kind(opens: np.ndarray, closes: np.ndarray, j: int) -> int | None:
    return confirmed_harami_on_signal_day(opens, closes, j)


def collect_ema_harami_events(daily: pl.DataFrame, period: int) -> tuple[list[EmaHaramiEvent], list[EmaHaramiEvent]]:
    work = attach_ema_columns(build_ema_work_frame(daily, (period,)), (period,))
    prev_col = ema_dev_prev_column(period)
    if work.is_empty() or prev_col not in work.columns:
        return [], []
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    thresholds = build_pair_thresholds_frame(work, prev_col)
    if thresholds.is_empty():
        return [], []

    bullish: list[EmaHaramiEvent] = []
    bearish: list[EmaHaramiEvent] = []

    for pair in work["pair"].unique().to_list():
        sub = work.filter(pl.col("pair") == pair).sort("day_utc")
        th = thresholds.filter(pl.col("pair") == pair)
        if th.is_empty():
            continue
        t = th.row(0, named=True)
        n = sub.height
        if n < HALF_WINDOW_DAYS * 2 + 1:
            continue

        dev = sub[prev_col].to_numpy().astype(np.float64, copy=False)
        ema_buckets = assign_ema_dev_buckets_vectorized(
            dev,
            np.full(n, float(t["t1_up"])),
            np.full(n, float(t["t2_up"])),
            np.full(n, float(t["t1_down"])),
            np.full(n, float(t["t2_down"])),
            np.full(n, float(t["near_abs"])),
        )
        opens = sub["day_open"].to_numpy().astype(np.float64, copy=False)
        closes = sub["day_close"].to_numpy().astype(np.float64, copy=False)
        days = sub["day_utc"].to_list()

        for j in range(HALF_WINDOW_DAYS, n - HALF_WINDOW_DAYS):
            ema_b = int(ema_buckets[j])
            required = EMA_HARAMI_HARAMI_BY_BUCKET.get(ema_b)
            if required is None:
                continue
            harami = _harami_kind(opens, closes, j)
            if harami != required:
                continue
            day_str = days[j].strftime("%Y-%m-%d") if hasattr(days[j], "strftime") else str(days[j])[:10]
            event = EmaHaramiEvent(
                pair=str(pair),
                kind=BULLISH_HARAMI_LABEL if harami == BUCKET_BULLISH else BEARISH_HARAMI_LABEL,
                signal_idx=j,
                ema_bucket=ema_b,
                day_signal=day_str,
            )
            if harami == BUCKET_BULLISH:
                bullish.append(event)
            else:
                bearish.append(event)

    return bullish, bearish


def _sample_events(events: list[EmaHaramiEvent], n: int, rng: np.random.Generator) -> list[EmaHaramiEvent]:
    if not events:
        return []
    if len(events) <= n:
        return list(events)
    idx = rng.choice(len(events), size=n, replace=False)
    return [events[int(i)] for i in sorted(idx)]


def _ema_values(closes: np.ndarray, period: int) -> np.ndarray:
    if closes.size == 0:
        return closes
    alpha = 2.0 / (period + 1.0)
    out = np.empty(closes.size, dtype=np.float64)
    out[0] = closes[0]
    for i in range(1, closes.size):
        out[i] = alpha * closes[i] + (1.0 - alpha) * out[i - 1]
    return out


def _draw_candles(
    ax: plt.Axes,
    xs: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> None:
    width = 0.65
    for x, o, h, l, c in zip(xs, opens, highs, lows, closes):
        color = COLOR_UP if c >= o else COLOR_DOWN
        ax.plot([x, x], [l, h], color=color, linewidth=0.7, solid_capstyle="round")
        body_lo = min(o, c)
        body_hi = max(o, c)
        height = body_hi - body_lo
        if height <= 0.0:
            height = max((h - l) * 0.04, abs(o) * 1e-6, 1e-8)
            body_lo = o - height / 2.0
        ax.bar(x, height, bottom=body_lo, width=width, color=color, edgecolor=color, linewidth=0.4)


def _plot_event_panel(
    ax: plt.Axes,
    pair_daily: pl.DataFrame,
    event: EmaHaramiEvent,
    period: int,
) -> None:
    sub = pair_daily
    j = event.signal_idx
    lo = j - HALF_WINDOW_DAYS
    hi = j + HALF_WINDOW_DAYS
    win = sub.slice(lo, hi - lo + 1)
    xs = np.arange(-HALF_WINDOW_DAYS, HALF_WINDOW_DAYS + 1, dtype=np.float64)
    opens = win["day_open"].to_numpy().astype(np.float64, copy=False)
    highs = win["day_high"].to_numpy().astype(np.float64, copy=False)
    lows = win["day_low"].to_numpy().astype(np.float64, copy=False)
    closes = win["day_close"].to_numpy().astype(np.float64, copy=False)

    _draw_candles(ax, xs, opens, highs, lows, closes)
    ema_col = ema_period_column(period)
    if ema_col in sub.columns:
        ema = sub[ema_col].to_numpy().astype(np.float64, copy=False)[lo : hi + 1]
    else:
        ema = _ema_values(sub["day_close"].to_numpy().astype(np.float64, copy=False), period)[
            lo : hi + 1
        ]
    ax.plot(xs, ema, color=COLOR_EMA, linewidth=1.0, label=f"EMA{period}")

    ax.axvline(0.0, color=COLOR_SIGNAL, linewidth=0.9, linestyle="--", alpha=0.85)
    ax.axvline(-1.0, color="#059669", linewidth=0.75, linestyle=":", alpha=0.8)
    ax.axvline(-2.0, color=COLOR_PATTERN, linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axvline(-3.0, color=COLOR_PATTERN, linewidth=0.7, linestyle=":", alpha=0.7)

    ax.set_xlim(-HALF_WINDOW_DAYS - 0.5, HALF_WINDOW_DAYS + 0.5)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["-10", "-5", "t+1", "+5", "+10"], fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_title(
        f"{event.pair} · b{event.ema_bucket} · {event.day_signal}",
        fontsize=6.5,
        pad=2,
    )


def _fill_section(
    fig: plt.Figure,
    gs: GridSpec,
    row_offset: int,
    events: list[EmaHaramiEvent],
    pair_frames: dict[str, pl.DataFrame],
    period: int,
) -> None:
    for idx, event in enumerate(events):
        row, col = divmod(idx, N_COLS)
        ax = fig.add_subplot(gs[row_offset + row, col])
        pair_sub = pair_frames.get(event.pair)
        if pair_sub is None:
            ax.axis("off")
            continue
        _plot_event_panel(ax, pair_sub, event, period)
        if col == 0:
            ax.set_ylabel("price", fontsize=7)

    for idx in range(len(events), SECTION_ROWS * N_COLS):
        row, col = divmod(idx, N_COLS)
        fig.add_subplot(gs[row_offset + row, col]).axis("off")


def _section_banner(fig: plt.Figure, gs: GridSpec, row: int, text: str) -> None:
    ax = fig.add_subplot(gs[row, :])
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )


def save_ema_harami_sample_candles_plot(
    daily: pl.DataFrame,
    path: Path,
    *,
    period: int,
    n_per_side: int = N_PER_SIDE,
    seed: int = RANDOM_SEED,
) -> Path | None:
    bullish, bearish = collect_ema_harami_events(daily, period)
    rng = np.random.default_rng(seed)
    picked_bull = _sample_events(bullish, n_per_side, rng)
    picked_bear = _sample_events(bearish, n_per_side, rng)
    if not picked_bull and not picked_bear:
        log.warning("[ema_harami] нет событий для свечного полотна: %s", path)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = attach_ema_columns(build_ema_work_frame(daily, (period,)), (period,))
    pair_frames = {
        str(p): enriched.filter(pl.col("pair") == p).sort("day_utc")
        for p in enriched["pair"].unique().to_list()
    }

    fig_h = SECTION_ROWS * 2.35 * 2 + 1.8
    fig = plt.figure(figsize=(N_COLS * 3.0, fig_h), dpi=PLOT_DPI)
    gs = GridSpec(
        SECTION_ROWS * 2 + 1,
        N_COLS,
        figure=fig,
        height_ratios=(
            [1.0] * SECTION_ROWS
            + [HEADER_ROW_RATIO]
            + [1.0] * SECTION_ROWS
        ),
        hspace=0.55,
        wspace=0.35,
    )

    _fill_section(fig, gs, 0, picked_bull, pair_frames, period)
    _section_banner(
        fig,
        gs,
        SECTION_ROWS,
        "Bullish Harami (подтв.) — 20 случайных (ниже EMA, b4–b6)  |  "
        "Bearish Harami (подтв.) — 20 случайных (выше EMA, b0–b2)",
    )
    _fill_section(fig, gs, SECTION_ROWS + 1, picked_bear, pair_frames, period)

    fig.suptitle(
        f"EMA({period}) + Harami (подтв.) — свечи ±{HALF_WINDOW_DAYS} дней вокруг сигнала t+1\n"
        f"верх: Bullish (подтв.) · низ: Bearish (подтв.) · "
        f"t+1=сигнал, t=подтв., t−2/t−1=паттерн",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(top=0.94, bottom=0.04)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "[ema_harami] свечное полотно: %s (bull=%d bear=%d из %d/%d)",
        path,
        len(picked_bull),
        len(picked_bear),
        len(bullish),
        len(bearish),
    )
    return path
