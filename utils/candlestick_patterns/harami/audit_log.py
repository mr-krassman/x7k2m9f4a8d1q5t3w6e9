"""Аудит-лог: OHLC свечей t−1 / t и сигнал на день t+1."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from crypto_research.utils.candlestick_patterns.harami.constants import SCENARIO_ROWS
from crypto_research.utils.candlestick_patterns.harami.detection import (
    is_bearish_harami,
    is_bullish_harami,
)
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("harami_audit")

_AUDIT_COLUMNS: tuple[str, ...] = (
    "pair",
    "signal",
    "day_tm1",
    "open_tm1",
    "close_tm1",
    "high_tm1",
    "low_tm1",
    "body_tm1",
    "day_t",
    "open_t",
    "close_t",
    "high_t",
    "low_t",
    "body_t",
    "body_t_inside_tm1",
    "day_signal",
    "open_signal",
    "close_signal",
    "return_pct_signal",
    "price_up",
    "price_down",
)

_AUDIT_HEADER = (
    "=== Harami — аудит сигналов (OHLC паттерна и день t+1) ===",
    "",
    "Столбцы:",
    "  day_tm1 / day_t — свечи паттерна (t−1 и t); day_signal — день оценки доходности (t+1).",
    "  body_* — |close−open|; body_t_inside_tm1 — тело t внутри тела t−1.",
    "  price_up / price_down — return_pct_signal > 0 / < 0 (колонки «Цена росла» / «Цена падала»).",
    "",
)


def _body_size(open_: float, close: float) -> float:
    return abs(close - open_)


def _body_inside(open1: float, close1: float, open2: float, close2: float) -> bool:
    b1_lo = min(open1, close1)
    b1_hi = max(open1, close1)
    b2_lo = min(open2, close2)
    b2_hi = max(open2, close2)
    return b2_lo >= b1_lo and b2_hi <= b1_hi


def _day_str(day) -> str:
    if hasattr(day, "strftime"):
        return day.strftime("%Y-%m-%d")
    return str(day)[:10]


def collect_harami_audit_frame(daily: pl.DataFrame) -> pl.DataFrame:
    if daily.is_empty():
        return pl.DataFrame({col: [] for col in _AUDIT_COLUMNS})

    work = daily
    if "pair" not in work.columns:
        work = work.with_columns(pl.lit("_single").alias("pair"))

    rows: list[dict[str, object]] = []
    for pair in sorted(work["pair"].unique().to_list()):
        sub = work.filter(pl.col("pair") == pair).sort("day_utc")
        n = sub.height
        if n < 3:
            continue
        days = sub["day_utc"].to_list()
        opens = sub["day_open"].to_numpy()
        closes = sub["day_close"].to_numpy()
        highs = sub["day_high"].to_numpy()
        lows = sub["day_low"].to_numpy()
        returns = sub["return_pct"].to_numpy()
        for i in range(1, n - 1):
            o1, c1 = float(opens[i - 1]), float(closes[i - 1])
            o2, c2 = float(opens[i]), float(closes[i])
            if is_bullish_harami(o1, c1, o2, c2):
                signal = SCENARIO_ROWS[0]
            elif is_bearish_harami(o1, c1, o2, c2):
                signal = SCENARIO_ROWS[1]
            else:
                continue
            ret = float(returns[i + 1])
            rows.append({
                "pair": str(pair),
                "signal": signal,
                "day_tm1": _day_str(days[i - 1]),
                "open_tm1": o1,
                "close_tm1": c1,
                "high_tm1": float(highs[i - 1]),
                "low_tm1": float(lows[i - 1]),
                "body_tm1": _body_size(o1, c1),
                "day_t": _day_str(days[i]),
                "open_t": o2,
                "close_t": c2,
                "high_t": float(highs[i]),
                "low_t": float(lows[i]),
                "body_t": _body_size(o2, c2),
                "body_t_inside_tm1": _body_inside(o1, c1, o2, c2),
                "day_signal": _day_str(days[i + 1]),
                "open_signal": float(opens[i + 1]),
                "close_signal": float(closes[i + 1]),
                "return_pct_signal": ret,
                "price_up": int(ret > 0),
                "price_down": int(ret < 0),
            })
    if not rows:
        return pl.DataFrame({col: [] for col in _AUDIT_COLUMNS})
    return pl.DataFrame(rows).select(_AUDIT_COLUMNS)


def format_harami_audit_log(audit: pl.DataFrame) -> str:
    lines = list(_AUDIT_HEADER)
    lines.append(f"Всего строк (событий): {audit.height}")
    lines.append("")
    if audit.is_empty():
        lines.append("(нет событий Harami)")
        return "\n".join(lines)
    lines.append("\t".join(_AUDIT_COLUMNS))
    for row in audit.iter_rows(named=True):
        cells = [
            str(row["pair"]),
            str(row["signal"]),
            str(row["day_tm1"]),
            f"{row['open_tm1']:.8g}",
            f"{row['close_tm1']:.8g}",
            f"{row['high_tm1']:.8g}",
            f"{row['low_tm1']:.8g}",
            f"{row['body_tm1']:.8g}",
            str(row["day_t"]),
            f"{row['open_t']:.8g}",
            f"{row['close_t']:.8g}",
            f"{row['high_t']:.8g}",
            f"{row['low_t']:.8g}",
            f"{row['body_t']:.8g}",
            "1" if row["body_t_inside_tm1"] else "0",
            str(row["day_signal"]),
            f"{row['open_signal']:.8g}",
            f"{row['close_signal']:.8g}",
            f"{row['return_pct_signal']:.6f}",
            str(row["price_up"]),
            str(row["price_down"]),
        ]
        lines.append("\t".join(cells))
    return "\n".join(lines)


def save_harami_audit_log(daily: pl.DataFrame, path: Path) -> Path:
    audit = collect_harami_audit_frame(daily)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = format_harami_audit_log(audit)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[harami] аудит сигналов сохранён: %s (%d строк)", path, audit.height)
    return path
