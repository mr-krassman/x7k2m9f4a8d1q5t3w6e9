"""Отдельный отчёт: итог train → val по дням недели."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import (
    TRAIN_MAX_PAIR_START,
    VAL_MAX_PAIR_START,
    weekday_train_val_summary_log_path,
)
from crypto_research.utils.pipeline.weekday_effects import format_pairs_lines
from crypto_research.utils.pipeline.weekday_report import WeekdayReportContext
from crypto_research.utils.weekday.summary import compute_weekday_summary, format_summary_table

log = get_logger("weekday_train_val_summary")


def _format_pairs_block(label: str, pairs: list[str]) -> list[str]:
    lines = format_pairs_lines(pairs)
    if not lines:
        return [f"{label}: —"]
    lines[0] = lines[0].replace("Пары: ", f"{label}: ", 1)
    return lines


def build_summary_report_header(
    train_pairs: list[str],
    val_pairs: list[str],
    from_date: datetime,
    to_date: datetime,
) -> list[str]:
    lines = [
        "=== Отчёт: train → val (дни недели) ===",
        f"Train пар: {len(train_pairs)} (первая свеча ≤ {TRAIN_MAX_PAIR_START} UTC)",
        f"Val пар: {len(val_pairs)} (пул ≤ {VAL_MAX_PAIR_START} UTC, вне train-cohort)",
        f"Период теста (UTC): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
        "",
        "Train (discovery): средний return (open→close) и p-value по train-парам.",
        "Val (confirmation): знак по val-парам и pooled эффект (val) vs train.",
        "",
    ]
    lines.extend(_format_pairs_block("Train", train_pairs))
    lines.append("")
    lines.extend(_format_pairs_block("Val", val_pairs))
    lines.append("")
    return lines


def assemble_train_val_summary_report(
    train_pairs: list[str],
    val_pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
) -> str:
    rows = compute_weekday_summary(train_daily, val_daily)
    parts = (
        build_summary_report_header(train_pairs, val_pairs, from_date, to_date)
        + format_summary_table(rows)
    )
    return "\n".join(parts)


def save_train_val_summary_report(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[3] Итог train→val сохранён: %s", path)
    return path


def run_train_val_summary_report(
    ctx: WeekdayReportContext,
    train_daily: pl.DataFrame,
    val_daily: pl.DataFrame,
    train_pairs: list[str],
    val_pairs: list[str],
) -> Path:
    text = assemble_train_val_summary_report(
        train_pairs,
        val_pairs,
        ctx.from_date,
        ctx.to_date,
        train_daily,
        val_daily,
    )
    path = weekday_train_val_summary_log_path(
        len(train_pairs),
        len(val_pairs),
        ctx.from_date,
        ctx.to_date,
    )
    return save_train_val_summary_report(text, path)
