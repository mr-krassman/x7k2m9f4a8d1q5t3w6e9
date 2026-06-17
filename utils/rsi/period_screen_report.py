"""Отчёт этапа 0: выбор периода RSI по стабильности сигнала."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.utils.ema_spreads.constants import SCREEN_STATS_COLS
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import (
    rsi_period_screen_log_path,
    rsi_period_screen_plot_path,
)
from crypto_research.utils.rsi.constants import (
    DEFAULT_SCREEN_RSI_PERIODS,
    RSI_BUCKET_THRESHOLDS_NOTE,
    SCREEN_MIN_POOLED_DELTA_PP,
)
from crypto_research.utils.rsi.period_screen_plots import (
    save_stability_index_chart,
    save_yearly_agreement_heatmap,
)
from crypto_research.utils.rsi.period_stability import (
    compute_period_stability,
    material_cells_for_period,
    rank_period_stabilities,
)
from crypto_research.utils.weekday.bands import MeanBands

log = get_logger("rsi_period_screen")
_PAIRS_PER_LINE = 11

_METHOD = (
    f"Этап 0 — выбор периода RSI. Колонки: «{SCREEN_STATS_COLS[0]}» и "
    f"«{SCREEN_STATS_COLS[1]}» (Δ к BASE, п.п.). Значимая ячейка = бакет×колонка "
    f"с |Δ| ≥ {SCREEN_MIN_POOLED_DELTA_PP:g} п.п. Индекс (равный вес 25%): "
    f"ср.|Δ| (норм. к макс.), ср.кварталы %, ср.пары %, число значимых ячеек (норм. к макс.)."
)


def _format_pairs(pairs: list[str]) -> list[str]:
    sorted_pairs = sorted(pairs)
    if not sorted_pairs:
        return ["Пары: —"]
    out: list[str] = []
    for i in range(0, len(sorted_pairs), _PAIRS_PER_LINE):
        chunk = sorted_pairs[i : i + _PAIRS_PER_LINE]
        prefix = "Пары: " if i == 0 else "      "
        out.append(f"{prefix}{', '.join(chunk)}")
    return out


def _summary_table_lines(ranked: list) -> list[str]:
    col_rank, col_rsi, col_delta = 4, 5, 8
    col_quarters, col_pairs, col_cells, col_idx = 11, 9, 6, 8
    header = (
        f"{'#':>{col_rank}} | {'RSI':>{col_rsi}} | {'ср.|Δ|':>{col_delta}} | "
        f"{'Ср.кварт.':>{col_quarters}} | {'Ср.пары':>{col_pairs}} | "
        f"{'Знач.':>{col_cells}} | {'Индекс%':>{col_idx}}"
    )
    sep = "-" * len(header)
    lines = [
        "=== Сводка: стабильность по периодам RSI ===",
        _METHOD,
        "",
        header,
        sep,
    ]
    for m in ranked:
        lines.append(
            f"{m.rank:>{col_rank}} | {m.period:>{col_rsi}} | "
            f"{m.avg_abs_delta_label:>{col_delta}} | "
            f"{m.avg_quarters_label:>{col_quarters}} | {m.avg_pairs_label:>{col_pairs}} | "
            f"{m.significant_cell_count:>{col_cells}} | "
            f"{m.stability_index_pct:>{col_idx}.1f}"
        )
    lines.extend(
        [
            "",
            "Знач. — число значимых ячеек (бакет × колонка с |Δ| ≥ порога).",
            "",
            (
                f"Рекомендация этапа 0: RSI({ranked[0].period}) — индекс "
                f"{ranked[0].stability_index_pct:.1f}%, ср.|Δ| {ranked[0].avg_abs_delta_label} п.п., "
                f"ср.кварталы {ranked[0].avg_quarters_label}, ср.пары {ranked[0].avg_pairs_label}, "
                f"значимых ячеек {ranked[0].significant_cell_count}."
            ),
            "Дальше: исследования с выбранным периодом RSI (README).",
            "",
        ]
    )
    return lines


def _material_detail_lines(
    ranked: list,
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
) -> list[str]:
    lines = [
        "=== Значимые ячейки: «Цена росла» / «Цена падала» (|Δ| ≥ порога) ===",
        RSI_BUCKET_THRESHOLDS_NOTE,
        "Формат: бакет | колонка | Δ п.п. | кварталы X/Y | пары N/M",
        "",
    ]
    for m in ranked:
        cells, labels = material_cells_for_period(daily, m.period, pair_bands)
        lines.append(f"--- RSI({m.period}) ---")
        for cell in sorted(cells, key=lambda c: (c.bucket, c.column)):
            pairs_label = (
                "n/a"
                if not cell.pairs_eligible
                else f"{cell.pairs_match}/{cell.pairs_eligible}"
            )
            q_label = (
                "n/a"
                if cell.quarters_total == 0
                else f"{cell.quarters_match}/{cell.quarters_total}"
            )
            row_label = labels[cell.bucket] if cell.bucket < len(labels) else f"b{cell.bucket}"
            lines.append(
                f"  b{cell.bucket} {row_label} | {cell.column} | {cell.delta_pp:+.1f} п.п. | "
                f"кварталы {q_label} | пары {pairs_label}"
            )
        lines.append("")
    return lines


def run_rsi_period_screen_report(
    daily: pl.DataFrame,
    pairs: list[str],
    pair_bands: dict[str, MeanBands],
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    max_pair_start: datetime | None = None,
) -> Path:
    screen_periods = periods or DEFAULT_SCREEN_RSI_PERIODS
    raw_metrics = []
    for period in screen_periods:
        m = compute_period_stability(daily, period, pair_bands)
        if m is not None:
            raw_metrics.append(m)
    if not raw_metrics:
        raise RuntimeError("Недостаточно данных для скрининга периодов RSI")

    ranked = rank_period_stabilities(raw_metrics)

    header = [
        "=== Отчёт: rsi_period_screen (этап 0 — выбор периода RSI) ===",
        f"Пар: {len(pairs)}",
        f"Период скрининга (UTC, train): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
        f"Кандидаты RSI: {', '.join(str(p) for p in screen_periods)}",
    ]
    if max_pair_start is not None:
        header.append(
            f"Фильтр пар: первая свеча не позже {max_pair_start:%Y-%m-%d} (UTC)"
        )
    header.extend(_format_pairs(pairs))
    header.append("")

    body = _summary_table_lines(ranked)
    body.extend(_material_detail_lines(ranked, daily, pair_bands))

    plot_index = rsi_period_screen_plot_path(
        len(pairs), from_date, to_date, screen_periods, "stability_index"
    )
    plot_heat = rsi_period_screen_plot_path(
        len(pairs), from_date, to_date, screen_periods, "yearly_agreement"
    )
    save_stability_index_chart(
        ranked, plot_index, n_pairs=len(pairs), from_date=from_date, to_date=to_date
    )
    save_yearly_agreement_heatmap(
        daily,
        screen_periods,
        pair_bands,
        plot_heat,
        n_pairs=len(pairs),
        from_date=from_date,
        to_date=to_date,
    )

    body.append("=== Графики ===")
    body.append(f"Индекс стабильности: {plot_index}")
    body.append(f"Ср. согласие материальных ячеек по годам: {plot_heat}")
    body.append("")

    path = rsi_period_screen_log_path(len(pairs), from_date, to_date, screen_periods)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(header + body)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("Отчёт выбора периода RSI: %s", path)
    return path
