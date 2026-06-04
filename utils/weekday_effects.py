from datetime import datetime
from pathlib import Path

import polars as pl

from crypto_research.stats.day_return_stats import MeanBands, build_weekday_table
from crypto_research.stats.year_repeatability import MIN_YEAR_BASE_DAYS, MIN_YEAR_ROW_DAYS
from crypto_research.utils.logger import get_logger
from crypto_research.utils.paths import weekday_plot_path, weekday_stats_log_path

log = get_logger("weekday_effects")
_PAIRS_PER_LINE = 11


def format_pairs_lines(pairs: list[str], per_line: int = _PAIRS_PER_LINE) -> list[str]:
    sorted_pairs = sorted(pairs)
    if not sorted_pairs:
        return ["Пары: —"]
    out: list[str] = []
    for i in range(0, len(sorted_pairs), per_line):
        chunk = sorted_pairs[i : i + per_line]
        prefix = "Пары: " if i == 0 else "      "
        out.append(f"{prefix}{', '.join(chunk)}")
    return out


_REPORT_FOOTER = [
    "=== Пояснение к строкам таблицы ===",
    "",
    "Строка, например «Ср (ср: росла +0.5%, n=8693)»:",
    "",
    "  Ср              — день недели по UTC (Пн … Вс)",
    "  ср: росла +0.5% — средняя дневная доходность (close−open)/open×100% по всем дням строки; "
    "«росла»/«падала» — по знаку среднего",
    "  n=8693          — число дней в выборке (все пары × все календарные дни с этим weekday)",
    "",
    "Пороги μ×0.5 / μ×1.5 в колонках — отдельно по каждой паре "
    "(среднее дней роста / падения после обрезки перцентилей 5–95%).",
    "",
    "=== Пояснение к ячейкам таблиц ===",
    "",
    "В ячейке, например 50.7 (2/5) [36]:",
    "",
    "  50.7     — доля дней с условием колонки (%)",
    "  (2/5)    — в Y годах с достаточной историей знак отклонения доли колонки от BASE "
    "совпал с общим; в X — совпал",
    "  [36]     — 36 пар из отобранных, у которых на своих данных тот же знак Δ",
    "",
    "Колонки close: сила роста/падения относительно μ×0.5 и μ×1.5 своей пары. "
    "Колонки High/Low: intraday-движение от open дня до тех же порогов.",
    "",
    f"Год в знаменатель (X/Y) попадает только если в этом году ≥ {MIN_YEAR_BASE_DAYS} дней для BASE "
    f"(все дни выборки) и ≥ {MIN_YEAR_ROW_DAYS} дней для строки (день недели). "
    "Иначе год пропускают — мало данных, сравнение ненадёжное.",
    "",
    f"Для [пар] аналогично: ≥ {MIN_YEAR_BASE_DAYS} / ≥ {MIN_YEAR_ROW_DAYS} дней по каждой паре.",
]


def _graph_footer_lines(plot_filename: str) -> list[str]:
    return [
        "",
        f"=== График {plot_filename} ===",
        "",
        "Cumulative Simple Return (%) — простая накопленная доходность в процентах (п.п.), "
        "без реинвестирования.",
        "Каждый день weekday: фиксированный номинал (1×); "
        "R_t = Σ r_i, где r_i — дневной return (close−open)/open×100%.",
        "",
    ]


def build_report_header(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    max_pair_start: datetime | None = None,
) -> list[str]:
    lines = [
        "=== Отчёт: эффекты дня недели ===",
        f"Пар: {len(pairs)}",
        f"Период теста (UTC): {from_date:%Y-%m-%d} .. {to_date:%Y-%m-%d}",
    ]
    if max_pair_start is not None:
        lines.append(
            f"Фильтр пар: первая свеча не позже {max_pair_start:%Y-%m-%d} (UTC)"
        )
    lines.extend(format_pairs_lines(pairs))
    lines.append("")
    return lines


def compute_weekday_effects(
    weekday_daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
) -> list[str]:
    return build_weekday_table(
        weekday_daily,
        pair_bands,
        auto_width=True,
        table_intro=False,
    )


def assemble_weekday_report(
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    table_lines: list[str],
    max_pair_start: datetime | None = None,
) -> str:
    plot_name = weekday_plot_path(len(pairs), from_date, to_date).name
    parts = (
        build_report_header(pairs, from_date, to_date, max_pair_start)
        + table_lines
        + _REPORT_FOOTER
        + _graph_footer_lines(plot_name)
    )
    return "\n".join(parts)


def save_weekday_statistics(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    log.info("[2] Таблицы дней недели сохранены: %s", path)
    return path


def run_weekday_effects(
    daily: pl.DataFrame,
    pair_bands: dict[str, MeanBands],
    pairs: list[str],
    from_date: datetime,
    to_date: datetime,
    max_pair_start: datetime | None = None,
) -> Path:
    from crypto_research.utils.daily_pool import build_weekday_daily
    from crypto_research.utils.weekday_plots import save_weekday_nav_plots

    n_pairs = len(pairs)
    table_lines = compute_weekday_effects(build_weekday_daily(daily), pair_bands)
    text = assemble_weekday_report(pairs, from_date, to_date, table_lines, max_pair_start)
    log_path = save_weekday_statistics(text, weekday_stats_log_path(n_pairs, from_date, to_date))
    save_weekday_nav_plots(daily, pairs, from_date, to_date, weekday_plot_path(n_pairs, from_date, to_date))
    return log_path
