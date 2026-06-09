"""Константы исследования ema_spreads."""

from __future__ import annotations

SELECTED_EMA_PERIOD = 5
DEFAULT_EMA_PERIODS: tuple[int, ...] = (SELECTED_EMA_PERIOD,)
DEFAULT_SCREEN_EMA_PERIODS: tuple[int, ...] = (5, 9, 12, 20, 50, 100, 200)

SCREEN_MIN_POOLED_DELTA_PP = 1.5

EMA_SCENARIO_ROWS: tuple[str, ...] = (
    "dev > t2⁺",
    "t1⁺ < dev ≤ t2⁺",
    "near < dev ≤ t1⁺",
    "|dev| ≤ near",
    "t2⁻ ≤ dev < −near",
    "t1⁻ ≤ dev < t2⁻",
    "dev < t1⁻",
)
N_EMA_SCENARIOS = len(EMA_SCENARIO_ROWS)

EMA_BUCKET_THRESHOLDS_NOTE = (
    "Пороги t1⁺/t2⁺ — ⅓ и ⅔ квантили dev>0 пары; t1⁻/t2⁻ — ⅓ и ⅔ dev<0 (t1⁻ ≤ t2⁻); "
    "near = max(Q10(|dev|), 0.05%). Бакеты b0–b6 — см. строки таблицы."
)

RETURN_STATS_COLS: tuple[str, ...] = (
    "Цена росла",
    "Цена росла > mean",
    "Цена росла mean",
    "Цена росла < mean",
    "Цена падала",
    "Цена падала > mean",
    "Цена падала mean",
    "Цена падала < mean",
    "Цена росла до mean",
    "Цена была выше mean",
    "Цена падала до mean",
    "Цена была ниже mean",
)

SCREEN_STATS_COLS: tuple[str, ...] = (RETURN_STATS_COLS[0], RETURN_STATS_COLS[4])

COL_WIDTH = 25
ROW_TITLE = "dev к EMA, %"

REPEATABILITY_NOTE = (
    "В ячейке: (годы X/Y) — знак отклонения доли колонки от BASE совпал с общим "
    "(порог: 40/15 дн. в году, BASE/строка)."
)
