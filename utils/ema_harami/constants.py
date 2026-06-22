"""Константы исследования ema_harami (те же бакеты EMA(9), что ema_spreads)."""

from __future__ import annotations

from crypto_research.utils.candlestick_patterns.harami.constants import (
    BUCKET_BEARISH,
    BUCKET_BULLISH,
)
from crypto_research.utils.ema_spreads.constants import (
    EMA_SCENARIO_ROWS,
    N_EMA_SCENARIOS,
    SELECTED_EMA_PERIOD,
)

BULLISH_HARAMI_LABEL = "Bullish Harami (подтв.)"
BEARISH_HARAMI_LABEL = "Bearish Harami (подтв.)"

# Индексы бакетов = b0..b6 как в ema_spreads; b3 (|dev|≤near) не используется.
ABOVE_EMA_BUCKETS: frozenset[int] = frozenset({0, 1, 2})
BELOW_EMA_BUCKETS: frozenset[int] = frozenset({4, 5, 6})

EMA_HARAMI_HARAMI_BY_BUCKET: dict[int, int] = {
    **{b: BUCKET_BEARISH for b in ABOVE_EMA_BUCKETS},
    **{b: BUCKET_BULLISH for b in BELOW_EMA_BUCKETS},
}

N_EMA_HARAMI_SCENARIOS = N_EMA_SCENARIOS

SCENARIO_ROWS: tuple[str, ...] = tuple(
    f"{EMA_SCENARIO_ROWS[i]} × "
    + (
        "—"
        if i == 3
        else (BEARISH_HARAMI_LABEL if i in ABOVE_EMA_BUCKETS else BULLISH_HARAMI_LABEL)
    )
    for i in range(N_EMA_SCENARIOS)
)

STUDY_NOTE = (
    f"EMA({SELECTED_EMA_PERIOD}): те же бакеты b0–b6 и пороги t1/t2/near, что в ema_spreads "
    "(dev вчера на день сигнала t+1 = конец дня t). "
    "Harami с подтверждением: паттерн (t−2, t−1), подтверждение на t "
    "(Bullish — close(t) > open(t−2); Bearish — close(t) < open(t−2)); сигнал и доходность — день t+1. "
    "Ниже EMA (b4–b6) — только Bullish; выше EMA (b0–b2) — только Bearish. "
    "b3 (|dev|≤near) исключён. BASE для Δ — все дни с EMA-бакетом (как ema_spreads). "
    "Сравнение с ema_summary_ema9.log — по тем же номерам бакетов."
)
