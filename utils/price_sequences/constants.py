"""Сценарии серий роста/падения (1…6 дней)."""

MAX_STREAK_DAYS = 6

SCENARIO_ROWS: tuple[str, ...] = tuple(
    f"После {d}д падения" for d in range(MAX_STREAK_DAYS, 0, -1)
) + tuple(f"После {d}д роста" for d in range(MAX_STREAK_DAYS, 0, -1))
