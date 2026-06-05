"""Парсинг опций графика weekday NAV."""

from __future__ import annotations

_WEEKDAY_ALIASES: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
    "пн": 0,
    "вт": 1,
    "ср": 2,
    "чт": 3,
    "пт": 4,
    "сб": 5,
    "вс": 6,
}


def parse_highlight_weekdays(tokens: list[str] | None) -> frozenset[int]:
    if not tokens:
        return frozenset()
    out: set[int] = set()
    for raw in tokens:
        key = raw.strip().lower().replace(".", "")
        if not key:
            continue
        if key not in _WEEKDAY_ALIASES:
            valid = "пн, вт, ср, чт, пт, сб, вс, mon..sun, 0..6"
            raise ValueError(f"Неизвестный день недели: {raw!r}. Допустимо: {valid}")
        out.add(_WEEKDAY_ALIASES[key])
    return frozenset(out)
