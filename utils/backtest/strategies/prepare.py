"""Подготовка данных и параметров перед run() стратегии."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class StrategyPrepareResult:
    """Результат prepare(): фильтры загрузки и strategy-specific extras."""

    pair_filter: list[str] | None = None
    max_pair_start: datetime | None = None
    load_from_date: datetime | None = None
    extras: dict = field(default_factory=dict)
