"""Базовый контракт стратегии backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from crypto_research.utils.backtest.context import BacktestContext
    from crypto_research.utils.backtest.report import BacktestResult


class StrategyHandler(ABC):
    @abstractmethod
    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        """Проверка CLI-аргументов; parser.error() при нарушении."""

    @abstractmethod
    def run(
        self,
        ctx: BacktestContext,
        daily,
        pairs: list[str],
        daily_benchmark_49,
        n_benchmark_pairs: int,
        pairs_by_weekday: dict[int, list[str]] | None,
    ) -> BacktestResult:
        ...
