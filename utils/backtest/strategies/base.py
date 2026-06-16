"""Базовый контракт стратегии backtester."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from crypto_research.utils.backtest.context import BacktestContext
    from crypto_research.utils.backtest.report import BacktestResult
    from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult


class StrategyHandler(ABC):
    name: str

    @abstractmethod
    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        """Проверка CLI-аргументов; parser.error() при нарушении."""

    def prepare(self, ctx: BacktestContext) -> StrategyPrepareResult:
        """Загрузка train-порогов, optimistic-отбор и пр. перед run()."""
        from crypto_research.utils.backtest.strategies.prepare import StrategyPrepareResult

        return StrategyPrepareResult()

    @abstractmethod
    def run(
        self,
        ctx: BacktestContext,
        daily,
        pairs: list[str],
        daily_benchmark_49,
        n_benchmark_pairs: int,
        prepare: StrategyPrepareResult,
    ) -> BacktestResult:
        ...
