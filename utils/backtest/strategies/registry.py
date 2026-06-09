"""Реестр стратегий backtester."""

from crypto_research.utils.backtest.strategies.day_of_week import STRATEGY_NAME
from crypto_research.utils.backtest.strategies.handler_day_of_week import DayOfWeekStrategyHandler

STRATEGY_HANDLERS = {
    STRATEGY_NAME: DayOfWeekStrategyHandler(),
}
