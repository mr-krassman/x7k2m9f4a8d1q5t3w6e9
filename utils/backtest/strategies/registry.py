"""Реестр стратегий backtester."""

from crypto_research.utils.backtest.strategies.day_of_week import STRATEGY_NAME as DOW_NAME
from crypto_research.utils.backtest.strategies.day_of_week_ml import STRATEGY_NAME as DOW_ML_NAME
from crypto_research.utils.backtest.strategies.ema_spreads import STRATEGY_NAME as EMA_NAME
from crypto_research.utils.backtest.strategies.handler_day_of_week import DayOfWeekStrategyHandler
from crypto_research.utils.backtest.strategies.handler_day_of_week_ml import DayOfWeekMlStrategyHandler
from crypto_research.utils.backtest.strategies.handler_ema_spreads import EmaSpreadsStrategyHandler

STRATEGY_HANDLERS = {
    DOW_NAME: DayOfWeekStrategyHandler(),
    DOW_ML_NAME: DayOfWeekMlStrategyHandler(),
    EMA_NAME: EmaSpreadsStrategyHandler(),
}
