"""Реестр стратегий backtester."""

from crypto_research.utils.backtest.strategies.day_of_week import STRATEGY_NAME as DOW_NAME
from crypto_research.utils.backtest.strategies.ema_spreads import STRATEGY_NAME as EMA_NAME
from crypto_research.utils.backtest.strategies.rsi_spreads import STRATEGY_NAME as RSI_NAME
from crypto_research.utils.backtest.strategies.price_sequences import STRATEGY_NAME as PS_NAME
from crypto_research.utils.backtest.strategies.volume_spreads import STRATEGY_NAME as VOL_NAME
from crypto_research.utils.backtest.strategies.handler_combined_algo import CombinedAlgoStrategyHandler
from crypto_research.utils.backtest.strategies.handler_day_of_week import DayOfWeekStrategyHandler
from crypto_research.utils.backtest.strategies.handler_ema_spreads import EmaSpreadsStrategyHandler
from crypto_research.utils.backtest.strategies.handler_price_sequences import PriceSequencesStrategyHandler
from crypto_research.utils.backtest.strategies.handler_rsi_spreads import RsiSpreadsStrategyHandler
from crypto_research.utils.backtest.strategies.handler_volume_spreads import VolumeSpreadsStrategyHandler
from crypto_research.utils.backtest.strategies.handler_ml import MlStrategyHandler
from crypto_research.utils.ml.registry import (
    ML_STUDY_DAY_OF_WEEK,
    ML_STUDY_EMA_SPREADS,
    ML_STUDY_PRICE_SEQUENCES,
    ML_STUDY_RSI_SPREADS,
    ML_STUDY_VOLUME_SPREADS,
    is_ml_study_id,
)

_ML_HANDLER = MlStrategyHandler()
_COMBINED_ALGO_HANDLER = CombinedAlgoStrategyHandler()

STRATEGY_HANDLERS = {
    DOW_NAME: DayOfWeekStrategyHandler(),
    EMA_NAME: EmaSpreadsStrategyHandler(),
    RSI_NAME: RsiSpreadsStrategyHandler(),
    PS_NAME: PriceSequencesStrategyHandler(),
    VOL_NAME: VolumeSpreadsStrategyHandler(),
    ML_STUDY_DAY_OF_WEEK: _ML_HANDLER,
    ML_STUDY_EMA_SPREADS: _ML_HANDLER,
    ML_STUDY_RSI_SPREADS: _ML_HANDLER,
    ML_STUDY_PRICE_SEQUENCES: _ML_HANDLER,
    ML_STUDY_VOLUME_SPREADS: _ML_HANDLER,
}


def get_strategy_handler(strategy: str, *, algo_spec=None, ml_spec=None):
    if algo_spec is not None:
        return _COMBINED_ALGO_HANDLER
    if ml_spec is not None or is_ml_study_id(strategy):
        return _ML_HANDLER
    if strategy not in STRATEGY_HANDLERS:
        raise KeyError(f"Неизвестная стратегия: {strategy}")
    return STRATEGY_HANDLERS[strategy]
