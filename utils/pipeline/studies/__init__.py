"""Реестр исследований report_generator."""

from crypto_research.utils.pipeline.study_ids import (
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_PERIOD_SCREEN,
    STUDY_EMA_SPREADS,
    STUDY_EMA_HARAMI,
    STUDY_HARAMI,
    STUDY_PRICE_SEQUENCES,
    STUDY_RSI_PERIOD_SCREEN,
    STUDY_RSI_SPREADS,
    STUDY_VOLATILITY_PERIOD_SCREEN,
    STUDY_VOLATILITY_SPREADS,
    STUDY_VOLUME_EMA_PERIOD_SCREEN,
    STUDY_VOLUME_SPREADS,
)
from crypto_research.utils.pipeline.studies.day_of_week import DayOfWeekStudy
from crypto_research.utils.pipeline.studies.ema_period_screen import EmaPeriodScreenStudy
from crypto_research.utils.pipeline.studies.ema_harami import EmaHaramiStudy
from crypto_research.utils.pipeline.studies.ema_spreads import EmaSpreadsStudy
from crypto_research.utils.pipeline.studies.harami import HaramiStudy
from crypto_research.utils.pipeline.studies.price_sequences import PriceSequencesStudy
from crypto_research.utils.pipeline.studies.rsi_period_screen import RsiPeriodScreenStudy
from crypto_research.utils.pipeline.studies.rsi_spreads import RsiSpreadsStudy
from crypto_research.utils.pipeline.studies.volatility_period_screen import VolatilityPeriodScreenStudy
from crypto_research.utils.pipeline.studies.volatility_spreads import VolatilitySpreadsStudy
from crypto_research.utils.pipeline.studies.volume_ema_period_screen import VolumeEmaPeriodScreenStudy
from crypto_research.utils.pipeline.studies.volume_spreads import VolumeSpreadsStudy

STUDY_HANDLERS = {
    STUDY_DAY_OF_WEEK: DayOfWeekStudy(),
    STUDY_EMA_SPREADS: EmaSpreadsStudy(),
    STUDY_EMA_PERIOD_SCREEN: EmaPeriodScreenStudy(),
    STUDY_RSI_PERIOD_SCREEN: RsiPeriodScreenStudy(),
    STUDY_RSI_SPREADS: RsiSpreadsStudy(),
    STUDY_VOLUME_EMA_PERIOD_SCREEN: VolumeEmaPeriodScreenStudy(),
    STUDY_VOLUME_SPREADS: VolumeSpreadsStudy(),
    STUDY_PRICE_SEQUENCES: PriceSequencesStudy(),
    STUDY_HARAMI: HaramiStudy(),
    STUDY_EMA_HARAMI: EmaHaramiStudy(),
    STUDY_VOLATILITY_PERIOD_SCREEN: VolatilityPeriodScreenStudy(),
    STUDY_VOLATILITY_SPREADS: VolatilitySpreadsStudy(),
}
