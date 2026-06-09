"""Реестр исследований report_generator."""

from crypto_research.utils.pipeline.study_ids import (
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_PERIOD_SCREEN,
    STUDY_EMA_SPREADS,
)
from crypto_research.utils.pipeline.studies.day_of_week import DayOfWeekStudy
from crypto_research.utils.pipeline.studies.ema_period_screen import EmaPeriodScreenStudy
from crypto_research.utils.pipeline.studies.ema_spreads import EmaSpreadsStudy

STUDY_HANDLERS = {
    STUDY_DAY_OF_WEEK: DayOfWeekStudy(),
    STUDY_EMA_SPREADS: EmaSpreadsStudy(),
    STUDY_EMA_PERIOD_SCREEN: EmaPeriodScreenStudy(),
}
