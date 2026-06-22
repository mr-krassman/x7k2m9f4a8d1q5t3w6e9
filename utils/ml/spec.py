"""Спецификация ML-исследования (реэкспорт из registry)."""

from crypto_research.utils.ml.registry import (
    CATEGORICAL_FEATURES,
    FEATURE_EMA_DEV_PAIR_NORM,
    FEATURE_PAIR_ID,
    FEATURE_RSI_PAIR_NORM,
    FEATURE_STREAK_PAIR_NORM,
    FEATURE_VOL_LOG_REL_PAIR,
    FEATURE_WEEKDAY_ENC,
    ML_STUDY_CHOICES,
    ML_STUDY_DAY_OF_WEEK,
    ML_STUDY_EMA_SPREADS,
    ML_STUDY_PRICE_SEQUENCES,
    ML_STUDY_RSI_SPREADS,
    ML_STUDY_VOLUME_SPREADS,
    MlStudySpec,
    ml_spec_to_dict,
    resolve_ml_study,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_EMA_DEV_PAIR_NORM",
    "FEATURE_PAIR_ID",
    "FEATURE_RSI_PAIR_NORM",
    "FEATURE_STREAK_PAIR_NORM",
    "FEATURE_VOL_LOG_REL_PAIR",
    "FEATURE_WEEKDAY_ENC",
    "ML_STUDY_CHOICES",
    "ML_STUDY_DAY_OF_WEEK",
    "ML_STUDY_EMA_SPREADS",
    "ML_STUDY_PRICE_SEQUENCES",
    "ML_STUDY_RSI_SPREADS",
    "ML_STUDY_VOLUME_SPREADS",
    "MlStudySpec",
    "ml_spec_to_dict",
    "resolve_ml_study",
]
