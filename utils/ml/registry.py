"""Единый реестр ML-исследований и комбинированных bundle (пути, фичи, policy)."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_research.utils.pipeline.study_ids import (
    STUDY_COMBINED,
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_SPREADS,
    STUDY_PRICE_SEQUENCES,
    STUDY_RSI_SPREADS,
)

ML_STUDY_DAY_OF_WEEK = "day_of_week_ml"
ML_STUDY_EMA_SPREADS = "ema_spreads_ml"
ML_STUDY_RSI_SPREADS = "rsi_spreads_ml"
ML_STUDY_PRICE_SEQUENCES = "price_sequences_ml"

ML_STUDY_ORDER: tuple[str, ...] = (
    ML_STUDY_DAY_OF_WEEK,
    ML_STUDY_EMA_SPREADS,
    ML_STUDY_RSI_SPREADS,
    ML_STUDY_PRICE_SEQUENCES,
)
ML_STUDY_CHOICES: tuple[str, ...] = ML_STUDY_ORDER

FEATURE_PAIR_ID = "pair_id"
FEATURE_WEEKDAY_ENC = "weekday_enc"
FEATURE_EMA_DEV_PAIR_NORM = "ema_dev_pair_norm"
FEATURE_RSI_PAIR_NORM = "rsi_pair_norm"
FEATURE_STREAK_PAIR_NORM = "streak_pair_norm"

CATEGORICAL_FEATURES: frozenset[str] = frozenset({FEATURE_PAIR_ID, FEATURE_WEEKDAY_ENC})

RULE_BASED_STRATEGIES: frozenset[str] = frozenset(
    {"day_of_week", "ema_spreads", "rsi_spreads", "price_sequences"}
)


@dataclass(frozen=True)
class MlStudyEntry:
    study_id: str
    short_id: str
    extra_features: tuple[str, ...]
    output_study: str
    legacy_output_tag: bool
    policy_mode: str  # "weekday" | "global"
    predictive_feature: str | None = None
    predictive_plot_slug: str | None = None


@dataclass(frozen=True)
class CombinedBundleEntry:
    bundle_id: str
    studies: tuple[str, ...]
    feature_slug: str
    policy_mode: str = "weekday"


ML_STUDY_REGISTRY: dict[str, MlStudyEntry] = {
    ML_STUDY_DAY_OF_WEEK: MlStudyEntry(
        study_id=ML_STUDY_DAY_OF_WEEK,
        short_id="dow",
        extra_features=(FEATURE_WEEKDAY_ENC,),
        output_study=STUDY_DAY_OF_WEEK,
        legacy_output_tag=True,
        policy_mode="weekday",
    ),
    ML_STUDY_EMA_SPREADS: MlStudyEntry(
        study_id=ML_STUDY_EMA_SPREADS,
        short_id="ema",
        extra_features=(FEATURE_EMA_DEV_PAIR_NORM,),
        output_study=STUDY_EMA_SPREADS,
        legacy_output_tag=False,
        policy_mode="global",
        predictive_feature=FEATURE_EMA_DEV_PAIR_NORM,
        predictive_plot_slug="ema_dev",
    ),
    ML_STUDY_RSI_SPREADS: MlStudyEntry(
        study_id=ML_STUDY_RSI_SPREADS,
        short_id="rsi",
        extra_features=(FEATURE_RSI_PAIR_NORM,),
        output_study=STUDY_RSI_SPREADS,
        legacy_output_tag=False,
        policy_mode="global",
        predictive_feature=FEATURE_RSI_PAIR_NORM,
        predictive_plot_slug="rsi",
    ),
    ML_STUDY_PRICE_SEQUENCES: MlStudyEntry(
        study_id=ML_STUDY_PRICE_SEQUENCES,
        short_id="streak",
        extra_features=(FEATURE_STREAK_PAIR_NORM,),
        output_study=STUDY_PRICE_SEQUENCES,
        legacy_output_tag=False,
        policy_mode="global",
        predictive_feature=FEATURE_STREAK_PAIR_NORM,
        predictive_plot_slug="streak",
    ),
}

COMBINED_BUNDLE_REGISTRY: dict[tuple[str, ...], CombinedBundleEntry] = {
    (ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS): CombinedBundleEntry(
        bundle_id="dow_ema_sp",
        studies=(ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS),
        feature_slug="dow_ema",
        policy_mode="weekday",
    ),
    (ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS, ML_STUDY_RSI_SPREADS): CombinedBundleEntry(
        bundle_id="dow_ema_rsi_sp",
        studies=(ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS, ML_STUDY_RSI_SPREADS),
        feature_slug="dow_ema_rsi",
        policy_mode="weekday",
    ),
    (
        ML_STUDY_DAY_OF_WEEK,
        ML_STUDY_EMA_SPREADS,
        ML_STUDY_RSI_SPREADS,
        ML_STUDY_PRICE_SEQUENCES,
    ): CombinedBundleEntry(
        bundle_id="dow_ema_rsi_streak",
        studies=(
            ML_STUDY_DAY_OF_WEEK,
            ML_STUDY_EMA_SPREADS,
            ML_STUDY_RSI_SPREADS,
            ML_STUDY_PRICE_SEQUENCES,
        ),
        feature_slug="dow_ema_rsi_streak",
        policy_mode="weekday",
    ),
}

BUNDLE_ID_TO_STUDIES: dict[str, tuple[str, ...]] = {
    entry.bundle_id: entry.studies for entry in COMBINED_BUNDLE_REGISTRY.values()
}

COMPARE_MODEL_CHOICES: tuple[str, ...] = ML_STUDY_ORDER + tuple(
    entry.bundle_id for entry in COMBINED_BUNDLE_REGISTRY.values()
)


@dataclass(frozen=True)
class MlStudySpec:
    studies: tuple[str, ...]
    feature_columns: tuple[str, ...]
    output_study: str
    feature_slug: str
    legacy_output_tag: bool
    ml_subdir: str
    bundle_id: str | None = None
    policy_mode: str = "weekday"
    predictive_feature: str | None = None
    predictive_plot_slug: str | None = None


def is_ml_study_id(name: str) -> bool:
    return name in ML_STUDY_REGISTRY


def is_combined_bundle_id(name: str) -> bool:
    return name in BUNDLE_ID_TO_STUDIES


def is_ml_backtest_strategy(name: str) -> bool:
    return is_ml_study_id(name) or is_combined_bundle_id(name)


def canonical_ml_studies(studies: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(studies))
    unknown = set(ordered) - set(ML_STUDY_CHOICES)
    if unknown:
        raise ValueError(f"Неизвестные ML-исследования: {sorted(unknown)}")
    return tuple(study for study in ML_STUDY_ORDER if study in ordered)


def combined_bundle_for(studies: tuple[str, ...]) -> CombinedBundleEntry | None:
    if len(studies) <= 1:
        return None
    return COMBINED_BUNDLE_REGISTRY.get(studies)


def resolve_ml_study(studies: list[str] | tuple[str, ...]) -> MlStudySpec:
    if not studies:
        studies = (ML_STUDY_DAY_OF_WEEK,)
    ordered = canonical_ml_studies(studies)
    features: list[str] = [FEATURE_PAIR_ID]
    for study_id in ordered:
        features.extend(ML_STUDY_REGISTRY[study_id].extra_features)
    bundle = combined_bundle_for(ordered)
    if bundle is not None:
        from crypto_research.utils.ml.numeric_features import active_numeric_specs, resolve_predictive

        numeric_specs = active_numeric_specs(features)
        predictive_feature, predictive_plot_slug = resolve_predictive(numeric_specs)
        return MlStudySpec(
            studies=bundle.studies,
            feature_columns=tuple(features),
            output_study=STUDY_COMBINED,
            feature_slug=bundle.feature_slug,
            legacy_output_tag=False,
            ml_subdir="ml",
            bundle_id=bundle.bundle_id,
            policy_mode=bundle.policy_mode,
            predictive_feature=predictive_feature,
            predictive_plot_slug=predictive_plot_slug,
        )

    entry = ML_STUDY_REGISTRY[ordered[0]]
    feature_slug = entry.short_id
    return MlStudySpec(
        studies=ordered,
        feature_columns=tuple(features),
        output_study=entry.output_study,
        feature_slug=feature_slug,
        legacy_output_tag=entry.legacy_output_tag,
        ml_subdir="ml",
        bundle_id=None,
        policy_mode=entry.policy_mode,
        predictive_feature=entry.predictive_feature,
        predictive_plot_slug=entry.predictive_plot_slug,
    )


def immediate_parent_bundle_id(spec: MlStudySpec) -> str | None:
    if not spec.bundle_id or len(spec.studies) <= 1:
        return None
    parent = combined_bundle_for(spec.studies[:-1])
    return parent.bundle_id if parent else None


def compare_model_id(spec: MlStudySpec) -> str:
    return spec.bundle_id or spec.studies[0]


def auto_compare_model_ids(spec: MlStudySpec) -> tuple[str, ...]:
    current = compare_model_id(spec)
    parent = immediate_parent_bundle_id(spec)
    if parent is None:
        return (current,)
    return (parent, current)


def resolve_compare_model(model_id: str) -> MlStudySpec:
    """Идентификатор для --compare-models: ML study или combined bundle_id."""
    if is_combined_bundle_id(model_id):
        return resolve_ml_study(BUNDLE_ID_TO_STUDIES[model_id])
    if is_ml_study_id(model_id):
        return resolve_ml_study((model_id,))
    raise ValueError(
        f"Неизвестная модель для сравнения: {model_id!r}. "
        f"Допустимо: {list(COMPARE_MODEL_CHOICES)}"
    )


def ml_spec_to_dict(spec: MlStudySpec) -> dict[str, object]:
    return {
        "studies": list(spec.studies),
        "feature_columns": list(spec.feature_columns),
        "output_study": spec.output_study,
        "feature_slug": spec.feature_slug,
        "bundle_id": spec.bundle_id,
        "ml_subdir": spec.ml_subdir,
        "policy_mode": spec.policy_mode,
    }


