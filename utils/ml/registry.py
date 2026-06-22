"""Единый реестр ML-исследований и комбинированных bundle (пути, фичи, policy)."""

from __future__ import annotations

from dataclasses import dataclass

from crypto_research.utils.pipeline.study_ids import (
    STUDY_COMBINED,
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_SPREADS,
    STUDY_PRICE_SEQUENCES,
    STUDY_RSI_SPREADS,
    STUDY_VOLUME_SPREADS,
)

ML_STUDY_DAY_OF_WEEK = "day_of_week_ml"
ML_STUDY_EMA_SPREADS = "ema_spreads_ml"
ML_STUDY_RSI_SPREADS = "rsi_spreads_ml"
ML_STUDY_PRICE_SEQUENCES = "price_sequences_ml"
ML_STUDY_VOLUME_SPREADS = "volume_spreads_ml"

ML_STUDY_ORDER: tuple[str, ...] = (
    ML_STUDY_DAY_OF_WEEK,
    ML_STUDY_EMA_SPREADS,
    ML_STUDY_RSI_SPREADS,
    ML_STUDY_PRICE_SEQUENCES,
    ML_STUDY_VOLUME_SPREADS,
)
ML_STUDY_CHOICES: tuple[str, ...] = ML_STUDY_ORDER

FEATURE_PAIR_ID = "pair_id"
FEATURE_WEEKDAY_ENC = "weekday_enc"
FEATURE_EMA_DEV_PAIR_NORM = "ema_dev_pair_norm"
FEATURE_RSI_PAIR_NORM = "rsi_pair_norm"
FEATURE_STREAK_PAIR_NORM = "streak_pair_norm"
FEATURE_VOL_LOG_REL_PAIR = "vol_log_rel_pair"

CATEGORICAL_FEATURES: frozenset[str] = frozenset({FEATURE_PAIR_ID, FEATURE_WEEKDAY_ENC})

RULE_BASED_STRATEGIES: frozenset[str] = frozenset(
    {"day_of_week", "ema_spreads", "rsi_spreads", "price_sequences", "volume_spreads"}
)

RULE_BASED_TO_ML_STUDY: dict[str, str] = {
    "day_of_week": ML_STUDY_DAY_OF_WEEK,
    "ema_spreads": ML_STUDY_EMA_SPREADS,
    "rsi_spreads": ML_STUDY_RSI_SPREADS,
    "price_sequences": ML_STUDY_PRICE_SEQUENCES,
    "volume_spreads": ML_STUDY_VOLUME_SPREADS,
}


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
    policy_mode: str = "global"


ML_STUDY_REGISTRY: dict[str, MlStudyEntry] = {
    ML_STUDY_DAY_OF_WEEK: MlStudyEntry(
        study_id=ML_STUDY_DAY_OF_WEEK,
        short_id="dow",
        extra_features=(FEATURE_WEEKDAY_ENC,),
        output_study=STUDY_DAY_OF_WEEK,
        legacy_output_tag=True,
        policy_mode="global",
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
    ML_STUDY_VOLUME_SPREADS: MlStudyEntry(
        study_id=ML_STUDY_VOLUME_SPREADS,
        short_id="vol",
        extra_features=(FEATURE_VOL_LOG_REL_PAIR,),
        output_study=STUDY_VOLUME_SPREADS,
        legacy_output_tag=False,
        policy_mode="global",
        predictive_feature=FEATURE_VOL_LOG_REL_PAIR,
        predictive_plot_slug="vol_log_rel",
    ),
}

COMBINED_BUNDLE_REGISTRY: dict[tuple[str, ...], CombinedBundleEntry] = {
    (ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS): CombinedBundleEntry(
        bundle_id="dow_ema_sp",
        studies=(ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS),
        feature_slug="dow_ema",
    ),
    (ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS, ML_STUDY_RSI_SPREADS): CombinedBundleEntry(
        bundle_id="dow_ema_rsi_sp",
        studies=(ML_STUDY_DAY_OF_WEEK, ML_STUDY_EMA_SPREADS, ML_STUDY_RSI_SPREADS),
        feature_slug="dow_ema_rsi",
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
    ),
    (
        ML_STUDY_DAY_OF_WEEK,
        ML_STUDY_EMA_SPREADS,
        ML_STUDY_RSI_SPREADS,
        ML_STUDY_PRICE_SEQUENCES,
        ML_STUDY_VOLUME_SPREADS,
    ): CombinedBundleEntry(
        bundle_id="dow_ema_rsi_streak_vol",
        studies=(
            ML_STUDY_DAY_OF_WEEK,
            ML_STUDY_EMA_SPREADS,
            ML_STUDY_RSI_SPREADS,
            ML_STUDY_PRICE_SEQUENCES,
            ML_STUDY_VOLUME_SPREADS,
        ),
        feature_slug="dow_ema_rsi_streak_vol",
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
    policy_mode: str = "global"
    predictive_feature: str | None = None
    predictive_plot_slug: str | None = None


def is_ml_study_id(name: str) -> bool:
    return name in ML_STUDY_REGISTRY


def is_combined_bundle_id(name: str) -> bool:
    if name in BUNDLE_ID_TO_STUDIES:
        return True
    return ml_studies_from_bundle_slug(name) is not None


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


def bundle_slug_from_ml_studies(studies: tuple[str, ...]) -> str:
    return "_".join(ML_STUDY_REGISTRY[study_id].short_id for study_id in studies)


def ml_studies_from_bundle_slug(slug: str) -> tuple[str, ...] | None:
    if not slug:
        return None
    short_to_study = {ML_STUDY_REGISTRY[study_id].short_id: study_id for study_id in ML_STUDY_ORDER}
    studies: list[str] = []
    for token in slug.split("_"):
        study_id = short_to_study.get(token)
        if study_id is None:
            return None
        studies.append(study_id)
    if len(studies) <= 1:
        return None
    return canonical_ml_studies(studies)


def bundle_id_for_ml_studies(studies: tuple[str, ...]) -> tuple[str, str]:
    """Вернуть (bundle_id, feature_slug) для combined ML: зарегистрированный или динамический."""
    entry = combined_bundle_for(studies)
    if entry is not None:
        return entry.bundle_id, entry.feature_slug
    slug = bundle_slug_from_ml_studies(studies)
    return slug, slug


def resolve_ml_study(studies: list[str] | tuple[str, ...]) -> MlStudySpec:
    if not studies:
        studies = (ML_STUDY_DAY_OF_WEEK,)
    ordered = canonical_ml_studies(studies)
    features: list[str] = [FEATURE_PAIR_ID]
    for study_id in ordered:
        features.extend(ML_STUDY_REGISTRY[study_id].extra_features)

    if len(ordered) <= 1:
        entry = ML_STUDY_REGISTRY[ordered[0]]
        return MlStudySpec(
            studies=ordered,
            feature_columns=tuple(features),
            output_study=entry.output_study,
            feature_slug=entry.short_id,
            legacy_output_tag=entry.legacy_output_tag,
            ml_subdir="ml",
            bundle_id=None,
            policy_mode=entry.policy_mode,
            predictive_feature=entry.predictive_feature,
            predictive_plot_slug=entry.predictive_plot_slug,
        )

    from crypto_research.utils.ml.numeric_features import active_numeric_specs, resolve_predictive

    bundle_id, feature_slug = bundle_id_for_ml_studies(ordered)
    registered = combined_bundle_for(ordered)
    policy_mode = registered.policy_mode if registered is not None else "global"
    numeric_specs = active_numeric_specs(features)
    predictive_feature, predictive_plot_slug = resolve_predictive(numeric_specs)
    return MlStudySpec(
        studies=ordered,
        feature_columns=tuple(features),
        output_study=STUDY_COMBINED,
        feature_slug=feature_slug,
        legacy_output_tag=False,
        ml_subdir="ml",
        bundle_id=bundle_id,
        policy_mode=policy_mode,
        predictive_feature=predictive_feature,
        predictive_plot_slug=predictive_plot_slug,
    )


def immediate_parent_bundle_id(spec: MlStudySpec) -> str | None:
    if len(spec.studies) <= 1:
        return None
    parent_studies = spec.studies[:-1]
    if len(parent_studies) == 1:
        return parent_studies[0]
    bundle_id, _ = bundle_id_for_ml_studies(parent_studies)
    return bundle_id


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
        studies = BUNDLE_ID_TO_STUDIES.get(model_id)
        if studies is None:
            parsed = ml_studies_from_bundle_slug(model_id)
            if parsed is None:
                raise ValueError(f"Не удалось разобрать combined bundle_id: {model_id!r}")
            studies = parsed
        return resolve_ml_study(studies)
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


