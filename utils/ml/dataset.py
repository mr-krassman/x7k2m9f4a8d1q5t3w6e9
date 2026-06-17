"""Датасет для ML: направление дневной доходности open→close."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD
from crypto_research.utils.ml.ema_dev_norm import PairEmaDevBounds
from crypto_research.utils.ml.numeric_features import (
    active_numeric_specs,
    attach_normalized_features,
    needs_day_close,
)
from crypto_research.utils.ml.pair_bounds import PairBounds
from crypto_research.utils.ml.registry import FEATURE_EMA_DEV_PAIR_NORM, FEATURE_PAIR_ID, FEATURE_WEEKDAY_ENC
from crypto_research.utils.ml.spec import (
    CATEGORICAL_FEATURES,
    ML_STUDY_DAY_OF_WEEK,
    MlStudySpec,
    resolve_ml_study,
)
from crypto_research.utils.pipeline.daily_pool import build_pooled_daily
from crypto_research.utils.pipeline.dates import datetime_to_ms, parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import (
    _DEFAULT_WORKERS,
    load_pairs_klines,
    resolve_pairs,
)
from crypto_research.utils.pipeline.load_summary import log_load_summary
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import FULL_POOL_FROM, FULL_POOL_TO
from crypto_research.utils.rsi.constants import SELECTED_RSI_PERIOD

log = get_logger("ml_dataset")


@dataclass(frozen=True)
class DirectionDataset:
    frame: pl.DataFrame
    pairs: list[str]
    pair_encoder: LabelEncoder
    weekday_encoder: LabelEncoder | None
    feature_columns: tuple[str, ...]
    ml_spec: MlStudySpec
    target_column: str = "direction_up"
    pair_norm_bounds: dict[str, dict[str, PairBounds | PairEmaDevBounds]] | None = None

    @property
    def pair_ema_dev_bounds(self) -> dict[str, PairEmaDevBounds] | None:
        if not self.pair_norm_bounds:
            return None
        bounds = self.pair_norm_bounds.get(FEATURE_EMA_DEV_PAIR_NORM)
        return bounds  # type: ignore[return-value]

    @property
    def ema_period(self) -> int | None:
        return SELECTED_EMA_PERIOD if FEATURE_EMA_DEV_PAIR_NORM in self.feature_columns else None

    @property
    def rsi_period(self) -> int | None:
        from crypto_research.utils.ml.registry import FEATURE_RSI_PAIR_NORM

        return SELECTED_RSI_PERIOD if FEATURE_RSI_PAIR_NORM in self.feature_columns else None


WeekdayDirectionDataset = DirectionDataset


def _normalize_weekday_expr() -> pl.Expr:
    wd = pl.col("weekday")
    return pl.when(wd >= 1).then(wd - 1).otherwise(wd).cast(pl.Int64).alias("weekday")


def load_full_pool_daily(
    data_dir: Path,
    *,
    max_pair_start: datetime | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    workers: int = _DEFAULT_WORKERS,
) -> tuple[pl.DataFrame, list[str]]:
    """Пары с первой свечой ≤ from_date, дневные ряды за [from_date, to_date]."""
    period_from = from_date or parse_iso_utc(FULL_POOL_FROM)
    period_to = to_date or parse_iso_utc(FULL_POOL_TO)
    pair_start_limit = max_pair_start or period_from
    pairs = resolve_pairs(data_dir, None, pair_start_limit, workers=workers)
    log.info(
        "[ml] pair filter: first_candle <= %s → %s pairs",
        pair_start_limit.date(),
        len(pairs),
    )
    klines = load_pairs_klines(
        data_dir,
        pairs,
        from_ms=datetime_to_ms(period_from),
        to_ms=datetime_to_ms(period_to),
        workers=workers,
    )
    log_load_summary(klines)
    daily = build_pooled_daily(klines)
    resolved = sorted(klines.keys())
    log.info(
        "[ml] pooled daily: pairs=%s rows=%s period=%s..%s",
        len(resolved),
        daily.height,
        period_from.date(),
        period_to.date(),
    )
    return daily, resolved


def _base_weekday_frame(daily: pl.DataFrame, *, need_close: bool) -> pl.DataFrame:
    cols = ["return_pct", "day_utc", "day_open", "day_high", "day_low", "pair"]
    if need_close:
        cols.append("day_close")
    frame = daily.select(cols).with_columns(pl.col("day_utc").dt.weekday().alias("weekday"))
    return frame.with_columns(_normalize_weekday_expr())


def build_direction_dataset(
    daily: pl.DataFrame,
    spec: MlStudySpec,
    *,
    pair_ema_dev_bounds: dict[str, PairEmaDevBounds] | None = None,
    pair_norm_bounds: dict[str, dict[str, PairBounds | PairEmaDevBounds]] | None = None,
) -> DirectionDataset:
    """Фичи по spec, таргет direction_up (1 = close > open)."""
    numeric_cols = [s.column for s in active_numeric_specs(spec.feature_columns)]
    resolved_bounds = pair_norm_bounds
    if resolved_bounds is None and pair_ema_dev_bounds is not None:
        resolved_bounds = {FEATURE_EMA_DEV_PAIR_NORM: pair_ema_dev_bounds}

    weekday_daily = _base_weekday_frame(daily, need_close=needs_day_close(spec.feature_columns))
    weekday_daily = attach_normalized_features(weekday_daily, spec.feature_columns, resolved_bounds)

    frame = (
        weekday_daily.with_columns(
            (pl.col("return_pct") > 0).cast(pl.Int8).alias("direction_up"),
        )
        .filter(pl.col("return_pct").is_finite())
    )
    for col in numeric_cols:
        frame = frame.filter(pl.col(col).is_finite())

    frame = frame.select(
        "day_utc",
        "pair",
        "weekday",
        "return_pct",
        "direction_up",
        *numeric_cols,
    ).sort("day_utc", "pair")

    pair_encoder = LabelEncoder()
    pair_ids = pair_encoder.fit_transform(frame["pair"].to_list())
    frame = frame.with_columns(
        pl.Series(FEATURE_PAIR_ID, pair_ids, dtype=pl.Int32).cast(pl.Utf8).cast(pl.Categorical),
    )

    weekday_encoder: LabelEncoder | None = None
    if FEATURE_WEEKDAY_ENC in spec.feature_columns:
        weekday_encoder = LabelEncoder()
        weekday_ids = weekday_encoder.fit_transform(frame["weekday"].to_list())
        frame = frame.with_columns(
            pl.Series(FEATURE_WEEKDAY_ENC, weekday_ids, dtype=pl.Int32)
            .cast(pl.Utf8)
            .cast(pl.Categorical),
        )
    else:
        weekday_encoder = LabelEncoder()
        weekday_encoder.fit(frame["weekday"].to_list())

    for col in numeric_cols:
        norm_col = frame[col]
        log.info(
            "[ml] %s: min=%.3f max=%.3f mean=%.3f",
            col,
            float(norm_col.min()),
            float(norm_col.max()),
            float(norm_col.mean()),
        )

    log.info(
        "[ml] dataset studies=%s features=%s rows=%s pairs=%s up_rate=%.3f",
        list(spec.studies),
        list(spec.feature_columns),
        frame.height,
        frame["pair"].n_unique(),
        frame["direction_up"].mean(),
    )
    return DirectionDataset(
        frame=frame,
        pairs=sorted(frame["pair"].unique().to_list()),
        pair_encoder=pair_encoder,
        weekday_encoder=weekday_encoder,
        feature_columns=spec.feature_columns,
        ml_spec=spec,
        pair_norm_bounds=resolved_bounds,
    )


def build_weekday_direction_dataset(daily: pl.DataFrame) -> DirectionDataset:
    return build_direction_dataset(daily, resolve_ml_study([ML_STUDY_DAY_OF_WEEK]))


def categorical_feature_names(feature_columns: tuple[str, ...] | list[str]) -> list[str]:
    return [col for col in feature_columns if col in CATEGORICAL_FEATURES]


def dataset_to_numpy(
    dataset: DirectionDataset,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    """X, y, prediction_times, evaluation_times в порядке сортировки frame."""
    df = dataset.frame
    x = df.select(*dataset.feature_columns).to_pandas()
    for col in dataset.feature_columns:
        if col in CATEGORICAL_FEATURES:
            x[col] = x[col].astype("category")
        else:
            x[col] = x[col].astype(np.float64)
    y = df[dataset.target_column].to_numpy()
    day_times = pd.Series(df["day_utc"].to_pandas())
    return x, y, day_times, day_times
