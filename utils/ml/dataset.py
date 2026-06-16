"""Датасет для ML: день недели → направление дневной доходности open→close."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder

from crypto_research.utils.pipeline.daily_pool import build_pooled_daily, build_weekday_daily
from crypto_research.utils.pipeline.dates import datetime_to_ms, parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import (
    _DEFAULT_WORKERS,
    load_pairs_klines,
    resolve_pairs,
)
from crypto_research.utils.pipeline.load_summary import log_load_summary
from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import FULL_POOL_FROM, FULL_POOL_TO

log = get_logger("ml_dataset")

CATEGORICAL_FEATURES: tuple[str, ...] = ("weekday_enc", "pair_id")


@dataclass(frozen=True)
class WeekdayDirectionDataset:
    frame: pl.DataFrame
    pairs: list[str]
    pair_encoder: LabelEncoder
    weekday_encoder: LabelEncoder
    feature_columns: tuple[str, ...]
    target_column: str = "direction_up"


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


def build_weekday_direction_dataset(daily: pl.DataFrame) -> WeekdayDirectionDataset:
    """Фича weekday, таргет direction_up (1 = close > open)."""
    weekday_daily = build_weekday_daily(daily).with_columns(_normalize_weekday_expr())
    frame = (
        weekday_daily.with_columns(
            (pl.col("return_pct") > 0).cast(pl.Int8).alias("direction_up"),
        )
        .filter(pl.col("return_pct").is_finite())
        .select("day_utc", "pair", "weekday", "return_pct", "direction_up")
        .sort("day_utc", "pair")
    )
    pair_encoder = LabelEncoder()
    weekday_encoder = LabelEncoder()
    pair_ids = pair_encoder.fit_transform(frame["pair"].to_list())
    weekday_ids = weekday_encoder.fit_transform(frame["weekday"].to_list())
    frame = frame.with_columns(
        pl.Series("pair_id", pair_ids, dtype=pl.Int32).cast(pl.Utf8).cast(pl.Categorical),
        pl.Series("weekday_enc", weekday_ids, dtype=pl.Int32).cast(pl.Utf8).cast(pl.Categorical),
    )
    log.info(
        "[ml] dataset: rows=%s pairs=%s weekdays=%s up_rate=%.3f",
        frame.height,
        frame["pair"].n_unique(),
        len(weekday_encoder.classes_),
        frame["direction_up"].mean(),
    )
    return WeekdayDirectionDataset(
        frame=frame,
        pairs=sorted(frame["pair"].unique().to_list()),
        pair_encoder=pair_encoder,
        weekday_encoder=weekday_encoder,
        feature_columns=CATEGORICAL_FEATURES,
    )


def dataset_to_numpy(
    dataset: WeekdayDirectionDataset,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    """X, y, prediction_times, evaluation_times в порядке сортировки frame."""
    df = dataset.frame
    x = df.select(*dataset.feature_columns).to_pandas()
    for col in dataset.feature_columns:
        if not pd.api.types.is_categorical_dtype(x[col]):
            x[col] = x[col].astype("category")
    y = df[dataset.target_column].to_numpy()
    day_times = pd.Series(df["day_utc"].to_pandas())
    return x, y, day_times, day_times
