"""Загрузка frozen ML-моделей и holdout OOS для сравнения."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from crypto_research.utils.ml.dataset import build_direction_dataset, dataset_to_numpy
from crypto_research.utils.ml.numeric_features import bounds_map_from_bundle
from crypto_research.utils.ml.registry import MlStudySpec, resolve_compare_model
from crypto_research.utils.pipeline.paths import ml_model_bundle_path, ml_policy_path


@dataclass(frozen=True)
class CompareModelEntry:
    model_id: str
    spec: MlStudySpec
    oos: pl.DataFrame
    t_long: float
    t_short: float


def _weekday_expr() -> pl.Expr:
    return ((pl.col("day_utc").dt.weekday() - 1) % 7).cast(pl.Int64).alias("weekday")


def _load_policy_thresholds(policy_path: Path, spec: MlStudySpec) -> tuple[float, float]:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        spec.policy_mode == "global"
        and "global_thresholds" in payload
        and "selected_pairs_global" in payload
    ):
        th = payload["global_thresholds"]
        return float(th["t_long"]), float(th["t_short"])
    th = payload["thresholds"]
    return float(th["t_long"]), float(th["t_short"])


def predict_holdout_oos(
    daily: pl.DataFrame,
    *,
    spec: MlStudySpec,
    bundle_path: Path,
) -> pl.DataFrame:
    with bundle_path.open("rb") as f:
        bundle = pickle.load(f)
    bounds = bounds_map_from_bundle(bundle, spec.feature_columns)
    dataset = build_direction_dataset(daily, spec, pair_norm_bounds=bounds)
    x, _, _, _ = dataset_to_numpy(dataset)
    model = bundle["model"]
    y_prob = model.predict_proba(x)[:, 1]
    return (
        dataset.frame.select("day_utc", "pair", "direction_up")
        .rename({"direction_up": "y_true"})
        .with_columns(
            pl.Series("y_prob", y_prob),
            _weekday_expr(),
        )
        .sort("day_utc", "pair")
    )


def load_compare_model(
    model_id: str,
    daily: pl.DataFrame,
    *,
    n_pairs: int,
    train_from: datetime,
    train_to: datetime,
    test_to: datetime,
    bundle_path: Path | None = None,
    policy_path: Path | None = None,
) -> CompareModelEntry:
    spec = resolve_compare_model(model_id)
    bundle_path = bundle_path or ml_model_bundle_path(spec, n_pairs, train_from, train_to)
    policy_path = policy_path or ml_policy_path(spec, n_pairs, train_from, test_to)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"model bundle не найден: {bundle_path}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy не найден: {policy_path}")
    t_long, t_short = _load_policy_thresholds(policy_path, spec)
    oos = predict_holdout_oos(daily, spec=spec, bundle_path=bundle_path)
    return CompareModelEntry(
        model_id=model_id,
        spec=spec,
        oos=oos,
        t_long=t_long,
        t_short=t_short,
    )


def align_probabilities(base: pl.DataFrame, other: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    merged = base.select("day_utc", "pair", pl.col("y_prob").alias("p_base")).join(
        other.select("day_utc", "pair", pl.col("y_prob").alias("p_other")),
        on=["day_utc", "pair"],
        how="inner",
    )
    return merged["p_base"].to_numpy(), merged["p_other"].to_numpy()


def signal_rates(y_prob: np.ndarray, t_long: float, t_short: float) -> dict[str, float]:
    long_m = y_prob >= t_long
    short_m = y_prob <= t_short
    flat_m = ~(long_m | short_m)
    n = max(int(y_prob.size), 1)
    return {
        "long": float(long_m.sum()) / n,
        "flat": float(flat_m.sum()) / n,
        "short": float(short_m.sum()) / n,
    }
