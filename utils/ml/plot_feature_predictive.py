#!/usr/bin/env python3
"""Построить график ROC AUC / accuracy vs непрерывный признак (train + val) из bundle."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

_REPO_PARENT = Path(__file__).resolve().parents[3]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.ml.dataset import (
    build_direction_dataset,
    dataset_to_numpy,
    load_full_pool_daily,
)
from crypto_research.utils.ml.ema_dev_norm import pair_ema_dev_bounds_from_dict
from crypto_research.utils.ml.feature_predictive_plot import save_feature_curve_plot
from crypto_research.utils.ml.spec import FEATURE_EMA_DEV_PAIR_NORM, resolve_ml_study
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import _DEFAULT_WORKERS
from crypto_research.utils.pipeline.paths import (
    DEFAULT_DATA_DIR,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
    VAL_MAX_PAIR_START,
    ml_feature_predictive_plot_path,
    ml_model_bundle_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="График ML vs непрерывный признак (train + val).")
    parser.add_argument("--model-bundle", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-from", default=TEMPORAL_TRAIN_FROM)
    parser.add_argument("--train-to", default=TEMPORAL_TRAIN_TO)
    parser.add_argument("--test-from", default=TEMPORAL_VAL_FROM)
    parser.add_argument("--test-to", default=TEMPORAL_VAL_TO)
    parser.add_argument("--max-pair-start", default=VAL_MAX_PAIR_START)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _predict_oos(daily, spec, bounds, model) -> pl.DataFrame:
    dataset = build_direction_dataset(daily, spec, pair_ema_dev_bounds=bounds)
    x, _, _, _ = dataset_to_numpy(dataset)
    y_prob = model.predict_proba(x)[:, 1]
    cols = ["day_utc", "pair", "direction_up"]
    if spec.predictive_feature is not None:
        cols.append(spec.predictive_feature)
    return dataset.frame.select(*cols).rename({"direction_up": "y_true"}).with_columns(
        pl.Series("y_prob", y_prob)
    )


def main() -> int:
    args = parse_args()
    spec = resolve_ml_study(["ema_spreads_ml"])
    if not spec.predictive_feature or not spec.predictive_plot_slug:
        raise SystemExit("Исследование не задаёт predictive_feature")
    train_from = parse_iso_utc(args.train_from)
    train_to = parse_iso_utc(args.train_to)
    test_from = parse_iso_utc(args.test_from)
    test_to = parse_iso_utc(args.test_to)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    data_dir = args.data_dir.expanduser().resolve()
    pair_start = parse_iso_utc(args.max_pair_start)

    bundle_path = args.model_bundle or ml_model_bundle_path(spec, 49, train_from, train_to)
    with bundle_path.open("rb") as f:
        bundle = pickle.load(f)
    model: lgb.LGBMClassifier = bundle["model"]
    bounds = (
        pair_ema_dev_bounds_from_dict(bundle["pair_ema_dev_bounds"])
        if FEATURE_EMA_DEV_PAIR_NORM in spec.feature_columns
        else None
    )

    train_daily, _ = load_full_pool_daily(
        data_dir, max_pair_start=pair_start, from_date=train_from, to_date=train_to, workers=workers
    )
    test_daily, test_pairs = load_full_pool_daily(
        data_dir, max_pair_start=pair_start, from_date=test_from, to_date=test_to, workers=workers
    )
    train_oos = _predict_oos(train_daily, spec, bounds, model)
    val_oos = _predict_oos(test_daily, spec, bounds, model)

    out = args.output or ml_feature_predictive_plot_path(
        spec, len(test_pairs), test_from, test_to, spec.predictive_plot_slug
    )
    save_feature_curve_plot(
        train_oos,
        val_oos,
        out,
        feature_column=spec.predictive_feature,
        train_title=f"Train {args.train_from}..{args.train_to}",
        val_title=f"Val {args.test_from}..{args.test_to}",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
