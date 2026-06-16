#!/usr/bin/env python3
"""Оркестратор ML: weekday → direction_up, CPCV + LightGBM."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.ml.cpcv_train import (
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_N_SPLITS,
    DEFAULT_N_TEST_GROUPS,
    CPCVTrainResult,
    _default_lgbm_params,
    train_lightgbm_cpcv,
)
from crypto_research.utils.ml.dataset import (
    build_weekday_direction_dataset,
    dataset_to_numpy,
    load_full_pool_daily,
)
from crypto_research.utils.ml.oos_paths import (
    oos_calibration_metrics,
    save_oos_calibration_plot,
    save_oos_probability_plot,
    save_roc_auc_comparison_plot,
)
from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.load_pairs import _DEFAULT_WORKERS
from crypto_research.utils.pipeline.logger import add_file_logging, get_logger
from crypto_research.utils.pipeline.paths import (
    DEFAULT_DATA_DIR,
    FULL_POOL_FROM,
    FULL_POOL_TO,
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_TRAIN_TO,
    TEMPORAL_VAL_FROM,
    TEMPORAL_VAL_TO,
    VAL_MAX_PAIR_START,
    weekday_ml_log_path,
    weekday_ml_model_bundle_path,
    weekday_ml_metrics_path,
    weekday_ml_oos_calibration_plot_path,
    weekday_ml_oos_plot_path,
    weekday_ml_roc_auc_plot_path,
    weekday_ml_train_test_metrics_path,
)

log = get_logger("ml_research")
JSON_FLOAT_PRECISION = 4


def parse_ml_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML: день недели → направление дневной доходности, CPCV + LightGBM.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Папка с *_klines_1m.jsonl (по умолчанию: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--from-date",
        default=FULL_POOL_FROM,
        help=(
            f"Начало периода (ISO UTC) и отбор пар (первая свеча ≤ этой даты); "
            f"по умолчанию {FULL_POOL_FROM} (~24 пары), для пула 49 пар — 2023-01-01"
        ),
    )
    parser.add_argument(
        "--to-date",
        default=FULL_POOL_TO,
        help=f"Конец периода (ISO UTC), по умолчанию {FULL_POOL_TO}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Потоки для параллельной загрузки JSONL",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Число временных блоков CPCV (n_folds)",
    )
    parser.add_argument(
        "--n-test-groups",
        type=int,
        default=DEFAULT_N_TEST_GROUPS,
        help="Число тестовых блоков в каждой комбинации CPCV",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=DEFAULT_EMBARGO_DAYS,
        help="Embargo после тестового окна (дни, на уровне календарных дней)",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=50,
        help="Сколько первых строк датасета вывести в лог (без сжатия Polars)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON с метриками CPCV (по умолчанию research_outputs/day_of_week/ml/)",
    )
    parser.add_argument(
        "--train-test",
        "--train-val",
        dest="train_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Режим train/test (по умолчанию): CPCV на train, финальный fit, holdout на test. "
            "Один период CPCV без holdout: --no-train-test."
        ),
    )
    parser.add_argument(
        "--train-from",
        default=TEMPORAL_TRAIN_FROM,
        help=f"Начало train-периода (UTC), по умолчанию {TEMPORAL_TRAIN_FROM}",
    )
    parser.add_argument(
        "--train-to",
        default=TEMPORAL_TRAIN_TO,
        help=f"Конец train-периода (UTC), по умолчанию {TEMPORAL_TRAIN_TO}",
    )
    parser.add_argument(
        "--test-from",
        "--val-from",
        dest="test_from",
        default=TEMPORAL_VAL_FROM,
        help=f"Начало holdout test-периода (UTC), по умолчанию {TEMPORAL_VAL_FROM}",
    )
    parser.add_argument(
        "--test-to",
        "--val-to",
        dest="test_to",
        default=TEMPORAL_VAL_TO,
        help=f"Конец holdout test-периода (UTC), по умолчанию {TEMPORAL_VAL_TO}",
    )
    parser.add_argument(
        "--max-pair-start",
        default=VAL_MAX_PAIR_START,
        help=f"Отбор пула пар по первой свече (UTC), по умолчанию {VAL_MAX_PAIR_START} (49 пар).",
    )
    return parser.parse_args()


def _log_dataset_preview(frame: pl.DataFrame, *, head: int) -> None:
    log.info("Dataset columns (%d): %s", len(frame.columns), list(frame.columns))
    preview = frame.head(head)
    with pl.Config(
        tbl_rows=head,
        tbl_cols=-1,
        fmt_str_lengths=100,
        set_fmt_float="full",
    ):
        table = str(preview)
    log.info("Dataset head (%d rows):\n%s", preview.height, table)


def _round_json_floats(value, *, digits: int = JSON_FLOAT_PRECISION):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_json_floats(v, digits=digits) for v in value]
    if isinstance(value, dict):
        return {k: _round_json_floats(v, digits=digits) for k, v in value.items()}
    return value


def _save_result(result: CPCVTrainResult, path: Path, *, n_pairs: int, n_rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_pairs": n_pairs,
        "n_rows": n_rows,
        "n_splits": result.n_splits,
        "n_test_groups": result.n_test_groups,
        "n_folds": result.n_folds,
        "mean_metrics": result.mean_metrics,
        "weekday_metrics": result.weekday_metrics,
        "calibration_metrics": result.calibration_metrics,
        "fold_metrics": list(result.fold_metrics),
        "oos_paths_shape": list(result.oos_paths.shape) if result.oos_paths is not None else None,
        "oos_plot_path": str(result.oos_plot_path) if result.oos_plot_path else None,
        "oos_calibration_plot_path": (
            str(result.oos_calibration_plot_path) if result.oos_calibration_plot_path else None
        ),
    }
    path.write_text(
        json.dumps(_round_json_floats(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int8)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "n_test": int(y_true.size),
        "pred_up_rate": float(y_pred.mean()),
        "base_rate_up": float(y_true.mean()),
        "mean_p_up": float(y_prob.mean()),
        "mean_p_down": float(1.0 - y_prob.mean()),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else float("nan")
    return out


def _pooled_metrics_from_oos_paths(oos_paths: np.ndarray | None) -> dict[str, float] | None:
    if oos_paths is None or oos_paths.size == 0:
        return None
    y_prob = oos_paths[:, 1]
    y_true = oos_paths[:, 2].astype(np.int8)
    return _binary_metrics(y_true, y_prob)


def _fold_stability_table(fold_metrics: tuple[dict[str, float], ...]) -> dict[str, dict[str, float]]:
    if not fold_metrics:
        return {}
    out: dict[str, dict[str, float]] = {}
    for key in ("accuracy", "roc_auc", "log_loss"):
        values = np.array([float(m[key]) for m in fold_metrics], dtype=float)
        q1, q3 = np.percentile(values, [25, 75])
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        out[key] = {
            "mean": mean,
            "std": std,
            "min": float(values.min()),
            "max": float(values.max()),
            "iqr": float(q3 - q1),
            "cv_pct": float((std / abs(mean)) * 100.0) if mean != 0 else float("nan"),
        }
    return out


def _fold_threshold_hits(fold_metrics: tuple[dict[str, float], ...]) -> dict[str, int]:
    auc_hits = sum(1 for m in fold_metrics if float(m["roc_auc"]) > 0.5)
    acc_hits = sum(1 for m in fold_metrics if float(m["accuracy"]) > 0.5)
    return {"roc_auc_gt_0_5": auc_hits, "accuracy_gt_0_5": acc_hits, "n_folds": len(fold_metrics)}


def _weekday_holdout_metrics(oos: pl.DataFrame) -> dict[str, dict[str, float]]:
    oos_wd = oos.with_columns(((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday"))
    out: dict[str, dict[str, float]] = {}
    for wd, name in enumerate(WEEKDAY_NAMES):
        sub = oos_wd.filter(pl.col("weekday") == wd)
        if sub.is_empty():
            continue
        out[name] = _binary_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
    return out


def _group_metrics(
    oos: pl.DataFrame,
    group_col: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in oos[group_col].unique().sort().to_list():
        sub = oos.filter(pl.col(group_col) == key)
        out[str(key)] = _binary_metrics(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
    return out


def _weekday_pair_metrics(oos: pl.DataFrame) -> dict[str, dict[str, float]]:
    df = oos.with_columns(((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday"))
    grouped = df.group_by("weekday", "pair").agg(
        pl.col("y_true").alias("y_true_list"),
        pl.col("y_prob").alias("y_prob_list"),
    )
    out: dict[str, dict[str, float]] = {}
    for row in grouped.iter_rows(named=True):
        key = f"{WEEKDAY_NAMES[int(row['weekday'])]}::{row['pair']}"
        y_true = np.array(row["y_true_list"], dtype=np.int8)
        y_prob = np.array(row["y_prob_list"], dtype=float)
        out[key] = _binary_metrics(y_true, y_prob)
    return out


def _weekday_base_rate_from_frame(frame: pl.DataFrame) -> dict[str, float]:
    wd_frame = frame.with_columns(((pl.col("day_utc").dt.weekday() - 1) % 7).alias("weekday"))
    out: dict[str, float] = {}
    for wd, name in enumerate(WEEKDAY_NAMES):
        sub = wd_frame.filter(pl.col("weekday") == wd)
        if sub.is_empty():
            continue
        out[name] = float(sub["direction_up"].mean())
    return out


def _attach_train_and_test_base_rates(
    weekday_metrics: dict[str, dict[str, float]] | None,
    *,
    train_weekday_base: dict[str, float],
) -> dict[str, dict[str, float]] | None:
    if weekday_metrics is None:
        return None
    out: dict[str, dict[str, float]] = {}
    for name, metrics in weekday_metrics.items():
        m = dict(metrics)
        test_rate = m.pop("base_rate_up", None)
        m["base_rate_up_test"] = float(test_rate) if test_rate is not None else float("nan")
        m["base_rate_up_train"] = float(train_weekday_base.get(name, float("nan")))
        out[name] = m
    return out


def _default_train_test_output_path(n_pairs: int, train_from, test_to) -> Path:
    return weekday_ml_train_test_metrics_path(n_pairs, train_from, test_to)


def run_ml_train_test_pipeline(args: argparse.Namespace) -> Path:
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    pair_start_limit = parse_iso_utc(args.max_pair_start)
    train_from = parse_iso_utc(args.train_from)
    train_to = parse_iso_utc(args.train_to)
    test_from = parse_iso_utc(args.test_from)
    test_to = parse_iso_utc(args.test_to)

    train_daily, train_pairs = load_full_pool_daily(
        args.data_dir.expanduser().resolve(),
        max_pair_start=pair_start_limit,
        from_date=train_from,
        to_date=train_to,
        workers=workers,
    )
    test_daily, test_pairs = load_full_pool_daily(
        args.data_dir.expanduser().resolve(),
        max_pair_start=pair_start_limit,
        from_date=test_from,
        to_date=test_to,
        workers=workers,
    )
    if train_pairs != test_pairs:
        raise RuntimeError("Train/test загрузили разный пул пар; проверьте max-pair-start и data-dir")

    train_dataset = build_weekday_direction_dataset(train_daily)
    test_dataset = build_weekday_direction_dataset(test_daily)
    train_weekday_base = _weekday_base_rate_from_frame(train_dataset.frame)
    _log_dataset_preview(train_dataset.frame, head=args.preview_rows)

    train_cpcv = train_lightgbm_cpcv(
        train_dataset,
        n_splits=args.n_splits,
        n_test_groups=args.n_test_groups,
        embargo_days=args.embargo_days,
        oos_plot_path=weekday_ml_oos_plot_path(len(train_pairs), train_from, train_to),
        oos_calibration_plot_path=weekday_ml_oos_calibration_plot_path(len(train_pairs), train_from, train_to),
    )

    x_train, y_train, _, _ = dataset_to_numpy(train_dataset)
    x_test, y_test, _, _ = dataset_to_numpy(test_dataset)
    model = lgb.LGBMClassifier(**_default_lgbm_params())
    model.fit(x_train, y_train, categorical_feature=list(train_dataset.feature_columns))
    y_train_prob = model.predict_proba(x_train)[:, 1]
    y_test_prob = model.predict_proba(x_test)[:, 1]

    train_fit_oos = train_dataset.frame.select("day_utc", "pair", "direction_up").rename(
        {"direction_up": "y_true"}
    ).with_columns(
        pl.Series("y_prob", y_train_prob),
        pl.lit(1).alias("n_folds"),
    )

    test_oos = test_dataset.frame.select("day_utc", "pair", "direction_up").rename(
        {"direction_up": "y_true"}
    ).with_columns(
        pl.Series("y_prob", y_test_prob),
        pl.lit(1).alias("n_folds"),
    )

    test_plot_path = save_oos_probability_plot(
        test_oos,
        weekday_ml_oos_plot_path(len(train_pairs), test_from, test_to),
    )
    test_calibration_plot_path = save_oos_calibration_plot(
        test_oos,
        weekday_ml_oos_calibration_plot_path(len(train_pairs), test_from, test_to),
    )
    train_oos = train_cpcv.oos_paths
    if train_oos is None:
        raise RuntimeError("Train CPCV не вернул OOS-предсказания для ROC AUC plot")
    roc_auc_plot_path = save_roc_auc_comparison_plot(
        train_oos[:, 2].astype(np.int8),
        train_oos[:, 1],
        y_test,
        y_test_prob,
        weekday_ml_roc_auc_plot_path(len(train_pairs), train_from, test_to),
    )
    test_metrics = _binary_metrics(y_test, y_test_prob)
    train_fit_metrics = _binary_metrics(y_train, y_train_prob)
    train_fit_calibration = oos_calibration_metrics(train_fit_oos)
    test_calibration = oos_calibration_metrics(test_oos)
    test_weekday = _weekday_holdout_metrics(test_oos)
    train_weekday_metrics = _attach_train_and_test_base_rates(
        train_cpcv.weekday_metrics,
        train_weekday_base=train_weekday_base,
    )
    test_weekday = _attach_train_and_test_base_rates(
        test_weekday,
        train_weekday_base=train_weekday_base,
    )

    model_path = weekday_ml_model_bundle_path(len(train_pairs), train_from, train_to)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "feature_columns": list(train_dataset.feature_columns),
        "pair_encoder_classes": train_dataset.pair_encoder.classes_.tolist(),
        "weekday_encoder_classes": train_dataset.weekday_encoder.classes_.tolist(),
        "train_period": {"from": args.train_from, "to": args.train_to},
    }
    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    payload = {
        "mode": "train_test",
        "n_pairs": len(train_pairs),
        "pair_start_limit": args.max_pair_start,
        "train_period": {"from": args.train_from, "to": args.train_to},
        "holdout_test_period": {"from": args.test_from, "to": args.test_to},
        "train_cpcv": {
            "n_rows": train_dataset.frame.height,
            "n_splits": train_cpcv.n_splits,
            "n_test_groups": train_cpcv.n_test_groups,
            "n_folds": train_cpcv.n_folds,
            "mean_metrics": train_cpcv.mean_metrics,
            "pooled_oos_metrics": _pooled_metrics_from_oos_paths(train_cpcv.oos_paths),
            "fold_stability": _fold_stability_table(train_cpcv.fold_metrics),
            "fold_threshold_hits": _fold_threshold_hits(train_cpcv.fold_metrics),
            "weekday_metrics": train_weekday_metrics,
            "pair_metrics": _group_metrics(train_cpcv.oos_predictions, "pair") if train_cpcv.oos_predictions is not None else {},
            "weekday_pair_metrics": _weekday_pair_metrics(train_cpcv.oos_predictions) if train_cpcv.oos_predictions is not None else {},
            "calibration_metrics": train_cpcv.calibration_metrics,
            "fold_metrics": list(train_cpcv.fold_metrics),
            "oos_plot_path": str(train_cpcv.oos_plot_path) if train_cpcv.oos_plot_path else None,
            "oos_calibration_plot_path": (
                str(train_cpcv.oos_calibration_plot_path) if train_cpcv.oos_calibration_plot_path else None
            ),
            "roc_auc_plot_path": str(roc_auc_plot_path),
        },
        "final_model_path": str(model_path),
        "train_fit": {
            "n_rows": train_dataset.frame.height,
            "metrics": train_fit_metrics,
            "calibration_metrics": train_fit_calibration,
            "pair_metrics": _group_metrics(train_fit_oos, "pair"),
            "weekday_pair_metrics": _weekday_pair_metrics(train_fit_oos),
        },
        "holdout_test": {
            "n_rows": test_dataset.frame.height,
            "metrics": test_metrics,
            "weekday_metrics": test_weekday,
            "pair_metrics": _group_metrics(test_oos, "pair"),
            "weekday_pair_metrics": _weekday_pair_metrics(test_oos),
            "calibration_metrics": test_calibration,
            "oos_plot_path": str(test_plot_path),
            "oos_calibration_plot_path": str(test_calibration_plot_path),
        },
    }
    out_path = args.output or _default_train_test_output_path(len(train_pairs), train_from, test_to)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_round_json_floats(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[ml] train/test metrics saved: %s", out_path)
    return out_path


def run_ml_pipeline(args: argparse.Namespace) -> CPCVTrainResult:
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    from_date = parse_iso_utc(args.from_date)
    to_date = parse_iso_utc(args.to_date)

    daily, pairs = load_full_pool_daily(
        args.data_dir.expanduser().resolve(),
        from_date=from_date,
        to_date=to_date,
        workers=workers,
    )
    log.info("Loaded %d pairs with %d daily rows", len(pairs), daily.height)
    dataset = build_weekday_direction_dataset(daily)
    _log_dataset_preview(dataset.frame, head=args.preview_rows)
    result = train_lightgbm_cpcv(
        dataset,
        n_splits=args.n_splits,
        n_test_groups=args.n_test_groups,
        embargo_days=args.embargo_days,
        oos_plot_path=weekday_ml_oos_plot_path(len(pairs), from_date, to_date),
        oos_calibration_plot_path=weekday_ml_oos_calibration_plot_path(
            len(pairs), from_date, to_date
        ),
    )

    out_path = args.output
    if out_path is None:
        out_path = weekday_ml_metrics_path(len(pairs), from_date, to_date)
    saved = _save_result(result, out_path, n_pairs=len(pairs), n_rows=dataset.frame.height)
    log.info("[ml] metrics saved: %s", saved)
    return result


def main() -> int:
    args = parse_ml_args()
    if args.train_test:
        from_date = parse_iso_utc(args.train_from)
        to_date = parse_iso_utc(args.test_to)
    else:
        from_date = parse_iso_utc(args.from_date)
        to_date = parse_iso_utc(args.to_date)
    log_file = add_file_logging(weekday_ml_log_path(from_date, to_date))
    log.info("[ml] log file: %s", log_file)
    if args.train_test:
        run_ml_train_test_pipeline(args)
    else:
        run_ml_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
