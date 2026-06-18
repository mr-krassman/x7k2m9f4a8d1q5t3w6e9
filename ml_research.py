#!/usr/bin/env python3
"""Оркестратор ML: weekday → direction_up, CPCV + LightGBM."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

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
    train_lightgbm_cpcv,
)
from crypto_research.utils.ml.dataset import (
    DirectionDataset,
    build_direction_dataset,
    build_weekday_direction_dataset,
    dataset_to_numpy,
    load_full_pool_daily,
)
from crypto_research.utils.ml.numeric_features import (
    active_numeric_specs,
    bounds_map_from_bundle,
    bounds_map_to_bundle,
    fit_bounds_for_features,
    resolve_predictive,
)
from crypto_research.utils.ml.plot_registry import (
    ML_PLOT_CHOICES,
    ML_PLOT_FEATURE_PREDICTIVE,
    ML_PLOT_ROC_AUC,
    ML_PLOT_WEEKDAY_PAIR_SUMMARY,
    MlPlotContext,
    resolve_plot_ids,
    run_selected_ml_plots,
)
from crypto_research.utils.ml.learning_curve import (
    fit_lightgbm_full_train,
    fit_lightgbm_with_eval_curve,
)
from crypto_research.utils.ml.registry import (
    COMPARE_MODEL_CHOICES,
    ML_STUDY_CHOICES,
    ml_spec_to_dict,
    resolve_ml_study,
)
from crypto_research.utils.ml.model_compare import load_compare_model, signal_rates
from crypto_research.utils.ml.model_compare_plot import save_all_compare_plots
from crypto_research.utils.ml.oos_paths import (
    oos_calibration_metrics,
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
    TRAIN_EVAL_FROM,
    VAL_MAX_PAIR_START,
    ml_feature_predictive_plot_path,
    ml_learning_curve_log_path,
    ml_learning_curve_plot_path,
    ml_log_path,
    ml_model_bundle_path,
    ml_metrics_path,
    ml_oos_calibration_plot_path,
    ml_oos_plot_path,
    ml_plots_dir,
    ml_train_test_metrics_path,
    ml_compare_dir,
)

log = get_logger("ml_research")
JSON_FLOAT_PRECISION = 4


def parse_ml_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML: direction_up, CPCV + LightGBM (day_of_week_ml / ema_spreads_ml / rsi_spreads_ml).",
    )
    parser.add_argument(
        "studies",
        nargs="*",
        metavar="STUDY",
        help="Исследования: day_of_week_ml, ema_spreads_ml, rsi_spreads_ml; combined: day_of_week_ml ema_spreads_ml [rsi_spreads_ml]",
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
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Early stopping на хвосте train перед финальным fit (по умолчанию выкл.)",
    )
    parser.add_argument(
        "--train-eval-from",
        default=TRAIN_EVAL_FROM,
        help=f"Начало eval-хвоста train для --early-stopping (UTC), по умолчанию {TRAIN_EVAL_FROM}",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Раунды без улучшения eval logloss при --early-stopping",
    )
    parser.add_argument("--n-pairs", type=int, default=49, help="Размер пула пар (пути bundle/policy).")
    parser.add_argument(
        "--compare-models",
        nargs="+",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Сравнить frozen-модели на holdout test (без обучения). "
            f"ID: {', '.join(COMPARE_MODEL_CHOICES)}. "
            "Пример: --compare-models dow_ema_sp dow_ema_rsi_sp"
        ),
    )
    parser.add_argument(
        "--compare-baseline",
        default=None,
        help="Первая модель в списке для ΔP(up); по умолчанию — первый аргумент --compare-models.",
    )
    parser.add_argument(
        "--compare-model-bundle",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Переопределить путь к bundle: ema_spreads_ml=/path/to.pkl",
    )
    parser.add_argument(
        "--compare-policy",
        action="append",
        default=[],
        metavar="MODEL_ID=PATH",
        help="Переопределить путь к policy.json.",
    )
    parser.add_argument(
        "--compare-output-dir",
        type=Path,
        default=None,
        help="Каталог для графиков сравнения (по умолчанию research_outputs/ml_compare/…).",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=None,
        metavar="PLOT",
        help=(
            "Какие графики строить (по умолчанию — стандартный набор holdout). "
            f"ID: {', '.join(ML_PLOT_CHOICES)}; "
            "алиас: тепловая_карта_корреляционной_матрицы. "
            "Пример: --plots correlation_matrix_heatmap shape_summary_plot"
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Только графики по frozen bundle (без обучения); обязателен --plots.",
    )
    parser.add_argument(
        "--plot-metrics-over-feature",
        "--plot_metrics_over_feature",
        dest="plot_metrics_over_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "График ROC AUC / accuracy по бинам predictive-фичи (train-fit | holdout test). "
            "Для ema_spreads_ml — ema_dev_pair_norm, для rsi_spreads_ml — rsi_pair_norm."
        ),
    )
    return parser.parse_args()


def _validate_ml_args(args: argparse.Namespace) -> None:
    if args.compare_models:
        return
    if args.plots_only and not args.plots:
        raise SystemExit("--plots-only требует --plots с именами графиков.")
    if not args.studies:
        raise SystemExit(
            "Укажите STUDY (day_of_week_ml, …) или --compare-models для сравнения frozen-моделей."
        )
    unknown = set(args.studies) - set(ML_STUDY_CHOICES)
    if unknown:
        raise SystemExit(f"Неизвестные ML-исследования: {sorted(unknown)}")
    if args.plots:
        resolve_plot_ids(args.plots)


def _effective_plot_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.plots:
        return resolve_plot_ids(args.plots)
    plot_ids = list(resolve_plot_ids(None))
    if args.plot_metrics_over_feature and ML_PLOT_FEATURE_PREDICTIVE not in plot_ids:
        plot_ids.append(ML_PLOT_FEATURE_PREDICTIVE)
    return tuple(plot_ids)


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


def _save_result(
    result: CPCVTrainResult,
    path: Path,
    *,
    n_pairs: int,
    n_rows: int,
    spec,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ml_spec": ml_spec_to_dict(spec),
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


def _model_oos_frame(dataset: DirectionDataset, y_prob: np.ndarray) -> pl.DataFrame:
    cols = ["day_utc", "pair", "direction_up"]
    for ns in active_numeric_specs(dataset.feature_columns):
        if ns.column not in cols:
            cols.append(ns.column)
    return dataset.frame.select(*cols).rename({"direction_up": "y_true"}).with_columns(
        pl.Series("y_prob", y_prob),
        pl.lit(1).alias("n_folds"),
    )


def _model_feature_importance(model, feature_columns: list[str]) -> dict[str, float]:
    names = list(getattr(model, "feature_name_", feature_columns))
    gains = list(getattr(model, "feature_importances_", []))
    return {name: float(gain) for name, gain in zip(names, gains)}


def _default_train_test_output_path(spec, n_pairs: int, train_from, test_to) -> Path:
    return ml_train_test_metrics_path(spec, n_pairs, train_from, test_to)


def _load_frozen_model_context(args: argparse.Namespace) -> MlPlotContext:
    spec = resolve_ml_study(args.studies)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    pair_start_limit = parse_iso_utc(args.max_pair_start)
    train_from = parse_iso_utc(args.train_from)
    train_to = parse_iso_utc(args.train_to)
    test_from = parse_iso_utc(args.test_from)
    test_to = parse_iso_utc(args.test_to)

    model_path = ml_model_bundle_path(spec, args.n_pairs, train_from, train_to)
    if not model_path.is_file():
        raise FileNotFoundError(f"model bundle не найден: {model_path}")
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

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
        raise RuntimeError("Train/test загрузили разный пул пар")

    bounds = bounds_map_from_bundle(bundle, spec.feature_columns)
    train_dataset = build_direction_dataset(train_daily, spec, pair_norm_bounds=bounds)
    test_dataset = build_direction_dataset(test_daily, spec, pair_norm_bounds=bounds)
    model = bundle["model"]
    x_train, y_train, _, _ = dataset_to_numpy(train_dataset)
    x_test, y_test, _, _ = dataset_to_numpy(test_dataset)
    y_train_prob = model.predict_proba(x_train)[:, 1]
    y_test_prob = model.predict_proba(x_test)[:, 1]

    return MlPlotContext(
        spec=spec,
        args=args,
        n_pairs=len(train_pairs),
        train_from=train_from,
        train_to=train_to,
        test_from=test_from,
        test_to=test_to,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_fit_oos=_model_oos_frame(train_dataset, y_train_prob),
        test_oos=_model_oos_frame(test_dataset, y_test_prob),
        y_test=y_test,
        y_test_prob=y_test_prob,
    )


def run_ml_plots_only_pipeline(args: argparse.Namespace) -> dict[str, object]:
    plot_ids = _effective_plot_ids(args)
    ctx = _load_frozen_model_context(args)
    plot_paths = run_selected_ml_plots(plot_ids, ctx)
    summary_path = ml_plots_dir(ctx.spec) / "plots_only_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "plots_only",
        "ml_spec": ml_spec_to_dict(ctx.spec),
        "plot_ids": list(plot_ids),
        "holdout_test_period": {"from": args.test_from, "to": args.test_to},
        "n_pairs": ctx.n_pairs,
        "plot_paths": plot_paths,
    }
    summary_path.write_text(
        json.dumps(_round_json_floats(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[ml] plots-only summary saved: %s", summary_path)
    return plot_paths


def _build_model_bundle(
    model,
    dataset: DirectionDataset,
    *,
    train_period: dict[str, str],
    early_stopping: dict[str, object] | None = None,
) -> dict[str, object]:
    bundle: dict[str, object] = {
        "model": model,
        "feature_columns": list(dataset.feature_columns),
        "ml_spec": ml_spec_to_dict(dataset.ml_spec),
        "pair_encoder_classes": dataset.pair_encoder.classes_.tolist(),
        "train_period": train_period,
    }
    if dataset.weekday_encoder is not None and "weekday_enc" in dataset.feature_columns:
        bundle["weekday_encoder_classes"] = dataset.weekday_encoder.classes_.tolist()
    if dataset.ema_period is not None:
        bundle["ema_period"] = dataset.ema_period
    if dataset.rsi_period is not None:
        bundle["rsi_period"] = dataset.rsi_period
    bundle.update(bounds_map_to_bundle(dataset.pair_norm_bounds))
    if early_stopping is not None:
        bundle["early_stopping"] = early_stopping
    return bundle


def run_ml_train_test_pipeline(args: argparse.Namespace) -> Path:
    spec = resolve_ml_study(args.studies)
    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    pair_start_limit = parse_iso_utc(args.max_pair_start)
    train_from = parse_iso_utc(args.train_from)
    train_to = parse_iso_utc(args.train_to)
    eval_from = parse_iso_utc(args.train_eval_from)
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
    if args.early_stopping:
        if eval_from <= train_from or eval_from >= train_to:
            raise RuntimeError(
                f"--train-eval-from ({args.train_eval_from}) должен быть строго внутри train "
                f"({args.train_from} .. {args.train_to})"
            )

    pair_bounds = fit_bounds_for_features(train_daily, spec.feature_columns)
    train_dataset = build_direction_dataset(train_daily, spec, pair_norm_bounds=pair_bounds)
    test_dataset = build_direction_dataset(test_daily, spec, pair_norm_bounds=pair_bounds)
    train_weekday_base = _weekday_base_rate_from_frame(train_dataset.frame)
    _log_dataset_preview(train_dataset.frame, head=args.preview_rows)

    train_cpcv = train_lightgbm_cpcv(
        train_dataset,
        n_splits=args.n_splits,
        n_test_groups=args.n_test_groups,
        embargo_days=args.embargo_days,
        oos_plot_path=ml_oos_plot_path(spec, len(train_pairs), train_from, train_to),
        oos_calibration_plot_path=ml_oos_calibration_plot_path(spec, len(train_pairs), train_from, train_to),
    )

    x_train, y_train, _, _ = dataset_to_numpy(train_dataset)
    x_test, y_test, _, _ = dataset_to_numpy(test_dataset)
    if args.early_stopping:
        fit_result = fit_lightgbm_with_eval_curve(
            train_dataset,
            eval_from=eval_from,
            eval_to=train_to,
            plot_path=ml_learning_curve_plot_path(spec, len(train_pairs), train_from, train_to),
            log_path=ml_learning_curve_log_path(spec, len(train_pairs), train_from, train_to),
            early_stopping_rounds=args.early_stopping_rounds,
        )
    else:
        fit_result = fit_lightgbm_full_train(train_dataset)
    model = fit_result.model
    y_train_prob = model.predict_proba(x_train)[:, 1]
    y_test_prob = model.predict_proba(x_test)[:, 1]

    train_fit_oos = _model_oos_frame(train_dataset, y_train_prob)
    test_oos = _model_oos_frame(test_dataset, y_test_prob)

    plot_ids = _effective_plot_ids(args)
    if ML_PLOT_ROC_AUC in plot_ids and train_cpcv.oos_paths is None:
        raise RuntimeError("Train CPCV не вернул OOS-предсказания для roc_auc")
    if ML_PLOT_WEEKDAY_PAIR_SUMMARY in plot_ids and train_cpcv.oos_predictions is None:
        raise RuntimeError("Train CPCV не вернул OOS-предсказания для weekday_pair_summary")

    plot_ctx = MlPlotContext(
        spec=spec,
        args=args,
        n_pairs=len(train_pairs),
        train_from=train_from,
        train_to=train_to,
        test_from=test_from,
        test_to=test_to,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_fit_oos=train_fit_oos,
        test_oos=test_oos,
        y_test=y_test,
        y_test_prob=y_test_prob,
        train_cpcv=train_cpcv,
        fit_result=fit_result,
    )
    plot_paths = run_selected_ml_plots(plot_ids, plot_ctx)
    feature_predictive_plot_paths = {
        k.removeprefix("feature_predictive_"): str(v)
        for k, v in plot_paths.items()
        if k.startswith("feature_predictive_")
    }
    feature_predictive_plot_path = None
    if spec.predictive_plot_slug and spec.predictive_plot_slug in feature_predictive_plot_paths:
        feature_predictive_plot_path = feature_predictive_plot_paths[spec.predictive_plot_slug]
    elif feature_predictive_plot_paths:
        feature_predictive_plot_path = next(iter(feature_predictive_plot_paths.values()))

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

    model_path = ml_model_bundle_path(spec, len(train_pairs), train_from, train_to)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    early_stopping_meta = None
    if args.early_stopping:
        early_stopping_meta = {
            "eval_period": fit_result.eval_period,
            "best_iteration": fit_result.best_iteration,
            "n_fit_rows": fit_result.n_fit_rows,
            "n_eval_rows": fit_result.n_eval_rows,
            "learning_curve_plot_path": (
                str(fit_result.learning_curve_plot_path) if fit_result.learning_curve_plot_path else None
            ),
            "learning_curve_log_path": (
                str(fit_result.learning_curve_log_path) if fit_result.learning_curve_log_path else None
            ),
        }
    model_bundle = _build_model_bundle(
        model,
        train_dataset,
        train_period={"from": args.train_from, "to": args.train_to},
        early_stopping=early_stopping_meta,
    )
    with model_path.open("wb") as f:
        pickle.dump(model_bundle, f)

    payload = {
        "mode": "train_test",
        "ml_spec": ml_spec_to_dict(spec),
        "n_pairs": len(train_pairs),
        "pair_start_limit": args.max_pair_start,
        "train_period": {"from": args.train_from, "to": args.train_to},
        "holdout_test_period": {"from": args.test_from, "to": args.test_to},
        "plot_ids": list(plot_ids),
        "plot_paths": plot_paths,
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
            "roc_auc_plot_path": plot_paths.get("roc_auc"),
            "weekday_pair_summary_plot_path": plot_paths.get("weekday_pair_summary_train"),
        },
        "final_model_path": str(model_path),
        "feature_importance": _model_feature_importance(model, list(train_dataset.feature_columns)),
        "train_fit": {
            "n_rows": train_dataset.frame.height,
            "n_estimators": fit_result.best_iteration,
            "early_stopping": early_stopping_meta,
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
            "oos_plot_path": plot_paths.get("oos_prob"),
            "oos_calibration_plot_path": plot_paths.get("oos_calibration"),
            "weekday_pair_summary_plot_path": plot_paths.get("weekday_pair_summary"),
            "correlation_matrix_plot_path": plot_paths.get("correlation_matrix_heatmap"),
            "shape_summary_plot_path": plot_paths.get("shape_summary_plot"),
            "feature_correlations": plot_paths.get("feature_correlations"),
            "feature_predictive_plot_path": (
                str(feature_predictive_plot_path) if feature_predictive_plot_path else None
            ),
            "feature_predictive_plot_paths": feature_predictive_plot_paths or None,
        },
    }
    out_path = args.output or _default_train_test_output_path(spec, len(train_pairs), train_from, test_to)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_round_json_floats(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[ml] train/test metrics saved: %s", out_path)
    return out_path


def _parse_path_overrides(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Ожидается MODEL_ID=PATH, получено: {item!r}")
        model_id, raw = item.split("=", 1)
        out[model_id.strip()] = Path(raw.strip()).expanduser()
    return out


def run_compare_models_pipeline(args: argparse.Namespace) -> dict[str, str]:
    if not args.compare_models or len(args.compare_models) < 2:
        raise RuntimeError("--compare-models: укажите минимум 2 модели")

    model_ids = list(dict.fromkeys(args.compare_models))
    if args.compare_baseline is not None:
        if args.compare_baseline not in model_ids:
            raise RuntimeError("--compare-baseline должен быть среди --compare-models")
        model_ids = [args.compare_baseline] + [m for m in model_ids if m != args.compare_baseline]

    workers = args.workers if args.workers is not None else _DEFAULT_WORKERS
    pair_start_limit = parse_iso_utc(args.max_pair_start)
    train_from = parse_iso_utc(args.train_from)
    train_to = parse_iso_utc(args.train_to)
    test_from = parse_iso_utc(args.test_from)
    test_to = parse_iso_utc(args.test_to)
    n_pairs = args.n_pairs

    bundle_overrides = _parse_path_overrides(args.compare_model_bundle)
    policy_overrides = _parse_path_overrides(args.compare_policy)

    test_daily, test_pairs = load_full_pool_daily(
        args.data_dir.expanduser().resolve(),
        max_pair_start=pair_start_limit,
        from_date=test_from,
        to_date=test_to,
        workers=workers,
    )
    log.info("[ml] compare holdout: pairs=%s rows=%s", len(test_pairs), test_daily.height)

    entries = []
    for model_id in model_ids:
        entry = load_compare_model(
            model_id,
            test_daily,
            n_pairs=n_pairs,
            train_from=train_from,
            train_to=train_to,
            test_to=test_to,
            bundle_path=bundle_overrides.get(model_id),
            policy_path=policy_overrides.get(model_id),
        )
        y_prob = entry.oos["y_prob"].to_numpy()
        log.info(
            "[ml] compare %s: rows=%s mean_p_up=%.4f pred_up_rate=%.4f t_long=%.4f t_short=%.4f",
            model_id,
            entry.oos.height,
            float(np.mean(y_prob)),
            float((y_prob >= 0.5).mean()),
            entry.t_long,
            entry.t_short,
        )
        entries.append(entry)

    plots_root = args.compare_output_dir or ml_compare_dir(model_ids, test_from, test_to)
    plots_paths = save_all_compare_plots(entries, plots_root / "plots")
    summary_path = plots_root / "compare_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_ids": model_ids,
        "holdout_test_period": {"from": args.test_from, "to": args.test_to},
        "n_pairs": len(test_pairs),
        "models": [
            {
                "model_id": e.model_id,
                "t_long": e.t_long,
                "t_short": e.t_short,
                "mean_p_up": float(e.oos["y_prob"].mean()),
                "pred_up_rate": float((e.oos["y_prob"].to_numpy() >= 0.5).mean()),
                **{
                    f"signal_{k}": v
                    for k, v in signal_rates(
                        e.oos["y_prob"].to_numpy(), e.t_long, e.t_short
                    ).items()
                },
            }
            for e in entries
        ],
        "plot_paths": plots_paths,
    }
    summary_path.write_text(
        json.dumps(_round_json_floats(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[ml] compare summary saved: %s", summary_path)
    return plots_paths


def run_ml_pipeline(args: argparse.Namespace) -> CPCVTrainResult:
    spec = resolve_ml_study(args.studies)
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
    pair_bounds = fit_bounds_for_features(daily, spec.feature_columns)
    dataset = build_direction_dataset(daily, spec, pair_norm_bounds=pair_bounds)
    _log_dataset_preview(dataset.frame, head=args.preview_rows)
    result = train_lightgbm_cpcv(
        dataset,
        n_splits=args.n_splits,
        n_test_groups=args.n_test_groups,
        embargo_days=args.embargo_days,
        oos_plot_path=ml_oos_plot_path(spec, len(pairs), from_date, to_date),
        oos_calibration_plot_path=ml_oos_calibration_plot_path(spec, len(pairs), from_date, to_date),
    )

    out_path = args.output
    if out_path is None:
        out_path = ml_metrics_path(spec, len(pairs), from_date, to_date)
    saved = _save_result(result, out_path, n_pairs=len(pairs), n_rows=dataset.frame.height, spec=spec)
    log.info("[ml] metrics saved: %s", saved)
    return result


def main() -> int:
    args = parse_ml_args()
    _validate_ml_args(args)
    if args.compare_models:
        test_from = parse_iso_utc(args.test_from)
        test_to = parse_iso_utc(args.test_to)
        model_ids = list(dict.fromkeys(args.compare_models))
        log_root = args.compare_output_dir or ml_compare_dir(model_ids, test_from, test_to)
        log_file = add_file_logging(log_root / "compare.log")
        log.info("[ml] compare models: %s", model_ids)
        log.info("[ml] log file: %s", log_file)
        run_compare_models_pipeline(args)
        return 0

    spec = resolve_ml_study(args.studies)
    log.info("[ml] studies=%s features=%s output=%s", list(spec.studies), list(spec.feature_columns), spec.output_study)
    if args.plots_only:
        if not args.train_test:
            log.warning("[ml] --plots-only: принудительно используем holdout test-период")
        from_date = parse_iso_utc(args.train_from)
        to_date = parse_iso_utc(args.test_to)
        log_file = add_file_logging(ml_log_path(spec, from_date, to_date))
        log.info("[ml] log file: %s", log_file)
        run_ml_plots_only_pipeline(args)
        return 0

    if args.train_test:
        from_date = parse_iso_utc(args.train_from)
        to_date = parse_iso_utc(args.test_to)
    else:
        from_date = parse_iso_utc(args.from_date)
        to_date = parse_iso_utc(args.to_date)
    log_file = add_file_logging(ml_log_path(spec, from_date, to_date))
    log.info("[ml] log file: %s", log_file)
    if args.train_test:
        run_ml_train_test_pipeline(args)
    else:
        run_ml_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
