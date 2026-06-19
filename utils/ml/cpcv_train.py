"""Обучение LightGBM с Combinatorial Purged Cross-Validation (CPCV).

Задача модуля — честно оценить модель «день недели → направление дневной доходности
open→close» на объединённом пуле пар без утечки из будущего.

Общая схема (Lopez de Prado, *Advances in Financial Machine Learning*):

1. **Датасет** — все пары в одном frame, отсортированы по (day_utc, pair).
   Единственный признак: `weekday_enc` (категориальный). Таргет: `direction_up`.

2. **CPCV по календарным дням** — временная ось режется на n_splits блоков *дней*
   (не отдельных строк). Из блоков выбираются все комбинации n_test_groups для теста:
   C(7, 2) = 21 сценарий train/test. Это строже, чем одно скользящее окно.

3. **Развёртка дней → строки** — один тестовый день включает все пары этого дня,
   чтобы не резать календарный день посередине и не смешивать train/test внутри дня.

4. **Embargo** — после тестового окна train-наблюдения на embargo_days удаляются,
   чтобы снизить автокорреляцию между соседними по времени фолдами.

5. **LightGBM** — на каждом фолде отдельная модель; метрики усредняются по фолдам.

6. **OOS paths** — предсказания всех фолдов объединяются, дубликаты (day, pair)
   усредняются, строится график P(up) во времени.

Почему purge_horizon=None: признак weekday известен на open, таргет — close того же дня;
между соседними днями нет перекрывающегося горизонта метки. При добавлении lag-признаков
понадобится построчный purge по prediction_times / evaluation_times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from purgedcv import CombinatorialPurgedCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.ml.dataset import DirectionDataset, categorical_feature_names, dataset_to_numpy
from crypto_research.utils.ml.oos_paths import (
    build_oos_predictions,
    collect_fold_predictions,
    oos_calibration_metrics,
    oos_paths_array,
    save_oos_calibration_plot,
    save_oos_probability_plot,
)
from crypto_research.utils.pipeline.logger import get_logger

log = get_logger("ml_cpcv")

# Число временных блоков (дней) для CPCV. 7 — компромисс между охватом и размером train.
DEFAULT_N_SPLITS = 7
# Сколько блоков одновременно в тесте. C(7, 2) = 21 комбинации train/test.
DEFAULT_N_TEST_GROUPS = 2
# Дней «тишины» после теста: train не берёт наблюдения сразу за тестовым окном.
DEFAULT_EMBARGO_DAYS = 1
# Сколько строк X/y печатать в подробном логе одного фолда.
LOG_PREVIEW_ROWS = 3


def _y_distribution(y: np.ndarray) -> dict[int, int]:
    """Распределение классов {0: n_down, 1: n_up} для логов."""
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _day_range(frame: pl.DataFrame, row_indices: np.ndarray) -> tuple[str, str] | None:
    """Минимальная и максимальная дата в выборке строк (для лога фолда)."""
    if row_indices.size == 0:
        return None
    days = frame["day_utc"].gather(row_indices.tolist())
    return str(days.min()), str(days.max())


def _log_xy(x: pd.DataFrame, y: np.ndarray, *, target_name: str) -> None:
    """Этап 1 логирования: что именно пойдёт в модель до разбиения на фолды."""
    log.info(
        "[ml] X: shape=%s columns=%s dtypes=%s",
        x.shape,
        list(x.columns),
        {col: str(dtype) for col, dtype in x.dtypes.items()},
    )
    log.info("[ml] X head (%d rows):\n%s", LOG_PREVIEW_ROWS, x.head(LOG_PREVIEW_ROWS))
    log.info(
        "[ml] y (%s): shape=%s dtype=%s distribution=%s",
        target_name,
        y.shape,
        y.dtype,
        _y_distribution(y),
    )
    log.info("[ml] y head (%d values): %s", LOG_PREVIEW_ROWS, y[:LOG_PREVIEW_ROWS].tolist())


def _log_cv(
    cv: CombinatorialPurgedCV,
    *,
    n_splits: int,
    n_test_groups: int,
    embargo_days: int,
    n_folds: int,
    n_days: int,
    pred_times: pd.Series,
) -> None:
    """Этап 2 логирования: конфигурация CPCV и размеры train/test по дням в каждом фолде."""
    log.info(
        "[ml] CV: CombinatorialPurgedCV n_splits=%s n_test_groups=%s embargo_days=%s "
        "n_folds=C(%s,%s)=%s n_days=%s",
        n_splits,
        n_test_groups,
        embargo_days,
        n_splits,
        n_test_groups,
        n_folds,
        n_days,
    )
    log.info(
        "[ml] CV time range: %s .. %s (prediction_times=evaluation_times, purge=None)",
        pred_times.min(),
        pred_times.max(),
    )
    fold_sizes: list[str] = []
    for fold_idx, (train_day_idx, test_day_idx) in enumerate(cv.split(np.arange(n_days))):
        fold_sizes.append(
            f"fold{fold_idx + 1}: train_days={train_day_idx.size} test_days={test_day_idx.size}"
        )
    log.info("[ml] CV day splits:\n  %s", "\n  ".join(fold_sizes))


def _log_fold_inputs(
    fold_idx: int,
    n_folds: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    day_range_train: tuple[str, str] | None,
    day_range_test: tuple[str, str] | None,
    detailed: bool,
) -> None:
    """Этап 3 логирования: состав train/test конкретного фолда перед fit/predict."""
    log.info(
        "[ml] fold %s/%s → model.fit: X_train=%s y_train=%s days=%s dist=%s",
        fold_idx + 1,
        n_folds,
        x_train.shape,
        y_train.shape,
        day_range_train,
        _y_distribution(y_train),
    )
    log.info(
        "[ml] fold %s/%s → model.predict: X_test=%s y_test=%s days=%s dist=%s",
        fold_idx + 1,
        n_folds,
        x_test.shape,
        y_test.shape,
        day_range_test,
        _y_distribution(y_test),
    )
    if not detailed:
        return
    log.info(
        "[ml] fold %s X_train head:\n%s",
        fold_idx + 1,
        x_train.head(LOG_PREVIEW_ROWS),
    )
    log.info(
        "[ml] fold %s y_train head: %s",
        fold_idx + 1,
        y_train[:LOG_PREVIEW_ROWS].tolist(),
    )
    log.info(
        "[ml] fold %s X_test head:\n%s",
        fold_idx + 1,
        x_test.head(LOG_PREVIEW_ROWS),
    )
    log.info(
        "[ml] fold %s y_test head: %s",
        fold_idx + 1,
        y_test[:LOG_PREVIEW_ROWS].tolist(),
    )


@dataclass(frozen=True)
class CPCVTrainResult:
    """Итог обучения: метрики по фолдам, средние, OOS-предсказания и путь к графику."""

    n_splits: int
    n_test_groups: int
    n_folds: int
    fold_metrics: tuple[dict[str, float], ...]
    mean_metrics: dict[str, float]
    oos_paths: np.ndarray | None
    oos_predictions: pl.DataFrame | None = None
    # Метрики OOS по дням недели (ключи — WEEKDAY_NAMES: Пн … Вс).
    weekday_metrics: dict[str, dict[str, float]] | None = None
    calibration_metrics: dict[str, float | dict[str, dict[str, float]]] | None = None
    oos_plot_path: Path | None = None
    oos_calibration_plot_path: Path | None = None


def _default_lgbm_params() -> dict:
    """Базовые гиперпараметры LightGBM для бинарной классификации.

    num_leaves=7 — мало листьев при малом числе признаков.
    max_depth=5, min_child_samples=80, reg_alpha/reg_lambda — L1/L2 регуляризация
    против переобучения на хвосте train (см. learning curve).
    """
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": 100,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 5,
        "min_child_samples": 200,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "random_state": 42,
    }


def _day_level_times(frame: pl.DataFrame) -> pd.Series:
    """Уникальные календарные дни в хронологическом порядке.

    CPCV в purgedcv работает на уровне *наблюдений* в Series времени; мы подаём
    по одному timestamp на каждый уникальный день. Затем индексы дней разворачиваем
    на все строки (пары) через _expand_day_indices.

    Важно: берём to_pandas() напрямую, без cast(pl.Int64) — иначе микросекунды
    интерпретируются как наносекунды и даты съезжают в 1970 год.
    """
    days = frame.select("day_utc").unique().sort("day_utc")["day_utc"]
    return pd.Series(days.to_pandas())


def _expand_day_indices(frame: pl.DataFrame, day_indices: np.ndarray) -> np.ndarray:
    """Переводит индексы дней CPCV в индексы строк frame.

    Зачем: в одном календарном дне — несколько пар (до 49 строк). CPCV выдаёт
    train_day_idx / test_day_idx в пространстве уникальных дней; модель же обучается
    на строках. Все пары выбранного дня попадают в один фолд целиком.
    """
    if day_indices.size == 0:
        return np.array([], dtype=np.int64)
    selected_days = (
        frame.select("day_utc")
        .unique()
        .sort("day_utc")
        .gather(day_indices)["day_utc"]
        .to_list()
    )
    return np.flatnonzero(frame["day_utc"].is_in(selected_days).to_numpy())


def _build_day_cpcv(
    frame: pl.DataFrame,
    *,
    n_splits: int,
    n_test_groups: int,
    embargo_days: int,
) -> CombinatorialPurgedCV:
    """Создаёт CombinatorialPurgedCV на оси календарных дней.

    prediction_times = evaluation_times = день UTC, потому что:
    - признак (weekday) известен на открытии дня;
    - таргет (direction_up) фиксируется на закрытии того же дня.

    embargo — post-test буфер: train не использует дни сразу после тестового окна.
    purge_horizon=None — для текущего набора признаков горизонт метки не перекрывается
    между соседними днями.
    """
    pred_times = _day_level_times(frame)
    eval_times = pred_times
    embargo = pd.Timedelta(days=embargo_days) if embargo_days > 0 else None
    return CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        prediction_times=pred_times,
        evaluation_times=eval_times,
        purge_horizon=None,
        embargo=embargo,
    )


def _fold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Метрики одного фолда на out-of-sample тесте."""
    y_pred = (y_prob >= 0.5).astype(np.int8)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "n_test": int(y_true.size),
        "pred_up_rate": float(y_pred.mean()),
    }
    if np.unique(y_true).size > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def _mean_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Средние метрики по всем успешно обученным фолдам (не по всем C(n,k) комбинациям)."""
    keys = ("accuracy", "log_loss", "roc_auc")
    return {
        key: float(np.nanmean([m[key] for m in fold_metrics]))
        for key in keys
    }


def _oos_metrics_by_weekday(oos: pl.DataFrame) -> dict[str, dict[str, float]]:
    """OOS-метрики отдельно для каждого weekday (0=Пн … 6=Вс)."""
    if oos.is_empty():
        return {}
    weekday = ((oos["day_utc"].dt.weekday() - 1) % 7).to_numpy()
    y_true = oos["y_true"].to_numpy()
    y_prob = oos["y_prob"].to_numpy()
    out: dict[str, dict[str, float]] = {}
    for wd in range(len(WEEKDAY_NAMES)):
        mask = weekday == wd
        if not np.any(mask):
            continue
        metrics = _fold_metrics(y_true[mask], y_prob[mask])
        metrics["base_rate_up"] = float(y_true[mask].mean())
        metrics["mean_p_up"] = float(y_prob[mask].mean())
        metrics["mean_p_down"] = float(1.0 - y_prob[mask].mean())
        metrics["weekday"] = float(wd)
        out[WEEKDAY_NAMES[wd]] = metrics
    return out


def train_lightgbm_cpcv(
    dataset: DirectionDataset,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    n_test_groups: int = DEFAULT_N_TEST_GROUPS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    lgbm_params: dict | None = None,
    reconstruct_paths: bool = True,
    oos_plot_path: Path | None = None,
    oos_calibration_plot_path: Path | None = None,
) -> CPCVTrainResult:
    """Главный пайплайн: CPCV + LightGBM + сбор OOS-предсказаний.

    Этапы:
        1. Подготовка X, y из dataset (pandas + category для LightGBM).
        2. Настройка CombinatorialPurgedCV по уникальным дням.
        3. Цикл по фолдам: expand дней → строки, fit, predict, метрики.
        4. Усреднение метрик; объединение OOS-предсказаний и график.
    """
    frame = dataset.frame

    # --- Этап 1: матрица признаков и вектор таргета ---
    # X — только фичи для модели; y — direction_up; порядок строк = frame (day_utc, pair).
    x, y, _, _ = dataset_to_numpy(dataset)
    _log_xy(x, y, target_name=dataset.target_column)

    # --- Этап 2: конфигурация CPCV ---
    # pred_times нужен и для CV, и для проверки диапазона дат в логе.
    pred_times = _day_level_times(frame)
    cv = _build_day_cpcv(
        frame,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_days=embargo_days,
    )
    n_folds = cv.get_n_splits()
    n_days = frame.select("day_utc").unique().height
    _log_cv(
        cv,
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        embargo_days=embargo_days,
        n_folds=n_folds,
        n_days=n_days,
        pred_times=pred_times,
    )

    # --- Этап 3: гиперпараметры LightGBM ---
    params = _default_lgbm_params()
    if lgbm_params:
        params.update(lgbm_params)
    cat_features = categorical_feature_names(dataset.feature_columns)

    # Индексы 0..n_days-1 — «заглушка» для cv.split: CPCV смотрит только на длину
    # и на prediction_times, привязанные при создании cv.
    day_dummy = np.arange(n_days)

    fold_metrics: list[dict[str, float]] = []
    fold_predictions: list[pl.DataFrame] = []
    first_valid_fold_logged = False

    # --- Этап 4: цикл по комбинациям train/test фолдов CPCV ---
    for fold_idx, (train_day_idx, test_day_idx) in enumerate(cv.split(day_dummy)):
        # Индексы дней → индексы всех строк (пар) этих дней.
        train_idx = _expand_day_indices(frame, train_day_idx)
        test_idx = _expand_day_indices(frame, test_day_idx)

        # После embargo часть комбинаций может дать пустой train — такой фолд пропускаем.
        if train_idx.size < 2 or test_idx.size < 1:
            log.warning(
                "[ml] fold %s skipped: train=%s test=%s",
                fold_idx + 1,
                train_idx.size,
                test_idx.size,
            )
            continue

        x_train = x.iloc[train_idx]
        y_train = y[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y[test_idx]

        # Подробный лог (head X/y) только для первого валидного фолда — иначе слишком много текста.
        _log_fold_inputs(
            fold_idx,
            n_folds,
            x_train,
            y_train,
            x_test,
            y_test,
            day_range_train=_day_range(frame, train_idx),
            day_range_test=_day_range(frame, test_idx),
            detailed=not first_valid_fold_logged,
        )
        first_valid_fold_logged = True

        # На каждом фолде — новая модель (нет переноса весов между комбинациями CPCV).
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            categorical_feature=cat_features,
        )
        y_prob = model.predict_proba(x_test)[:, 1]

        # Сохраняем OOS-предсказания с привязкой ко времени и паре для последующего merge.
        fold_predictions.append(
            collect_fold_predictions(frame, test_idx, y_prob, y_test, fold_idx)
        )

        metrics = _fold_metrics(y_test, y_prob)
        metrics["fold"] = float(fold_idx)
        fold_metrics.append(metrics)
        log.info(
            "[ml] fold %s/%s train=%s test=%s acc=%.4f auc=%.4f",
            fold_idx + 1,
            n_folds,
            train_idx.size,
            test_idx.size,
            metrics["accuracy"],
            metrics["roc_auc"],
        )

    if not fold_metrics:
        raise RuntimeError("CPCV не дал ни одного валидного фолда для обучения")

    # --- Этап 5: агрегированные метрики по фолдам ---
    mean = _mean_metrics(fold_metrics)
    log.info(
        "[ml] CPCV mean: folds=%s acc=%.4f logloss=%.4f auc=%.4f",
        len(fold_metrics),
        mean["accuracy"],
        mean["log_loss"],
        mean["roc_auc"],
    )

    # --- Этап 6: OOS paths — объединение предсказаний всех фолдов ---
    # Один (day, pair) может встретиться в тесте нескольких комбинаций CPCV;
    # y_prob усредняется (см. build_oos_predictions в oos_paths.py).
    oos_paths = None
    oos_predictions = None
    saved_plot_path = None
    saved_calibration_plot_path = None
    weekday_metrics: dict[str, dict[str, float]] | None = None
    calibration_metrics: dict[str, float | dict[str, dict[str, float]]] | None = None
    if reconstruct_paths and fold_predictions:
        oos_df = build_oos_predictions(fold_predictions)
        oos_predictions = oos_df
        oos_paths = oos_paths_array(oos_df)
        weekday_metrics = _oos_metrics_by_weekday(oos_df)
        calibration_metrics = oos_calibration_metrics(oos_df)
        log.info(
            "[ml] OOS paths: rows=%s cols=[time_ns, y_prob, y_true] folds_merged=%s",
            oos_paths.shape[0],
            len(fold_predictions),
        )
        log.info(
            "[ml] OOS calibration: brier=%.4f ece=%.4f",
            calibration_metrics["brier_score"],
            calibration_metrics["ece"],
        )
        for name, metrics in weekday_metrics.items():
            log.info(
                "[ml] OOS weekday %s: n=%s base_up=%.4f pred_up=%.4f acc=%.4f logloss=%.4f auc=%.4f",
                name,
                metrics["n_test"],
                metrics["base_rate_up"],
                metrics["pred_up_rate"],
                metrics["accuracy"],
                metrics["log_loss"],
                metrics["roc_auc"],
            )
        if oos_plot_path is not None:
            saved_plot_path = save_oos_probability_plot(oos_df, oos_plot_path)
        if oos_calibration_plot_path is not None:
            saved_calibration_plot_path = save_oos_calibration_plot(
                oos_df,
                oos_calibration_plot_path,
            )

    return CPCVTrainResult(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        n_folds=n_folds,
        fold_metrics=tuple(fold_metrics),
        mean_metrics=mean,
        oos_predictions=oos_predictions,
        weekday_metrics=weekday_metrics,
        calibration_metrics=calibration_metrics,
        oos_paths=oos_paths,
        oos_plot_path=saved_plot_path,
        oos_calibration_plot_path=saved_calibration_plot_path,
    )
