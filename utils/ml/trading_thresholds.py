"""Trading thresholds t_long/t_short from train_test metrics (prob_return_dependence)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.pipeline.paths import ml_train_test_metrics_path

log = get_logger("ml_trading_thresholds")

DEFAULT_POLICY_THRESHOLD = 0.5


def resolve_prob_return_threshold_pair(
    t_long: float | None,
    t_short: float | None,
) -> tuple[float, float]:
    """Если хотя бы один порог не найден — симметричная политика 0.5 / 0.5."""
    if t_long is None or t_short is None:
        return DEFAULT_POLICY_THRESHOLD, DEFAULT_POLICY_THRESHOLD
    return float(t_long), float(t_short)


def resolve_prob_return_thresholds_dict(
    th: dict[str, float | str | int | None],
) -> dict[str, float | str | int | None]:
    t_long, t_short = resolve_prob_return_threshold_pair(
        float(th["t_long"]) if th.get("t_long") is not None else None,
        float(th["t_short"]) if th.get("t_short") is not None else None,
    )
    out = dict(th)
    out["t_long"] = t_long
    out["t_short"] = t_short
    return out


def _thresholds_from_payload(payload: dict[str, object]) -> dict[str, float | str | int | None] | None:
    th = payload.get("prob_return_thresholds")
    if isinstance(th, dict):
        return th
    # Обратная совместимость со старыми train_test.json (дубль в plot_paths).
    plot_paths = payload.get("plot_paths")
    if isinstance(plot_paths, dict):
        nested = plot_paths.get("prob_return_thresholds")
        if isinstance(nested, dict):
            return nested
    return None


def load_prob_return_thresholds(
    spec,
    n_pairs: int,
    train_from: datetime,
    test_to: datetime,
    *,
    metrics_path: Path | None = None,
) -> tuple[float, float, dict[str, float | str | int | None]]:
    path = metrics_path or ml_train_test_metrics_path(spec, n_pairs, train_from, test_to)
    if not path.is_file():
        raise FileNotFoundError(f"train_test metrics не найден: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    th = _thresholds_from_payload(payload)
    if th is None:
        raise KeyError(f"prob_return_thresholds не найден в train_test metrics: {path}")
    raw_long = th.get("t_long")
    raw_short = th.get("t_short")
    t_long, t_short = resolve_prob_return_threshold_pair(
        float(raw_long) if raw_long is not None else None,
        float(raw_short) if raw_short is not None else None,
    )
    if raw_long is None or raw_short is None:
        log.info(
            "[ml] prob_return_thresholds неполные в %s (%r), fallback t_long=t_short=0.5",
            path,
            th,
        )
    else:
        log.info(
            "[ml] trading thresholds from %s: t_long=%.4f t_short=%.4f",
            path,
            t_long,
            t_short,
        )
    th = resolve_prob_return_thresholds_dict(th)
    return t_long, t_short, th


def try_load_prob_return_thresholds(
    spec,
    n_pairs: int,
    train_from: datetime,
    test_to: datetime,
    *,
    metrics_path: Path | None = None,
) -> tuple[float, float] | None:
    """Для compare: None если metrics старые или без prob_return_thresholds."""
    path = metrics_path or ml_train_test_metrics_path(spec, n_pairs, train_from, test_to)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    th = _thresholds_from_payload(payload)
    if th is None:
        return None
    raw_long = th.get("t_long")
    raw_short = th.get("t_short")
    return resolve_prob_return_threshold_pair(
        float(raw_long) if raw_long is not None else None,
        float(raw_short) if raw_short is not None else None,
    )
