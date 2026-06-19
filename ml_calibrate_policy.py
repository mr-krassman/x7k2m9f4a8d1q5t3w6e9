#!/usr/bin/env python3
"""Калибровка frozen policy для day_of_week_ml по train CPCV метрикам."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.backtest.analytics import WEEKDAY_NAMES
from crypto_research.utils.ml.registry import ML_STUDY_CHOICES, resolve_ml_study
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import (
    TEMPORAL_TRAIN_FROM,
    TEMPORAL_VAL_TO,
    ml_policy_path,
    ml_train_test_metrics_path,
)

JSON_FLOAT_PRECISION = 4


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Калибровка frozen policy для ML-стратегий (day_of_week_ml / ema_spreads_ml / rsi_spreads_ml / price_sequences_ml / combined).",
    )
    parser.add_argument(
        "studies",
        nargs="*",
        choices=list(ML_STUDY_CHOICES),
        metavar="STUDY",
        help="ML-исследование (по умолчанию day_of_week_ml)",
    )
    parser.add_argument("--n-pairs", type=int, default=49, help="Размер пула пар.")
    parser.add_argument("--train-from", default=TEMPORAL_TRAIN_FROM, help="Начало train периода.")
    parser.add_argument("--test-to", default=TEMPORAL_VAL_TO, help="Конец holdout test периода.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Путь к *_train_test.json.")
    parser.add_argument("--output", type=Path, default=None, help="Путь к policy.json.")
    parser.add_argument(
        "--score-quantile",
        type=float,
        default=None,
        help=(
            "Глобальный квантиль score по всем (weekday,pair). "
            "По умолчанию: 0.6 (weekday) / 0.1 (global policy)."
        ),
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=60,
        help="Минимум наблюдений n_test для пары (в CPCV ~65–118 на weekday×pair).",
    )
    parser.add_argument("--long-quantile", type=float, default=None, help="Квантиль long-порогов.")
    parser.add_argument("--short-quantile", type=float, default=None, help="Квантиль short-порогов.")
    return parser.parse_args()


def _study_from_payload(payload: dict[str, object]):
    ml_spec = payload.get("ml_spec")
    if isinstance(ml_spec, dict):
        studies = ml_spec.get("studies")
        if isinstance(studies, list) and studies:
            return resolve_ml_study(studies)
    return resolve_ml_study(["day_of_week_ml"])


def _default_selection_params(spec) -> dict[str, float | int]:
    if spec.policy_mode == "global":
        return {
            "score_quantile": 0.1,
            "min_obs": 60,
            "long_quantile": 0.999,
            "short_quantile": 0.0001,
        }
    return {
        "score_quantile": 0.6,
        "min_obs": 60,
        "long_quantile": 0.25,
        "short_quantile": 0.75,
    }


def _parse_weekday_pair_key(key: str) -> tuple[int, str]:
    wd_name, pair = key.split("::", 1)
    return WEEKDAY_NAMES.index(wd_name), pair


def _side(mean_p_up: float) -> str:
    if mean_p_up > 0.5:
        return "long"
    if mean_p_up < 0.5:
        return "short"
    return "flat"


def _select_pairs_by_weekday(
    weekday_pair_metrics: dict[str, dict[str, float]],
    *,
    score_quantile: float,
    min_obs: int,
) -> tuple[dict[int, list[str]], list[dict[str, float | int | str]], float]:
    by_wd: dict[int, list[tuple[str, float, float, int]]] = {wd: [] for wd in range(7)}
    details: list[dict[str, float | int | str]] = []
    for key, metrics in weekday_pair_metrics.items():
        wd, pair = _parse_weekday_pair_key(key)
        mean_p = float(metrics["mean_p_up"])
        base = float(metrics["base_rate_up"])
        acc = float(metrics["accuracy"])
        n_obs = int(metrics["n_test"])
        edge = mean_p - base
        confidence = abs(mean_p - 0.5)
        score = 0.7 * abs(edge) + 0.3 * max(0.0, acc - 0.5) + 0.2 * confidence
        by_wd[wd].append((pair, score, mean_p, n_obs))
        details.append(
            {
                "weekday": WEEKDAY_NAMES[wd],
                "pair": pair,
                "score": score,
                "mean_p_up": mean_p,
                "base_rate_up": base,
                "edge_p_up": edge,
                "accuracy": acc,
                "n_test": n_obs,
                "side": _side(mean_p),
            }
        )

    all_scores = np.array([row["score"] for row in details], dtype=float)
    score_cutoff = float(np.quantile(all_scores, score_quantile))

    selected: dict[int, list[str]] = {}
    for wd, rows in by_wd.items():
        if not rows:
            selected[wd] = []
            continue
        pick = sorted(
            pair for pair, score, _, n_obs in rows if score >= score_cutoff and n_obs >= min_obs
        )
        if not pick:
            # Fallback: берём лучшую пару weekday, чтобы стратегия не стала пустой.
            best_pair = max(rows, key=lambda x: x[1])[0]
            pick = [best_pair]
        selected[wd] = pick
    return selected, details, score_cutoff


def _dynamic_thresholds(
    weekday_pair_metrics: dict[str, dict[str, float]],
    selected_pairs_by_weekday: dict[int, list[str]],
    *,
    long_quantile: float,
    short_quantile: float,
) -> dict[str, float]:
    long_probs: list[float] = []
    short_probs: list[float] = []
    for key, metrics in weekday_pair_metrics.items():
        wd, pair = _parse_weekday_pair_key(key)
        if pair not in selected_pairs_by_weekday.get(wd, []):
            continue
        p = float(metrics["mean_p_up"])
        if p > 0.5:
            long_probs.append(p)
        elif p < 0.5:
            short_probs.append(p)

    if not long_probs or not short_probs:
        all_probs = [float(v["mean_p_up"]) for v in weekday_pair_metrics.values()]
        long_probs = [p for p in all_probs if p > 0.5]
        short_probs = [p for p in all_probs if p < 0.5]

    t_long = max(0.5, float(np.quantile(np.array(long_probs, dtype=float), long_quantile)))
    t_short = min(0.5, float(np.quantile(np.array(short_probs, dtype=float), short_quantile)))
    return {"t_long": t_long, "t_short": t_short}


def _select_pairs_global(
    weekday_pair_metrics: dict[str, dict[str, float]],
    *,
    score_quantile: float,
    min_obs: int,
) -> tuple[list[str], list[dict[str, float | int | str]], float]:
    by_pair: dict[str, list[tuple[float, float, float, int]]] = {}
    for key, metrics in weekday_pair_metrics.items():
        _, pair = _parse_weekday_pair_key(key)
        mean_p = float(metrics["mean_p_up"])
        base = float(metrics["base_rate_up"])
        acc = float(metrics["accuracy"])
        n_obs = int(metrics["n_test"])
        edge = mean_p - base
        confidence = abs(mean_p - 0.5)
        score = 0.7 * abs(edge) + 0.3 * max(0.0, acc - 0.5) + 0.2 * confidence
        by_pair.setdefault(pair, []).append((score, mean_p, base, n_obs))

    details: list[dict[str, float | int | str]] = []
    for pair, rows in by_pair.items():
        weights = np.array([max(r[3], 1) for r in rows], dtype=float)
        score_arr = np.array([r[0] for r in rows], dtype=float)
        mean_arr = np.array([r[1] for r in rows], dtype=float)
        base_arr = np.array([r[2] for r in rows], dtype=float)
        n_obs_total = int(sum(r[3] for r in rows))
        score_agg = float(np.average(score_arr, weights=weights))
        mean_p_agg = float(np.average(mean_arr, weights=weights))
        base_agg = float(np.average(base_arr, weights=weights))
        details.append(
            {
                "pair": pair,
                "score": score_agg,
                "mean_p_up": mean_p_agg,
                "base_rate_up": base_agg,
                "edge_p_up": mean_p_agg - base_agg,
                "n_test_total": n_obs_total,
                "weekday_coverage": len(rows),
                "side": _side(mean_p_agg),
            }
        )

    all_scores = np.array([row["score"] for row in details], dtype=float)
    score_cutoff = float(np.quantile(all_scores, score_quantile))
    selected = sorted(
        row["pair"]
        for row in details
        if float(row["score"]) >= score_cutoff and int(row["n_test_total"]) >= min_obs
    )
    if not selected and details:
        selected = [str(max(details, key=lambda row: float(row["score"]))["pair"])]
    return selected, details, score_cutoff


def _dynamic_thresholds_global(
    pair_score_table: list[dict[str, float | int | str]],
    selected_pairs: list[str],
    *,
    long_quantile: float,
    short_quantile: float,
) -> dict[str, float]:
    selected = set(selected_pairs)
    long_probs: list[float] = []
    short_probs: list[float] = []
    for row in pair_score_table:
        pair = str(row["pair"])
        if pair not in selected:
            continue
        p = float(row["mean_p_up"])
        if p > 0.5:
            long_probs.append(p)
        elif p < 0.5:
            short_probs.append(p)

    if not long_probs or not short_probs:
        all_probs = [float(row["mean_p_up"]) for row in pair_score_table]
        long_probs = [p for p in all_probs if p > 0.5]
        short_probs = [p for p in all_probs if p < 0.5]

    t_long = max(0.5, float(np.quantile(np.array(long_probs, dtype=float), long_quantile)))
    t_short = min(0.5, float(np.quantile(np.array(short_probs, dtype=float), short_quantile)))
    return {"t_long": t_long, "t_short": t_short}


def write_policy_from_train_test_payload(
    payload: dict[str, object],
    spec,
    *,
    n_pairs: int,
    train_from,
    test_to,
    output: Path | None = None,
    metrics_path: Path | None = None,
    score_quantile: float | None = None,
    min_obs: int | None = None,
    long_quantile: float | None = None,
    short_quantile: float | None = None,
) -> Path:
    defaults = _default_selection_params(spec)
    score_quantile = score_quantile if score_quantile is not None else float(defaults["score_quantile"])
    min_obs = min_obs if min_obs is not None else int(defaults["min_obs"])
    long_quantile = long_quantile if long_quantile is not None else float(defaults["long_quantile"])
    short_quantile = short_quantile if short_quantile is not None else float(defaults["short_quantile"])

    weekday_pair_metrics = payload["train_cpcv"]["weekday_pair_metrics"]
    selected_pairs_by_weekday, details, score_cutoff = _select_pairs_by_weekday(
        weekday_pair_metrics,
        score_quantile=score_quantile,
        min_obs=min_obs,
    )
    thresholds = _dynamic_thresholds(
        weekday_pair_metrics,
        selected_pairs_by_weekday,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
    )
    selected_pairs_global, global_details, global_score_cutoff = _select_pairs_global(
        weekday_pair_metrics,
        score_quantile=score_quantile,
        min_obs=min_obs,
    )
    thresholds_global = _dynamic_thresholds_global(
        global_details,
        selected_pairs_global,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
    )

    source_metrics = metrics_path or payload.get("source_metrics_json")
    out = {
        "mode": f"{spec.studies[0]}_policy_v1",
        "policy_mode": spec.policy_mode,
        "ml_spec": payload.get("ml_spec"),
        "source_metrics_json": str(source_metrics) if source_metrics else None,
        "n_pairs": n_pairs,
        "train_period": payload["train_period"],
        "holdout_test_period": payload["holdout_test_period"],
        "thresholds": thresholds,
        "selection_policy": {
            "score_quantile": score_quantile,
            "score_cutoff": score_cutoff,
            "min_obs": min_obs,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
        },
        "selected_pairs_by_weekday": {str(k): v for k, v in selected_pairs_by_weekday.items()},
        "selected_pairs_count_by_weekday": {
            str(k): len(v) for k, v in selected_pairs_by_weekday.items()
        },
        "weekday_pair_score_table": details,
        "selected_pairs_global": selected_pairs_global,
        "selected_pairs_global_count": len(selected_pairs_global),
        "global_selection_policy": {
            "score_quantile": score_quantile,
            "score_cutoff": global_score_cutoff,
            "min_obs": min_obs,
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
        },
        "global_thresholds": thresholds_global,
        "pair_score_table": global_details,
    }
    output_path = output or ml_policy_path(spec, n_pairs, train_from, test_to)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_round_json_floats(out), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    args = parse_args()
    studies = args.studies or ["day_of_week_ml"]
    spec = resolve_ml_study(studies)
    defaults = _default_selection_params(spec)
    score_quantile = args.score_quantile if args.score_quantile is not None else float(defaults["score_quantile"])
    min_obs = args.min_obs if args.min_obs is not None else int(defaults["min_obs"])
    long_quantile = args.long_quantile if args.long_quantile is not None else float(defaults["long_quantile"])
    short_quantile = args.short_quantile if args.short_quantile is not None else float(defaults["short_quantile"])

    train_from = parse_iso_utc(args.train_from)
    test_to = parse_iso_utc(args.test_to)
    metrics_path = args.metrics_json or ml_train_test_metrics_path(
        spec, args.n_pairs, train_from, test_to
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload_spec = _study_from_payload(payload)
    if payload_spec.studies != spec.studies:
        raise SystemExit(
            f"metrics-json для {list(payload_spec.studies)}, CLI запрошен {list(spec.studies)}"
        )
    output_path = write_policy_from_train_test_payload(
        payload,
        spec,
        n_pairs=args.n_pairs,
        train_from=train_from,
        test_to=test_to,
        output=args.output,
        metrics_path=metrics_path,
        score_quantile=score_quantile,
        min_obs=min_obs,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
    )
    print(f"saved_policy={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

