import argparse
from pathlib import Path

from crypto_research.utils.pipeline.paths import DEFAULT_DATA_DIR
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_CONSERVATIVE,
    SCENARIO_MAXIMAL,
    SCENARIO_OPTIMISTIC,
    normalize_scenario,
)


def parse_backtest_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оркестратор бэктестов crypto_research.")
    parser.add_argument(
        "strategy",
        choices=["day_of_week", "day_of_week_ml", "ema_spreads"],
        help="Название стратегии (совпадает с исследованием в report_generator)",
    )
    parser.add_argument(
        "--scenario",
        choices=[SCENARIO_MAXIMAL, SCENARIO_CONSERVATIVE, SCENARIO_OPTIMISTIC, "maximal_val"],
        default=SCENARIO_MAXIMAL,
        type=normalize_scenario,
        help=(
            "maximal — 49 пар, полный период; "
            "conservative — 49 пар, val (консервативный); "
            "optimistic — val, train-отбор пар (оптимистичный)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Папка с *_klines_1m.jsonl (по умолчанию: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Начало периода (ISO UTC); по умолчанию — полный пул исследования",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Конец периода (ISO UTC); по умолчанию — полный пул исследования",
    )
    parser.add_argument(
        "--max-pair-start",
        default=None,
        help="Фильтр пар по первой свече (ISO UTC); по умолчанию 2023-01-01",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Список пар (btcusdt ethusdt). Без аргумента — все файлы в data-dir",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Потоки для параллельной загрузки JSONL",
    )
    parser.add_argument(
        "--ema-period",
        type=int,
        default=None,
        help="Период EMA (только ema_spreads; по умолчанию 9)",
    )
    parser.add_argument(
        "--ml-policy-path",
        type=Path,
        default=None,
        help="Путь к frozen policy JSON (только day_of_week_ml).",
    )
    parser.add_argument(
        "--ml-model-path",
        type=Path,
        default=None,
        help="Путь к model bundle PKL (только day_of_week_ml).",
    )
    args = parser.parse_args()
    from crypto_research.utils.backtest.strategies.registry import STRATEGY_HANDLERS

    if args.strategy not in STRATEGY_HANDLERS:
        parser.error(f"Неизвестная стратегия: {args.strategy}")
    if args.ema_period is not None and args.strategy != "ema_spreads":
        parser.error("--ema-period применим только к ema_spreads")
    if args.strategy != "day_of_week_ml" and (
        args.ml_policy_path is not None or args.ml_model_path is not None
    ):
        parser.error("--ml-policy-path/--ml-model-path применимы только к day_of_week_ml")
    if args.strategy == "day_of_week_ml" and normalize_scenario(args.scenario) != SCENARIO_MAXIMAL:
        parser.error("--scenario не применим к day_of_week_ml: период holdout test задаётся по умолчанию")
    if args.ema_period is None and args.strategy == "ema_spreads":
        from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD

        args.ema_period = SELECTED_EMA_PERIOD
    STRATEGY_HANDLERS[args.strategy].validate_args(parser, args)
    return args
