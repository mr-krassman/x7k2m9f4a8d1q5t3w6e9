import argparse
from pathlib import Path

from crypto_research.utils.backtest.bundle_registry import parse_backtest_strategy_args
from crypto_research.utils.backtest.scenarios import (
    SCENARIO_CONSERVATIVE,
    SCENARIO_MAXIMAL,
    SCENARIO_OPTIMISTIC,
    normalize_scenario,
)
from crypto_research.utils.backtest.strategies.registry import get_strategy_handler
from crypto_research.utils.pipeline.paths import DEFAULT_DATA_DIR


def _strategy_help() -> str:
    return (
        "rule-based: day_of_week | ema_spreads | rsi_spreads | price_sequences | volume_spreads | "
        "любое подмножество 2+ rule-based --mode and|or; "
        "ML: day_of_week_ml | ema_spreads_ml | rsi_spreads_ml | price_sequences_ml | volume_spreads_ml | "
        "любое подмножество 2+ ML (порядок не важен)"
    )


def parse_backtest_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оркестратор бэктестов crypto_research.")
    parser.add_argument(
        "strategies",
        nargs="+",
        metavar="STRATEGY",
        help=_strategy_help(),
    )
    parser.add_argument(
        "--mode",
        choices=["and", "or"],
        default=None,
        help="Режим combined rule-based (2+ стратегии): and | or.",
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
        "--rsi-period",
        type=int,
        default=None,
        help="Период RSI (только rsi_spreads; по умолчанию 9)",
    )
    parser.add_argument(
        "--vol-period",
        type=int,
        default=None,
        help="Период EMA(volume) (только volume_spreads; по умолчанию 50)",
    )
    parser.add_argument(
        "--ml-metrics-path",
        type=Path,
        default=None,
        help="Путь к train_test metrics JSON (только ML; пороги prob_return_thresholds).",
    )
    parser.add_argument(
        "--ml-model-path",
        type=Path,
        default=None,
        help="Путь к model bundle PKL (только ML-стратегии).",
    )
    args = parser.parse_args()

    try:
        parsed = parse_backtest_strategy_args(args.strategies, combine_mode=args.mode)
    except ValueError as exc:
        parser.error(str(exc))

    args.strategy = parsed.strategy_key
    args.ml_spec = parsed.ml_spec
    args.algo_spec = parsed.algo_spec
    args.bundle_id = parsed.bundle_id
    args.bundle_kind = parsed.bundle_kind
    args.combine_mode = parsed.combine_mode

    if args.mode is not None and parsed.algo_spec is None:
        parser.error("--mode применим только к combined rule-based (2+ стратегии)")

    if parsed.algo_spec is not None:
        args.scenario = SCENARIO_OPTIMISTIC

    handler = get_strategy_handler(
        args.strategy,
        algo_spec=args.algo_spec,
        ml_spec=args.ml_spec,
    )

    if args.ema_period is not None and args.strategy not in ("ema_spreads",) and args.algo_spec is None:
        parser.error("--ema-period применим только к ema_spreads")
    if args.rsi_period is not None and args.strategy not in ("rsi_spreads",) and args.algo_spec is None:
        parser.error("--rsi-period применим только к rsi_spreads")
    if args.vol_period is not None and args.strategy not in ("volume_spreads",) and args.algo_spec is None:
        parser.error("--vol-period применим только к volume_spreads")
    if args.ml_spec is None and args.algo_spec is None and (
        args.ml_metrics_path is not None or args.ml_model_path is not None
    ):
        parser.error("--ml-metrics-path/--ml-model-path применимы только к ML-стратегиям")
    if args.ml_spec is not None and normalize_scenario(args.scenario) != SCENARIO_MAXIMAL:
        parser.error("--scenario не применим к ML-стратегиям: период holdout test задаётся по умолчанию")
    if args.ema_period is None and args.strategy == "ema_spreads":
        from crypto_research.utils.ema_spreads.constants import SELECTED_EMA_PERIOD

        args.ema_period = SELECTED_EMA_PERIOD
    if args.rsi_period is None and args.strategy == "rsi_spreads":
        from crypto_research.utils.rsi.constants import SELECTED_RSI_PERIOD

        args.rsi_period = SELECTED_RSI_PERIOD
    if args.vol_period is None and args.strategy == "volume_spreads":
        from crypto_research.utils.volume.constants import SELECTED_VOLUME_EMA_PERIOD

        args.vol_period = SELECTED_VOLUME_EMA_PERIOD
    handler.validate_args(parser, args)
    return args
