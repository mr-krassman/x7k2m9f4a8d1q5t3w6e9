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
        choices=["day_of_week"],
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
    return parser.parse_args()
