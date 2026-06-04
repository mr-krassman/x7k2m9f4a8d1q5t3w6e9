import argparse
from pathlib import Path

from crypto_research.utils.paths import DEFAULT_DATA_DIR


def parse_report_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оркестратор отчётов crypto_research.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Папка с *_klines_1m.jsonl (по умолчанию: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="Начало периода (ISO UTC), например 2024-01-01",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="Конец периода (ISO UTC), например 2024-12-31",
    )
    parser.add_argument(
        "--max-pair-start",
        default=None,
        help=(
            "Только пары, у которых первая свеча не позже этой даты (ISO UTC), "
            "например 2023-01-01"
        ),
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
        help="Потоки для параллельной загрузки JSONL (по умолчанию min(8, CPU))",
    )
    return parser.parse_args()
