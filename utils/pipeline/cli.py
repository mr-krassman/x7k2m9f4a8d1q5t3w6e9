import argparse
from pathlib import Path

from crypto_research.utils.pipeline.paths import DEFAULT_DATA_DIR, TRAIN_MAX_PAIR_START, VAL_MAX_PAIR_START


def parse_report_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оркестратор отчётов crypto_research.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--train",
        action="store_true",
        help=f"Train: пары с первой свечой не позже {TRAIN_MAX_PAIR_START} (UTC); суффикс _train",
    )
    split_group.add_argument(
        "--val",
        action="store_true",
        help=f"Val: пары с первой свечой не позже {VAL_MAX_PAIR_START} (UTC); суффикс _val",
    )
    split_group.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Сводка: универсальность среди пар + устойчивость во времени; "
            "отдельный weekday_summary.log"
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
        help="Начало периода (ISO UTC), например 2024-01-01; для --summary не нужен",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Конец периода (ISO UTC), например 2024-12-31; для --summary не нужен",
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
    parser.add_argument(
        "--highlight-weekdays",
        nargs="+",
        default=None,
        metavar="DAY",
        help=(
            "Выделить дни на NAV-графике более толстыми линиями "
            "(пн, вт, ср, чт, пт, сб, вс или mon..sun)"
        ),
    )
    parser.add_argument(
        "--main-plot-only",
        action="store_true",
        help="Только главный NAV-график (без мини-графиков по парам)",
    )
    args = parser.parse_args()
    if not args.summary and (not args.from_date or not args.to_date):
        parser.error("--from-date и --to-date обязательны вне режима --summary")
    return args
