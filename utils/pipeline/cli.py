import argparse
from pathlib import Path

from crypto_research.utils.pipeline.paths import DEFAULT_DATA_DIR, TRAIN_MAX_PAIR_START, VAL_MAX_PAIR_START
from crypto_research.utils.ema_spreads.constants import (
    DEFAULT_EMA_PERIODS,
    DEFAULT_SCREEN_EMA_PERIODS,
)
from crypto_research.utils.volume.constants import DEFAULT_SCREEN_VOLUME_EMA_PERIODS
from crypto_research.utils.volatility.constants import (
    DEFAULT_SCREEN_RANGE_SMA_PERIODS,
    SELECTED_RANGE_SMA_PERIOD,
)
from crypto_research.utils.rsi.constants import DEFAULT_RSI_PERIODS, DEFAULT_SCREEN_RSI_PERIODS
from crypto_research.utils.pipeline.study_ids import (
    ALL_STUDIES,
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_PERIOD_SCREEN,
    STUDY_EMA_SPREADS,
    STUDY_EMA_HARAMI,
    STUDY_HARAMI,
)


def parse_report_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оркестратор отчётов crypto_research.")
    parser.add_argument(
        "study",
        choices=list(ALL_STUDIES),
        help=(
            "Исследование: day_of_week, ema_spreads, ema_period_screen, rsi_period_screen, "
            "rsi_spreads, volume_ema_period_screen, volume_spreads, volatility_period_screen, "
            "volatility_spreads, harami, ema_harami, price_sequences"
        ),
    )
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
    parser.add_argument(
        "--select-pairs-by-train",
        action="store_true",
        help=(
            "Добавить в отчёт секцию отбора пар для оптимистичного сценария day_of_week: "
            "знак Δ к BASE + подтверждение в 2/3 годов train-периода"
        ),
    )
    parser.add_argument(
        "--ema-periods",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Периоды EMA: ema_spreads "
            f"({ ' '.join(str(p) for p in DEFAULT_EMA_PERIODS) }); "
            f"ema_period_screen ({' '.join(str(p) for p in DEFAULT_SCREEN_EMA_PERIODS)}); "
            f"volume_ema_period_screen ({' '.join(str(p) for p in DEFAULT_SCREEN_VOLUME_EMA_PERIODS)}); "
            f"volatility_period_screen ({' '.join(str(p) for p in DEFAULT_SCREEN_RANGE_SMA_PERIODS)}); "
            f"volatility_spreads (SMA {SELECTED_RANGE_SMA_PERIOD})"
        ),
    )
    parser.add_argument(
        "--rsi-periods",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Периоды RSI: rsi_spreads "
            f"({' '.join(str(p) for p in DEFAULT_RSI_PERIODS)}); "
            f"rsi_period_screen ({' '.join(str(p) for p in DEFAULT_SCREEN_RSI_PERIODS)})"
        ),
    )
    args = parser.parse_args()
    from crypto_research.utils.pipeline.studies import STUDY_HANDLERS

    STUDY_HANDLERS[args.study].validate_args(parser, args)
    return args
