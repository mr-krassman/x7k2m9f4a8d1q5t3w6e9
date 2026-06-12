import os
from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.study_ids import (
    STUDY_DAY_OF_WEEK,
    STUDY_EMA_PERIOD_SCREEN,
    STUDY_EMA_SPREADS,
)

RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
_JSONL_GLOB = "*_klines_1m.jsonl"
_LEGACY_DATA_DIR = RESEARCH_ROOT.parent / "load_data_from_bybit" / "data"

_STUDY_STATS_REL: dict[str, Path] = {
    STUDY_DAY_OF_WEEK: Path("day_of_week") / "statistics",
    STUDY_EMA_SPREADS: Path("ema") / "ema_spreads" / "statistics",
    STUDY_EMA_PERIOD_SCREEN: Path("ema") / "ema_spreads" / "statistics",
}


def resolve_default_data_dir() -> Path:
    if os.environ.get("CRYPTO_DATA_DIR"):
        return Path(os.environ["CRYPTO_DATA_DIR"]).expanduser().resolve()
    local = RESEARCH_ROOT / "data"
    if any(local.glob(_JSONL_GLOB)):
        return local
    if _LEGACY_DATA_DIR.is_dir() and any(_LEGACY_DATA_DIR.glob(_JSONL_GLOB)):
        return _LEGACY_DATA_DIR
    return local


DEFAULT_DATA_DIR = resolve_default_data_dir()

TRAIN_MAX_PAIR_START = "2022-01-01"
VAL_MAX_PAIR_START = "2023-01-01"

# Универсальность среди пар: train 24 / val 25, один период
PAIR_UNIVERSALITY_FROM = "2022-01-01"
PAIR_UNIVERSALITY_TO = "2026-05-31"

# Устойчивость во времени: 49 пар, два периода
TEMPORAL_POOL_MAX_PAIR_START = VAL_MAX_PAIR_START
TEMPORAL_TRAIN_FROM = "2022-01-01"
TEMPORAL_TRAIN_TO = "2024-04-01"
TEMPORAL_VAL_FROM = "2024-04-01"
TEMPORAL_VAL_TO = "2026-05-31"

# Полный пул: все пары, весь период исследования
FULL_POOL_MAX_PAIR_START = VAL_MAX_PAIR_START
FULL_POOL_FROM = PAIR_UNIVERSALITY_FROM
FULL_POOL_TO = PAIR_UNIVERSALITY_TO


def study_research_root(study: str) -> Path:
    """Корень артефактов исследования в research_outputs (без statistics/backtest)."""
    stats = _STUDY_STATS_REL[study]
    return RESEARCH_ROOT / "research_outputs" / stats.parent


def study_stats_dir(study: str) -> Path:
    return RESEARCH_ROOT / "research_outputs" / _STUDY_STATS_REL[study]


def study_plots_dir(study: str) -> Path:
    return study_stats_dir(study) / "plots"


def study_backtest_dir(study: str) -> Path:
    return study_research_root(study) / "backtest"


def study_backtest_plots_dir(study: str) -> Path:
    return study_backtest_dir(study) / "plots"


# Обратная совместимость (те же пути, что до унификации)
WEEKDAY_STATS_DIR = study_stats_dir(STUDY_DAY_OF_WEEK)
WEEKDAY_PLOTS_DIR = study_plots_dir(STUDY_DAY_OF_WEEK)
EMA_SPREADS_STATS_DIR = study_stats_dir(STUDY_EMA_SPREADS)


def weekday_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> str:
    tag = f"{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    if split:
        tag = f"{tag}_{split}"
    return tag


def weekday_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date, split)
    return WEEKDAY_STATS_DIR / f"weekday_statistics_{tag}.log"


def weekday_summary_log_path() -> Path:
    return WEEKDAY_STATS_DIR / "weekday_summary.log"


def weekday_train_val_summary_log_path(
    n_train: int,
    n_val: int,
    from_date: datetime,
    to_date: datetime,
) -> Path:
    tag = f"{n_train}train_{n_val}val_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return WEEKDAY_STATS_DIR / f"weekday_train_val_summary_{tag}.log"


def weekday_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    split: str | None = None,
) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date, split)
    return WEEKDAY_PLOTS_DIR / f"dow_intraday_session_nav_{tag}.png"


def weekday_check_plot_path(check: str) -> Path:
    """Train+val NAV на одном полотне для проверки (pair_universality | temporal_stability)."""
    return WEEKDAY_PLOTS_DIR / f"dow_check_{check}_train_val_nav.png"


def ema_spreads_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> str:
    periods_tag = "_".join(str(p) for p in periods)
    return f"{n_pairs}pairs_ema{periods_tag}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def ema_spreads_stats_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return EMA_SPREADS_STATS_DIR / f"ema_spreads_{tag}.log"


def ema_period_screen_log_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return EMA_SPREADS_STATS_DIR / f"ema_period_screen_{tag}.log"


def ema_summary_log_path(ema_period: int) -> Path:
    return EMA_SPREADS_STATS_DIR / f"ema_summary_ema{ema_period}.log"


def ema_summary_plot_path(ema_period: int, plot_kind: str) -> Path:
    return study_plots_dir(STUDY_EMA_SPREADS) / f"ema_summary_{plot_kind}_ema{ema_period}.png"


def ema_period_screen_plot_path(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    periods: tuple[int, ...],
    plot_kind: str,
) -> Path:
    tag = ema_spreads_output_tag(n_pairs, from_date, to_date, periods)
    return study_plots_dir(STUDY_EMA_SPREADS) / f"ema_period_screen_{plot_kind}_{tag}.png"
