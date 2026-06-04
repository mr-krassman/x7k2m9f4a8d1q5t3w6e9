import os
from datetime import datetime
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
_JSONL_GLOB = "*_klines_1m.jsonl"
_LEGACY_DATA_DIR = RESEARCH_ROOT.parent / "load_data_from_bybit" / "data"


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
WEEKDAY_STATS_DIR = RESEARCH_ROOT / "research_outputs" / "day_of_week" / "statistics"
WEEKDAY_PLOTS_DIR = WEEKDAY_STATS_DIR / "plots"

TRAIN_MAX_PAIR_START = "2022-01-01"
VAL_MAX_PAIR_START = "2023-01-01"


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
