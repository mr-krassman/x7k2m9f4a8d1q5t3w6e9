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


def weekday_output_tag(n_pairs: int, from_date: datetime, to_date: datetime) -> str:
    return f"{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def weekday_stats_log_path(n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date)
    return WEEKDAY_STATS_DIR / f"weekday_statistics_{tag}.log"


def weekday_plot_path(n_pairs: int, from_date: datetime, to_date: datetime) -> Path:
    tag = weekday_output_tag(n_pairs, from_date, to_date)
    return WEEKDAY_PLOTS_DIR / f"dow_intraday_session_nav_{tag}.png"
