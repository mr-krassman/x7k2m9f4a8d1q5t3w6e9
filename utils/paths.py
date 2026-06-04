import os
from datetime import datetime
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("CRYPTO_DATA_DIR", RESEARCH_ROOT / "data"))
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
