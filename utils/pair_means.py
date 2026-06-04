import polars as pl

from statistic.day_return_stats import MeanBands, build_pair_bands_map
from crypto_research.utils.logger import get_logger

log = get_logger("pair_means")


def compute_pair_means(daily: pl.DataFrame) -> dict[str, MeanBands]:
    bands = build_pair_bands_map(daily, report_split=None)
    log.info("[1b] Mean по парам: пар=%s", len(bands))
    return bands
