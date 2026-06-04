import polars as pl

from crypto_research.utils.pipeline.logger import get_logger
from crypto_research.utils.weekday.bands import MeanBands, build_pair_bands_map

log = get_logger("pair_means")


def compute_pair_means(daily: pl.DataFrame) -> dict[str, MeanBands]:
    bands = build_pair_bands_map(daily)
    log.info("[1b] Mean по парам: пар=%s", len(bands))
    return bands
