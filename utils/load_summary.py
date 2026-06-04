from datetime import datetime, timezone

import polars as pl

from crypto_research.utils.logger import get_logger

log = get_logger("load")


def log_load_summary(data: dict[str, pl.DataFrame]) -> None:
    total_rows = sum(df.height for df in data.values())
    total_mb = sum(df.estimated_size() for df in data.values()) / 2**20
    log.info("[1] Загрузка данных: пар=%s строк=%s ~%.1f MB", len(data), total_rows, total_mb)
    for pair, df in sorted(data.items()):
        if df.height == 0:
            log.info("  %s: пусто", pair)
            continue
        t0 = datetime.fromtimestamp(df["start_ms"].min() / 1000, tz=timezone.utc)
        t1 = datetime.fromtimestamp(df["start_ms"].max() / 1000, tz=timezone.utc)
        log.info("  %s: rows=%s %s .. %s", pair, df.height, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
