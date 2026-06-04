import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable

import polars as pl

from crypto_research.utils.pipeline.dates import datetime_to_ms

_BATCH = 65536
_SCHEMA = {
    "start_ms": pl.Int64,
    "open": pl.Utf8,
    "high": pl.Utf8,
    "low": pl.Utf8,
    "close": pl.Utf8,
    "volume": pl.Utf8,
}
_FLOAT = ("open", "high", "low", "close", "volume")
_SUFFIX = "_klines_1m.jsonl"
_DEFAULT_WORKERS = min(8, os.cpu_count() or 4)


def pair_jsonl_path(data_dir: Path, pair: str) -> Path:
    return data_dir / f"{pair}{_SUFFIX}"


def pair_name_from_jsonl(path: Path) -> str:
    return path.name.removesuffix(_SUFFIX)


def list_pair_names(data_dir: str | Path) -> list[str]:
    root = Path(data_dir).expanduser().resolve()
    return sorted(pair_name_from_jsonl(p) for p in root.glob(f"*{_SUFFIX}"))


def read_first_start_ms(jsonl_path: Path) -> int | None:
    try:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ms = obj.get("start_ms")
                if ms is not None:
                    return int(ms)
    except OSError:
        return None
    return None


def _pair_passes_max_start(pair: str, path: Path, limit_ms: int) -> str | None:
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла данных: {path}")
    first_ms = read_first_start_ms(path)
    if first_ms is None or first_ms > limit_ms:
        return None
    return pair


def resolve_pairs(
    data_dir: Path,
    pair_names: list[str] | None,
    max_pair_start: datetime | None,
    *,
    workers: int = _DEFAULT_WORKERS,
) -> list[str]:
    candidates = pair_names if pair_names else list_pair_names(data_dir)
    if not candidates:
        raise FileNotFoundError(
            f"Нет *_klines_1m.jsonl в {data_dir}. "
            "Укажите --data-dir или export CRYPTO_DATA_DIR=/path/to/jsonl"
        )
    if max_pair_start is None:
        return candidates
    limit_ms = datetime_to_ms(max_pair_start)
    paths = [(pair, pair_jsonl_path(data_dir, pair)) for pair in candidates]

    def task(item: tuple[str, Path]) -> str | None:
        return _pair_passes_max_start(item[0], item[1], limit_ms)

    if workers <= 1 or len(paths) <= 1:
        selected = [p for p in (task(x) for x in paths) if p is not None]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            selected = [p for p in ex.map(task, paths) if p is not None]
    if not selected:
        raise RuntimeError("После фильтра max-pair-start не осталось ни одной пары")
    return selected


def _cast_f32(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col(c).cast(pl.Float32) for c in _FLOAT)


def _load_one(path: Path, from_ms: int | None, to_ms: int | None) -> pl.DataFrame:
    if from_ms is None and to_ms is None:
        return _cast_f32(pl.read_ndjson(path, schema=_SCHEMA, batch_size=_BATCH))
    lf = _cast_f32(pl.scan_ndjson(path, schema=_SCHEMA, batch_size=_BATCH))
    if from_ms is not None:
        lf = lf.filter(pl.col("start_ms") >= from_ms)
    if to_ms is not None:
        lf = lf.filter(pl.col("start_ms") <= to_ms)
    return lf.collect(engine="streaming")


def load_pairs_klines(
    data_dir: str | Path,
    pairs: Iterable[str],
    *,
    from_ms: int | None = None,
    to_ms: int | None = None,
    workers: int = _DEFAULT_WORKERS,
) -> dict[str, pl.DataFrame]:
    root = Path(data_dir).expanduser().resolve()
    pair_list = list(pairs)
    if not pair_list:
        return {}

    def task(pair: str) -> tuple[str, pl.DataFrame]:
        return pair, _load_one(root / f"{pair}{_SUFFIX}", from_ms, to_ms)

    if workers <= 1 or len(pair_list) == 1:
        return dict(task(pair) for pair in pair_list)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(task, pair_list))


def load_klines_for_period(
    data_dir: Path,
    from_date: datetime,
    to_date: datetime,
    pairs: list[str] | None,
    max_pair_start: datetime | None,
    *,
    workers: int = _DEFAULT_WORKERS,
) -> dict[str, pl.DataFrame]:
    resolved = resolve_pairs(data_dir, pairs, max_pair_start, workers=workers)
    from_ms = datetime_to_ms(from_date)
    to_ms = datetime_to_ms(to_date)
    if from_ms > to_ms:
        raise ValueError(f"from_date ({from_date}) позже to_date ({to_date})")
    return load_pairs_klines(data_dir, resolved, from_ms=from_ms, to_ms=to_ms, workers=workers)
