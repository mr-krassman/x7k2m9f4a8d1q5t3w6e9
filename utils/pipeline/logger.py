import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"
_ROOT = "crypto_research"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger(_ROOT)
    if root.handlers:
        root.setLevel(level)
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, _DATE))
    root.addHandler(handler)


def add_file_logging(log_path: Path, level: int = logging.INFO) -> Path:
    """Дописывает все логи crypto_research.* в файл (stderr остаётся)."""
    setup_logging(level)
    root = logging.getLogger(_ROOT)
    resolved = log_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved:
            return resolved
    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FMT, _DATE))
    root.addHandler(file_handler)
    return resolved


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{_ROOT}.{name}")
