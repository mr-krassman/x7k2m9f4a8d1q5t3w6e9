import logging
import sys

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


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"{_ROOT}.{name}")
