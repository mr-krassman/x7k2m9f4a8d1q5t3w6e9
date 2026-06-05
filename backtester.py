#!/usr/bin/env python3
"""Оркестратор бэктестов crypto_research."""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.backtest.cli import parse_backtest_args
from crypto_research.utils.backtest.context import build_backtest_context
from crypto_research.utils.backtest.runner import run_backtest


def run(args) -> None:
    ctx = build_backtest_context(args)
    run_backtest(ctx)


def main() -> int:
    run(parse_backtest_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
