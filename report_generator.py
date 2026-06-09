#!/usr/bin/env python3
"""Оркестратор отчётов crypto_research."""

import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from crypto_research.utils.pipeline.cli import parse_report_args
from crypto_research.utils.pipeline.report_runner import run_report


def main() -> int:
    run_report(parse_report_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
