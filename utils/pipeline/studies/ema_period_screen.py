"""Этап 0: выбор периода EMA по стабильности сигнала."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crypto_research.utils.ema_spreads.period_screen_report import run_ema_period_screen_report
from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import FULL_POOL_MAX_PAIR_START
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.studies.base import StudyHandler
from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets


class EmaPeriodScreenStudy(StudyHandler):
    supports_summary = False

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.from_date or not args.to_date:
            parser.error("ema_period_screen: --from-date и --to-date обязательны")
        if args.summary:
            parser.error("ema_period_screen не поддерживает --summary")
        if args.train or args.val:
            parser.error("ema_period_screen не поддерживает --train / --val")
        if args.select_pairs_by_train:
            parser.error("--select-pairs-by-train только для day_of_week")

    def resolve_max_pair_start(
        self,
        max_pair_start: datetime | None,
        *,
        summary: bool,
    ) -> datetime | None:
        if summary or max_pair_start is not None:
            return max_pair_start
        return parse_iso_utc(FULL_POOL_MAX_PAIR_START)

    def run(self, ctx: ReportContext, dataset: StudyDataset) -> Path:
        return run_ema_period_screen_report(
            dataset.daily,
            dataset.pairs,
            dataset.pair_bands,
            ctx.from_date,
            ctx.to_date,
            ctx.ema_periods,
            ctx.max_pair_start,
        )

    def run_summary(self, ctx: ReportContext, summary: SummaryDatasets) -> Path:
        raise NotImplementedError("ema_period_screen не поддерживает --summary")
