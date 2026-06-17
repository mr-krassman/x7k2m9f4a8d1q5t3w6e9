"""Исследование rsi_spreads."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.dates import parse_iso_utc
from crypto_research.utils.pipeline.paths import FULL_POOL_MAX_PAIR_START
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.rsi_train_val_summary import run_summary_report
from crypto_research.utils.pipeline.studies.base import StudyHandler
from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets
from crypto_research.utils.rsi.report import run_rsi_spreads_report


class RsiSpreadsStudy(StudyHandler):
    supports_summary = True

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.summary and (not args.from_date or not args.to_date):
            parser.error("rsi_spreads: --from-date и --to-date обязательны вне режима --summary")
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
        return run_rsi_spreads_report(
            dataset.daily,
            dataset.pair_bands,
            dataset.pairs,
            ctx.from_date,
            ctx.to_date,
            ctx.rsi_periods,
            ctx.max_pair_start,
        )

    def run_summary(self, ctx: ReportContext, summary: SummaryDatasets) -> Path:
        return run_summary_report(
            ctx,
            summary.train_daily,
            summary.val_daily,
            summary.train_pairs,
            summary.val_pairs,
            summary.temporal_train,
            summary.temporal_val,
            summary.temporal_pairs,
            summary.full_daily,
            summary.full_pairs,
        )
