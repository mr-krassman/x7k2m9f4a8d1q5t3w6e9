"""Исследование volatility_spreads — подтверждение сигналов range/SMA."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.studies.base import StudyHandler
from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets
from crypto_research.utils.pipeline.volatility_train_val_summary import run_summary_report


class VolatilitySpreadsStudy(StudyHandler):
    supports_summary = True

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.summary:
            parser.error("volatility_spreads: используйте --summary")
        if args.train or args.val:
            parser.error("volatility_spreads не поддерживает --train / --val")
        if args.select_pairs_by_train:
            parser.error("--select-pairs-by-train только для day_of_week")

    def resolve_max_pair_start(
        self,
        max_pair_start: datetime | None,
        *,
        summary: bool,
    ) -> datetime | None:
        return max_pair_start

    def run(self, ctx: ReportContext, dataset: StudyDataset) -> Path:
        raise NotImplementedError("volatility_spreads: используйте --summary")

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
