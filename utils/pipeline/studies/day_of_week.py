"""Исследование day_of_week."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.studies.base import StudyHandler
from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets
from crypto_research.utils.pipeline.weekday_effects import run_weekday_effects
from crypto_research.utils.pipeline.weekday_train_val_summary import run_summary_report


class DayOfWeekStudy(StudyHandler):
    supports_summary = True

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.summary and (not args.from_date or not args.to_date):
            parser.error("--from-date и --to-date обязательны вне режима --summary")

    def run(self, ctx: ReportContext, dataset: StudyDataset) -> Path:
        return run_weekday_effects(
            dataset.daily,
            dataset.pair_bands,
            pairs=dataset.pairs,
            from_date=ctx.from_date,
            to_date=ctx.to_date,
            max_pair_start=ctx.max_pair_start,
            split=ctx.split,
            highlight_weekdays=ctx.highlight_weekdays,
            main_plot_only=ctx.main_plot_only,
            select_pairs_by_train=ctx.select_pairs_by_train,
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
