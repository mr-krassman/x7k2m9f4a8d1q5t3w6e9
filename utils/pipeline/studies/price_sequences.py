"""Исследование price_sequences: серии роста/падения 1…6 дней."""

from __future__ import annotations

import argparse
from pathlib import Path

from crypto_research.utils.pipeline.price_sequences_train_val_summary import run_summary_report
from crypto_research.utils.pipeline.report_context import ReportContext
from crypto_research.utils.pipeline.studies.base import StudyHandler
from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets
from crypto_research.utils.price_sequences.report import run_price_sequences_report


class PriceSequencesStudy(StudyHandler):
    supports_summary = True

    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        if not args.summary and (not args.from_date or not args.to_date):
            parser.error("--from-date и --to-date обязательны вне режима --summary")
        if args.select_pairs_by_train:
            parser.error("--select-pairs-by-train только для day_of_week")

    def run(self, ctx: ReportContext, dataset: StudyDataset) -> Path:
        return run_price_sequences_report(
            dataset.daily,
            dataset.pair_bands,
            dataset.pairs,
            ctx.from_date,
            ctx.to_date,
            ctx.max_pair_start,
            ctx.split,
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
