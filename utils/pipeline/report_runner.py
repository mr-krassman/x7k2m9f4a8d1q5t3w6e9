"""Единая точка запуска report_generator."""

from __future__ import annotations

from crypto_research.utils.pipeline.report_context import build_report_context
from crypto_research.utils.pipeline.studies import STUDY_HANDLERS
from crypto_research.utils.pipeline.study_dataset import load_primary_dataset, load_summary_datasets


def run_report(args) -> None:
    ctx = build_report_context(args)
    handler = STUDY_HANDLERS[ctx.study]

    if ctx.summary:
        if not handler.supports_summary:
            raise ValueError(f"Исследование {ctx.study!r} не поддерживает --summary")
        summary = load_summary_datasets(ctx)
        handler.run_summary(ctx, summary)
        return

    dataset = load_primary_dataset(ctx)
    handler.run(ctx, dataset)
