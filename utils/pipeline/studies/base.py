"""Базовый контракт исследования report_generator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from crypto_research.utils.pipeline.report_context import ReportContext
    from crypto_research.utils.pipeline.study_dataset import StudyDataset, SummaryDatasets


class StudyHandler(ABC):
    supports_summary: bool = False

    @abstractmethod
    def validate_args(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        """Проверка CLI-аргументов; parser.error() при нарушении."""

    def resolve_max_pair_start(
        self,
        max_pair_start: datetime | None,
        *,
        summary: bool,
    ) -> datetime | None:
        return max_pair_start

    @abstractmethod
    def run(self, ctx: ReportContext, dataset: StudyDataset) -> Path:
        ...

    def run_summary(self, ctx: ReportContext, summary: SummaryDatasets) -> Path:
        raise NotImplementedError(f"Исследование {type(self).__name__} не поддерживает --summary")
