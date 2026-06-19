from datetime import datetime
from pathlib import Path

from crypto_research.utils.ml.registry import (
    ML_STUDY_DAY_OF_WEEK,
    ML_STUDY_EMA_SPREADS,
    ML_STUDY_PRICE_SEQUENCES,
    ML_STUDY_RSI_SPREADS,
    is_ml_study_id,
)
from crypto_research.utils.pipeline.paths import study_backtest_dir, study_backtest_plots_dir

# CLI-имя ML-бэктеста → каталог research_outputs (как у rule-based стратегий)
_ML_BACKTEST_OUTPUT_STUDY: dict[str, str] = {
    ML_STUDY_DAY_OF_WEEK: "day_of_week",
    ML_STUDY_EMA_SPREADS: "ema_spreads",
    ML_STUDY_RSI_SPREADS: "rsi_spreads",
    ML_STUDY_PRICE_SEQUENCES: "price_sequences",
}


def strategy_backtest_dir(
    strategy: str,
    *,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
) -> Path:
    if bundle_id is not None and bundle_kind is not None:
        from crypto_research.utils.backtest.bundle_registry import bundle_backtest_dir

        return bundle_backtest_dir(bundle_id, bundle_kind)  # type: ignore[arg-type]
    if output_study is not None:
        return study_backtest_dir(output_study)
    study = _ML_BACKTEST_OUTPUT_STUDY.get(strategy, strategy)
    return study_backtest_dir(study)


def strategy_backtest_plots_dir(
    strategy: str,
    *,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
) -> Path:
    if bundle_id is not None and bundle_kind is not None:
        from crypto_research.utils.backtest.bundle_registry import bundle_backtest_plots_dir

        return bundle_backtest_plots_dir(bundle_id, bundle_kind)  # type: ignore[arg-type]
    if output_study is not None:
        return study_backtest_plots_dir(output_study)
    study = _ML_BACKTEST_OUTPUT_STUDY.get(strategy, strategy)
    return study_backtest_plots_dir(study)


def backtest_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    optimistic_segment: str | None = None,
) -> str:
    if scenario == "optimistic" and selected_pairs is not None:
        segment = optimistic_segment or "b6"
        return f"optimistic_{segment}_{len(selected_pairs)}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    if scenario == "optimistic" and pairs_by_weekday:
        thu = len(pairs_by_weekday.get(3, ()))
        pt = len(pairs_by_weekday.get(4, ()))
        sb = len(pairs_by_weekday.get(5, ()))
        return f"optimistic_thu{thu}_pt{pt}_sb{sb}_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
    return f"{n_pairs}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"


def _tag(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    combine_mode: str | None = None,
    bundle_kind: str | None = None,
    optimistic_segment: str | None = None,
) -> str:
    tag = backtest_output_tag(
        n_pairs,
        from_date,
        to_date,
        scenario=scenario,
        pairs_by_weekday=pairs_by_weekday,
        selected_pairs=selected_pairs,
        optimistic_segment=optimistic_segment,
    )
    if combine_mode:
        tag = f"{combine_mode}_{tag}"
    if bundle_kind == "ml" or (bundle_kind is None and is_ml_study_id(strategy)):
        return f"ml_{tag}"
    return tag


def _path_kw(
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> dict:
    return {
        "scenario": scenario,
        "pairs_by_weekday": pairs_by_weekday,
        "selected_pairs": selected_pairs,
        "output_study": output_study,
        "bundle_id": bundle_id,
        "bundle_kind": bundle_kind,
        "combine_mode": combine_mode,
        "optimistic_segment": optimistic_segment,
    }


def backtest_report_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> Path:
    tag = _tag(
        strategy,
        n_pairs,
        from_date,
        to_date,
        scenario,
        pairs_by_weekday,
        selected_pairs,
        combine_mode=combine_mode,
        bundle_kind=bundle_kind,
        optimistic_segment=optimistic_segment,
    )
    return strategy_backtest_dir(
        strategy,
        output_study=output_study,
        bundle_id=bundle_id,
        bundle_kind=bundle_kind,
    ) / f"{strategy}_backtest_{tag}.log"


def backtest_equity_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> Path:
    tag = _tag(
        strategy,
        n_pairs,
        from_date,
        to_date,
        scenario,
        pairs_by_weekday,
        selected_pairs,
        combine_mode=combine_mode,
        bundle_kind=bundle_kind,
        optimistic_segment=optimistic_segment,
    )
    return strategy_backtest_plots_dir(
        strategy,
        output_study=output_study,
        bundle_id=bundle_id,
        bundle_kind=bundle_kind,
    ) / f"equity_curve_{tag}.png"


def backtest_drawdown_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> Path:
    tag = _tag(
        strategy,
        n_pairs,
        from_date,
        to_date,
        scenario,
        pairs_by_weekday,
        selected_pairs,
        combine_mode=combine_mode,
        bundle_kind=bundle_kind,
        optimistic_segment=optimistic_segment,
    )
    return strategy_backtest_plots_dir(
        strategy,
        output_study=output_study,
        bundle_id=bundle_id,
        bundle_kind=bundle_kind,
    ) / f"drawdown_{tag}.png"


def backtest_returns_hist_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> Path:
    tag = _tag(
        strategy,
        n_pairs,
        from_date,
        to_date,
        scenario,
        pairs_by_weekday,
        selected_pairs,
        combine_mode=combine_mode,
        bundle_kind=bundle_kind,
        optimistic_segment=optimistic_segment,
    )
    return strategy_backtest_plots_dir(
        strategy,
        output_study=output_study,
        bundle_id=bundle_id,
        bundle_kind=bundle_kind,
    ) / f"returns_hist_{tag}.png"


def backtest_weekday_corr_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
    output_study: str | None = None,
    bundle_id: str | None = None,
    bundle_kind: str | None = None,
    combine_mode: str | None = None,
    optimistic_segment: str | None = None,
) -> Path:
    tag = _tag(
        strategy,
        n_pairs,
        from_date,
        to_date,
        scenario,
        pairs_by_weekday,
        selected_pairs,
        combine_mode=combine_mode,
        bundle_kind=bundle_kind,
        optimistic_segment=optimistic_segment,
    )
    return strategy_backtest_plots_dir(
        strategy,
        output_study=output_study,
        bundle_id=bundle_id,
        bundle_kind=bundle_kind,
    ) / f"weekday_corr_{tag}.png"
