from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.paths import study_backtest_dir, study_backtest_plots_dir


def strategy_backtest_dir(strategy: str) -> Path:
    study = "day_of_week" if strategy == "day_of_week_ml" else strategy
    return study_backtest_dir(study)


def strategy_backtest_plots_dir(strategy: str) -> Path:
    study = "day_of_week" if strategy == "day_of_week_ml" else strategy
    return study_backtest_plots_dir(study)


def backtest_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> str:
    if scenario == "optimistic" and selected_pairs is not None:
        return f"optimistic_b6_{len(selected_pairs)}pairs_{from_date:%Y%m%d}_{to_date:%Y%m%d}"
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
) -> str:
    tag = backtest_output_tag(
        n_pairs,
        from_date,
        to_date,
        scenario=scenario,
        pairs_by_weekday=pairs_by_weekday,
        selected_pairs=selected_pairs,
    )
    if strategy == "day_of_week_ml":
        return f"ml_{tag}"
    return tag


def backtest_report_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> Path:
    tag = _tag(
        strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday, selected_pairs
    )
    return strategy_backtest_dir(strategy) / f"{strategy}_backtest_{tag}.log"


def backtest_equity_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> Path:
    tag = _tag(
        strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday, selected_pairs
    )
    return strategy_backtest_plots_dir(strategy) / f"equity_curve_{tag}.png"


def backtest_drawdown_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> Path:
    tag = _tag(
        strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday, selected_pairs
    )
    return strategy_backtest_plots_dir(strategy) / f"drawdown_{tag}.png"


def backtest_returns_hist_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> Path:
    tag = _tag(
        strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday, selected_pairs
    )
    return strategy_backtest_plots_dir(strategy) / f"returns_hist_{tag}.png"


def backtest_weekday_corr_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
    selected_pairs: list[str] | None = None,
) -> Path:
    tag = _tag(
        strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday, selected_pairs
    )
    return strategy_backtest_plots_dir(strategy) / f"weekday_corr_{tag}.png"
