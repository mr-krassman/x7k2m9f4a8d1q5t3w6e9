from datetime import datetime
from pathlib import Path

from crypto_research.utils.pipeline.paths import RESEARCH_ROOT

BACKTEST_ROOT = RESEARCH_ROOT / "research_outputs"


def strategy_backtest_dir(strategy: str) -> Path:
    return BACKTEST_ROOT / strategy / "backtest"


def strategy_backtest_plots_dir(strategy: str) -> Path:
    return strategy_backtest_dir(strategy) / "plots"


def backtest_output_tag(
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> str:
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
) -> str:
    return backtest_output_tag(
        n_pairs,
        from_date,
        to_date,
        scenario=scenario,
        pairs_by_weekday=pairs_by_weekday,
    )


def backtest_report_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> Path:
    tag = _tag(strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday)
    return strategy_backtest_dir(strategy) / f"{strategy}_backtest_{tag}.log"


def backtest_equity_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> Path:
    tag = _tag(strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday)
    return strategy_backtest_plots_dir(strategy) / f"equity_curve_{tag}.png"


def backtest_drawdown_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> Path:
    tag = _tag(strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday)
    return strategy_backtest_plots_dir(strategy) / f"drawdown_{tag}.png"


def backtest_returns_hist_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> Path:
    tag = _tag(strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday)
    return strategy_backtest_plots_dir(strategy) / f"returns_hist_{tag}.png"


def backtest_weekday_corr_plot_path(
    strategy: str,
    n_pairs: int,
    from_date: datetime,
    to_date: datetime,
    *,
    scenario: str | None = None,
    pairs_by_weekday: dict[int, list[str]] | None = None,
) -> Path:
    tag = _tag(strategy, n_pairs, from_date, to_date, scenario, pairs_by_weekday)
    return strategy_backtest_plots_dir(strategy) / f"weekday_corr_{tag}.png"
