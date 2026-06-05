"""Форматирование оси дат: год на январе, месяцы Apr/Jul/Oct на кварталах."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator


def quarter_tick_label(value: float, _pos: int) -> str:
    dt = mdates.num2date(value)
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    if dt.month == 1:
        return dt.strftime("%Y")
    return dt.strftime("%b")


def format_date_axis(ax: plt.Axes, *, rotate: int = 0, labelsize: int | None = None) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(FuncFormatter(quarter_tick_label))
    ax.xaxis.set_minor_locator(NullLocator())
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotate,
        ha="right" if rotate else "center",
        fontsize=labelsize,
    )
