"""Loading and reshaping the modelling panel.

The panel is stored long (one row per station-day) because that is how it is
built and how it is served. Every forecasting routine here works on the wide
form instead -- a dates x series matrix -- because that is the shape in which
lags, seasonal indices and rolling statistics are one numpy slice each.
"""

from __future__ import annotations

import pandas as pd

from metro_pulse.config import INTERIM_DIR

SERIES_SEPARATOR = ":"


def series_id(line_id: str, station_id: str) -> str:
    """``("1", "zaragoza")`` -> ``"1:zaragoza"``.

    A station served by several lines is several series: Pantitlán on Line 1 and
    Pantitlán on Line 9 have different demand and different closures.
    """
    return f"{line_id}{SERIES_SEPARATOR}{station_id}"


def split_series_id(value: str) -> tuple[str, str]:
    line_id, station_id = value.split(SERIES_SEPARATOR, 1)
    return line_id, station_id


def load_panel(operator: str = "metro") -> pd.DataFrame:
    """Read the panel written by :mod:`metro_pulse.clean.build`."""
    path = INTERIM_DIR / f"{operator}_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m metro_pulse.clean.build --operator {operator}`."
        )
    panel = pd.read_parquet(path)
    panel["series_id"] = panel["line_id"] + SERIES_SEPARATOR + panel["station_id"]
    return panel


def to_wide(panel: pd.DataFrame, value: str = "target") -> pd.DataFrame:
    """Pivot to a ``dates x series`` matrix, keeping the calendar contiguous.

    Days a station was closed stay in the matrix as ``NaN``: dropping them would
    silently shift every lag by however many days the station was shut.
    """
    ids = (
        panel["series_id"]
        if "series_id" in panel.columns
        else panel["line_id"] + SERIES_SEPARATOR + panel["station_id"]
    )
    wide = panel.assign(series_id=ids).pivot_table(
        index="date", columns="series_id", values=value, aggfunc="first", dropna=False
    )
    full_calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full_calendar).rename_axis(index="date", columns="series_id")
