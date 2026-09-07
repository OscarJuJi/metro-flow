"""Data-quality flags that are not service interruptions.

Only one such defect exists in the Metro panel, and it is worth handling
explicitly rather than silently: throughout December 2020 the row for
``Deportivo Oceanía`` on Line B is labelled ``Oceanía``, so that month has two
``Oceanía`` rows a day and no ``Deportivo Oceanía`` row at all.

Neither magnitude nor file order resolves which row is which -- in November 2020
the two stations are not even written in a consistent order within the day -- so
the 62 rows are flagged instead of guessed at. They keep their place in the
panel, but they are not used as training targets, as evaluation targets or as
lag features.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PANEL_KEY = ["date", "line_id", "station_id"]


@dataclass
class DuplicateReport:
    """What the duplicate scan found, for logging and for tests."""

    n_rows: int
    n_dates: int
    stations: list[str]
    date_min: str | None
    date_max: str | None

    def summary(self) -> str:
        if not self.n_rows:
            return "no duplicate station-days"
        return (
            f"{self.n_rows} rows on {self.n_dates} dates "
            f"({self.date_min}..{self.date_max}) for {self.stations}"
        )


def find_duplicate_station_days(canonical: pd.DataFrame) -> tuple[pd.Index, DuplicateReport]:
    """Locate station-days that appear more than once in the panel."""
    duplicated = canonical.duplicated(subset=PANEL_KEY, keep=False)
    rows = canonical.index[duplicated]
    if not len(rows):
        return rows, DuplicateReport(0, 0, [], None, None)

    affected = canonical.loc[rows]
    report = DuplicateReport(
        n_rows=len(rows),
        n_dates=affected["date"].nunique(),
        stations=sorted(affected["station_id"].unique().tolist()),
        date_min=affected["date"].min().strftime("%Y-%m-%d"),
        date_max=affected["date"].max().strftime("%Y-%m-%d"),
    )
    return rows, report
