"""Schema contract for the raw ridership CSV.

The portal ships a data dictionary that does *not* match the file it describes
(it declares ``dia`` and ``ano`` columns that the CSV does not have). So the
contract enforced here is the one observed in the actual file, and it is checked
on every ingest: if the portal changes the schema, the pipeline stops instead of
silently training on a different dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = ("fecha", "anio", "mes", "linea", "estacion", "afluencia")

# Sanity floor, not an exact figure: the panel grows by one month at a time and
# stood at 1,180,920 rows on 2026-08-28. A file smaller than this means a
# truncated download or a republished subset.
MIN_EXPECTED_ROWS = 1_000_000


class RidershipSchemaError(ValueError):
    """The raw file does not honour the contract this pipeline was built against."""


@dataclass
class SchemaReport:
    """What the ingested file actually contains, for logging and for tests."""

    n_rows: int
    date_min: str
    date_max: str
    n_raw_line_labels: int
    n_raw_station_labels: int
    n_zero_rows: int
    rows_per_year: dict[int, int] = field(default_factory=dict)

    @property
    def zero_share(self) -> float:
        return self.n_zero_rows / self.n_rows if self.n_rows else 0.0

    def summary(self) -> str:
        return (
            f"rows={self.n_rows:,} | span={self.date_min}..{self.date_max} | "
            f"raw line labels={self.n_raw_line_labels} | raw station labels="
            f"{self.n_raw_station_labels} | zero rows={self.n_zero_rows:,} "
            f"({self.zero_share:.1%})"
        )


def validate_ridership_frame(
    frame: pd.DataFrame, *, min_rows: int = MIN_EXPECTED_ROWS
) -> SchemaReport:
    """Validate the raw ridership frame and describe it.

    Raises :class:`RidershipSchemaError` on any breach of the contract; returns a
    :class:`SchemaReport` describing the file when it passes.
    """
    actual = tuple(frame.columns)
    if actual != EXPECTED_COLUMNS:
        raise RidershipSchemaError(
            f"Unexpected columns.\n  expected: {EXPECTED_COLUMNS}\n  found:    {actual}"
        )

    if len(frame) < min_rows:
        raise RidershipSchemaError(
            f"Only {len(frame):,} rows; expected at least {min_rows:,}. "
            "The download is probably truncated or the portal republished a subset."
        )

    null_counts = frame.isna().sum()
    if null_counts.any():
        offenders = {col: int(n) for col, n in null_counts.items() if n}
        raise RidershipSchemaError(f"Null values are not expected in this source: {offenders}")

    dates = pd.to_datetime(frame["fecha"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        bad = frame.loc[dates.isna(), "fecha"].head(5).tolist()
        raise RidershipSchemaError(f"Unparseable dates in 'fecha', e.g. {bad}")

    if not pd.api.types.is_integer_dtype(frame["afluencia"]):
        raise RidershipSchemaError(
            f"'afluencia' must be an integer count, found dtype {frame['afluencia'].dtype}"
        )
    if (frame["afluencia"] < 0).any():
        raise RidershipSchemaError("Negative ridership values are not physically meaningful")

    year_mismatch = (dates.dt.year != frame["anio"]).sum()
    if year_mismatch:
        raise RidershipSchemaError(
            f"{year_mismatch:,} rows where 'anio' disagrees with the year in 'fecha'"
        )

    return SchemaReport(
        n_rows=len(frame),
        date_min=dates.min().strftime("%Y-%m-%d"),
        date_max=dates.max().strftime("%Y-%m-%d"),
        n_raw_line_labels=frame["linea"].nunique(),
        n_raw_station_labels=frame["estacion"].nunique(),
        n_zero_rows=int((frame["afluencia"] == 0).sum()),
        rows_per_year=frame["anio"].value_counts().sort_index().to_dict(),
    )
