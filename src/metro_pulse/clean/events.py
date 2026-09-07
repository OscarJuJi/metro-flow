"""Extract the network's service interruptions from the derived status column.

These windows are not an input to the pipeline -- the mask is derived from the
data, not from a hand-written list. They are an *output*: a labelled record of
what actually happened, used to sanity-check the mask against documented history
and, in the anomaly-detection stage, as ground truth for measuring recall.

Run as ``python -m metro_pulse.clean.events`` to refresh
``reference/<operator>_service_events.csv``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from metro_pulse.clean.service import CLOSED, NOT_MEASURED, NOT_YET_OPEN, find_runs
from metro_pulse.config import INTERIM_DIR, OPERATORS, PROJECT_ROOT, OperatorSource

INTERRUPTION_STATUSES = (CLOSED, NOT_YET_OPEN, NOT_MEASURED)


def extract_interruptions(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, uninterrupted window of the same non-operating status)."""
    records = []
    ordered = panel.sort_values(["line_id", "station_id", "date"])
    for (line_id, station_id), group in ordered.groupby(["line_id", "station_id"], observed=True):
        dates = group["date"].to_numpy()
        for status in INTERRUPTION_STATUSES:
            flags = (group["status"] == status).to_numpy()
            for start, end in find_runs(flags):
                records.append(
                    {
                        "line_id": line_id,
                        "station_id": station_id,
                        "status": status,
                        "start": pd.Timestamp(dates[start]).date(),
                        "end": pd.Timestamp(dates[end]).date(),
                        "days": end - start + 1,
                    }
                )
    return pd.DataFrame.from_records(records, columns=[
        "line_id", "station_id", "status", "start", "end", "days"
    ])


def group_into_events(interruptions: pd.DataFrame, *, min_days: int = 7) -> pd.DataFrame:
    """Collapse per-station windows into network events sharing a line and span."""
    if interruptions.empty:
        return interruptions.assign(n_stations=[], stations=[])
    long_enough = interruptions[interruptions["days"] >= min_days]
    grouped = long_enough.groupby(["line_id", "status", "start", "end", "days"], observed=True)
    events = grouped.agg(
        n_stations=("station_id", "nunique"),
        stations=("station_id", lambda s: "|".join(sorted(s.unique()))),
    ).reset_index()
    return events.sort_values(["start", "line_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    parser.add_argument("--min-days", type=int, default=7)
    args = parser.parse_args(argv)

    operator: OperatorSource = OPERATORS[args.operator]
    panel = pd.read_parquet(INTERIM_DIR / f"{operator.key}_panel.parquet")

    interruptions = extract_interruptions(panel)
    events = group_into_events(interruptions, min_days=args.min_days)

    destination = PROJECT_ROOT / "reference" / f"{operator.key}_service_events.csv"
    events.to_csv(destination, index=False, encoding="utf-8")

    print(f"{len(interruptions):,} station-level interruptions")
    print(f"{len(events):,} network events of at least {args.min_days} days -> {destination}")
    print()
    print(events.drop(columns="stations").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
