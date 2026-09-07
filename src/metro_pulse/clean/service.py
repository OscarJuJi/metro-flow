"""Derive, from the data itself, whether a station was in service on a given day.

5.3% of the raw rows report zero ridership, and those zeros are not demand: they
are the operating history of the network written in the same column as the
measurements. Training on them teaches the model to predict closures; evaluating
on them rewards a model for predicting zero. So every station-day is classified
before anything else happens:

``not_yet_open``
    A leading run of zeros, at least ``min_closure_days`` long, from the first
    day of the panel -- Line 12 reports zeros from 2010-01-01 until it opened on
    2012-10-30.
``not_measured``
    A day on which *every* station reports zero. The counting system was down,
    not the network: the only such window is 2017-09-20..27, after the
    September 19th earthquake.
``closed``
    Any other zero run at least ``min_closure_days`` long -- the Line 12
    collapse, the Line 1 modernization, the pandemic station closures.
``ambiguous_label``
    Station-days whose label could not be resolved (see :mod:`.quality`).
``operating``
    Everything else. A zero here is a real anomaly, not a closure, and is kept
    as both a training target and a positive label for the detector.

The threshold sits at 4 days because the run-length distribution has a natural
gap there: 182 runs last exactly one day and 97 last two or three, but only 14
last four to six, and everything from seven days up is a documented closure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OPERATING = "operating"
NOT_YET_OPEN = "not_yet_open"
CLOSED = "closed"
NOT_MEASURED = "not_measured"
AMBIGUOUS_LABEL = "ambiguous_label"

STATUSES = (OPERATING, NOT_YET_OPEN, CLOSED, NOT_MEASURED, AMBIGUOUS_LABEL)

DEFAULT_MIN_CLOSURE_DAYS = 4

# "Every station reported zero" only implies a measurement outage when there are
# enough independent stations for the coincidence to be impossible. On a panel of
# one or two stations the same test would relabel every ordinary closure.
DEFAULT_MIN_NETWORK_SIZE = 10


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of ``True``, as inclusive ``(start, end)`` index pairs.

    >>> find_runs(np.array([False, True, True, False, True]))
    [(1, 2), (4, 4)]
    """
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def network_outage_dates(
    panel: pd.DataFrame, *, min_network_size: int = DEFAULT_MIN_NETWORK_SIZE
) -> pd.DatetimeIndex:
    """Days on which every station in the panel reports zero.

    A network-wide zero is a measurement outage: the Metro did not carry zero
    passengers on those days, it simply did not count them. Panels smaller than
    ``min_network_size`` stations cannot support that inference, so none is made.
    """
    n_stations = len(panel.groupby(["line_id", "station_id"], observed=True).size())
    if n_stations < min_network_size:
        return pd.DatetimeIndex([])

    by_day = panel.groupby("date", observed=True)["ridership"].agg(
        n="size", zeros=lambda values: int((values == 0).sum())
    )
    return pd.DatetimeIndex(by_day.index[by_day["zeros"] == by_day["n"]])


def derive_service_status(
    panel: pd.DataFrame,
    *,
    min_closure_days: int = DEFAULT_MIN_CLOSURE_DAYS,
    min_network_size: int = DEFAULT_MIN_NETWORK_SIZE,
) -> pd.Series:
    """Classify every station-day of a complete daily panel.

    Expects columns ``date``, ``line_id``, ``station_id``, ``ridership`` and one
    row per (pair, date). Returns a status Series aligned to ``panel.index``.
    """
    required = {"date", "line_id", "station_id", "ridership"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"derive_service_status needs columns {sorted(missing)}")

    status = pd.Series(OPERATING, index=panel.index, dtype="object")
    outage_dates = set(network_outage_dates(panel, min_network_size=min_network_size))

    ordered = panel.sort_values(["line_id", "station_id", "date"])
    for _, group in ordered.groupby(["line_id", "station_id"], observed=True):
        zeros = (group["ridership"].to_numpy() == 0)
        if not zeros.any():
            continue
        positions = group.index.to_numpy()
        for start, end in find_runs(zeros):
            length = end - start + 1
            if length < min_closure_days:
                # A short zero run is an anomaly, not an interruption -- including
                # one that opens the panel. The record starting on a holiday when
                # a station happened to carry nobody is not evidence that the
                # station did not exist yet.
                continue
            label = NOT_YET_OPEN if start == 0 else CLOSED
            status.loc[positions[start : end + 1]] = label

    # A measurement outage overrides any per-station verdict: on those days we
    # know nothing about any station, closed or not.
    status.loc[panel["date"].isin(outage_dates)] = NOT_MEASURED
    return status


def summarize_status(status: pd.Series) -> pd.DataFrame:
    """Row counts and shares per status, for logging and for the EDA notebook."""
    counts = status.value_counts()
    return pd.DataFrame({"rows": counts, "share": (counts / len(status)).round(4)})
