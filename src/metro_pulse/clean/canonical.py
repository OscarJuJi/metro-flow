"""Canonicalize the raw line and station labels into stable identifiers.

The raw panel carries 24 distinct ``linea`` labels for 12 lines and 167 distinct
``estacion`` labels for 163 stations, because the 2021-2023 slice is mojibake
(see :mod:`metro_pulse.text`). Repairing and normalizing collapses them to
12 lines / 163 stations / 195 line-station pairs, which matches the 195 rows the
file publishes every single day.
"""

from __future__ import annotations

import re

import pandas as pd

from metro_pulse.config import NetworkShape
from metro_pulse.text import fix_mojibake, normalize_key

LINE_LABEL = re.compile(r"^linea\s+(?P<code>\d{1,2}|[ab])$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

class CanonicalizationError(ValueError):
    """A label could not be mapped onto the known network."""


def parse_line_id(raw_line: str) -> str:
    """``"LÃ\xadnea 12"`` -> ``"12"``, ``"Linea A"`` -> ``"A"``."""
    match = LINE_LABEL.match(normalize_key(raw_line))
    if not match:
        raise CanonicalizationError(f"Unrecognized line label: {raw_line!r}")
    return match.group("code").upper()


def slugify_station(raw_station: str) -> str:
    """``"La Villa/Basílica"`` -> ``"la-villa-basilica"``."""
    slug = _NON_ALNUM.sub("-", normalize_key(raw_station)).strip("-")
    if not slug:
        raise CanonicalizationError(f"Station label reduces to an empty slug: {raw_station!r}")
    return slug


def display_name(raw_station: str) -> str:
    """The human-facing station name, with its mojibake repaired."""
    return fix_mojibake(str(raw_station)).strip()


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``date``, ``line_id``, ``station_id`` and ``station_name`` columns.

    The raw label columns are kept so the transformation stays auditable.
    """
    out = frame.copy()
    out["date"] = pd.to_datetime(out["fecha"], format="%Y-%m-%d")
    out["line_id"] = out["linea"].map(parse_line_id)
    out["station_id"] = out["estacion"].map(slugify_station)
    out["station_name"] = out["estacion"].map(display_name)
    out["ridership"] = out["afluencia"].astype("int64")
    return out


def build_station_registry(canonical: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per line-station pair, with its observed date span.

    Written to ``reference/stations.csv`` so the mapping can be reviewed by a
    human instead of living inside the code.
    """
    grouped = canonical.groupby(["line_id", "station_id"], observed=True)
    registry = grouped.agg(
        station_name=("station_name", lambda s: s.mode().iat[0]),
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_days=("date", "nunique"),
        n_raw_labels=("estacion", "nunique"),
    ).reset_index()
    registry["_order"] = registry["line_id"].map(line_sort_rank)
    registry = registry.sort_values(["_order", "line_id", "station_id"])
    return registry.drop(columns="_order").reset_index(drop=True)


def line_sort_rank(line_id: str) -> int:
    """Order line ids the way the network is usually listed: 1..12, then A, B."""
    return 100 + ord(line_id) if line_id.isalpha() else int(line_id)


def assert_expected_network(canonical: pd.DataFrame, network: NetworkShape) -> dict[str, int]:
    """Fail loudly if canonicalization did not reconstruct the operator's network.

    The expected counts belong to the operator, not to this module: hardcoding
    the Metro's twelve lines here would make every other network a special case.
    """
    counts = {
        "n_lines": canonical["line_id"].nunique(),
        "n_stations": canonical["station_id"].nunique(),
        "n_pairs": len(canonical.groupby(["line_id", "station_id"], observed=True).size()),
    }
    expected = {
        "n_lines": network.n_lines,
        "n_stations": network.n_stations,
        "n_pairs": network.n_pairs,
    }
    mismatches = {k: (counts[k], expected[k]) for k in expected if counts[k] != expected[k]}
    if mismatches:
        detail = ", ".join(
            f"{k}: got {got}, expected {want}" for k, (got, want) in mismatches.items()
        )
        raise CanonicalizationError(
            f"Canonicalization did not reconstruct the known network ({detail}). "
            "Either the portal added a line/station, or a new spelling variant slipped through."
        )
    return counts
