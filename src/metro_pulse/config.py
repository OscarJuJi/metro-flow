"""Project-wide paths and dataset coordinates.

Every path is derived from the repository root so the pipeline behaves identically
whether it runs from a clone, from Docker or from a GitHub Actions runner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("METRO_PULSE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("METRO_PULSE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

CKAN_BASE_URL = "https://datos.cdmx.gob.mx"


@dataclass(frozen=True)
class NetworkShape:
    """How many lines, stations and line-station pairs an operator should have.

    Canonicalization is checked against these, so a new station, a renamed line
    or a spelling variant that slips through stops the pipeline instead of
    quietly changing the panel underneath the models.
    """

    n_lines: int
    n_stations: int
    n_pairs: int


@dataclass(frozen=True)
class OperatorSource:
    """A CKAN dataset + resource pair for one transit operator.

    The resource is addressed by *name* rather than by UUID: the CDMX portal
    republishes resources periodically and the UUIDs change when it does.
    """

    key: str
    ckan_dataset_id: str
    resource_name: str
    label: str
    network: NetworkShape


METRO = OperatorSource(
    key="metro",
    ckan_dataset_id="afluencia-diaria-del-metro-cdmx",
    resource_name="Afluencia Diaria del Metro (Simple)",
    label="STC Metro",
    network=NetworkShape(n_lines=12, n_stations=163, n_pairs=195),
)

# Metrobús is *not* a second operator here, despite sharing a portal and a
# publication cadence: its open data is aggregated to the line
# (``fecha, anio, mes, linea, afluencia``) and carries no station column at all.
# Nothing in this project -- the service mask, the per-station forecast, the
# detector -- has anything to work with at that granularity.

OPERATORS: dict[str, OperatorSource] = {source.key: source for source in (METRO,)}


def ensure_dirs() -> None:
    """Create the data directories the pipeline writes to."""
    for directory in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
