"""Load the serving artifacts once, and say something useful when they are absent.

Everything the API returns was computed by :mod:`metro_pulse.models.train`. This
module is the only place that knows where those files live, so a missing
artifact produces one clear error naming the command that creates it, rather
than a ``FileNotFoundError`` from somewhere inside a request handler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from metro_pulse.config import ARTIFACTS_DIR

REQUIRED = ("forecast.parquet", "anomalies.parquet", "metadata.json")


class ArtifactsMissing(RuntimeError):
    """The service was started before anything was trained."""


@dataclass(frozen=True)
class Store:
    """The trained state of the service, held in memory for the process lifetime."""

    metadata: dict
    forecast: pd.DataFrame
    anomalies: pd.DataFrame
    coordinates: pd.DataFrame

    @property
    def stations(self) -> pd.DataFrame:
        """One row per station, with coordinates when they were ingested."""
        stations = (
            self.forecast[["series_id", "line_id", "station_id", "station_name"]]
            .drop_duplicates()
            .sort_values(["line_id", "station_id"])
        )
        if self.coordinates.empty:
            return stations.assign(longitude=None, latitude=None)
        return stations.merge(self.coordinates, on="series_id", how="left")


@lru_cache(maxsize=1)
def load_store() -> Store:
    """Read the artifacts from disk. Cached: they only change on a redeploy."""
    missing = [name for name in REQUIRED if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        raise ArtifactsMissing(
            f"{', '.join(missing)} not found in {ARTIFACTS_DIR}. "
            "Run `python -m metro_pulse.models.train` (which needs the ingest and "
            "clean steps to have run first)."
        )

    coordinates_path = ARTIFACTS_DIR / "coordinates.parquet"
    return Store(
        metadata=json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8")),
        forecast=pd.read_parquet(ARTIFACTS_DIR / "forecast.parquet"),
        anomalies=pd.read_parquet(ARTIFACTS_DIR / "anomalies.parquet"),
        coordinates=(
            pd.read_parquet(coordinates_path)
            if coordinates_path.exists()
            else pd.DataFrame(columns=["series_id", "longitude", "latitude"])
        ),
    )
