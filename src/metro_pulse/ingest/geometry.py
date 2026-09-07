"""Station coordinates, for the map the dashboard draws.

The ridership panel says how many people used a station; it does not say where
the station is. The city publishes that separately, and only as SHP and KMZ --
no CSV, no GeoJSON. KMZ is the cheaper of the two to read: it is a zip holding a
KML, which is XML, so it needs nothing beyond the standard library, whereas the
shapefile would drag in a geospatial stack for 195 points.

Run as ``python -m metro_pulse.ingest.geometry``.

The parse is checked against the ridership panel rather than trusted: the KMZ
carries exactly 195 placemarks and the panel exactly 195 line-station pairs, so
they must correspond one to one. Two of them did not, and both were worth
knowing about -- one station has been renamed since the ridership series was
set up, and one is simply misspelled in the published geometry. Those two live
in ``reference/station_aliases.csv`` where a human can read and challenge them,
rather than in a dictionary literal in this file.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile

import pandas as pd
import requests

from metro_pulse.clean.canonical import slugify_station
from metro_pulse.config import METRO, PROJECT_ROOT, OperatorSource
from metro_pulse.ingest.ckan import resolve_resource

GEOMETRY_DATASET = "lineas-y-estaciones-del-metro"
GEOMETRY_RESOURCE = "Líneas y Estaciones de STC Metro (KMZ)"
STATIONS_MEMBER = "STC_Metro_estaciones.kmz"
ALIASES_FILE = "station_aliases.csv"

PLACEMARK = re.compile(r"<Placemark.*?</Placemark>", re.S)
FIELD = r"<td>{}</td>\s*<td>(.*?)</td>"
NAME_FIELD = re.compile(FIELD.format("NOMBRE"), re.S)
LINE_FIELD = re.compile(FIELD.format("LINEA"), re.S)
COORDINATES = re.compile(r"<coordinates>\s*([-\d.]+),([-\d.]+)")


class GeometryError(RuntimeError):
    """The published geometry does not line up with the ridership panel."""


def parse_line_code(raw: str) -> str:
    """``"01"`` -> ``"1"``, ``"12"`` -> ``"12"``, ``"A"`` -> ``"A"``."""
    code = raw.strip().upper()
    return code.lstrip("0") or code


def extract_kml(kmz_bytes: bytes) -> str:
    """Pull the station KML out of the nested KMZ the portal publishes."""
    outer = zipfile.ZipFile(io.BytesIO(kmz_bytes))
    member = next((n for n in outer.namelist() if n.endswith(STATIONS_MEMBER)), None)
    if member is None:
        raise GeometryError(
            f"{STATIONS_MEMBER} not found in the archive; it holds {outer.namelist()}"
        )
    inner = zipfile.ZipFile(io.BytesIO(outer.read(member)))
    kml_name = next((n for n in inner.namelist() if n.lower().endswith(".kml")), None)
    if kml_name is None:
        raise GeometryError(f"no KML inside {member}; it holds {inner.namelist()}")
    return inner.read(kml_name).decode("utf-8")


def parse_stations(kml: str) -> pd.DataFrame:
    """One row per placemark: line, station, longitude, latitude."""
    records = []
    for placemark in PLACEMARK.findall(kml):
        name = NAME_FIELD.search(placemark)
        line = LINE_FIELD.search(placemark)
        point = COORDINATES.search(placemark)
        if not (name and line and point):
            continue
        records.append(
            {
                "line_id": parse_line_code(line.group(1)),
                "station_id": slugify_station(name.group(1)),
                "station_name": name.group(1).strip(),
                "longitude": float(point.group(1)),
                "latitude": float(point.group(2)),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise GeometryError("no station placemarks parsed out of the KML")
    return frame


def load_aliases(source: str = "geometry") -> pd.DataFrame:
    """Known one-to-one station renamings and misspellings, from the reference table."""
    path = PROJECT_ROOT / "reference" / ALIASES_FILE
    if not path.exists():
        return pd.DataFrame(columns=["line_id", "alias_station_id", "canonical_station_id"])
    aliases = pd.read_csv(path, encoding="utf-8", dtype={"line_id": "string"})
    return aliases[aliases["source"] == source]


def apply_aliases(geometry: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """Rewrite known alternative station ids onto the ones the panel uses."""
    if aliases.empty:
        return geometry
    mapping = {
        (row.line_id, row.alias_station_id): row.canonical_station_id
        for row in aliases.itertuples()
    }
    resolved = geometry.copy()
    resolved["station_id"] = [
        mapping.get((line, station), station)
        for line, station in zip(resolved["line_id"], resolved["station_id"], strict=True)
    ]
    return resolved


def reconcile(geometry: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Join geometry onto the panel's stations, refusing to lose or invent any."""
    keys = ["line_id", "station_id"]
    merged = registry[[*keys, "station_name"]].merge(
        geometry[[*keys, "longitude", "latitude"]], on=keys, how="outer", indicator=True
    )

    missing = merged[merged["_merge"] == "left_only"]
    extra = merged[merged["_merge"] == "right_only"]
    if len(missing) or len(extra):
        raise GeometryError(
            f"{len(missing)} panel stations have no coordinates and {len(extra)} "
            "coordinates match no panel station.\n"
            f"  without coordinates: {missing['station_id'].tolist()}\n"
            f"  unmatched geometry:  {extra['station_id'].tolist()}"
        )

    if not merged["latitude"].between(19.0, 19.8).all():
        raise GeometryError("a station falls outside the Valley of Mexico; check the parse")
    if not merged["longitude"].between(-99.4, -98.8).all():
        raise GeometryError("a station falls outside the Valley of Mexico; check the parse")

    return merged.drop(columns="_merge")


def ingest_geometry(operator: OperatorSource = METRO) -> pd.DataFrame:
    """Download, parse and reconcile the station coordinates."""
    registry_path = PROJECT_ROOT / "reference" / f"{operator.key}_stations.csv"
    if not registry_path.exists():
        raise GeometryError(
            f"{registry_path} not found. Run `python -m metro_pulse.clean.build` first: "
            "the coordinates are checked against the stations the panel actually has."
        )

    print(f"[1/4] Resolving '{GEOMETRY_RESOURCE}' in {GEOMETRY_DATASET} ...")
    resource = resolve_resource(GEOMETRY_DATASET, GEOMETRY_RESOURCE)
    print(f"      -> {resource.url}")

    print("[2/4] Downloading and unpacking the KMZ ...")
    response = requests.get(resource.url, timeout=120)
    response.raise_for_status()
    kml = extract_kml(response.content)

    print("[3/4] Parsing placemarks ...")
    geometry = parse_stations(kml)
    print(f"      {len(geometry)} stations across {geometry['line_id'].nunique()} lines")

    print("[4/4] Reconciling against the ridership panel ...")
    aliases = load_aliases()
    geometry = apply_aliases(geometry, aliases)
    print(f"      {len(aliases)} known aliases applied")
    registry = pd.read_csv(registry_path, encoding="utf-8", dtype={"line_id": "string"})
    reconciled = reconcile(geometry, registry)
    print(f"      {len(reconciled)} line-station pairs matched, none missing, none extra")
    return reconciled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro")
    args = parser.parse_args(argv)

    reconciled = ingest_geometry(METRO)
    destination = PROJECT_ROOT / "reference" / f"{args.operator}_station_coordinates.csv"
    reconciled.to_csv(destination, index=False, encoding="utf-8")

    print()
    print(f"Written to {destination}")
    print(reconciled.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
