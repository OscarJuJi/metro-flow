"""Download, validate and freeze the raw ridership panel.

Run as ``python -m metro_pulse.ingest.download`` (add ``--operator metrobus`` for
the Metrobús panel, which shares the same schema).

The step is idempotent: it records the resource's ``last_modified`` stamp and the
file digest, and skips the transfer when the portal has nothing new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from metro_pulse.config import OPERATORS, RAW_DIR, OperatorSource, ensure_dirs
from metro_pulse.ingest.ckan import CkanResource, resolve_resource
from metro_pulse.ingest.schema import SchemaReport, validate_ridership_frame

CHUNK_SIZE = 1 << 20  # 1 MiB
HTML_SNIFF_PREFIXES = (b"<!doctype", b"<html", b"<?xml")


class DownloadError(RuntimeError):
    """The transfer completed but the payload is not the CSV we asked for."""


@dataclass
class IngestResult:
    """Everything the ingest step learned, persisted next to the data."""

    operator: str
    resource_name: str
    resource_url: str
    resource_last_modified: str | None
    csv_path: str
    parquet_path: str
    sha256: str
    n_bytes: int
    report: dict


def _sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def _download(resource: CkanResource, destination: Path) -> tuple[str, int]:
    """Stream a resource to disk, returning its digest and size in bytes."""
    digest = hashlib.sha256()
    n_bytes = 0
    next_report = 10 << 20

    with requests.get(resource.url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                if n_bytes == 0 and chunk[:16].lower().startswith(HTML_SNIFF_PREFIXES):
                    raise DownloadError(
                        f"{resource.url} served an HTML page instead of a CSV. "
                        "The resource UUID has probably rotated; "
                        "re-resolve it through the CKAN API."
                    )
                handle.write(chunk)
                digest.update(chunk)
                n_bytes += len(chunk)
                if n_bytes >= next_report:
                    print(f"    ... {n_bytes / (1 << 20):.0f} MB", flush=True)
                    next_report += 10 << 20

    if resource.size and abs(resource.size - n_bytes) > CHUNK_SIZE:
        raise DownloadError(
            f"Size mismatch: portal announced {resource.size:,} bytes, received {n_bytes:,}"
        )
    return digest.hexdigest(), n_bytes


def read_ridership_csv(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV with an explicit encoding.

    Windows defaults to cp1252, which turns every accented station name into
    mojibake; the source file is UTF-8 and is read as such everywhere.
    """
    return pd.read_csv(
        csv_path,
        encoding="utf-8",
        dtype={"fecha": "string", "mes": "string", "linea": "string", "estacion": "string"},
    )


def ingest_operator(operator: OperatorSource, *, force: bool = False) -> IngestResult:
    """Resolve, download, validate and freeze one operator's ridership panel."""
    ensure_dirs()
    csv_path = RAW_DIR / f"{operator.key}_ridership.csv"
    parquet_path = RAW_DIR / f"{operator.key}_ridership.parquet"
    sidecar = _sidecar_path(csv_path)

    print(f"[1/4] Resolving '{operator.resource_name}' in {operator.ckan_dataset_id} ...")
    resource = resolve_resource(operator.ckan_dataset_id, operator.resource_name)
    print(f"      -> {resource.url}")
    print(f"      last_modified={resource.last_modified} size={resource.size}")

    previous = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
    unchanged = (
        previous is not None
        and previous.get("resource_last_modified") == resource.last_modified
        and csv_path.exists()
        and parquet_path.exists()
    )
    if unchanged and not force:
        print("[2/4] Portal reports no change since the last ingest; reusing local copy.")
        sha256, n_bytes = previous["sha256"], previous["n_bytes"]
    else:
        print(f"[2/4] Downloading to {csv_path} ...")
        sha256, n_bytes = _download(resource, csv_path)
        print(f"      {n_bytes:,} bytes | sha256={sha256[:16]}...")

    print("[3/4] Validating schema ...")
    frame = read_ridership_csv(csv_path)
    report: SchemaReport = validate_ridership_frame(frame)
    print(f"      {report.summary()}")

    print(f"[4/4] Writing {parquet_path} ...")
    frame.to_parquet(parquet_path, index=False, compression="snappy")

    result = IngestResult(
        operator=operator.key,
        resource_name=resource.name,
        resource_url=resource.url,
        resource_last_modified=resource.last_modified,
        csv_path=str(csv_path),
        parquet_path=str(parquet_path),
        sha256=sha256,
        n_bytes=n_bytes,
        report=asdict(report),
    )
    sidecar.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    parser.add_argument("--force", action="store_true", help="re-download even if unchanged")
    args = parser.parse_args(argv)

    result = ingest_operator(OPERATORS[args.operator], force=args.force)
    print("\nIngest complete.")
    print(f"  parquet : {result.parquet_path}")
    print(f"  rows    : {result.report['n_rows']:,}")
    print(f"  span    : {result.report['date_min']} .. {result.report['date_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
