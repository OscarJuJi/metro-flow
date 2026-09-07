"""Minimal CKAN client for the CDMX open data portal.

Resources are looked up by name instead of by UUID. The portal rotates resource
UUIDs whenever it republishes a dataset, so a hardcoded download URL is a time
bomb: it keeps returning ``200`` while serving the portal's HTML error page.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from metro_pulse.config import CKAN_BASE_URL
from metro_pulse.text import normalize_key

DEFAULT_TIMEOUT = 60


class CkanError(RuntimeError):
    """The portal answered, but not with what we asked for."""


@dataclass(frozen=True)
class CkanResource:
    """One downloadable file inside a CKAN dataset."""

    resource_id: str
    name: str
    url: str
    fmt: str
    size: int | None
    last_modified: str | None


def fetch_package(dataset_id: str, base_url: str = CKAN_BASE_URL) -> dict:
    """Return the raw ``package_show`` payload for a dataset."""
    response = requests.get(
        f"{base_url}/api/3/action/package_show",
        params={"id": dataset_id},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise CkanError(f"CKAN reported failure for dataset {dataset_id!r}: {payload!r}")
    return payload["result"]


def list_resources(dataset_id: str, base_url: str = CKAN_BASE_URL) -> list[CkanResource]:
    """List every resource published under a dataset."""
    package = fetch_package(dataset_id, base_url=base_url)
    return [
        CkanResource(
            resource_id=item.get("id", ""),
            name=item.get("name", ""),
            url=item.get("url", ""),
            fmt=(item.get("format") or "").upper(),
            size=item.get("size"),
            last_modified=item.get("last_modified") or item.get("created"),
        )
        for item in package.get("resources", [])
    ]


def resolve_resource(
    dataset_id: str, resource_name: str, base_url: str = CKAN_BASE_URL
) -> CkanResource:
    """Find one resource by name, tolerating accent and spacing drift.

    Raises :class:`CkanError` listing what the portal *does* publish, so a rename
    upstream produces a diagnosable failure instead of a mystery.
    """
    resources = list_resources(dataset_id, base_url=base_url)
    wanted = normalize_key(resource_name)
    for resource in resources:
        if normalize_key(resource.name) == wanted:
            return resource

    available = "\n".join(f"  - {r.name!r} ({r.fmt})" for r in resources) or "  (none)"
    raise CkanError(
        f"Resource {resource_name!r} not found in dataset {dataset_id!r}.\n"
        f"The portal currently publishes:\n{available}"
    )
