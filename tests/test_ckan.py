"""Resource lookup must survive the portal rotating UUIDs and renaming files."""

import pytest

from metro_pulse.ingest import ckan
from metro_pulse.ingest.ckan import CkanError, resolve_resource

PACKAGE = {
    "resources": [
        {
            "id": "aaa",
            "name": "Diccionario de Datos (Afluencia simple)",
            "url": "https://example.mx/dict.csv",
            "format": "csv",
            "size": 1124,
            "last_modified": "2026-08-28T05:03:34",
        },
        {
            "id": "bbb",
            "name": "Afluencia Diaria del Metro (Simple)",
            "url": "https://example.mx/ridership.csv",
            "format": "csv",
            "size": 59917547,
            "last_modified": "2026-08-28T07:52:22",
        },
    ]
}


@pytest.fixture
def portal(monkeypatch):
    monkeypatch.setattr(ckan, "fetch_package", lambda dataset_id, base_url=None: PACKAGE)


def test_resolves_by_name_not_by_uuid(portal):
    resource = resolve_resource("any-dataset", "Afluencia Diaria del Metro (Simple)")
    assert resource.resource_id == "bbb"
    assert resource.url.endswith("ridership.csv")
    assert resource.fmt == "CSV"
    assert resource.size == 59917547


def test_resolution_tolerates_accent_and_spacing_drift(portal):
    # e.g. the Metrobús resource, whose name carries an accent upstream.
    resource = resolve_resource("any-dataset", "afluencia  diaria del métro (simple)")
    assert resource.resource_id == "bbb"


def test_a_rename_upstream_fails_loudly_and_lists_what_exists(portal):
    with pytest.raises(CkanError) as excinfo:
        resolve_resource("any-dataset", "Afluencia Horaria del Metro")
    message = str(excinfo.value)
    assert "not found" in message
    assert "Afluencia Diaria del Metro (Simple)" in message
