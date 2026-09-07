"""Guards around the transfer itself.

The portal answers 200 with an HTML error page when a resource UUID has rotated.
Writing that to disk as ``*.csv`` is the silent failure this module exists to prevent.
"""

import pytest

from metro_pulse.ingest import download as dl
from metro_pulse.ingest.ckan import CkanResource
from metro_pulse.ingest.download import DownloadError, _download


class FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield from self._chunks


def serve(monkeypatch, chunks):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **kw: FakeResponse(chunks))


def resource(size=None):
    return CkanResource(
        resource_id="bbb",
        name="Afluencia Diaria del Metro (Simple)",
        url="https://example.mx/ridership.csv",
        fmt="CSV",
        size=size,
        last_modified="2026-08-28T07:52:22",
    )


def test_writes_the_payload_and_reports_its_digest(monkeypatch, tmp_path):
    payload = b"fecha,anio,mes,linea,estacion,afluencia\n2010-01-01,2010,Enero,Linea 1,Zaragoza,1\n"
    serve(monkeypatch, [payload])
    destination = tmp_path / "ridership.csv"

    sha256, n_bytes = _download(resource(size=len(payload)), destination)

    assert destination.read_bytes() == payload
    assert n_bytes == len(payload)
    assert len(sha256) == 64


def test_rejects_an_html_error_page_served_as_csv(monkeypatch, tmp_path):
    serve(monkeypatch, [b"<!DOCTYPE html>\n<html><body>Not found</body></html>"])

    with pytest.raises(DownloadError, match="HTML page"):
        _download(resource(), tmp_path / "ridership.csv")


def test_rejects_a_truncated_transfer(monkeypatch, tmp_path):
    serve(monkeypatch, [b"fecha,anio\n2010-01-01,2010\n"])

    with pytest.raises(DownloadError, match="Size mismatch"):
        _download(resource(size=59_917_547), tmp_path / "ridership.csv")
