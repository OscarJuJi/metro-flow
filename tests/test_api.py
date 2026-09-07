"""The HTTP surface: what it returns, and how it behaves with nothing trained."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from metro_pulse.api import store as store_module
from metro_pulse.api.app import app, records
from metro_pulse.api.store import ArtifactsMissing, load_store
from metro_pulse.config import ARTIFACTS_DIR


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not (ARTIFACTS_DIR / "metadata.json").exists():
        pytest.skip("no artifacts; run `python -m metro_pulse.models.train` first")
    return TestClient(app)


class TestRecords:
    def test_dates_become_plain_iso_strings(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]), "value": [1]})
        assert records(frame) == [{"date": "2026-08-01", "value": 1}]

    def test_every_flavour_of_missing_becomes_null(self):
        """NaN, NaT and pd.NA are all invalid JSON and all reach this function."""
        frame = pd.DataFrame(
            {
                "a": [np.nan],
                "b": pd.Series([pd.NA], dtype="Int64"),
                "c": pd.Series([pd.NaT], dtype="datetime64[ns]"),
            }
        )
        assert records(frame) == [{"a": None, "b": None, "c": None}]

    def test_an_empty_frame_is_an_empty_list(self):
        assert records(pd.DataFrame(columns=["a"])) == []


class TestWithoutArtifacts:
    def test_health_reports_503_and_names_the_command_that_fixes_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_module, "ARTIFACTS_DIR", tmp_path)
        load_store.cache_clear()
        try:
            response = TestClient(app).get("/health")
            assert response.status_code == 503
            assert "metro_pulse.models.train" in response.json()["detail"]
        finally:
            load_store.cache_clear()

    def test_loading_a_missing_store_raises_a_named_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_module, "ARTIFACTS_DIR", tmp_path)
        load_store.cache_clear()
        try:
            with pytest.raises(ArtifactsMissing, match="not found"):
                load_store()
        finally:
            load_store.cache_clear()


class TestService:
    def test_health_reports_what_the_data_reaches(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert pd.Timestamp(body["data_through"]) > pd.Timestamp("2020-01-01")

    def test_metadata_describes_the_training_run(self, client):
        body = client.get("/metadata").json()
        assert body["n_series"] == 195
        assert body["forecast_from"] > body["data_through"]
        assert 0 < body["false_alarm_rate"] < 1


class TestStations:
    def test_lists_every_line_station_pair(self, client):
        stations = client.get("/stations").json()
        assert len(stations) == 195
        assert len({s["line_id"] for s in stations}) == 12

    def test_filters_to_one_line(self, client):
        stations = client.get("/stations?line=B").json()
        assert {s["line_id"] for s in stations} == {"B"}

    def test_line_ids_are_case_insensitive(self, client):
        assert client.get("/stations?line=b").json() == client.get("/stations?line=B").json()

    def test_an_unknown_line_is_a_404(self, client):
        assert client.get("/stations?line=42").status_code == 404

    def test_coordinates_are_present_for_the_map(self, client):
        stations = client.get("/stations").json()
        assert all(s["latitude"] is not None for s in stations)
        assert all(19.0 < s["latitude"] < 19.8 for s in stations)


class TestForecast:
    def test_covers_fourteen_days_for_every_station(self, client):
        forecast = client.get("/forecast").json()
        assert len(forecast) == 195 * 14

    def test_a_transfer_station_is_forecast_once_per_line(self, client):
        """Pantitlán is on four lines and they are four different demands."""
        rows = client.get("/forecast?station=pantitlan&horizon=1").json()
        assert {r["line_id"] for r in rows} == {"1", "5", "9", "A"}

    def test_predictions_are_non_negative_whole_people(self, client):
        for row in client.get("/forecast?line=4").json():
            assert row["prediction"] is None or row["prediction"] >= 0

    def test_an_unknown_station_is_a_404_that_points_at_the_station_list(self, client):
        response = client.get("/forecast?station=estacion-inexistente")
        assert response.status_code == 404
        assert "/stations" in response.json()["detail"]

    def test_a_horizon_beyond_the_trained_range_is_rejected_by_validation(self, client):
        assert client.get("/forecast?horizon=99").status_code == 422


class TestAnomalies:
    def test_defaults_to_everything_flagged_in_the_scored_window(self, client):
        """A quiet last day must not look like a broken service."""
        flagged = client.get("/anomalies").json()
        assert all(row["is_anomaly"] for row in flagged)
        dates = [row["date"] for row in flagged]
        assert dates == sorted(dates, reverse=True)

    def test_a_single_day_can_still_be_asked_for(self, client):
        through = client.get("/metadata").json()["data_through"]
        rows = client.get(f"/anomalies?date={through}&flagged_only=false").json()
        assert {row["date"] for row in rows} == {through}
        assert len(rows) == 195

    def test_every_flagged_row_says_why(self, client):
        for row in client.get("/anomalies").json():
            assert row["reason"] in {"extreme day", "sustained shortfall"}

    def test_a_station_query_returns_its_recent_history_scored(self, client):
        rows = client.get("/anomalies?station=pantitlan&flagged_only=false").json()
        assert len(rows) == 4 * 120  # four lines, the whole scoring window
        assert {"ridership", "expected", "score", "is_anomaly"} <= set(rows[0])

    def test_an_unknown_station_is_a_404(self, client):
        assert client.get("/anomalies?station=estacion-inexistente").status_code == 404

    def test_a_day_outside_the_scored_window_is_a_404(self, client):
        assert client.get("/anomalies?date=1999-01-01").status_code == 404


def test_the_dashboard_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Metro Pulse" in response.text
