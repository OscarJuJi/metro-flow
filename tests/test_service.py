"""The service mask decides what the model is allowed to learn from."""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.clean.service import (
    AMBIGUOUS_LABEL,
    CLOSED,
    NOT_MEASURED,
    NOT_YET_OPEN,
    OPERATING,
    derive_service_status,
    find_runs,
    network_outage_dates,
    summarize_status,
)


class TestFindRuns:
    def test_finds_maximal_runs_as_inclusive_bounds(self):
        assert find_runs(np.array([False, True, True, False, True])) == [(1, 2), (4, 4)]

    def test_handles_runs_touching_both_edges(self):
        assert find_runs(np.array([True, False, True])) == [(0, 0), (2, 2)]
        assert find_runs(np.array([True, True, True])) == [(0, 2)]

    def test_handles_empty_and_all_false(self):
        assert find_runs(np.array([], dtype=bool)) == []
        assert find_runs(np.array([False, False])) == []


def panel_from(series: dict[str, list[float]], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a rectangular two-column panel from ``{station_id: values}``."""
    dates = pd.date_range(start, periods=len(next(iter(series.values()))), freq="D")
    frames = [
        pd.DataFrame(
            {"date": dates, "line_id": "1", "station_id": station, "ridership": values}
        )
        for station, values in series.items()
    ]
    return pd.concat(frames, ignore_index=True)


class TestDeriveServiceStatus:
    def test_leading_zeros_are_a_station_that_does_not_exist_yet(self):
        panel = panel_from({"new-station": [0, 0, 0, 0, 100, 110, 120]})
        status = derive_service_status(panel)
        assert status.tolist() == [NOT_YET_OPEN] * 4 + [OPERATING] * 3

    def test_a_long_zero_run_mid_series_is_a_closure(self):
        panel = panel_from({"a": [100, 110, 0, 0, 0, 0, 120]})
        status = derive_service_status(panel)
        assert status.tolist() == [OPERATING, OPERATING] + [CLOSED] * 4 + [OPERATING]

    def test_a_short_zero_run_at_the_start_of_the_record_is_not_a_missing_station(self):
        """The panel opens on 1 January; a holiday with no riders is not non-existence."""
        panel = panel_from({"a": [0, 0, 100, 110, 120]})
        assert set(derive_service_status(panel)) == {OPERATING}

    def test_a_long_leading_zero_run_is_still_a_station_that_does_not_exist_yet(self):
        panel = panel_from({"a": [0, 0, 0, 0, 0, 100, 110]})
        status = derive_service_status(panel)
        assert status.tolist() == [NOT_YET_OPEN] * 5 + [OPERATING] * 2

    def test_a_short_zero_run_stays_operating_because_it_is_an_anomaly(self):
        # Three days of zero ridership at an open station is an incident to be
        # detected, not a closure to be masked out.
        panel = panel_from({"a": [100, 0, 0, 0, 120]})
        assert set(derive_service_status(panel)) == {OPERATING}

    @pytest.mark.parametrize(
        ("run_length", "expected"), [(3, OPERATING), (4, CLOSED), (5, CLOSED)]
    )
    def test_the_closure_threshold_is_applied_at_its_boundary(self, run_length, expected):
        panel = panel_from({"a": [100] + [0] * run_length + [100]})
        assert derive_service_status(panel).iloc[1] == expected

    def test_the_threshold_is_configurable(self):
        panel = panel_from({"a": [100, 0, 0, 100]})
        assert derive_service_status(panel, min_closure_days=2).iloc[1] == CLOSED

    def test_a_day_where_every_station_reports_zero_is_a_measurement_outage(self):
        # The 2017 earthquake pattern: the whole network reads zero for days.
        panel = panel_from({"a": [100, 0, 0, 120], "b": [200, 0, 0, 220]})
        status = derive_service_status(panel, min_network_size=2)
        assert status.tolist() == [OPERATING, NOT_MEASURED, NOT_MEASURED, OPERATING] * 2

    def test_a_measurement_outage_overrides_a_station_closure(self):
        # Station "a" is mid-closure while the network-wide count fails; we know
        # nothing about that day, so "closed" would be an unjustified claim.
        panel = panel_from({"a": [100, 0, 0, 0, 0, 100], "b": [200, 210, 0, 220, 230, 240]})
        status = derive_service_status(panel, min_network_size=2)
        by_station = {s: g.tolist() for s, g in status.groupby(panel["station_id"])}
        assert by_station["a"][2] == NOT_MEASURED
        assert by_station["a"][1] == CLOSED

    def test_missing_ridership_is_never_mistaken_for_a_zero(self):
        panel = panel_from({"a": [100, np.nan, np.nan, np.nan, np.nan, 100]})
        assert set(derive_service_status(panel)) == {OPERATING}

    def test_rejects_a_panel_missing_required_columns(self):
        with pytest.raises(ValueError, match="needs columns"):
            derive_service_status(pd.DataFrame({"date": [], "ridership": []}))


def test_a_small_panel_cannot_support_a_network_outage_inference():
    # One station reading zero is a closure, not evidence that counting failed.
    panel = panel_from({"a": [0, 0, 5]})
    assert network_outage_dates(panel).empty


def test_network_outage_dates_lists_only_fully_zero_days():
    panel = panel_from({"a": [0, 0, 5], "b": [0, 3, 5]})
    outages = network_outage_dates(panel, min_network_size=2)
    assert outages.strftime("%Y-%m-%d").tolist() == ["2020-01-01"]


def test_summarize_status_reports_rows_and_shares():
    summary = summarize_status(pd.Series([OPERATING, OPERATING, CLOSED, AMBIGUOUS_LABEL]))
    assert summary.loc[OPERATING, "rows"] == 2
    assert summary.loc[OPERATING, "share"] == 0.5
