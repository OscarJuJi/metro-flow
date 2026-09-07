"""Integration checks: does the derived mask agree with documented history?

These run against the real panel and are skipped when it has not been built, so
a fresh clone still has a green suite before any data is downloaded:

    python -m metro_pulse.ingest.download
    python -m metro_pulse.clean.build
"""

import pandas as pd
import pytest

from metro_pulse.clean.service import CLOSED, NOT_MEASURED, NOT_YET_OPEN, OPERATING
from metro_pulse.config import INTERIM_DIR

PANEL_PATH = INTERIM_DIR / "metro_panel.parquet"

N_PAIRS = 195
EARTHQUAKE_WINDOW = ("2017-09-20", "2017-09-27")


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        pytest.skip(f"{PANEL_PATH} not built; run the ingest and clean steps first")
    return pd.read_parquet(PANEL_PATH)


def test_the_panel_is_rectangular(panel):
    n_dates = panel["date"].nunique()
    assert len(panel) == N_PAIRS * n_dates
    assert panel.groupby(["line_id", "station_id"], observed=True).size().eq(n_dates).all()
    assert not panel.duplicated(subset=["date", "line_id", "station_id"]).any()


def test_the_network_has_twelve_lines_and_163_stations(panel):
    assert panel["line_id"].nunique() == 12
    assert panel["station_id"].nunique() == 163


def test_line_12_does_not_exist_before_it_opened(panel):
    """Line 12 opened on 2012-10-30; its first counted day in this source is 11-05."""
    line_12 = panel[panel["line_id"] == "12"].sort_values("date")
    not_yet_open = line_12[line_12["status"] == NOT_YET_OPEN]
    assert not_yet_open["date"].min() == pd.Timestamp("2010-01-01")
    assert not_yet_open["date"].max() == pd.Timestamp("2012-11-04")
    assert len(not_yet_open) == 20 * 1039  # every station of the line, every day

    first_operating = line_12[line_12["status"] == OPERATING]["date"].min()
    assert first_operating == pd.Timestamp("2012-11-05")


def test_the_2017_earthquake_reads_as_a_measurement_outage_not_a_closure(panel):
    """For eight days the entire network reports zero: nobody was counting."""
    window = panel[panel["date"].between(*EARTHQUAKE_WINDOW)]
    assert set(window["status"]) == {NOT_MEASURED}
    assert window["line_id"].nunique() == 12
    assert len(window) == N_PAIRS * 8

    outside = panel[panel["status"] == NOT_MEASURED]
    assert outside["date"].min() == pd.Timestamp(EARTHQUAKE_WINDOW[0])
    assert outside["date"].max() == pd.Timestamp(EARTHQUAKE_WINDOW[1])


def test_the_line_12_collapse_closes_the_whole_line(panel):
    """The Olivos overpass collapsed on 2021-05-03; service stops the next day."""
    line_12 = panel[(panel["line_id"] == "12") & (panel["date"] >= "2021-05-01")]
    closed = line_12[line_12["status"] == CLOSED]
    assert closed["date"].min() == pd.Timestamp("2021-05-04")
    assert closed["station_id"].nunique() == 20  # every station on the line


def test_the_line_1_modernization_closes_the_line_in_two_phases(panel):
    """Pantitlán-Salto del Agua from 2022-07, then Balderas-Observatorio from 2023-11."""
    line_1 = panel[(panel["line_id"] == "1") & (panel["date"] >= "2022-01-01")]
    closed = line_1[line_1["status"] == CLOSED]
    assert closed["date"].min() == pd.Timestamp("2022-07-09")

    # Phase one: 12 stations shut in July 2022, ten of which reopened in Oct 2023.
    assert closed[closed["date"] == "2022-08-01"]["station_id"].nunique() == 12
    assert closed[closed["date"] == "2023-11-01"]["station_id"].nunique() == 2

    # Phase two starts 2023-11-10 and adds eight more, on top of those two.
    newly_closed = closed[closed["date"] == "2023-11-10"]["station_id"].nunique()
    assert newly_closed == 10
    assert closed[closed["date"] == "2024-01-01"]["station_id"].nunique() == 10

    # By the end of the panel the whole line has been closed at some point.
    assert closed["station_id"].nunique() == 20


def test_the_control_centre_fire_hits_lines_1_2_and_3_on_the_same_day(panel):
    """The 2021-01-09 fire at the Puesto Central de Control took out three lines."""
    day = panel[(panel["date"] == "2021-01-09") & (panel["status"] == CLOSED)]
    assert set(day["line_id"]) == {"1", "2", "3"}
    assert len(day) == 65


def test_december_2020_oceania_stays_flagged_and_never_becomes_a_target(panel):
    ambiguous = panel[panel["status"] == "ambiguous_label"]
    assert len(ambiguous) == 62
    assert ambiguous["date"].dt.to_period("M").astype(str).unique().tolist() == ["2020-12"]
    assert ambiguous["target"].isna().all()
    assert set(ambiguous["station_id"]) == {"oceania", "deportivo-oceania"}


def test_the_target_is_defined_exactly_on_operating_days(panel):
    assert (panel["status"] == OPERATING).sum() == panel["target"].notna().sum()
    assert panel.loc[panel["status"] != OPERATING, "target"].isna().all()


def test_zero_ridership_survives_inside_operating_days_as_an_anomaly(panel):
    """The zeros that are not closures are the detector's positive labels."""
    anomalous_zeros = panel[(panel["status"] == OPERATING) & (panel["ridership"] == 0)]
    assert len(anomalous_zeros) > 0
    assert anomalous_zeros["target"].eq(0).all()
