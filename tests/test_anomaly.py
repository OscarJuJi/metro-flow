"""The anomaly detector: does it fire on incidents and stay quiet otherwise?"""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.models.anomaly import (
    MAD_TO_SIGMA,
    RobustAnomalyDetector,
    cusum_negative,
    median_absolute_deviation,
)

RNG = np.random.default_rng(31415)
WEEKLY_SHAPE = np.array([10_000, 11_000, 11_500, 11_200, 13_000, 7_000, 5_000], dtype="float64")


def wide_panel(n_days: int = 730, n_series: int = 12, noise: float = 300.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n_days, freq="D")  # a Monday
    weekly = np.tile(WEEKLY_SHAPE, n_days // 7 + 1)[:n_days]
    scales = np.linspace(0.6, 1.8, n_series)
    values = np.column_stack(
        [weekly * scale + RNG.normal(0, noise, n_days) for scale in scales]
    )
    columns = [f"{1 + i // 6}:station-{i}" for i in range(n_series)]
    return pd.DataFrame(values, index=dates, columns=columns)


class TestMedianAbsoluteDeviation:
    def test_measures_spread_around_the_median(self):
        values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        assert median_absolute_deviation(values)[0] == pytest.approx(1.0)

    def test_a_single_extreme_value_barely_moves_it(self):
        clean = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        spiked = np.array([[1.0], [2.0], [3.0], [4.0], [10_000.0]])
        assert median_absolute_deviation(spiked)[0] == pytest.approx(
            median_absolute_deviation(clean)[0]
        )
        assert spiked.std() > 10 * clean.std()

    def test_an_all_missing_series_yields_nan_rather_than_raising(self):
        values = np.full((5, 1), np.nan)
        assert np.isnan(median_absolute_deviation(values)[0])

    def test_the_sigma_factor_is_the_usual_normal_consistency_constant(self):
        normal = RNG.normal(0, 1, (20_000, 1))
        estimate = median_absolute_deviation(normal)[0] * MAD_TO_SIGMA
        assert estimate == pytest.approx(1.0, abs=0.05)


class TestCusum:
    def test_stays_at_zero_while_nothing_is_wrong(self):
        scores = np.zeros((20, 2))
        assert cusum_negative(scores).max() == 0.0
        assert cusum_negative(scores).min() == 0.0

    def test_accumulates_while_a_shortfall_persists(self):
        scores = np.full((10, 1), -2.0)
        accumulated = cusum_negative(scores, drift=0.5)
        assert accumulated[-1, 0] == pytest.approx(-15.0)
        assert accumulated[0, 0] > accumulated[-1, 0]

    def test_recovers_towards_zero_once_the_shortfall_stops(self):
        """It climbs back at the drift rate: 12.5 of debt takes 25 quiet days."""
        scores = np.concatenate([np.full((5, 1), -3.0), np.full((30, 1), 0.0)])
        accumulated = cusum_negative(scores, drift=0.5)
        assert accumulated[4, 0] == pytest.approx(-12.5)
        assert accumulated[4 + 10, 0] == pytest.approx(-7.5)
        assert accumulated[-1, 0] == 0.0

    def test_an_isolated_dip_never_reaches_an_alarming_total(self):
        scores = np.zeros((20, 1))
        scores[10] = -3.0
        assert cusum_negative(scores, drift=0.5).min() > -3.0

    def test_without_a_restart_one_closure_alarms_for_weeks_afterwards(self):
        """Two months shut, then normal: the debt outlives the incident."""
        scores = np.concatenate([np.full((60, 1), -8.0), np.zeros((30, 1))])
        accumulated = cusum_negative(scores, drift=0.5)
        assert accumulated[-1, 0] < -400  # still deep in alarm a month later

    def test_the_restart_alarms_periodically_during_an_outage_not_every_day(self):
        """Sixty days shut should raise a recurring alarm, not sixty of them."""
        scores = np.concatenate([np.full((60, 1), -8.0), np.zeros((30, 1))])
        without = (cusum_negative(scores, drift=0.5) <= -50).sum()
        with_reset = (cusum_negative(scores, drift=0.5, reset_at=50) <= -50).sum()
        assert without >= 60
        assert with_reset < 15

    def test_the_restart_goes_quiet_once_the_station_is_back(self):
        """After the last restart only a small debt remains, and it drains away."""
        scores = np.concatenate([np.full((60, 1), -8.0), np.zeros((30, 1))])
        with_reset = cusum_negative(scores, drift=0.5, reset_at=50)
        without = cusum_negative(scores, drift=0.5)

        assert not (with_reset[-25:] <= -50).any()  # no alarms after it reopened
        assert with_reset[-1, 0] > -50  # whereas the un-restarted statistic is
        assert without[-1, 0] < -400  # still hundreds deep and alarming daily

    def test_the_restart_still_catches_a_shortfall_that_is_under_way(self):
        scores = np.full((20, 1), -3.0)
        accumulated = cusum_negative(scores, drift=0.5, reset_at=20)
        assert (accumulated <= -20).any()

    def test_missing_scores_hold_the_statistic_instead_of_erasing_it(self):
        scores = np.concatenate([np.full((5, 1), -3.0), np.full((3, 1), np.nan)])
        accumulated = cusum_negative(scores, drift=0.5)
        assert accumulated[-1, 0] == pytest.approx(accumulated[4, 0])


@pytest.fixture(scope="module")
def fitted() -> RobustAnomalyDetector:
    return RobustAnomalyDetector().fit(wide_panel())


class TestDetector:
    def test_ordinary_days_are_flagged_at_about_the_requested_rate(self, fitted):
        quiet = wide_panel(n_days=200)
        flag_rate = fitted.detect(quiet).to_numpy().mean()
        assert flag_rate < 0.05

    def test_a_station_that_carries_nobody_is_flagged_immediately(self, fitted):
        upcoming = wide_panel(n_days=30)
        upcoming.iloc[10:, 0] = 0.0
        flags = fitted.detect(upcoming)
        assert flags.iloc[10, 0]

    def test_a_reopened_station_stops_being_flagged(self, fitted):
        """The bug the dashboard exposed: weeks of alerts after a closure ended."""
        upcoming = wide_panel(n_days=120)
        upcoming.iloc[10:70, 4] = 0.0  # two months shut, then back to normal
        flags = fitted.detect(upcoming)
        assert flags.iloc[10:70, 4].any(), "the closure itself must be flagged"
        assert not flags.iloc[-20:, 4].any(), "but not for weeks after it ended"

    def test_reasons_distinguish_an_extreme_day_from_a_sustained_fall(self, fitted):
        upcoming = wide_panel(n_days=40)
        upcoming.iloc[20, 0] = 0.0  # one catastrophic day
        upcoming.iloc[10:, 1] *= 0.7  # a slow bleed
        reasons = fitted.reasons(upcoming)
        assert reasons.iloc[20, 0] == "extreme day"
        assert "sustained shortfall" in set(reasons.iloc[:, 1])

    def test_a_sustained_partial_drop_is_caught_by_the_accumulator(self, fitted):
        """Half-empty for a fortnight never trips the daily score, but must not pass."""
        upcoming = wide_panel(n_days=40)
        upcoming.iloc[20:, 1] *= 0.75
        flags = fitted.detect(upcoming)
        assert flags.iloc[20:, 1].any()

    def test_a_surge_is_not_reported_as_an_outage(self, fitted):
        """The detector looks for shortfalls; a crowded day is not an incident."""
        upcoming = wide_panel(n_days=30)
        upcoming.iloc[15, 2] *= 3
        assert not fitted.detect(upcoming).iloc[15, 2]

    def test_lowering_the_false_alarm_rate_makes_it_stricter(self, fitted):
        upcoming = wide_panel(n_days=120)
        loose = fitted.set_false_alarm_rate(0.05).detect(upcoming).to_numpy().sum()
        strict = fitted.set_false_alarm_rate(0.0005).detect(upcoming).to_numpy().sum()
        assert strict <= loose
        fitted.set_false_alarm_rate(0.002)

    def test_the_threshold_is_read_off_the_data_not_assumed(self, fitted):
        """Residuals here are fat-tailed, so the cut-off should not land on 4 sigma."""
        assert fitted.threshold_ > 1.0
        assert fitted.cusum_threshold_ > 1.0

    def test_a_line_wide_failure_is_reported_as_one_line_not_six_stations(self, fitted):
        upcoming = wide_panel(n_days=30)
        line_1 = [c for c in upcoming.columns if c.startswith("1:")]
        upcoming.loc[upcoming.index[20:], line_1] = 0.0

        incidents = fitted.line_incidents(upcoming)
        assert incidents.loc[incidents.index[20], "1"]
        assert not incidents.loc[incidents.index[20], "2"]

    def test_expected_demand_is_exposed_for_the_dashboard(self, fitted):
        upcoming = wide_panel(n_days=10)
        expected = fitted.expected(upcoming)
        assert expected.shape == upcoming.shape
        assert (expected.to_numpy() > 0).all()

    def test_scoring_before_fitting_is_refused(self):
        with pytest.raises(ValueError, match="Call fit before scoring"):
            RobustAnomalyDetector().score(wide_panel(n_days=10))

    def test_setting_the_rate_before_fitting_is_refused(self):
        with pytest.raises(ValueError, match="Call fit before"):
            RobustAnomalyDetector().set_false_alarm_rate(0.01)

    def test_empty_history_is_refused(self):
        with pytest.raises(ValueError, match="at least one row"):
            RobustAnomalyDetector().fit(pd.DataFrame())

    def test_a_station_closed_throughout_calibration_is_scored_as_unknown(self):
        history = wide_panel()
        history.iloc[-300:, 3] = np.nan
        detector = RobustAnomalyDetector().fit(history)

        upcoming = wide_panel(n_days=20)
        assert detector.score(upcoming).iloc[:, 3].isna().all()
        assert not detector.detect(upcoming).iloc[:, 3].any()
