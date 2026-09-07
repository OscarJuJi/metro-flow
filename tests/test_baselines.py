"""The numpy baselines: what they predict, and what they refuse to predict."""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.models.baselines import HierarchicalSeasonalModel, SeasonalNaive

WEEKLY_SHAPE = [1000, 1100, 1150, 1120, 1300, 700, 500]  # Mon..Sun


def weekly_history(n_weeks: int = 12, start: str = "2024-01-01", scale: float = 1.0):
    """A perfectly periodic series starting on a Monday."""
    dates = pd.date_range(start, periods=n_weeks * 7, freq="D")
    values = np.tile(WEEKLY_SHAPE, n_weeks) * scale
    return pd.DataFrame({"1:a": values}, index=dates)


class TestSeasonalNaive:
    def test_reproduces_a_perfectly_weekly_series(self):
        history = weekly_history()
        prediction = SeasonalNaive().fit(history).predict(7)
        assert prediction["1:a"].tolist() == WEEKLY_SHAPE

    def test_forecast_dates_follow_the_end_of_the_history(self):
        history = weekly_history()
        prediction = SeasonalNaive().fit(history).predict(3)
        assert prediction.index[0] == history.index[-1] + pd.Timedelta(days=1)
        assert len(prediction) == 3

    def test_repeats_the_weekly_pattern_beyond_one_season(self):
        prediction = SeasonalNaive().fit(weekly_history()).predict(14)
        assert prediction["1:a"].tolist() == WEEKLY_SHAPE * 2

    def test_looks_further_back_when_the_most_recent_week_is_a_closure(self):
        history = weekly_history()
        history.iloc[-7:] = np.nan  # the station was shut for the final week
        prediction = SeasonalNaive().fit(history).predict(7)
        assert prediction["1:a"].tolist() == WEEKLY_SHAPE

    def test_gives_up_rather_than_resurrecting_a_stale_observation(self):
        # A station closed for longer than the lookback has no defensible forecast.
        history = weekly_history(n_weeks=20)
        history.iloc[-10 * 7 :] = np.nan
        prediction = SeasonalNaive(max_lookback_weeks=8).fit(history).predict(7)
        assert prediction["1:a"].isna().all()

    def test_rejects_empty_history_and_non_positive_horizons(self):
        with pytest.raises(ValueError, match="at least one row"):
            SeasonalNaive().fit(pd.DataFrame())
        with pytest.raises(ValueError, match="at least 1"):
            SeasonalNaive().fit(weekly_history()).predict(0)


class TestHierarchicalSeasonalModel:
    def test_recovers_the_weekly_shape_of_a_stationary_series(self):
        prediction = HierarchicalSeasonalModel().fit(weekly_history(n_weeks=26)).predict(7)
        assert prediction["1:a"].to_numpy() == pytest.approx(WEEKLY_SHAPE, rel=0.02)

    def test_tracks_the_level_of_a_series_whose_demand_has_changed(self):
        """The level is a trailing median, so it reflects now -- not the average of history."""
        old = weekly_history(n_weeks=26, start="2024-01-01", scale=1.0)
        new = weekly_history(n_weeks=13, start="2024-07-01", scale=0.5)
        history = pd.concat([old, new])
        history = history[~history.index.duplicated(keep="last")]

        prediction = HierarchicalSeasonalModel().fit(history).predict(7)
        assert prediction["1:a"].mean() < 0.7 * np.mean(WEEKLY_SHAPE)

    def test_a_single_outlier_does_not_move_the_seasonal_factors(self):
        history = weekly_history(n_weeks=26)
        clean = HierarchicalSeasonalModel().fit(history).predict(7)

        spiked = history.copy()
        spiked.iloc[-3] = 500_000  # one absurd day
        contaminated = HierarchicalSeasonalModel().fit(spiked).predict(7)

        assert contaminated["1:a"].to_numpy() == pytest.approx(clean["1:a"].to_numpy(), rel=0.05)

    def test_expected_demand_is_available_for_past_days_too(self):
        """The detector needs to ask what a day that already happened should have been."""
        history = weekly_history(n_weeks=26)
        model = HierarchicalSeasonalModel().fit(history)
        expected = model.expected(history.index[-14:])
        assert expected.shape == (14, 1)
        recent = history["1:a"].to_numpy()[-14:]
        assert expected["1:a"].to_numpy() == pytest.approx(recent, rel=0.05)

    def test_a_series_with_too_little_history_yields_no_forecast(self):
        history = weekly_history(n_weeks=1)
        prediction = HierarchicalSeasonalModel().fit(history).predict(7)
        assert prediction["1:a"].isna().all()
