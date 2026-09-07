"""The supervised table: every feature is computed at the origin, never later."""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.features.build import (
    build_prediction_frame,
    build_supervised_frame,
    feature_columns,
    same_weekday_lag,
    warmup_days,
)

RNG = np.random.default_rng(4242)


def wide_panel(n_days: int = 900, columns=("1:a", "1:b", "2:c")) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n_days, freq="D")  # a Monday
    weekly = np.tile([1000, 1100, 1150, 1120, 1300, 700, 500], n_days // 7 + 1)[:n_days]
    values = np.column_stack(
        [weekly * scale + RNG.normal(0, 20, n_days) for scale in (1.0, 1.5, 0.7)]
    )
    return pd.DataFrame(values, index=dates, columns=list(columns))


class TestSameWeekdayLag:
    @pytest.mark.parametrize(
        ("horizon", "expected"), [(1, 6), (2, 5), (6, 1), (7, 0), (8, 6), (13, 1), (14, 0)]
    )
    def test_points_at_the_most_recent_matching_weekday(self, horizon, expected):
        assert same_weekday_lag(horizon) == expected

    def test_is_never_negative_so_it_cannot_reach_past_the_origin(self):
        assert all(same_weekday_lag(h) >= 0 for h in range(1, 60))

    def test_lands_on_the_same_weekday_as_the_target(self):
        for horizon in range(1, 30):
            assert (horizon + same_weekday_lag(horizon)) % 7 == 0


class TestSupervisedFrame:
    def test_the_target_is_the_value_at_the_target_date(self):
        wide = wide_panel()
        frame = build_supervised_frame(wide, horizons=(1, 7))
        sample = frame.sample(50, random_state=0)
        for row in sample.itertuples():
            assert row.target == pytest.approx(wide.loc[row.date, row.series_id], rel=1e-5)

    def test_the_origin_is_exactly_the_horizon_before_the_target(self):
        frame = build_supervised_frame(wide_panel(), horizons=(1, 5, 14))
        offsets = (frame["date"] - frame["origin"]).dt.days
        assert (offsets == frame["horizon"]).all()

    def test_lag_zero_is_the_value_at_the_origin(self):
        wide = wide_panel()
        frame = build_supervised_frame(wide, horizons=(3,))
        sample = frame.sample(50, random_state=1)
        for row in sample.itertuples():
            assert row.lag_0 == pytest.approx(wide.loc[row.origin, row.series_id], rel=1e-5)

    def test_rows_without_a_target_are_dropped_rather_than_imputed(self):
        wide = wide_panel()
        wide.loc["2023-06-01":"2023-06-30", "1:a"] = np.nan
        frame = build_supervised_frame(wide, horizons=(1,))
        closed = frame[(frame["series_id"] == "1:a") & (frame["date"].dt.month == 6)]
        assert closed[closed["date"].dt.year == 2023].empty
        assert frame["target"].notna().all()

    def test_no_origin_is_close_enough_to_the_end_to_lack_its_target(self):
        wide = wide_panel()
        frame = build_supervised_frame(wide, horizons=range(1, 15))
        assert frame["date"].max() <= wide.index[-1]

    def test_rows_start_only_once_every_feature_is_defined(self):
        wide = wide_panel()
        frame = build_supervised_frame(wide, horizons=(1,))
        assert frame["origin"].min() >= wide.index[warmup_days()]

    def test_features_exclude_the_target_and_the_dates(self):
        frame = build_supervised_frame(wide_panel(), horizons=(1,))
        features = feature_columns(frame)
        assert "target" not in features
        assert "date" not in features
        assert "origin" not in features
        assert "same_weekday" in features
        assert "horizon" in features


class TestFeaturesCannotSeeTheFuture:
    """Rebuild the table with the future replaced by noise; the features must not move."""

    def test_features_at_an_origin_ignore_everything_after_it(self):
        wide = wide_panel()
        cutoff = wide.index[600]

        corrupted = wide.copy()
        corrupted.loc[corrupted.index > cutoff] = RNG.uniform(-1e6, 1e6, (299, 3))

        honest = build_supervised_frame(wide, horizons=(1, 7, 14))
        tampered = build_supervised_frame(corrupted, horizons=(1, 7, 14))

        keys = ["origin", "series_id", "horizon"]
        skip = {"line_id", "station_id", *keys}
        features = [c for c in feature_columns(honest) if c not in skip]
        honest = honest[honest["origin"] <= cutoff].set_index(keys)[features].sort_index()
        tampered = tampered[tampered["origin"] <= cutoff].set_index(keys)[features].sort_index()

        assert not honest.empty
        pd.testing.assert_frame_equal(honest, tampered)

    def test_the_check_above_would_notice_a_change_in_the_past(self):
        wide = wide_panel()
        cutoff = wide.index[600]
        corrupted = wide.copy()
        corrupted.loc[corrupted.index <= cutoff] *= 3

        honest = build_supervised_frame(wide, horizons=(1,))
        tampered = build_supervised_frame(corrupted, horizons=(1,))
        assert not np.allclose(honest["lag_0"], tampered["lag_0"])


class TestPredictionFrame:
    def test_forecasts_the_days_after_the_panel_from_its_last_day(self):
        wide = wide_panel()
        frame = build_prediction_frame(wide, horizons=range(1, 15))

        assert len(frame) == 14 * wide.shape[1]
        assert (frame["origin"] == wide.index[-1]).all()
        assert frame["date"].min() == wide.index[-1] + pd.Timedelta(days=1)
        assert frame["date"].max() == wide.index[-1] + pd.Timedelta(days=14)
        assert "target" not in frame.columns

    def test_its_features_match_the_training_table_at_the_same_origin(self):
        """Train and serve must agree, or the model is fed a different world."""
        wide = wide_panel()
        origin = wide.index[-15]

        training = build_supervised_frame(wide, horizons=(1,), origins=pd.DatetimeIndex([origin]))
        serving = build_prediction_frame(wide.loc[:origin], horizons=(1,))

        keys = ["series_id"]
        skip = {"line_id", "station_id", *keys}
        features = [c for c in feature_columns(training) if c not in skip]
        left = training.set_index(keys)[features].sort_index()
        right = serving.set_index(keys)[features].sort_index()
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
