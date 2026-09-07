"""The global gradient-boosted forecaster: contract, not accuracy.

Accuracy is the backtest's job. What is checked here is that the model refuses
to answer questions it was not trained for, that its output lines up with the
panel it was fitted on, and that it never forecasts negative ridership.

The model is deliberately tiny in these tests: the point is the wiring.
"""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.models.gbm import GradientBoostingForecaster

RNG = np.random.default_rng(99)
COLUMNS = ["1:a", "1:b", "2:c"]


def wide_panel(n_days: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n_days, freq="D")
    weekly = np.tile([1000, 1100, 1150, 1120, 1300, 700, 500], n_days // 7 + 1)[:n_days]
    values = np.column_stack(
        [weekly * scale + RNG.normal(0, 20, n_days) for scale in (1.0, 1.5, 0.7)]
    )
    return pd.DataFrame(values, index=dates, columns=COLUMNS)


def small_model(**kwargs) -> GradientBoostingForecaster:
    params = {"horizons": (1, 7), "n_estimators": 15, "num_leaves": 8, "train_years": 1.0}
    return GradientBoostingForecaster(**{**params, **kwargs})


@pytest.fixture(scope="module")
def fitted():
    return small_model().fit(wide_panel())


def test_predictions_cover_the_days_after_the_training_history(fitted):
    history = wide_panel()
    prediction = fitted.predict(7)
    assert list(prediction.columns) == COLUMNS
    assert prediction.index[0] == history.index[-1] + pd.Timedelta(days=1)
    assert len(prediction) == 7


def test_ridership_is_never_forecast_below_zero(fitted):
    assert (fitted.predict(7).to_numpy() >= 0).all()


def test_asking_beyond_the_trained_horizon_is_refused(fitted):
    with pytest.raises(ValueError, match="horizons up to 7"):
        fitted.predict(14)


def test_a_non_positive_horizon_is_refused(fitted):
    with pytest.raises(ValueError, match="at least 1"):
        fitted.predict(0)


def test_predicting_before_fitting_is_refused():
    with pytest.raises(ValueError, match="Call fit before predict"):
        small_model().predict(1)


def test_too_little_history_is_refused_with_the_number_needed():
    short = wide_panel(n_days=60)
    with pytest.raises(ValueError, match="needs at least"):
        small_model().fit(short)


def test_a_station_closed_for_the_whole_window_still_gets_a_column(fitted):
    """A closed station must not vanish from the output frame."""
    history = wide_panel()
    history.loc[history.index[-200:], "2:c"] = np.nan
    prediction = small_model().fit(history).predict(7)
    assert "2:c" in prediction.columns


def test_feature_importance_is_available_and_ranked(fitted):
    importance = fitted.feature_importance()
    assert len(importance) > 5
    assert importance.is_monotonic_decreasing


def test_feature_importance_before_fitting_is_refused():
    with pytest.raises(ValueError, match="Call fit before"):
        small_model().feature_importance()


def test_the_forecast_is_in_the_right_order_of_magnitude(fitted):
    """A sanity floor, not an accuracy claim: the backtest measures accuracy."""
    prediction = fitted.predict(7)
    assert 100 < prediction.to_numpy().mean() < 5_000
