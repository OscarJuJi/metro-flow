"""Accuracy metrics, and what they do with the days the network was shut."""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.backtest.metrics import (
    mae,
    mase,
    relative_mae,
    score_forecast,
    seasonal_naive_insample_mae,
    smape,
)


def frame(values, columns=("1:a", "1:b"), start="2024-01-01"):
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(values, index=index, columns=list(columns))


def test_mae_is_the_mean_absolute_error():
    truth = frame([[100.0, 200.0], [100.0, 200.0]])
    prediction = frame([[110.0, 200.0], [90.0, 180.0]])
    assert mae(truth, prediction) == pytest.approx((10 + 0 + 10 + 20) / 4)


def test_closed_days_are_not_scored():
    """A station that was shut has no demand to be wrong about."""
    truth = frame([[100.0, np.nan], [np.nan, np.nan]])
    prediction = frame([[100.0, 5_000.0], [5_000.0, 5_000.0]])
    assert mae(truth, prediction) == 0.0


def test_a_missing_forecast_counts_as_an_error_not_as_a_free_pass():
    truth = frame([[100.0, 100.0]])
    prediction = frame([[100.0, np.nan]])
    assert mae(truth, prediction) == pytest.approx(50.0)


def test_relative_mae_is_exactly_one_for_the_baseline_itself():
    truth = frame([[100.0, 200.0], [120.0, 180.0]])
    baseline = frame([[110.0, 190.0], [130.0, 200.0]])
    assert relative_mae(truth, baseline, baseline) == 1.0


def test_relative_mae_falls_below_one_when_the_model_beats_the_baseline():
    truth = frame([[100.0, 200.0]])
    baseline = frame([[120.0, 220.0]])
    better = frame([[110.0, 210.0]])
    assert relative_mae(truth, better, baseline) == pytest.approx(0.5)


def test_mase_scales_by_the_in_sample_seasonal_error():
    truth = frame([[100.0, 100.0]])
    prediction = frame([[110.0, 90.0]])
    assert mase(truth, prediction, insample_mae=20.0) == pytest.approx(0.5)


def test_mase_is_undefined_rather_than_infinite_when_the_denominator_is_degenerate():
    truth = frame([[100.0, 100.0]])
    prediction = frame([[110.0, 90.0]])
    assert np.isnan(mase(truth, prediction, insample_mae=0.0))
    assert np.isnan(mase(truth, prediction, insample_mae=float("nan")))


def test_seasonal_naive_insample_mae_uses_a_one_week_lag():
    values = np.arange(21, dtype="float64").reshape(-1, 1)
    history = pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=21), columns=["1:a"])
    assert seasonal_naive_insample_mae(history) == pytest.approx(7.0)


def test_seasonal_naive_insample_mae_is_undefined_for_a_history_shorter_than_a_week():
    history = frame([[100.0, 100.0]] * 3)
    assert np.isnan(seasonal_naive_insample_mae(history))


def test_smape_ignores_days_where_truth_and_forecast_are_both_zero():
    truth = frame([[0.0, 100.0]])
    prediction = frame([[0.0, 150.0]])
    assert smape(truth, prediction) == pytest.approx(40.0)


def test_score_forecast_reports_how_many_station_days_it_scored():
    truth = frame([[100.0, np.nan], [120.0, 180.0]])
    prediction = frame([[110.0, 100.0], [120.0, 180.0]])
    score = score_forecast(truth, prediction, baseline=prediction, insample_mae=10.0)
    assert score.n_observations == 3  # the closed station-day is not one of them
    assert score.mae == pytest.approx(10 / 3)
    assert score.relative_mae == 1.0


def test_a_flawless_baseline_leaves_the_ratio_undefined_rather_than_dividing_by_zero():
    truth = frame([[100.0, 200.0]])
    perfect = frame([[100.0, 200.0]])
    assert np.isnan(relative_mae(truth, perfect, baseline=perfect))
