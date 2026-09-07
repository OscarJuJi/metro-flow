"""Proof that no model can see the days it is scored on.

The argument is empirical rather than by inspection: replace every value after a
fold's cut-off with nonsense, refit, and require the forecasts to come out
bit-identical. A model that peeked would move.

The last test in this file keeps the argument honest by checking the opposite
direction -- corrupting the *training* window must change the forecasts -- so a
future refactor that quietly makes every prediction ``NaN`` cannot pass this
file by accident.
"""

import numpy as np
import pandas as pd
import pytest

from metro_pulse.backtest.run import BASELINES, run_fold
from metro_pulse.backtest.splitter import rolling_origin_folds
from metro_pulse.models.gbm import GradientBoostingForecaster

RNG = np.random.default_rng(20260905)
WEEKLY_SHAPE = np.array([1000, 1100, 1150, 1120, 1300, 700, 500], dtype="float64")


def small_gbm() -> GradientBoostingForecaster:
    """A deliberately tiny model: this file tests leakage, not accuracy."""
    return GradientBoostingForecaster(
        horizons=tuple(range(1, 15)), n_estimators=20, num_leaves=8, train_years=1.0
    )


# The learned model is checked alongside the baselines: it has by far the most
# machinery between the panel and the forecast, so it has the most room to leak.
CANDIDATES = {**BASELINES, "gbm": small_gbm}


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    """Three years of four weekly-seasonal series, with a closure in one of them."""
    dates = pd.date_range("2022-01-03", "2024-12-31", freq="D")  # starts on a Monday
    weekly = np.tile(WEEKLY_SHAPE, len(dates) // 7 + 1)[: len(dates)]
    columns = ["1:a", "1:b", "2:c", "2:d"]
    values = np.column_stack(
        [weekly * scale + RNG.normal(0, 25, len(dates)) for scale in (1.0, 1.4, 0.6, 2.1)]
    )
    frame = pd.DataFrame(values, index=dates, columns=columns)
    frame.loc["2023-05-01":"2023-08-31", "2:c"] = np.nan  # a four-month closure
    return frame


@pytest.fixture(scope="module")
def folds(wide):
    return rolling_origin_folds(wide.index, n_folds=4, horizon=14, min_train_days=365)


def corrupt_after(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Replace everything strictly after ``cutoff`` with values no model should see."""
    corrupted = frame.copy()
    future = corrupted.index > cutoff
    corrupted.loc[future] = RNG.uniform(-1e6, 1e6, corrupted.loc[future].shape)
    return corrupted


@pytest.mark.parametrize("model_name", sorted(CANDIDATES))
def test_forecasts_do_not_move_when_the_future_is_replaced_with_noise(wide, folds, model_name):
    models = {name: CANDIDATES[name] for name in {"seasonal_naive", model_name}}
    for fold in folds:
        _, honest = run_fold(wide, fold, models)
        _, tampered = run_fold(corrupt_after(wide, fold.train_end), fold, models)
        pd.testing.assert_frame_equal(honest[model_name], tampered[model_name])


def test_the_whole_backtest_is_unaffected_by_corrupting_the_last_folds_future(wide, folds):
    earliest = min(folds, key=lambda fold: fold.train_end)
    corrupted = corrupt_after(wide, earliest.train_end)
    _, honest = run_fold(wide, earliest, CANDIDATES)
    _, tampered = run_fold(corrupted, earliest, CANDIDATES)
    for name in CANDIDATES:
        pd.testing.assert_frame_equal(honest[name], tampered[name])


def test_a_forecast_is_produced_at_all_so_the_check_above_is_not_vacuous(wide, folds):
    _, predictions = run_fold(wide, folds[0], CANDIDATES)
    for name, prediction in predictions.items():
        assert prediction.notna().any().any(), f"{name} predicted nothing to compare"


@pytest.mark.parametrize("model_name", sorted(CANDIDATES))
def test_corrupting_the_training_window_does_change_the_forecast(wide, folds, model_name):
    """The counter-check: this suite would notice if the models stopped reacting."""
    fold = folds[-1]
    models = {name: CANDIDATES[name] for name in {"seasonal_naive", model_name}}
    _, honest = run_fold(wide, fold, models)

    tampered_frame = wide.copy()
    recent_training = (tampered_frame.index > fold.train_end - pd.Timedelta(days=30)) & (
        tampered_frame.index <= fold.train_end
    )
    tampered_frame.loc[recent_training] *= 10
    _, tampered = run_fold(tampered_frame, fold, models)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(honest[model_name], tampered[model_name])
