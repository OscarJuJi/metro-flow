"""The splitter is the single place where a temporal leak could be introduced."""

import pandas as pd
import pytest

from metro_pulse.backtest.splitter import Fold, LeakageError, rolling_origin_folds

DATES = pd.date_range("2020-01-01", "2024-12-31", freq="D")


def folds(**kwargs):
    params = {"n_folds": 4, "horizon": 14, "min_train_days": 365}
    return rolling_origin_folds(DATES, **{**params, **kwargs})


def test_every_fold_trains_strictly_before_it_is_scored():
    for fold in folds():
        assert fold.train_end < fold.test_start
        assert (fold.test_start - fold.train_end).days == 1


def test_folds_are_returned_oldest_first():
    ends = [fold.train_end for fold in folds()]
    assert ends == sorted(ends)


def test_test_windows_tile_the_evaluation_period_without_overlapping():
    windows = [(f.test_start, f.test_end) for f in folds(horizon=14)]
    for (_, earlier_end), (later_start, _) in zip(windows, windows[1:], strict=False):
        assert later_start > earlier_end


def test_the_last_fold_ends_on_the_last_available_day():
    assert max(f.test_end for f in folds()) == DATES[-1]


def test_each_test_window_is_exactly_the_horizon():
    for fold in folds(horizon=7):
        assert fold.horizon == 7
        assert len(pd.date_range(fold.test_start, fold.test_end)) == 7


def test_the_training_window_expands_with_each_fold():
    lengths = [(f.train_end - f.train_start).days for f in folds()]
    assert lengths == sorted(lengths)
    assert len({f.train_start for f in folds()}) == 1


def test_a_fold_that_would_leak_the_future_is_rejected_on_construction():
    with pytest.raises(LeakageError, match="not before the first scored day"):
        Fold(
            index=0,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-03-01"),
            test_start=pd.Timestamp("2020-02-15"),  # overlaps the training window
            test_end=pd.Timestamp("2020-03-15"),
        )


def test_a_fold_whose_test_starts_on_the_last_training_day_is_rejected():
    with pytest.raises(LeakageError):
        Fold(
            index=0,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-03-01"),
            test_start=pd.Timestamp("2020-03-01"),
            test_end=pd.Timestamp("2020-03-08"),
        )


def test_asking_for_more_history_than_exists_fails_with_an_actionable_message():
    with pytest.raises(ValueError, match="Not enough history"):
        rolling_origin_folds(DATES, n_folds=12, horizon=14, min_train_days=365 * 10)


def test_masks_select_the_intended_days():
    fold = folds(n_folds=1, horizon=7)[0]
    assert fold.train_mask(DATES).sum() == (fold.train_end - fold.train_start).days + 1
    assert fold.test_mask(DATES).sum() == 7
    assert not (fold.train_mask(DATES) & fold.test_mask(DATES)).any()
