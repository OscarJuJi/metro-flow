"""Rolling-origin evaluation windows.

A single train/test split would score the model on one arbitrary month. Instead
the origin walks forward: each fold trains on everything up to a cut-off and is
scored on the ``horizon`` days that follow, and the cut-off then advances.

The invariant the whole evaluation rests on is stated once, here, and enforced
by :meth:`Fold.validate`: **no fold may train on a day it is scored on, or on
any day after it.** Feature construction, model fitting and scoring all take
their dates from a fold, so there is exactly one place where a leak could be
introduced and exactly one place that has to be right.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LeakageError(AssertionError):
    """A fold would let the model see the future it is about to be scored on."""


@dataclass(frozen=True)
class Fold:
    """One expanding-window split: train on ``[train_start, train_end]``, score after it."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.train_start > self.train_end:
            raise LeakageError(f"fold {self.index}: empty training window")
        if self.test_start > self.test_end:
            raise LeakageError(f"fold {self.index}: empty test window")
        if self.test_start <= self.train_end:
            raise LeakageError(
                f"fold {self.index}: training data reaches {self.train_end:%Y-%m-%d}, "
                f"which is not before the first scored day {self.test_start:%Y-%m-%d}"
            )

    @property
    def horizon(self) -> int:
        return int((self.test_end - self.test_start).days) + 1

    def train_mask(self, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series((dates >= self.train_start) & (dates <= self.train_end), index=dates)

    def test_mask(self, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series((dates >= self.test_start) & (dates <= self.test_end), index=dates)

    def describe(self) -> str:
        return (
            f"fold {self.index:>2}: train {self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d} "
            f"-> test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}"
        )


def rolling_origin_folds(
    dates: pd.DatetimeIndex,
    *,
    n_folds: int = 12,
    horizon: int = 14,
    step: int | None = None,
    min_train_days: int = 365 * 3,
) -> list[Fold]:
    """Build expanding-window folds ending at the last available date.

    ``step`` defaults to ``horizon``, which makes the test windows tile the
    evaluation period without overlapping.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    dates = pd.DatetimeIndex(dates).sort_values()
    step = horizon if step is None else step
    day = pd.Timedelta(days=1)

    folds: list[Fold] = []
    for offset in range(n_folds):
        test_end = dates[-1] - (offset * step) * day
        test_start = test_end - (horizon - 1) * day
        train_end = test_start - day
        train_start = dates[0]

        if (train_end - train_start).days + 1 < min_train_days:
            raise ValueError(
                f"Not enough history for {n_folds} folds of {horizon} days: fold {offset} would "
                f"train on {(train_end - train_start).days + 1} days, below min_train_days="
                f"{min_train_days}. Shorten the horizon, ask for fewer folds, or lower the floor."
            )
        folds.append(
            Fold(
                index=offset,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return sorted(folds, key=lambda fold: fold.train_end)
