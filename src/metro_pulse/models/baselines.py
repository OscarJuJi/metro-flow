"""Forecasting baselines, written out in numpy.

Two models live here and they do different jobs.

:class:`SeasonalNaive` is the yardstick. Metro demand is dominated by the day of
the week, so "last Tuesday's count" is a genuinely strong forecast and any model
that cannot beat it is not worth deploying. It is the denominator of every skill
number this project reports.

:class:`HierarchicalSeasonalModel` is the *expected demand* model that the
anomaly detector is built on. It decomposes each series multiplicatively,

    ridership(t) ~ level(t) x weekday(t) x month(t)

and estimates every component with medians rather than means. That matters here:
the panel contains two-year closures and a pandemic, and a single mean would let
those events drag the level for months afterwards.

Both models take the wide ``dates x series`` matrix and treat ``NaN`` as "not
observed", never as zero.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEEK = 7


def _last_valid_per_phase(values: np.ndarray, period: int, max_cycles: int) -> np.ndarray:
    """For each phase of the cycle, the most recent non-NaN value of each series.

    ``values`` is ``(n_days, n_series)``; the result is ``(period, n_series)``
    where row ``p`` holds the newest observation whose row index satisfies
    ``index % period == p``. Closures are skipped, which is why this looks back
    over several cycles instead of taking a single lag.

    ``max_cycles`` bounds that search in *cycles*, not in days: each phase block
    already holds one row per cycle, so slicing it by a number of days would put
    the limit weeks away from where it was meant to be.
    """
    n_days, n_series = values.shape
    result = np.full((period, n_series), np.nan)
    row_index = np.arange(n_days)

    for phase in range(period):
        block = values[row_index % period == phase][-max_cycles:]
        if block.size == 0:
            continue
        observed = ~np.isnan(block)
        has_any = observed.any(axis=0)
        # Position of the last True, found by reversing the search axis.
        newest = block.shape[0] - 1 - np.argmax(observed[::-1], axis=0)
        result[phase] = np.where(has_any, block[newest, np.arange(n_series)], np.nan)
    return result


@dataclass
class SeasonalNaive:
    """Predict each future day with the most recent same-weekday observation.

    ``max_lookback_weeks`` bounds how far back the search goes, so a station
    closed for two years does not resurface with a two-year-old forecast.
    """

    season_length: int = WEEK
    max_lookback_weeks: int = 8

    columns_: pd.Index = field(init=False, repr=False)
    last_date_: pd.Timestamp = field(init=False, repr=False)
    phase_offset_: int = field(init=False, repr=False)
    by_phase_: np.ndarray = field(init=False, repr=False)

    def fit(self, history: pd.DataFrame) -> SeasonalNaive:
        if history.empty:
            raise ValueError("SeasonalNaive needs at least one row of history")
        self.columns_ = history.columns
        self.last_date_ = history.index[-1]
        self.phase_offset_ = len(history) - 1
        self.by_phase_ = _last_valid_per_phase(
            history.to_numpy(dtype="float64"),
            period=self.season_length,
            max_cycles=self.max_lookback_weeks,
        )
        return self

    def predict(self, n_steps: int) -> pd.DataFrame:
        """Forecast the ``n_steps`` days following the end of the history."""
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        steps = np.arange(1, n_steps + 1)
        phases = (self.phase_offset_ + steps) % self.season_length
        dates = self.last_date_ + pd.to_timedelta(steps, unit="D")
        return pd.DataFrame(self.by_phase_[phases], index=dates, columns=self.columns_)


@dataclass
class HierarchicalSeasonalModel:
    """Robust multiplicative decomposition: level x weekday x month.

    The level is a trailing median over ``level_window`` days, so it reflects
    where a station is *now* rather than where it averaged over its history.
    Seasonal factors are medians of the ratio between the observation and the
    level, which keeps a single anomalous day from moving them.
    """

    level_window: int = 56
    min_level_observations: int = 14
    min_factor_observations: int = 8

    columns_: pd.Index = field(init=False, repr=False)
    last_date_: pd.Timestamp = field(init=False, repr=False)
    level_: np.ndarray = field(init=False, repr=False)
    weekday_factor_: np.ndarray = field(init=False, repr=False)
    month_factor_: np.ndarray = field(init=False, repr=False)

    def fit(self, history: pd.DataFrame) -> HierarchicalSeasonalModel:
        if history.empty:
            raise ValueError("HierarchicalSeasonalModel needs at least one row of history")
        self.columns_ = history.columns
        self.last_date_ = history.index[-1]

        values = history.to_numpy(dtype="float64")
        level_series = self._rolling_level(history)

        # Each day is divided by the level *of its own era*. Dividing sixteen
        # years of history by a single present-day level would fold the network's
        # long decline into the seasonal factors and inflate every forecast.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.where(level_series > 0, values / level_series, np.nan)

        weekdays = history.index.dayofweek.to_numpy()
        self.weekday_factor_ = self._grouped_median(ratios, weekdays, n_groups=WEEK)

        deseasonalized = ratios / self.weekday_factor_[weekdays]
        months = history.index.month.to_numpy() - 1
        self.month_factor_ = self._grouped_median(deseasonalized, months, n_groups=12)

        self.level_ = self._current_level(history, level_series)
        return self

    def _rolling_level(self, history: pd.DataFrame) -> np.ndarray:
        """Trailing median of the last ``level_window`` days, evaluated at every day."""
        rolling = history.rolling(
            window=self.level_window, min_periods=self.min_level_observations
        ).median()
        return rolling.to_numpy(dtype="float64")

    def _current_level(self, history: pd.DataFrame, level_series: np.ndarray) -> np.ndarray:
        """The level to forecast from: the most recent one that could be computed.

        Carried forward when a station is closed at the end of the training
        window, so a station that reopens during the test period is forecast from
        the last level it actually had rather than from nothing.
        """
        levels = pd.DataFrame(level_series, index=history.index, columns=history.columns)
        return levels.ffill().iloc[-1].to_numpy(dtype="float64")

    def _grouped_median(self, ratios: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
        """Median ratio per calendar group, defaulting to 1.0 when unsupported."""
        factors = np.ones((n_groups, ratios.shape[1]))
        for group in range(n_groups):
            block = ratios[groups == group]
            if block.size == 0:
                continue
            observed = (~np.isnan(block)).sum(axis=0)
            with warnings.catch_warnings():
                # A series closed for the whole window is an all-NaN column here.
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(block, axis=0)
            factors[group] = np.where(
                (observed >= self.min_factor_observations) & np.isfinite(median), median, 1.0
            )
        return factors

    def expected(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Expected demand for any dates, past or future.

        Used both to forecast and, in the anomaly detector, to say what a day
        *should* have looked like.
        """
        dates = pd.DatetimeIndex(dates)
        weekday = self.weekday_factor_[dates.dayofweek.to_numpy()]
        month = self.month_factor_[dates.month.to_numpy() - 1]
        return pd.DataFrame(self.level_ * weekday * month, index=dates, columns=self.columns_)

    def predict(self, n_steps: int) -> pd.DataFrame:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        steps = np.arange(1, n_steps + 1)
        return self.expected(self.last_date_ + pd.to_timedelta(steps, unit="D"))
