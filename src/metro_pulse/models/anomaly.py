"""Detect operational incidents from the gap between demand and expectation.

The Metro publishes no real-time incident feed, so an incident has to be
inferred: a station whose ridership departs far enough from what the day of the
week, the month and its own recent level say it should have been.

Three ideas, each written out here rather than imported:

**Robust scaling.** The residual is divided by the median absolute deviation of
that station's own residuals, not by their standard deviation. A single
two-year closure inflates a standard deviation enough to hide every later
incident behind it; the MAD does not move.

**An empirical alarm threshold.** Ridership residuals are nowhere near normal:
their robust z-score has a standard deviation near 1.8 and 2.8% of ordinary
days fall below -4, where normal theory predicts 0.003%. A fixed cut-off
therefore buries an operator in alerts. Instead the detector asks for a *false
alarm rate* and reads the matching quantile off its own training scores, so the
alert volume is a design decision rather than a consequence of an assumption
the data does not honour.

**Persistence.** A one-day drop is noise; a drop that keeps accumulating is a
closure starting. A one-sided CUSUM adds up the shortfall and fires when the
running total passes a threshold, which detects a sustained fall the day-by-day
score would call unremarkable.

**Spatial agreement.** Stations do not fail independently. If several stations
on the same line deviate on the same day, that is a line incident, and saying
so is more useful than reporting nine separate station anomalies.

The detector is fitted only on days the network was running -- otherwise it
would learn that closures are normal -- but it *scores* every day, including
the closed ones, because in real time nobody has told you yet which is which.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metro_pulse.models.baselines import HierarchicalSeasonalModel

# For normally distributed data, MAD * 1.4826 estimates the standard deviation.
MAD_TO_SIGMA = 1.4826

DEFAULT_THRESHOLD = 4.0

# Residual statistics are calibrated on the tail of the training window, not on
# all of it. The expected-demand model carries a single current level per
# station, so its residuals are only meaningful near the end of the history it
# was fitted on; calibrating on three years of them measures the network's
# long-term decline instead of its day-to-day noise.
DEFAULT_CALIBRATION_DAYS = 180

# Roughly one alert per station every 500 operating days, which across 195
# stations is well under one alert a day for the network as a whole.
DEFAULT_FALSE_ALARM_RATE = 0.002


def median_absolute_deviation(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median of the absolute deviations from the median, ignoring NaN.

    Returns ``NaN`` for a series with no observations at all -- a station closed
    for the whole calibration window -- rather than raising.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nanmedian(values, axis=axis, keepdims=True)
        return np.nanmedian(np.abs(values - center), axis=axis)


def cusum_negative(
    scores: np.ndarray, drift: float = 0.5, reset_at: float | None = None
) -> np.ndarray:
    """One-sided CUSUM of downward deviations, accumulated down each column.

    ``S_t = min(0, S_{t-1} + z_t + drift)``: the statistic drifts back to zero
    while nothing is wrong and only runs away while the shortfall persists. NaN
    scores hold the statistic rather than resetting it, so a gap in the data
    does not erase an incident in progress.

    ``reset_at`` restarts the statistic at zero the step after it crosses
    ``-reset_at``. Without it a station that was shut for two months accumulates
    a debt so deep that it keeps raising alarms for weeks after it reopens and
    its daily scores are perfectly ordinary -- the textbook CUSUM restart, and
    the difference between reporting an incident and reporting it every day
    until the arithmetic forgives it.
    """
    accumulated = np.zeros_like(scores)
    running = np.zeros(scores.shape[1])
    for row in range(scores.shape[0]):
        step = np.where(np.isnan(scores[row]), 0.0, scores[row] + drift)
        running = np.minimum(0.0, running + step)
        accumulated[row] = running
        if reset_at is not None:
            running = np.where(running <= -reset_at, 0.0, running)
    return accumulated


@dataclass
class RobustAnomalyDetector:
    """Score every station-day by how far demand fell short of expectation."""

    false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE
    calibration_days: int = DEFAULT_CALIBRATION_DAYS
    cusum_drift: float = 0.5
    min_line_stations: int = 3
    min_scale: float = 1.0

    # Set either of these to pin a cut-off instead of calibrating it.
    threshold: float | None = None
    cusum_threshold: float | None = None

    expectation_: HierarchicalSeasonalModel = field(init=False, repr=False, default=None)
    columns_: pd.Index = field(init=False, repr=False, default=None)
    center_: np.ndarray = field(init=False, repr=False, default=None)
    scale_: np.ndarray = field(init=False, repr=False, default=None)
    threshold_: float = field(init=False, repr=False, default=DEFAULT_THRESHOLD)
    cusum_threshold_: float = field(init=False, repr=False, default=8.0)
    calibration_scores_: np.ndarray = field(init=False, repr=False, default=None)
    calibration_cusum_: np.ndarray = field(init=False, repr=False, default=None)

    def fit(self, operating: pd.DataFrame) -> RobustAnomalyDetector:
        """Learn expected demand and residual spread from days the network ran.

        ``operating`` is the wide panel with non-operating days already ``NaN``.
        """
        if operating.empty:
            raise ValueError("RobustAnomalyDetector needs at least one row of history")
        self.expectation_ = HierarchicalSeasonalModel().fit(operating)
        self.columns_ = operating.columns

        calibration = operating.iloc[-self.calibration_days :]
        residual = self._residual(calibration)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.center_ = np.nanmedian(residual, axis=0)
        scale = median_absolute_deviation(residual) * MAD_TO_SIGMA
        # A station with almost no variation would otherwise turn every ripple
        # into an extreme score.
        self.scale_ = np.where(np.isfinite(scale) & (scale > self.min_scale), scale, np.nan)

        calibration_scores = self.score(calibration).to_numpy()
        self.calibration_scores_ = calibration_scores.ravel()
        # Calibrated without the reset, because the reset needs the threshold
        # this very step is computing. The effect is one-directional: the
        # detector in use resets and therefore alarms no more often than the
        # rate it was calibrated for, never more.
        self.calibration_cusum_ = cusum_negative(
            calibration_scores, drift=self.cusum_drift
        ).ravel()
        self.set_false_alarm_rate(self.false_alarm_rate)
        return self

    def set_false_alarm_rate(self, rate: float) -> RobustAnomalyDetector:
        """Re-read both cut-offs off the stored training scores at a new alarm rate.

        Separating this from :meth:`fit` is what makes an operating curve cheap:
        the expensive part is estimating expected demand, and that does not
        depend on how many alerts the operator is willing to receive.
        """
        if self.calibration_scores_ is None:
            raise ValueError("Call fit before setting the false alarm rate")
        self.false_alarm_rate = rate
        self.threshold_ = self._quantile_cutoff(
            self.calibration_scores_, self.threshold, DEFAULT_THRESHOLD
        )
        self.cusum_threshold_ = self._quantile_cutoff(
            self.calibration_cusum_, self.cusum_threshold, 8.0
        )
        return self

    def _quantile_cutoff(
        self, values: np.ndarray, override: float | None, fallback: float
    ) -> float:
        if override is not None:
            return override
        usable = values[np.isfinite(values)]
        if usable.size == 0:
            return fallback
        return max(-float(np.quantile(usable, self.false_alarm_rate)), 1.0)

    def _residual(self, observed: pd.DataFrame) -> np.ndarray:
        expected = self.expectation_.expected(observed.index).reindex(columns=observed.columns)
        return observed.to_numpy(dtype="float64") - expected.to_numpy(dtype="float64")

    def expected(self, observed: pd.DataFrame) -> pd.DataFrame:
        """What each station-day should have been."""
        return self.expectation_.expected(observed.index).reindex(columns=observed.columns)

    def score(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Robust z-score of each station-day. Negative means demand fell short."""
        if self.expectation_ is None:
            raise ValueError("Call fit before scoring")
        residual = self._residual(observed)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (residual - self.center_) / self.scale_
        return pd.DataFrame(z, index=observed.index, columns=observed.columns)

    def cusum(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Accumulated shortfall, for detecting a drop that persists."""
        scores = self.score(observed)
        return pd.DataFrame(
            cusum_negative(
                scores.to_numpy(), drift=self.cusum_drift, reset_at=self.cusum_threshold_
            ),
            index=scores.index,
            columns=scores.columns,
        )

    def reasons(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Why each station-day was flagged: an extreme day, a sustained fall, or neither."""
        scores = self.score(observed)
        by_score = scores <= -self.threshold_
        by_drift = self.cusum(observed) <= -self.cusum_threshold_
        reason = pd.DataFrame("", index=scores.index, columns=scores.columns)
        reason = reason.mask(by_drift, "sustained shortfall")
        return reason.mask(by_score, "extreme day")

    def detect(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Flag station-days that are anomalously low, by score or by persistence."""
        scores = self.score(observed)
        return (scores <= -self.threshold_) | (
            self.cusum(observed) <= -self.cusum_threshold_
        )

    def line_incidents(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Days on which enough stations of one line deviate together to blame the line."""
        flags = self.detect(observed)
        lines = pd.Index([column.split(":", 1)[0] for column in flags.columns], name="line_id")
        counts = flags.T.groupby(lines).sum().T
        return counts >= self.min_line_stations
