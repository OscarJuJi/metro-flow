"""Forecast accuracy, measured only where the network was actually running.

Every function ignores ``NaN`` in the truth: a station that was closed has no
demand to be wrong about, and scoring those days would reward a model for
predicting the closure rather than the demand.

Two skill numbers are reported and they answer different questions.

``mase``
    The textbook Mean Absolute Scaled Error: test MAE divided by the *in-sample*
    one-step seasonal-naive MAE. Comparable with the forecasting literature.

``relative_mae``
    Test MAE divided by the seasonal naive's MAE *on the same test days*. This
    is the number that says whether the model beats the baseline on the days it
    was actually asked about, and it equals exactly 1.0 for the baseline itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SEASON_LENGTH = 7


def _aligned(truth: pd.DataFrame, prediction: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both frames onto the station-days where the truth is observed."""
    prediction = prediction.reindex(index=truth.index, columns=truth.columns)
    y = truth.to_numpy(dtype="float64").ravel()
    yhat = prediction.to_numpy(dtype="float64").ravel()
    observed = ~np.isnan(y)
    return y[observed], yhat[observed]


def mae(truth: pd.DataFrame, prediction: pd.DataFrame) -> float:
    """Mean absolute error. Unforecast station-days count as errors, not as zeros."""
    y, yhat = _aligned(truth, prediction)
    if y.size == 0:
        return float("nan")
    errors = np.abs(y - np.nan_to_num(yhat, nan=0.0))
    return float(errors.mean())


def smape(truth: pd.DataFrame, prediction: pd.DataFrame) -> float:
    """Symmetric MAPE in percent, skipping days where both values are zero."""
    y, yhat = _aligned(truth, prediction)
    yhat = np.nan_to_num(yhat, nan=0.0)
    denominator = (np.abs(y) + np.abs(yhat)) / 2
    usable = denominator > 0
    if not usable.any():
        return float("nan")
    return float(100 * np.mean(np.abs(y - yhat)[usable] / denominator[usable]))


def seasonal_naive_insample_mae(history: pd.DataFrame, season_length: int = SEASON_LENGTH) -> float:
    """MASE denominator: the one-step seasonal-naive error inside the training data."""
    values = history.to_numpy(dtype="float64")
    if values.shape[0] <= season_length:
        return float("nan")
    differences = np.abs(values[season_length:] - values[:-season_length])
    if np.isnan(differences).all():
        return float("nan")
    return float(np.nanmean(differences))


def mase(truth: pd.DataFrame, prediction: pd.DataFrame, insample_mae: float) -> float:
    """Mean absolute scaled error. Below 1.0 beats a same-weekday guess."""
    if not np.isfinite(insample_mae) or insample_mae <= 0:
        return float("nan")
    return mae(truth, prediction) / insample_mae


def relative_mae(
    truth: pd.DataFrame, prediction: pd.DataFrame, baseline: pd.DataFrame
) -> float:
    """Model MAE over baseline MAE on the same days. Exactly 1.0 for the baseline."""
    baseline_error = mae(truth, baseline)
    if not np.isfinite(baseline_error) or baseline_error <= 0:
        return float("nan")
    return mae(truth, prediction) / baseline_error


@dataclass
class ForecastScore:
    """One model's accuracy on one evaluation window."""

    n_observations: int
    mae: float
    smape: float
    mase: float
    relative_mae: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def score_forecast(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    baseline: pd.DataFrame,
    insample_mae: float,
) -> ForecastScore:
    """Score one forecast against the truth, the baseline and the training error."""
    y, _ = _aligned(truth, prediction)
    return ForecastScore(
        n_observations=int(y.size),
        mae=mae(truth, prediction),
        smape=smape(truth, prediction),
        mase=mase(truth, prediction, insample_mae),
        relative_mae=relative_mae(truth, prediction, baseline),
    )
