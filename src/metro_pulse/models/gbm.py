"""One gradient-boosted model for all 195 series.

Fitting a separate model per station would mean 195 models trained on ~6,000
rows each, none of which can learn that Sundays are quiet from any station but
its own. A single global model sees every station's history at once and uses
the station identity as a feature, so a station that reopens after a two-year
closure inherits everything the network knows about Sundays and holidays.

The horizon is a feature rather than a separate model, so one fitted object
answers every question from tomorrow to a fortnight out.

The objective is ``l1``: the model is scored on mean absolute error, so it is
trained on mean absolute error too, rather than on a squared loss that would
chase the outliers this data is full of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from metro_pulse.features.build import (
    build_prediction_frame,
    build_supervised_frame,
    feature_columns,
    warmup_days,
)

CATEGORICAL_FEATURES = ["line_id", "station_id"]
DEFAULT_HORIZONS = tuple(range(1, 15))


@dataclass
class GradientBoostingForecaster:
    """LightGBM over the supervised table built by :mod:`metro_pulse.features.build`."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    train_years: float = 3.0
    n_estimators: int = 400
    learning_rate: float = 0.06
    num_leaves: int = 96
    min_child_samples: int = 40
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    random_state: int = 20260905
    n_jobs: int = -1

    model_: lgb.LGBMRegressor = field(init=False, repr=False, default=None)
    categories_: dict[str, pd.Index] = field(init=False, repr=False, default_factory=dict)
    columns_: pd.Index = field(init=False, repr=False, default=None)
    features_: list[str] = field(init=False, repr=False, default_factory=list)
    last_date_: pd.Timestamp = field(init=False, repr=False, default=None)
    history_: pd.DataFrame = field(init=False, repr=False, default=None)

    def _as_categorical(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """Give the identity columns a stable category order across fit and predict."""
        frame = frame.copy()
        for column in CATEGORICAL_FEATURES:
            if fit:
                self.categories_[column] = pd.Index(sorted(frame[column].unique()))
            frame[column] = pd.Categorical(frame[column], categories=self.categories_[column])
        return frame

    def fit(self, history: pd.DataFrame) -> GradientBoostingForecaster:
        """Train on the most recent ``train_years`` of origins in ``history``."""
        needed = warmup_days() + max(self.horizons) + 30
        if len(history) < needed:
            raise ValueError(
                f"GradientBoostingForecaster needs at least {needed} days of history, "
                f"got {len(history)}"
            )

        cutoff = history.index[-1] - pd.DateOffset(years=self.train_years)
        origins = history.index[history.index >= cutoff]
        frame = build_supervised_frame(history, horizons=self.horizons, origins=origins)
        if frame.empty:
            raise ValueError("No usable training rows: every target in the window is missing")

        self.features_ = feature_columns(frame)
        design = self._as_categorical(frame[self.features_], fit=True)

        self.model_ = lgb.LGBMRegressor(
            objective="l1",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.model_.fit(design, frame["target"], categorical_feature=CATEGORICAL_FEATURES)

        self.columns_ = history.columns
        self.last_date_ = history.index[-1]
        self.history_ = history
        return self

    def predict(self, n_steps: int) -> pd.DataFrame:
        """Forecast the ``n_steps`` days after the end of the training history."""
        if self.model_ is None:
            raise ValueError("Call fit before predict")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if n_steps > max(self.horizons):
            raise ValueError(
                f"This model was trained for horizons up to {max(self.horizons)}; "
                f"{n_steps} steps were requested"
            )

        horizons = tuple(range(1, n_steps + 1))
        frame = build_prediction_frame(self.history_, horizons=horizons)
        design = self._as_categorical(frame[self.features_], fit=False)
        frame["prediction"] = np.clip(self.model_.predict(design), 0, None)

        wide = frame.pivot(index="date", columns="series_id", values="prediction")
        return wide.reindex(columns=self.columns_).rename_axis(
            index="date", columns="series_id"
        )

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, most important first."""
        if self.model_ is None:
            raise ValueError("Call fit before asking for importances")
        booster = self.model_.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=booster.feature_name()).sort_values(ascending=False)
