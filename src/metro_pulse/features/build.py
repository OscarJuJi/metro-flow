"""Turn the wide panel into a supervised table for a global model.

One model is trained for all 195 series at once, with the forecast horizon as a
feature. Each training row answers one question: *given everything known up to
day T, what was ridership at station s on day T+h?*

The rule that makes the table trustworthy is that every feature is computed at
the **origin** T and never at the target date. The only thing taken from the
target date is its calendar -- the day of the week and whether it is a holiday
are known years in advance and carry no information about demand that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metro_pulse.features.calendar import calendar_features

ORIGIN_LAGS = (0, 1, 2, 6, 13, 20, 27, 363)
ROLLING_WINDOWS = (7, 28, 91)

CALENDAR_COLUMNS = [
    "day_of_week",
    "effective_day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "is_payday",
    "is_day_after_payday",
    "is_holy_week",
    "is_year_end_break",
]


def same_weekday_lag(horizon: int) -> int:
    """How far back from the origin the most recent same-weekday-as-target day sits.

    For a two-day-ahead forecast that is five days before the origin; for a
    ten-day-ahead forecast, four. Always non-negative, so it never reaches past
    the origin into the future.
    """
    return 7 * int(np.ceil(horizon / 7)) - horizon


def warmup_days() -> int:
    """Days of history a row needs before all of its features are defined."""
    return max(max(ORIGIN_LAGS), max(ROLLING_WINDOWS))


def _rolling_frames(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    """Trailing statistics evaluated at every day, per series."""
    frames: dict[str, np.ndarray] = {}
    for window in ROLLING_WINDOWS:
        rolling = wide.rolling(window=window, min_periods=max(3, window // 4))
        frames[f"roll_mean_{window}"] = rolling.mean().to_numpy(dtype="float32")
        frames[f"roll_median_{window}"] = rolling.median().to_numpy(dtype="float32")
    frames["roll_std_7"] = wide.rolling(window=7, min_periods=3).std().to_numpy(dtype="float32")
    return frames


def _shifted(values: np.ndarray, lag: int) -> np.ndarray:
    """``values`` shifted down by ``lag`` rows, padded with NaN at the top."""
    if lag == 0:
        return values
    out = np.full_like(values, np.nan)
    out[lag:] = values[:-lag]
    return out


def _origin_blocks(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    """Everything computable at an origin: lags and trailing statistics."""
    values = wide.to_numpy(dtype="float32")
    lagged = {f"lag_{lag}": _shifted(values, lag) for lag in ORIGIN_LAGS}
    return {**lagged, **_rolling_frames(wide)}


def _identity_arrays(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series_ids = wide.columns.to_numpy()
    line_ids = np.array([sid.split(":", 1)[0] for sid in series_ids])
    station_ids = np.array([sid.split(":", 1)[1] for sid in series_ids])
    return series_ids, line_ids, station_ids


def build_supervised_frame(
    wide: pd.DataFrame,
    *,
    horizons: range | tuple[int, ...],
    origins: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build the (origin, series, horizon) training table from a wide panel.

    ``origins`` defaults to every date with enough history behind it and the
    longest horizon ahead of it. Rows whose target is missing -- a closed
    station -- are dropped: they are neither learnable nor scoreable.
    """
    values = wide.to_numpy(dtype="float32")
    dates = wide.index
    origin_block = _origin_blocks(wide)
    calendar = calendar_features(dates)
    series_ids, line_ids, station_ids = _identity_arrays(wide)
    n_series = values.shape[1]

    candidate = dates[warmup_days() : len(dates) - max(horizons)]
    if origins is not None:
        candidate = candidate.intersection(pd.DatetimeIndex(origins))
    origin_positions = dates.get_indexer(candidate)
    n_origins = len(origin_positions)

    blocks = []
    for horizon in horizons:
        target_positions = origin_positions + horizon
        block = {name: array[origin_positions].ravel() for name, array in origin_block.items()}
        block["same_weekday"] = _shifted(values, same_weekday_lag(horizon))[
            origin_positions
        ].ravel()
        block["horizon"] = np.full(n_origins * n_series, horizon, dtype="int16")
        block["origin"] = np.repeat(dates[origin_positions], n_series)
        block["date"] = np.repeat(dates[target_positions], n_series)
        block["series_id"] = np.tile(series_ids, n_origins)
        block["line_id"] = np.tile(line_ids, n_origins)
        block["station_id"] = np.tile(station_ids, n_origins)

        target_calendar = calendar.iloc[target_positions]
        for column in CALENDAR_COLUMNS:
            block[column] = np.repeat(target_calendar[column].to_numpy(), n_series)

        block["target"] = values[target_positions].ravel()
        blocks.append(pd.DataFrame(block))

    frame = pd.concat(blocks, ignore_index=True)
    return frame[frame["target"].notna()].reset_index(drop=True)


def build_prediction_frame(
    wide: pd.DataFrame, *, horizons: range | tuple[int, ...]
) -> pd.DataFrame:
    """Features for forecasting the days that follow the end of ``wide``.

    The origin is the last day of the panel, so the target dates lie beyond it
    and their calendar is generated rather than looked up.
    """
    values = wide.to_numpy(dtype="float32")
    origin_block = _origin_blocks(wide)
    series_ids, line_ids, station_ids = _identity_arrays(wide)
    n_series = values.shape[1]

    origin_position = len(wide) - 1
    origin_date = wide.index[-1]
    horizons = tuple(horizons)
    target_dates = origin_date + pd.to_timedelta(list(horizons), unit="D")
    calendar = calendar_features(pd.DatetimeIndex(target_dates))

    blocks = []
    for offset, horizon in enumerate(horizons):
        block = {name: array[origin_position] for name, array in origin_block.items()}
        block["same_weekday"] = _shifted(values, same_weekday_lag(horizon))[origin_position]
        block["horizon"] = np.full(n_series, horizon, dtype="int16")
        block["origin"] = np.repeat(origin_date, n_series)
        block["date"] = np.repeat(target_dates[offset], n_series)
        block["series_id"] = series_ids
        block["line_id"] = line_ids
        block["station_id"] = station_ids
        for column in CALENDAR_COLUMNS:
            block[column] = np.repeat(calendar[column].to_numpy()[offset], n_series)
        blocks.append(pd.DataFrame(block))

    return pd.concat(blocks, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model inputs: everything except the identifiers, the dates and the target."""
    excluded = {"target", "date", "origin", "series_id"}
    return [column for column in frame.columns if column not in excluded]
