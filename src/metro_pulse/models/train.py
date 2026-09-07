"""Fit the production models and write the artifacts the service reads.

Run as ``python -m metro_pulse.models.train``.

The source publishes once a month, so everything the API serves is computed
here, once, and stored: a 14-day forecast per station and the anomaly scores of
the recent past. Running LightGBM inside a request handler would burn CPU
recomputing an answer that cannot change until the next monthly release.

Outputs, all under ``artifacts/``:

``forecast.parquet``   one row per station and horizon
``anomalies.parquet``  recent station-days with their score and verdict
``model.joblib``       the fitted forecaster, for inspection and for reuse
``metadata.json``      what was trained, on what, and through which date
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

import joblib
import pandas as pd

from metro_pulse.config import ARTIFACTS_DIR, OPERATORS, PROJECT_ROOT, ensure_dirs
from metro_pulse.models.anomaly import RobustAnomalyDetector
from metro_pulse.models.gbm import DEFAULT_HORIZONS, GradientBoostingForecaster
from metro_pulse.panel import load_panel, split_series_id, to_wide

# The operating point recommended in the README: about nine alerts a day across
# the network, catching 90% of real interruptions within three days of onset.
SERVING_FALSE_ALARM_RATE = 0.01

ANOMALY_WINDOW_DAYS = 120


def station_names(panel: pd.DataFrame) -> pd.Series:
    """Display name per series id, for the API and the dashboard."""
    named = panel.drop_duplicates(subset=["line_id", "station_id"])
    return pd.Series(
        named["station_name"].to_numpy(),
        index=(named["line_id"] + ":" + named["station_id"]).to_numpy(),
    )


def station_coordinates(operator_key: str) -> pd.DataFrame:
    """Coordinates keyed by series id, or an empty frame if they were never ingested."""
    path = PROJECT_ROOT / "reference" / f"{operator_key}_station_coordinates.csv"
    if not path.exists():
        return pd.DataFrame(columns=["series_id", "longitude", "latitude"])
    coordinates = pd.read_csv(path, encoding="utf-8", dtype={"line_id": "string"})
    coordinates["series_id"] = coordinates["line_id"] + ":" + coordinates["station_id"]
    return coordinates[["series_id", "longitude", "latitude"]]


def long_forecast(prediction: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """Turn the wide forecast into the row-per-station-day the API serves."""
    long = (
        prediction.rename_axis(index="date", columns="series_id")
        .stack()
        .rename("prediction")
        .reset_index()
    )
    long["horizon"] = (long["date"] - long["date"].min()).dt.days + 1
    ids = long["series_id"].map(split_series_id)
    long["line_id"] = [line for line, _ in ids]
    long["station_id"] = [station for _, station in ids]
    long["station_name"] = long["series_id"].map(names)
    long["prediction"] = long["prediction"].round().astype("Int64")
    return long[
        ["date", "horizon", "series_id", "line_id", "station_id", "station_name", "prediction"]
    ]


def recent_anomalies(
    detector: RobustAnomalyDetector,
    observed: pd.DataFrame,
    names: pd.Series,
    *,
    window: int = ANOMALY_WINDOW_DAYS,
) -> pd.DataFrame:
    """Score the recent past so the dashboard can show what has been happening."""
    recent = observed.iloc[-window:]

    def stacked(frame: pd.DataFrame) -> pd.Series:
        return frame.rename_axis(index="date", columns="series_id").stack()

    # Concatenating on the shared (date, series) index rather than assigning raw
    # arrays: the four frames need not stack in the same order, and a silent
    # misalignment here would attach every score to the wrong station.
    long = pd.concat(
        {
            "ridership": stacked(recent),
            "expected": stacked(detector.expected(recent)),
            "score": stacked(detector.score(recent)),
            "is_anomaly": stacked(detector.detect(recent)),
            "reason": stacked(detector.reasons(recent)),
        },
        axis=1,
    ).reset_index()
    long["is_anomaly"] = long["is_anomaly"].fillna(False).astype(bool)
    ids = long["series_id"].map(split_series_id)
    long["line_id"] = [line for line, _ in ids]
    long["station_id"] = [station for _, station in ids]
    long["station_name"] = long["series_id"].map(names)
    return long


def train(operator_key: str = "metro", *, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> dict:
    """Fit both models on the whole panel and write every serving artifact."""
    ensure_dirs()
    panel = load_panel(operator_key)
    names = station_names(panel)
    operating = to_wide(panel, value="target")
    observed = to_wide(panel, value="ridership")
    data_through = panel["date"].max()

    print(f"[1/4] Fitting the forecaster on {len(operating):,} days ...")
    forecaster = GradientBoostingForecaster(horizons=horizons).fit(operating)

    print("[2/4] Forecasting the next", max(horizons), "days ...")
    forecast = long_forecast(forecaster.predict(max(horizons)), names)

    print("[3/4] Fitting the detector and scoring the recent past ...")
    detector = RobustAnomalyDetector(false_alarm_rate=SERVING_FALSE_ALARM_RATE).fit(operating)
    anomalies = recent_anomalies(detector, observed, names)

    print("[4/4] Writing artifacts ...")
    coordinates = station_coordinates(operator_key)
    forecast.to_parquet(ARTIFACTS_DIR / "forecast.parquet", index=False)
    anomalies.to_parquet(ARTIFACTS_DIR / "anomalies.parquet", index=False)
    if not coordinates.empty:
        coordinates.to_parquet(ARTIFACTS_DIR / "coordinates.parquet", index=False)
    joblib.dump(forecaster, ARTIFACTS_DIR / "model.joblib", compress=3)

    flagged = int(anomalies["is_anomaly"].sum())
    metadata = {
        "operator": operator_key,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_through": data_through.strftime("%Y-%m-%d"),
        "n_panel_rows": int(len(panel)),
        "n_series": int(operating.shape[1]),
        "horizons": list(horizons),
        "forecast_from": forecast["date"].min().strftime("%Y-%m-%d"),
        "forecast_to": forecast["date"].max().strftime("%Y-%m-%d"),
        "anomaly_window_days": ANOMALY_WINDOW_DAYS,
        "false_alarm_rate": SERVING_FALSE_ALARM_RATE,
        "anomaly_threshold": round(detector.threshold_, 3),
        "cusum_threshold": round(detector.cusum_threshold_, 3),
        "anomalies_flagged": flagged,
        "has_coordinates": not coordinates.empty,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", default="metro", choices=sorted(OPERATORS))
    args = parser.parse_args(argv)

    metadata = train(args.operator)
    print()
    print("Trained.")
    for key, value in metadata.items():
        print(f"  {key:<22} {value}")
    print(f"  artifacts             {ARTIFACTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
