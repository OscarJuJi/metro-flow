"""HTTP interface over the trained artifacts.

Run locally with::

    uvicorn metro_pulse.api.app:app --reload

Nothing here fits a model or reads the panel. The source publishes monthly, so
the forecasts and anomaly scores are computed once by
:mod:`metro_pulse.models.train` and this process just serves them; a request
handler that retrained would be spending CPU to recompute an unchanged answer.

``/`` serves the dashboard, ``/docs`` the generated OpenAPI page.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from metro_pulse.api.store import ArtifactsMissing, Store, load_store

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Metro Pulse",
    description=(
        "Demand forecasting and anomaly detection for the Mexico City Metro, "
        "built on the city's open ridership data."
    ),
    version="0.1.0",
)


def store() -> Store:
    """The loaded artifacts, or a 503 that says how to create them."""
    try:
        return load_store()
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def records(frame: pd.DataFrame) -> list[dict]:
    """Frame to JSON-safe records.

    Dates become ``YYYY-MM-DD`` strings, and every flavour of missing -- ``NaN``
    from a closed station, ``NaT``, ``pd.NA`` from a nullable integer column --
    becomes ``null``, because none of them is valid JSON.
    """
    payload = frame.copy()
    for column in payload.select_dtypes(include=["datetime"]):
        payload[column] = payload[column].dt.strftime("%Y-%m-%d")
    payload = payload.astype(object).where(payload.notna(), None)
    return payload.to_dict(orient="records")


@app.get("/health", tags=["service"])
def health() -> dict:
    """Liveness plus whether the artifacts are actually loadable."""
    try:
        loaded = load_store()
    except ArtifactsMissing as exc:
        return JSONResponse(status_code=503, content={"status": "no artifacts", "detail": str(exc)})
    return {
        "status": "ok",
        "data_through": loaded.metadata["data_through"],
        "trained_at": loaded.metadata["trained_at"],
    }


@app.get("/metadata", tags=["service"])
def metadata() -> dict:
    """What was trained, on what, and through which date."""
    return store().metadata


@app.get("/stations", tags=["network"])
def stations(line: str | None = Query(None, description="filter to one line, e.g. 3 or B")):
    """Every line-station pair the model covers, with coordinates where known."""
    frame = store().stations
    if line is not None:
        frame = frame[frame["line_id"] == line.upper()]
        if frame.empty:
            raise HTTPException(404, f"No station found on line {line!r}")
    return records(frame)


@app.get("/forecast", tags=["forecast"])
def forecast(
    station: str | None = Query(None, description="station id, e.g. pino-suarez"),
    line: str | None = Query(None, description="line id, e.g. 3 or B"),
    horizon: int | None = Query(None, ge=1, le=14, description="days ahead"),
):
    """Predicted ridership for the days following the end of the data."""
    frame = store().forecast
    if station is not None:
        frame = frame[frame["station_id"] == station]
        if frame.empty:
            raise HTTPException(404, f"Unknown station {station!r}; see /stations")
    if line is not None:
        frame = frame[frame["line_id"] == line.upper()]
    if horizon is not None:
        frame = frame[frame["horizon"] == horizon]
    return records(frame)


@app.get("/anomalies", tags=["anomalies"])
def anomalies(
    date: str | None = Query(None, description="a single day, YYYY-MM-DD; default: the latest"),
    station: str | None = Query(None, description="station id"),
    line: str | None = Query(None, description="line id"),
    flagged_only: bool = Query(True, description="only station-days the detector flagged"),
):
    """Recent station-days with observed ridership, expectation, score and verdict."""
    loaded = store()
    frame = loaded.anomalies

    if station is not None:
        frame = frame[frame["station_id"] == station]
        if frame.empty:
            raise HTTPException(404, f"Unknown station {station!r}; see /stations")
    elif date is not None:
        frame = frame[frame["date"] == pd.Timestamp(date)]
        if frame.empty:
            raise HTTPException(404, f"No scored data for {date}; see /metadata")
    # With no station and no date, "what is flagged" means the whole scored
    # window rather than only the last day. A well-behaved detector has quiet
    # days, and a dashboard that shows nothing on those is a dashboard that
    # looks broken exactly when the network is fine.
    if line is not None:
        frame = frame[frame["line_id"] == line.upper()]
    if flagged_only:
        frame = frame[frame["is_anomaly"]]
        frame = frame.sort_values(["date", "score"], ascending=[False, True])
    else:
        frame = frame.sort_values(["date", "series_id"])
    return records(frame)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
