"""The schema contract is the pipeline's tripwire against upstream changes."""

import pandas as pd
import pytest

from metro_pulse.ingest.schema import (
    EXPECTED_COLUMNS,
    RidershipSchemaError,
    validate_ridership_frame,
)


def make_frame(n_rows: int = 12, **overrides) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    frame = pd.DataFrame(
        {
            "fecha": dates.strftime("%Y-%m-%d"),
            "anio": dates.year,
            "mes": "Enero",
            "linea": "Linea 1",
            "estacion": "Zaragoza",
            "afluencia": range(1000, 1000 + n_rows),
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame[list(EXPECTED_COLUMNS)]


def validate(frame: pd.DataFrame):
    # Real files carry >1M rows; fixtures are tiny, so the floor is lowered here.
    return validate_ridership_frame(frame, min_rows=1)


def test_valid_frame_reports_what_it_contains():
    report = validate(make_frame())
    assert report.n_rows == 12
    assert report.date_min == "2024-01-01"
    assert report.n_raw_line_labels == 1
    assert report.n_zero_rows == 0


def test_counts_zero_rows_because_they_carry_the_service_history():
    frame = make_frame(n_rows=4)
    frame.loc[[0, 2], "afluencia"] = 0
    report = validate(frame)
    assert report.n_zero_rows == 2
    assert report.zero_share == 0.5


def test_rejects_a_renamed_or_reordered_column():
    frame = make_frame().rename(columns={"anio": "ano"})
    with pytest.raises(RidershipSchemaError, match="Unexpected columns"):
        validate(frame)


def test_rejects_a_truncated_download():
    with pytest.raises(RidershipSchemaError, match="truncated"):
        validate_ridership_frame(make_frame(n_rows=5), min_rows=1_000_000)


def test_rejects_nulls():
    frame = make_frame()
    frame.loc[0, "estacion"] = None
    with pytest.raises(RidershipSchemaError, match="Null values"):
        validate(frame)


def test_rejects_unparseable_dates():
    with pytest.raises(RidershipSchemaError, match="Unparseable dates"):
        validate(make_frame(fecha="01/01/2024"))


def test_rejects_non_integer_ridership():
    with pytest.raises(RidershipSchemaError, match="integer count"):
        validate(make_frame(afluencia=1234.5))


def test_rejects_negative_ridership():
    frame = make_frame()
    frame.loc[0, "afluencia"] = -1
    with pytest.raises(RidershipSchemaError, match="Negative"):
        validate(frame)


def test_rejects_year_column_disagreeing_with_the_date():
    with pytest.raises(RidershipSchemaError, match="disagrees"):
        validate(make_frame(anio=1999))
