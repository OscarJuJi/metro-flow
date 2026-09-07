"""Canonicalization is what collapses 24 line labels back onto 12 lines."""

import pandas as pd
import pytest

from metro_pulse.clean.canonical import (
    CanonicalizationError,
    assert_expected_network,
    build_station_registry,
    canonicalize,
    display_name,
    line_sort_rank,
    parse_line_id,
    slugify_station,
)
from metro_pulse.config import METRO, NetworkShape

MOJIBAKE_LINE_1 = "LÃ\xadnea 1"  # what the 2021-2023 slice literally stores
MOJIBAKE_SUAREZ = "Pino SuÃ¡rez"


def test_parse_line_id_maps_both_spellings_onto_one_line():
    assert parse_line_id("Linea 1") == "1"
    assert parse_line_id(MOJIBAKE_LINE_1) == "1"
    assert parse_line_id("Linea 12") == "12"
    assert parse_line_id("Linea A") == "A"
    assert parse_line_id("Linea B") == "B"


def test_parse_line_id_rejects_an_unknown_line():
    with pytest.raises(CanonicalizationError, match="Unrecognized line label"):
        parse_line_id("Tren Ligero")


def test_slugify_station_is_stable_across_spellings():
    assert slugify_station("Pino Suárez") == "pino-suarez"
    assert slugify_station(MOJIBAKE_SUAREZ) == "pino-suarez"
    assert slugify_station("La Villa/Basílica") == "la-villa-basilica"
    assert slugify_station("Etiopía/Plaza de la Transparencia").startswith("etiopia-")


def test_display_name_repairs_mojibake_for_humans():
    assert display_name(MOJIBAKE_SUAREZ) == "Pino Suárez"
    assert display_name("Zaragoza") == "Zaragoza"


def test_line_sort_rank_orders_numeric_lines_before_lettered_ones():
    ordered = sorted(["B", "12", "1", "A", "2"], key=line_sort_rank)
    assert ordered == ["1", "2", "12", "A", "B"]


def make_raw(rows):
    frame = pd.DataFrame(rows, columns=["fecha", "linea", "estacion", "afluencia"])
    frame["anio"] = frame["fecha"].str.slice(0, 4).astype(int)
    frame["mes"] = "Enero"
    return frame[["fecha", "anio", "mes", "linea", "estacion", "afluencia"]]


def test_canonicalize_merges_the_two_spellings_of_the_same_station_day():
    raw = make_raw(
        [
            ("2021-01-01", "Linea 1", "Pino Suárez", 10),
            ("2021-01-02", MOJIBAKE_LINE_1, MOJIBAKE_SUAREZ, 20),
        ]
    )
    canonical = canonicalize(raw)
    assert canonical["line_id"].unique().tolist() == ["1"]
    assert canonical["station_id"].unique().tolist() == ["pino-suarez"]
    assert canonical["station_name"].unique().tolist() == ["Pino Suárez"]


def test_assert_expected_network_rejects_an_incomplete_network():
    raw = make_raw([("2021-01-01", "Linea 1", "Zaragoza", 10)])
    with pytest.raises(CanonicalizationError, match="did not reconstruct"):
        assert_expected_network(canonicalize(raw), METRO.network)


def test_assert_expected_network_accepts_a_network_that_matches_its_declared_shape():
    raw = make_raw(
        [
            ("2021-01-01", "Linea 1", "Zaragoza", 10),
            ("2021-01-01", "Linea 2", "Tacuba", 20),
        ]
    )
    shape = NetworkShape(n_lines=2, n_stations=2, n_pairs=2)
    assert assert_expected_network(canonicalize(raw), shape) == {
        "n_lines": 2,
        "n_stations": 2,
        "n_pairs": 2,
    }


def test_station_registry_reports_the_observed_span_per_pair():
    raw = make_raw(
        [
            ("2012-11-05", "Linea 12", "Zapata", 10),
            ("2012-11-06", "Linea 12", "Zapata", 12),
            ("2012-11-05", "Linea 3", "Zapata", 30),
        ]
    )
    registry = build_station_registry(canonicalize(raw))
    assert len(registry) == 2  # Zapata is a transfer: one row per line
    line_12 = registry[registry.line_id == "12"].iloc[0]
    assert line_12.n_days == 2
    assert str(line_12.first_date.date()) == "2012-11-05"
