"""Station coordinates: parsing the KMZ, and refusing a join that loses stations."""

import io
import zipfile

import pandas as pd
import pytest

from metro_pulse.ingest.geometry import (
    GeometryError,
    apply_aliases,
    extract_kml,
    parse_line_code,
    parse_stations,
    reconcile,
)

PLACEMARK = """
<Placemark id="ID_{i}">
  <name>STC Metro</name>
  <description><![CDATA[
    <tr><td>NOMBRE</td><td>{name}</td></tr>
    <tr><td>LINEA</td><td>{line}</td></tr>
  ]]></description>
  <Point><coordinates> {lon},{lat},0</coordinates></Point>
</Placemark>
"""


def kml(stations) -> str:
    body = "".join(
        PLACEMARK.format(i=i, name=n, line=li, lon=lon, lat=lat)
        for i, (li, n, lon, lat) in enumerate(stations)
    )
    return f"<?xml version='1.0' encoding='UTF-8'?><kml><Document>{body}</Document></kml>"


def nested_kmz(kml_text: str, member: str = "stcmetro_kmz/STC_Metro_estaciones.kmz") -> bytes:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("doc.kml", kml_text.encode("utf-8"))
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        outer.writestr(member, inner_buffer.getvalue())
    return outer_buffer.getvalue()


class TestParseLineCode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("01", "1"), ("09", "9"), ("12", "12"), ("A", "A"), ("b", "B"), (" 03 ", "3")],
    )
    def test_matches_the_ids_the_panel_uses(self, raw, expected):
        assert parse_line_code(raw) == expected


class TestParsing:
    def test_reads_name_line_and_coordinates_out_of_a_placemark(self):
        text = kml([("01", "Pino Suárez", -99.1332, 19.4256)])
        stations = parse_stations(text)

        assert len(stations) == 1
        row = stations.iloc[0]
        assert row.line_id == "1"
        assert row.station_id == "pino-suarez"
        assert row.station_name == "Pino Suárez"
        assert row.longitude == pytest.approx(-99.1332)
        assert row.latitude == pytest.approx(19.4256)

    def test_longitude_comes_first_in_kml_which_is_the_usual_trap(self):
        stations = parse_stations(kml([("01", "Zaragoza", -99.08, 19.41)]))
        assert stations.iloc[0].longitude < 0
        assert stations.iloc[0].latitude > 0

    def test_a_kml_with_no_stations_is_an_error_not_an_empty_frame(self):
        with pytest.raises(GeometryError, match="no station placemarks"):
            parse_stations("<kml><Document></Document></kml>")

    def test_extracts_the_kml_from_the_nested_archive(self):
        text = kml([("01", "Balderas", -99.1494, 19.4269)])
        assert "Balderas" in extract_kml(nested_kmz(text))

    def test_an_archive_without_the_station_layer_is_reported_clearly(self):
        payload = nested_kmz(kml([]), member="stcmetro_kmz/STC_Metro_lineas.kmz")
        with pytest.raises(GeometryError, match="not found in the archive"):
            extract_kml(payload)


class TestAliases:
    def test_rewrites_a_renamed_station_onto_the_id_the_panel_uses(self):
        geometry = pd.DataFrame(
            {"line_id": ["3"], "station_id": ["ninos-heroes-poder-judicial-cdmx"]}
        )
        aliases = pd.DataFrame(
            {
                "line_id": ["3"],
                "alias_station_id": ["ninos-heroes-poder-judicial-cdmx"],
                "canonical_station_id": ["ninos-heroes"],
            }
        )
        assert apply_aliases(geometry, aliases).iloc[0].station_id == "ninos-heroes"

    def test_leaves_stations_with_no_alias_alone(self):
        geometry = pd.DataFrame({"line_id": ["1"], "station_id": ["zaragoza"]})
        aliases = pd.DataFrame(
            {
                "line_id": ["3"],
                "alias_station_id": ["ninos-heroes-poder-judicial-cdmx"],
                "canonical_station_id": ["ninos-heroes"],
            }
        )
        assert apply_aliases(geometry, aliases).iloc[0].station_id == "zaragoza"

    def test_an_alias_only_applies_on_its_own_line(self):
        geometry = pd.DataFrame({"line_id": ["9"], "station_id": ["mixhiuca"]})
        aliases = pd.DataFrame(
            {
                "line_id": ["1"],
                "alias_station_id": ["mixhiuca"],
                "canonical_station_id": ["mixiuhca"],
            }
        )
        assert apply_aliases(geometry, aliases).iloc[0].station_id == "mixhiuca"


class TestReconcile:
    def registry(self, pairs):
        return pd.DataFrame(
            [{"line_id": li, "station_id": st, "station_name": st} for li, st in pairs]
        )

    def geometry(self, pairs, lon=-99.13, lat=19.43):
        return pd.DataFrame(
            [
                {"line_id": li, "station_id": st, "longitude": lon, "latitude": lat}
                for li, st in pairs
            ]
        )

    def test_a_complete_match_passes_through(self):
        pairs = [("1", "zaragoza"), ("2", "tacuba")]
        merged = reconcile(self.geometry(pairs), self.registry(pairs))
        assert len(merged) == 2
        assert set(merged.columns) >= {"line_id", "station_id", "longitude", "latitude"}

    def test_a_station_without_coordinates_stops_the_pipeline(self):
        registry = self.registry([("1", "zaragoza"), ("2", "tacuba")])
        geometry = self.geometry([("1", "zaragoza")])
        with pytest.raises(GeometryError, match="tacuba"):
            reconcile(geometry, registry)

    def test_coordinates_for_a_station_the_panel_lacks_stop_the_pipeline(self):
        registry = self.registry([("1", "zaragoza")])
        geometry = self.geometry([("1", "zaragoza"), ("1", "ghost-station")])
        with pytest.raises(GeometryError, match="ghost-station"):
            reconcile(geometry, registry)

    def test_a_point_outside_the_valley_of_mexico_is_rejected(self):
        """A swapped latitude and longitude lands in the ocean; say so."""
        pairs = [("1", "zaragoza")]
        geometry = self.geometry(pairs, lon=19.43, lat=-99.13)
        with pytest.raises(GeometryError, match="outside the Valley of Mexico"):
            reconcile(geometry, self.registry(pairs))
