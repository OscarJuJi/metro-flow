"""Normalization must collapse the spelling drift present in the source data."""

from metro_pulse.text import normalize_key, strip_accents


def test_strip_accents_preserves_letters():
    assert strip_accents("Pino Suárez") == "Pino Suarez"
    assert strip_accents("Etiopía/Plaza de la Transparencia").startswith("Etiopia")


def test_normalize_key_unifies_the_two_line_spellings():
    # The raw file carries both spellings; 2021-2023 uses the accented one.
    assert normalize_key("Linea 1") == normalize_key("Línea 1")
    assert normalize_key("Linea B") == normalize_key("LÍNEA  b")


def test_normalize_key_unifies_separator_and_space_drift():
    assert normalize_key("La Villa/Basílica") == normalize_key("La Villa / Basilica")
    assert normalize_key("  Zaragoza  ") == "zaragoza"


def test_normalize_key_keeps_distinct_names_distinct():
    assert normalize_key("Zapata") != normalize_key("Zócalo")
