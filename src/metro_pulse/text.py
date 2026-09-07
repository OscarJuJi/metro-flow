"""Unicode repair and normalization shared by ingestion and cleaning.

The source panel was assembled from two exports and the 2021-2023 slice is
*mojibake*: its UTF-8 bytes were decoded as latin-1 and re-encoded as UTF-8, so
the file literally stores ``"LÃ­nea 1"`` where it means ``"Línea 1"``.
Stripping accents does not merge those two spellings -- the text has to be
repaired first. Every join key in this project therefore goes through
:func:`normalize_key`, which repairs before it normalizes.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_SEPARATORS = re.compile(r"\s*/\s*")


def fix_mojibake(text: str) -> str:
    """Undo one latin-1/UTF-8 double encoding, when the round trip is lossless.

    ``"LÃ­nea 1"`` -> ``"Línea 1"``. Text that is already correct fails the
    round trip (``"Pino Suárez"`` has no latin-1 encoding of its UTF-8 bytes) and
    is returned untouched, so this is safe to apply to every value.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def strip_accents(text: str) -> str:
    """Drop diacritics: ``"Pino Suárez"`` -> ``"Pino Suarez"``."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_key(text: str) -> str:
    """Collapse a label into a stable comparison key.

    Mojibake-repaired, lowercased, accent-free, single-spaced, with ``/``
    separators normalized so ``"La Villa/Basílica"`` and ``"La Villa / Basilica"``
    land on the same key.
    """
    cleaned = strip_accents(fix_mojibake(str(text))).strip().lower()
    cleaned = _SEPARATORS.sub("/", cleaned)
    return _WHITESPACE.sub(" ", cleaned)
