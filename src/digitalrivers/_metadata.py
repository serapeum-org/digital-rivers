"""Shared metadata keys and valid-value sets for typed result classes.

These constants are the single source of truth for GeoTIFF metadata keys
written by ``FlowDirection.persist_metadata`` (and equivalent methods on
``Accumulation``/``StreamRaster``). Keeping them in one module avoids
string-typo bugs across the typed-class files.
"""
from __future__ import annotations

__all__ = [
    "META_CLASS",
    "META_ROUTING",
    "META_ENCODING",
    "META_THRESHOLD",
    "VALID_ROUTING",
    "VALID_ENCODING",
]

META_CLASS = "DR_CLASS"
META_ROUTING = "DR_ROUTING"
META_ENCODING = "DR_ENCODING"
META_THRESHOLD = "DR_THRESHOLD"

VALID_ROUTING = frozenset({"d8", "dinf", "mfd_quinn", "mfd_holmgren", "rho8"})
VALID_ENCODING = frozenset({"digitalrivers", "taudem", "esri", "whitebox"})
