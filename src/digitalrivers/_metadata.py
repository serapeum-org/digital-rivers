"""Shared metadata keys and valid-value sets for typed result classes.

These constants are the single source of truth for GeoTIFF metadata keys
written by `FlowDirection.persist_metadata` (and equivalent methods on
`Accumulation`/`StreamRaster`). Keeping them in one module avoids
string-typo bugs across the typed-class files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "META_CLASS",
    "META_ROUTING",
    "META_ENCODING",
    "META_THRESHOLD",
    "VALID_ROUTING",
    "VALID_ENCODING",
    "resolve_no_val",
]

META_CLASS = "DR_CLASS"
META_ROUTING = "DR_ROUTING"
META_ENCODING = "DR_ENCODING"
META_THRESHOLD = "DR_THRESHOLD"

VALID_ROUTING = frozenset({"d8", "dinf", "mfd_quinn", "mfd_holmgren", "rho8"})
VALID_ENCODING = frozenset({"digitalrivers", "taudem", "esri", "whitebox"})


if TYPE_CHECKING:
    from pyramids.dataset import Dataset


def resolve_no_val(dataset: Dataset) -> float | int | None:
    """Return the dataset's no-data sentinel or None if not set.

    Pyramids exposes `Dataset.no_data_value` as a tuple-per-band. Callers
    that want the band-0 sentinel (or `None` for "no sentinel set") would
    otherwise hand-roll
    `ds.no_data_value[0] if ds.no_data_value else None` everywhere; this
    helper deduplicates that pattern across the typed-class files.

    Args:
        dataset: A pyramids `Dataset` (or any object exposing a
            `no_data_value` attribute). The attribute is treated as a
            tuple-per-band; only band 0 is returned.

    Returns:
        The band-0 no-data sentinel as `float` / `int`, or `None`
        when no sentinel has been set (the attribute is `None` or an
        empty tuple).

    Examples:
        - A pyramids Dataset built with `no_data_value` returns that
          sentinel:

            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from digitalrivers._metadata import resolve_no_val
            >>> ds = Dataset.create_from_array(
            ...     np.ones((2, 2), dtype=np.float32),
            ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
            ...     epsg=4326, no_data_value=-9999.0,
            ... )
            >>> float(resolve_no_val(ds))
            -9999.0

        - An object whose `no_data_value` attribute is `None` returns
          `None`:

            >>> from digitalrivers._metadata import resolve_no_val
            >>> class _Ds:
            ...     no_data_value = None
            >>> resolve_no_val(_Ds()) is None
            True

        - An empty tuple is also treated as "no sentinel":

            >>> from digitalrivers._metadata import resolve_no_val
            >>> class _Ds:
            ...     no_data_value = ()
            >>> resolve_no_val(_Ds()) is None
            True
    """
    nv = dataset.no_data_value
    if not nv:
        return None
    return nv[0]
