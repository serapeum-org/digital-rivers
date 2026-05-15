"""Typed stream-network raster.

``StreamRaster.__init__`` enforces the *ismulti guard* from TopoToolbox
MATLAB ``@STREAMobj/STREAMobj.m:36`` — stream extraction from a
multi-direction flow scheme is not well-defined, so the constructor rejects
any ``routing`` outside the supported single-direction set up front.
"""
from __future__ import annotations

from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers._metadata import (
    META_CLASS,
    META_ROUTING,
    META_THRESHOLD,
    VALID_ROUTING,
)


class StreamRaster(Dataset):
    """Boolean/int stream-network raster tagged with extraction threshold.

    Args:
        src: GDAL dataset wrapping the stream raster.
        access: ``"read_only"`` (default) or ``"write"``.
        threshold: Accumulation threshold used to extract this stream
            network. Stored for provenance and round-trip persistence.
            Required keyword-only.
        routing: Routing scheme of the ``FlowDirection`` that produced the
            upstream accumulation. Required keyword-only. Must be in
            ``_SUPPORTED_ROUTING``.

    Raises:
        ValueError: If ``routing`` is not a recognised value at all.
        TypeError: If ``routing`` is a multi-direction scheme. Convert the
            ``FlowDirection`` to D8 first.
    """

    threshold: float | int
    routing: str

    _SUPPORTED_ROUTING: frozenset[str] = frozenset({"d8"})

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        threshold: float | int,
        routing: str,
    ):
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        if routing not in self._SUPPORTED_ROUTING:
            raise TypeError(
                f"StreamRaster currently supports only single-direction routing "
                f"({sorted(self._SUPPORTED_ROUTING)}); got {routing!r}. "
                f"Convert the FlowDirection to D8 first."
            )
        super().__init__(src, access)
        self.threshold = threshold
        self.routing = routing

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        threshold: float | int,
        routing: str,
    ) -> StreamRaster:
        """Promote a plain ``Dataset`` into a ``StreamRaster``."""
        return cls(ds.raster, threshold=threshold, routing=routing)

    def to_dataset(self) -> Dataset:
        """Drop the typed wrapper and return the underlying ``Dataset``."""
        return Dataset(self.raster)

    def persist_metadata(self) -> None:
        """Write ``routing`` and ``threshold`` to the raster's metadata tags."""
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
            META_THRESHOLD: str(self.threshold),
        }

    @classmethod
    def open(
        cls,
        path: str,
        *,
        threshold: float | int | None = None,
        routing: str | None = None,
    ) -> StreamRaster:
        """Open a ``StreamRaster`` GeoTIFF.

        Resolution order: explicit kwargs > ``DR_*`` metadata tags > raise.
        ``threshold`` is parsed from the tag as a float (it was written via
        ``str(self.threshold)``).

        Raises:
            ValueError: If either ``routing`` or ``threshold`` cannot be
                resolved from kwargs or metadata tags.
        """
        ds = Dataset.read_file(path)
        md = ds.meta_data or {}
        resolved_routing = routing or md.get(META_ROUTING)
        if resolved_routing is None:
            raise ValueError(
                f"{path!r} carries no DR_ROUTING tag and no routing= was passed. "
                f"Pass routing= explicitly (one of {sorted(VALID_ROUTING)})."
            )
        if threshold is None:
            tag = md.get(META_THRESHOLD)
            if tag is None:
                raise ValueError(
                    f"{path!r} carries no DR_THRESHOLD tag and no threshold= was "
                    f"passed."
                )
            threshold = float(tag)
        return cls(ds.raster, threshold=threshold, routing=resolved_routing)

    def __repr__(self) -> str:
        return (
            f"<StreamRaster rows={self.rows} cols={self.columns} "
            f"threshold={self.threshold!r} routing={self.routing!r}>"
        )
