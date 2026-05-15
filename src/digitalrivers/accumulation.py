"""Typed flow-accumulation raster.

The ``Accumulation.routing`` attribute is for *provenance only*: it records
which routing scheme produced the upstream counts so that downstream
``Accumulation.streams(threshold)`` extraction can validate routing
compatibility. The accumulation surface itself (a scalar count or weighted
sum per cell) does not depend on the routing scheme.
"""
from __future__ import annotations

from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers._metadata import (
    META_CLASS,
    META_ROUTING,
    VALID_ROUTING,
)


class Accumulation(Dataset):
    """Flow-accumulation raster tagged with the producing routing scheme.

    Args:
        src: GDAL dataset wrapping the accumulation raster.
        access: ``"read_only"`` (default) or ``"write"``.
        routing: Routing scheme of the ``FlowDirection`` that produced this
            accumulation. Required keyword-only argument. Used as provenance
            so ``streams(threshold)`` can validate compatibility downstream.

    Raises:
        ValueError: If ``routing`` is not a recognised value.
    """

    routing: str

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        routing: str,
    ):
        super().__init__(src, access)
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        self.routing = routing

    @classmethod
    def from_dataset(cls, ds: Dataset, *, routing: str) -> Accumulation:
        """Promote a plain ``Dataset`` into an ``Accumulation``."""
        return cls(ds.raster, routing=routing)

    def to_dataset(self) -> Dataset:
        """Drop the typed wrapper and return the underlying ``Dataset``."""
        return Dataset(self.raster)

    def persist_metadata(self) -> None:
        """Write ``routing`` to the underlying raster's metadata tags."""
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
        }

    @classmethod
    def open(cls, path: str, *, routing: str | None = None) -> Accumulation:
        """Open an ``Accumulation`` GeoTIFF.

        Resolution order: explicit ``routing=`` > ``DR_ROUTING`` tag > raise.

        Raises:
            ValueError: If neither ``routing=`` nor a ``DR_ROUTING`` tag is
                available.
        """
        ds = Dataset.read_file(path)
        md = ds.meta_data or {}
        resolved_routing = routing or md.get(META_ROUTING)
        if resolved_routing is None:
            raise ValueError(
                f"{path!r} carries no DR_ROUTING tag and no routing= was passed. "
                f"Pass routing= explicitly (one of {sorted(VALID_ROUTING)})."
            )
        return cls(ds.raster, routing=resolved_routing)

    def streams(self, threshold: float | int) -> "StreamRaster":  # noqa: F821
        """Extract a typed ``StreamRaster`` from this accumulation surface.

        Args:
            threshold: Minimum upstream count for a cell to be classified as
                a stream cell.

        Raises:
            NotImplementedError: Implementation lands in P8.
        """
        raise NotImplementedError("Accumulation.streams is implemented in P8.")

    def __repr__(self) -> str:
        return (
            f"<Accumulation rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r}>"
        )
