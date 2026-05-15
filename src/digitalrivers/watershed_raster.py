"""Typed watershed raster (P13).

Subclass of :class:`pyramids.dataset.Dataset` that tags a watershed-label raster
with its basin count and the GeoDataFrame of outlet points. Carries the
producing FlowDirection's routing tag for provenance.
"""
from __future__ import annotations

from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers._metadata import (
    META_CLASS,
    META_ROUTING,
    VALID_ROUTING,
)


class WatershedRaster(Dataset):
    """Labelled-basins raster from :class:`FlowDirection.watershed`.

    Args:
        src: GDAL dataset wrapping the int32 basin-ID raster (0 = no basin).
        access: ``"read_only"`` (default) or ``"write"``.
        routing: Routing scheme of the source FlowDirection. Required
            keyword-only.
        outlets: ``GeoDataFrame`` with one row per pour point used to build the
            raster; required keyword-only.

    Attributes:
        routing: Routing scheme tag.
        outlets: ``GeoDataFrame`` of pour points and snap diagnostics.
        basin_count: Number of distinct basin labels (excluding background 0).
    """

    routing: str

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        routing: str,
        outlets,
    ):
        import numpy as np

        super().__init__(src, access)
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        self.routing = routing
        self.outlets = outlets
        arr = self.read_array()
        unique = np.unique(arr)
        self.basin_count = int((unique != 0).sum())

    @classmethod
    def from_dataset(cls, ds: Dataset, *, routing: str, outlets) -> "WatershedRaster":
        """Promote a plain ``Dataset`` into a ``WatershedRaster``."""
        return cls(ds.raster, routing=routing, outlets=outlets)

    def persist_metadata(self) -> None:
        """Persist the routing and class tags to the raster metadata."""
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
        }

    def to_polygons(self):
        """Vectorise the labelled raster to per-basin polygons.

        Each unique non-zero basin label becomes a single polygon (or
        MultiPolygon if the basin is disconnected). The output GeoDataFrame
        carries the basin ID in the ``basin_id`` column.

        Returns:
            ``geopandas.GeoDataFrame`` with columns ``basin_id`` (int) and
            ``geometry`` (Polygon / MultiPolygon).
        """
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        arr = self.read_array().astype(np.int32, copy=False)
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = arr.shape
        unique_ids = sorted({int(v) for v in np.unique(arr) if v != 0})
        records: list[dict] = []
        for bid in unique_ids:
            cells = np.argwhere(arr == bid)
            polygons: list[Polygon] = []
            for r, c in cells:
                # Build the four corners of this cell.
                x_left = x0 + c * dx
                x_right = x0 + (c + 1) * dx
                y_top = y0 + r * dy
                y_bot = y0 + (r + 1) * dy
                polygons.append(
                    Polygon(
                        [
                            (x_left, y_top),
                            (x_right, y_top),
                            (x_right, y_bot),
                            (x_left, y_bot),
                        ]
                    )
                )
            merged = unary_union(polygons)
            if not isinstance(merged, (Polygon, MultiPolygon)):
                merged = MultiPolygon([merged])
            records.append({"basin_id": bid, "geometry": merged})
        return gpd.GeoDataFrame(records, geometry="geometry", crs=self.epsg)

    def __repr__(self) -> str:
        return (
            f"<WatershedRaster rows={self.rows} cols={self.columns} "
            f"basin_count={self.basin_count} routing={self.routing!r}>"
        )
