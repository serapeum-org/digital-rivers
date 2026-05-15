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

    def statistics(
        self,
        dem=None,
        accumulation=None,
        slope=None,
        streams=None,
        metrics: list[str] | None = None,
    ):
        """Per-basin descriptor table.

        Returns one row per basin label with the requested metrics. Available
        metrics (subset of P17 spec):

        - ``area_km2``: number of cells × cell area (km²).
        - ``min_elev``, ``max_elev``, ``mean_elev``, ``std_elev``: elevation
          statistics from ``dem`` (required for the elev metrics).
        - ``hypsometric_integral``: Strahler (1952)
          ``(mean_elev - min_elev) / (max_elev - min_elev)``.
        - ``mean_slope``: mean of the ``slope`` raster across the basin.
        - ``drainage_density_km_per_km2``: ``stream_length_km / area_km2``
          (requires both ``streams`` and a cell-size; stream length uses cell
          centres at the cell size).
        - ``centroid_x``, ``centroid_y``: basin centroid in dataset CRS.

        Args:
            dem: Aligned DEM for elevation metrics.
            accumulation: Reserved for future longest-flow-path metric.
            slope: Aligned slope raster (m/m) for ``mean_slope``.
            streams: Aligned StreamRaster for ``drainage_density_km_per_km2``.
            metrics: Subset of the available metrics. ``None`` (default)
                returns every metric for which inputs were supplied.

        Returns:
            ``pandas.DataFrame`` indexed by basin_id with one column per metric.
        """
        import numpy as np
        import pandas as pd

        gt = self.geotransform
        cell_area_m2 = abs(gt[1] * gt[5])
        labels = self.read_array().astype(np.int32, copy=False)
        unique_ids = sorted({int(v) for v in np.unique(labels) if v != 0})

        available: dict[str, list] = {"basin_id": [], "area_km2": []}
        for bid in unique_ids:
            mask = labels == bid
            available["basin_id"].append(bid)
            available["area_km2"].append(int(mask.sum()) * cell_area_m2 / 1.0e6)

        if dem is not None:
            elev = dem.read_array().astype(np.float64, copy=False)
            no_val = dem.no_data_value[0] if dem.no_data_value else None
            if no_val is not None:
                elev = np.where(elev == no_val, np.nan, elev)
            for col in ("min_elev", "max_elev", "mean_elev", "std_elev",
                        "hypsometric_integral",
                        "centroid_x", "centroid_y"):
                available[col] = []
            x0, dx, _, y0, _, dy = gt
            for bid in unique_ids:
                mask = labels == bid
                vals = elev[mask]
                vals_finite = vals[np.isfinite(vals)]
                if vals_finite.size == 0:
                    available["min_elev"].append(np.nan)
                    available["max_elev"].append(np.nan)
                    available["mean_elev"].append(np.nan)
                    available["std_elev"].append(np.nan)
                    available["hypsometric_integral"].append(np.nan)
                else:
                    mn = float(vals_finite.min())
                    mx = float(vals_finite.max())
                    mean = float(vals_finite.mean())
                    std = float(vals_finite.std())
                    available["min_elev"].append(mn)
                    available["max_elev"].append(mx)
                    available["mean_elev"].append(mean)
                    available["std_elev"].append(std)
                    hi = (mean - mn) / (mx - mn) if mx > mn else 0.0
                    available["hypsometric_integral"].append(hi)
                rs, cs = np.where(mask)
                cx = x0 + (cs.mean() + 0.5) * dx
                cy = y0 + (rs.mean() + 0.5) * dy
                available["centroid_x"].append(float(cx))
                available["centroid_y"].append(float(cy))

        if slope is not None:
            slope_arr = slope.read_array().astype(np.float64, copy=False)
            no_val = slope.no_data_value[0] if slope.no_data_value else None
            if no_val is not None:
                slope_arr = np.where(slope_arr == no_val, np.nan, slope_arr)
            available["mean_slope"] = []
            for bid in unique_ids:
                mask = labels == bid
                vals = slope_arr[mask]
                vals = vals[np.isfinite(vals)]
                available["mean_slope"].append(
                    float(vals.mean()) if vals.size else np.nan
                )

        if streams is not None:
            sm = streams.read_array().astype(bool, copy=False)
            available["drainage_density_km_per_km2"] = []
            for bid in unique_ids:
                mask = labels == bid
                stream_count = int((sm & mask).sum())
                length_km = stream_count * abs(gt[1]) / 1000.0
                area_km2 = int(mask.sum()) * cell_area_m2 / 1.0e6
                if area_km2 > 0:
                    available["drainage_density_km_per_km2"].append(
                        length_km / area_km2
                    )
                else:
                    available["drainage_density_km_per_km2"].append(np.nan)

        df = pd.DataFrame(available).set_index("basin_id")
        if metrics is not None:
            wanted = [m for m in metrics if m in df.columns]
            df = df[wanted]
        return df

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
