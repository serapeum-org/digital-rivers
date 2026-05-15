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

    def streams(
        self,
        threshold: float | int,
        units: str = "cells",
        slope_dem: "Dataset | None" = None,  # noqa: F821
        area_slope_exponent: float | None = None,
    ) -> "StreamRaster":  # noqa: F821
        """Extract a stream-network raster from this accumulation surface.

        A cell is a stream cell when its accumulation (or its slope-area
        support, if ``slope_dem`` and ``area_slope_exponent`` are supplied)
        meets or exceeds the threshold.

        Args:
            threshold: Minimum accumulation for stream classification. Units
                determined by the ``units`` kwarg.
            units: ``"cells"`` (default — direct comparison with the raster),
                ``"km2"``, or ``"m2"``. Area units are converted to cell
                counts using the dataset's square cell size.
            slope_dem: Slope raster (m/m) for the Montgomery & Foufoula-
                Georgiou (1993) area-slope criterion. When supplied alongside
                ``area_slope_exponent``, the threshold is applied to
                ``acc * slope ** area_slope_exponent`` instead of ``acc``.
            area_slope_exponent: Theta in the area-slope formula
                ``A * S^theta >= k``. Typical value ≈ 2.

        Returns:
            StreamRaster carrying ``threshold`` (in cells) and this
            Accumulation's ``routing`` tag. The underlying raster is ``uint8``
            with ``1`` at stream cells and ``0`` at non-stream cells; the
            input's no-data positions are propagated.

        Raises:
            ValueError: If ``units`` is not recognised, or if only one of
                ``slope_dem`` / ``area_slope_exponent`` is supplied.
        """
        import numpy as np

        from digitalrivers.stream_raster import StreamRaster

        if units not in ("cells", "km2", "m2"):
            raise ValueError(
                f"units must be 'cells', 'km2', or 'm2'; got {units!r}"
            )
        if (slope_dem is None) != (area_slope_exponent is None):
            raise ValueError(
                "slope_dem and area_slope_exponent must both be supplied "
                "or both omitted"
            )

        if units == "cells":
            cells_threshold = float(threshold)
        else:
            gt = self.geotransform
            cell_area_m2 = abs(gt[1] * gt[5])
            if cell_area_m2 == 0:
                raise ValueError(
                    "Cannot convert area threshold: dataset has zero cell size"
                )
            if units == "km2":
                cells_threshold = float(threshold) * 1.0e6 / cell_area_m2
            else:  # m2
                cells_threshold = float(threshold) / cell_area_m2

        acc_arr = self.read_array().astype(np.float64, copy=False)
        finite = np.isfinite(acc_arr)
        no_val = self.no_data_value[0] if self.no_data_value else None
        if no_val is not None:
            valid = finite & (acc_arr != no_val)
        else:
            valid = finite

        if slope_dem is not None:
            slope_arr = slope_dem.read_array().astype(np.float64, copy=False)
            if slope_arr.shape != acc_arr.shape:
                raise ValueError(
                    f"slope_dem shape {slope_arr.shape} does not match "
                    f"accumulation shape {acc_arr.shape}"
                )
            support = acc_arr * np.power(np.maximum(slope_arr, 0.0),
                                         area_slope_exponent)
            mask = valid & (support >= cells_threshold)
        else:
            mask = valid & (acc_arr >= cells_threshold)

        stream_mask = mask.astype(np.uint8, copy=False)
        plain = Dataset.create_from_array(
            stream_mask,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=0,
        )
        return StreamRaster.from_dataset(
            plain, threshold=cells_threshold, routing=self.routing
        )

    def snap_pour_points(
        self,
        points,
        radius_cells: int | None = None,
        radius_m: float | None = None,
        method: str = "max_accumulation",
        streams=None,
        min_acc: float | None = None,
        report: bool = False,
    ):
        """Snap pour-point geometries to nearby high-accumulation / stream cells.

        For each input point, scan a square neighbourhood of the given radius
        and pick the cell that wins under the chosen ``method``:

        * ``"max_accumulation"`` (ArcGIS-style): the cell with the largest
          accumulation in the neighbourhood. First-seen-wins on ties.
        * ``"jenson"`` (Jenson & Domingue 1988): the nearest stream cell in
          the neighbourhood (squared Euclidean against cell centres). Requires
          a ``StreamRaster``.

        Args:
            points: ``GeoDataFrame`` of Point geometries in any CRS (will be
                reprojected to the dataset's CRS).
            radius_cells: Search-window radius in cells. Exactly one of
                ``radius_cells`` / ``radius_m`` must be supplied.
            radius_m: Search-window radius in map units (typically metres).
            method: ``"max_accumulation"`` (default) or ``"jenson"``.
            streams: ``StreamRaster`` required when ``method="jenson"``.
            min_acc: Optional floor on accepted snap-target accumulation.
                Candidates with ``acc < min_acc`` are excluded; if no candidate
                qualifies the point is left at its original location.
            report: Reserved for future per-point diagnostics.

        Returns:
            ``GeoDataFrame`` with the input columns plus ``pre_snap_geometry``
            (the original geometry), ``snapped_x``, ``snapped_y``,
            ``snap_distance_m`` (Euclidean distance moved, NaN if unmoved),
            and ``snap_acc`` (the accumulation at the snapped cell). The
            ``geometry`` column is updated to the snapped points.

        Raises:
            ValueError: If neither or both of ``radius_cells`` / ``radius_m``
                are supplied, or if ``method="jenson"`` and ``streams`` is
                ``None``, or if ``method`` is unknown.
        """
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point

        if method not in ("max_accumulation", "jenson"):
            raise ValueError(
                f"method must be 'max_accumulation' or 'jenson'; got {method!r}"
            )
        if (radius_cells is None) == (radius_m is None):
            raise ValueError(
                "Exactly one of radius_cells / radius_m must be supplied"
            )
        if method == "jenson" and streams is None:
            raise ValueError("method='jenson' requires the streams= argument")

        gt = self.geotransform
        cell_x = abs(gt[1])
        if radius_cells is None:
            r = max(int(round(radius_m / cell_x)), 0)
        else:
            r = int(radius_cells)

        acc_arr = self.read_array().astype(np.float64, copy=False)
        no_val = self.no_data_value[0] if self.no_data_value else None
        if streams is not None:
            stream_arr = streams.read_array().astype(bool, copy=False)
            if stream_arr.shape != acc_arr.shape:
                raise ValueError(
                    f"streams shape {stream_arr.shape} != accumulation shape "
                    f"{acc_arr.shape}"
                )
        else:
            stream_arr = None

        # Reproject input to dataset CRS if mismatched.
        target_epsg = self.epsg
        if (
            getattr(points, "crs", None) is not None
            and target_epsg is not None
            and points.crs.to_epsg() != target_epsg
        ):
            points = points.to_crs(target_epsg)

        rows, cols = acc_arr.shape
        x0 = gt[0]
        y0 = gt[3]
        dx = gt[1]
        dy = gt[5]

        pre_geom = list(points.geometry)
        snapped_xs: list[float] = []
        snapped_ys: list[float] = []
        snap_distances: list[float] = []
        snap_accs: list[float] = []

        for pt in pre_geom:
            px, py = float(pt.x), float(pt.y)
            col0 = int((px - x0) / dx)
            row0 = int((py - y0) / dy)
            if not (0 <= row0 < rows and 0 <= col0 < cols):
                snapped_xs.append(px)
                snapped_ys.append(py)
                snap_distances.append(np.nan)
                snap_accs.append(np.nan)
                continue

            r_lo = max(0, row0 - r)
            r_hi = min(rows - 1, row0 + r)
            c_lo = max(0, col0 - r)
            c_hi = min(cols - 1, col0 + r)

            best_r, best_c = row0, col0
            if method == "max_accumulation":
                best_acc = -np.inf
                for rr in range(r_lo, r_hi + 1):
                    for cc in range(c_lo, c_hi + 1):
                        v = acc_arr[rr, cc]
                        if no_val is not None and v == no_val:
                            continue
                        if min_acc is not None and v < min_acc:
                            continue
                        if v > best_acc:
                            best_acc = v
                            best_r, best_c = rr, cc
                if not np.isfinite(best_acc):
                    best_r, best_c = row0, col0
            else:  # jenson
                best_d2 = np.inf
                for rr in range(r_lo, r_hi + 1):
                    for cc in range(c_lo, c_hi + 1):
                        if not stream_arr[rr, cc]:
                            continue
                        if min_acc is not None and acc_arr[rr, cc] < min_acc:
                            continue
                        cx = x0 + (cc + 0.5) * dx
                        cy = y0 + (rr + 0.5) * dy
                        d2 = (cx - px) ** 2 + (cy - py) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_r, best_c = rr, cc
                if not np.isfinite(best_d2):
                    best_r, best_c = row0, col0

            snapped_x = x0 + (best_c + 0.5) * dx
            snapped_y = y0 + (best_r + 0.5) * dy
            distance = float(np.hypot(snapped_x - px, snapped_y - py))
            snapped_xs.append(snapped_x)
            snapped_ys.append(snapped_y)
            snap_distances.append(distance if distance > 0 else 0.0)
            snap_accs.append(float(acc_arr[best_r, best_c]))

        out = points.copy()
        out["pre_snap_geometry"] = pre_geom
        out["snapped_x"] = snapped_xs
        out["snapped_y"] = snapped_ys
        out["snap_distance_m"] = snap_distances
        out["snap_acc"] = snap_accs
        out["geometry"] = [Point(x, y) for x, y in zip(snapped_xs, snapped_ys)]
        return out

    def __repr__(self) -> str:
        return (
            f"<Accumulation rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r}>"
        )
