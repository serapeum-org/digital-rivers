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

    def __repr__(self) -> str:
        return (
            f"<Accumulation rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r}>"
        )
