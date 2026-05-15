"""Typed flow-direction raster carrying routing-scheme metadata.

The ``FlowDirection`` class is a thin subclass of ``pyramids.dataset.Dataset``
that tags the wrapped raster with the routing scheme (``d8`` / ``dinf`` /
``mfd_quinn`` / ``mfd_holmgren`` / ``rho8``) and the cell-value encoding
convention. The ``routing`` argument is required at construction; there is no
default. That is the safety property: it prevents a flow-direction raster of
unknown provenance from being silently reinterpreted as D8 by a downstream
consumer.
"""
from __future__ import annotations

from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers._metadata import (
    META_CLASS,
    META_ENCODING,
    META_ROUTING,
    VALID_ENCODING,
    VALID_ROUTING,
)


class FlowDirection(Dataset):
    """Flow-direction raster with routing-scheme metadata.

    Args:
        src: GDAL dataset wrapping a flow-direction raster.
        access: ``"read_only"`` (default) or ``"write"``.
        routing: Routing scheme used to produce this raster. One of
            ``"d8"``, ``"dinf"``, ``"mfd_quinn"``, ``"mfd_holmgren"``,
            ``"rho8"``. Required keyword-only argument — no default.
        encoding: Cell-value encoding convention. One of ``"digitalrivers"``,
            ``"taudem"``, ``"esri"``, ``"whitebox"``. Defaults to
            ``"digitalrivers"`` (the convention defined by ``DIR_OFFSETS`` in
            ``dem.py``).

    Raises:
        ValueError: If ``routing`` or ``encoding`` is not a recognised value.
    """

    routing: str
    encoding: str

    def __init__(
        self,
        src: gdal.Dataset,
        access: str = "read_only",
        *,
        routing: str,
        encoding: str = "digitalrivers",
    ):
        super().__init__(src, access)
        if routing not in VALID_ROUTING:
            raise ValueError(
                f"routing must be one of {sorted(VALID_ROUTING)}; got {routing!r}"
            )
        if encoding not in VALID_ENCODING:
            raise ValueError(
                f"encoding must be one of {sorted(VALID_ENCODING)}; got {encoding!r}"
            )
        self.routing = routing
        self.encoding = encoding

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *,
        routing: str,
        encoding: str = "digitalrivers",
    ) -> FlowDirection:
        """Promote a plain ``Dataset`` into a ``FlowDirection``.

        Args:
            ds: Dataset wrapping the flow-direction raster.
            routing: Routing scheme. Required keyword-only.
            encoding: Cell-value encoding convention.

        Returns:
            A ``FlowDirection`` sharing the same underlying GDAL dataset.
        """
        return cls(ds.raster, routing=routing, encoding=encoding)

    def to_dataset(self) -> Dataset:
        """Drop the typed wrapper and return the underlying ``Dataset``."""
        return Dataset(self.raster)

    def persist_metadata(self) -> None:
        """Write ``routing`` and ``encoding`` to the underlying raster tags.

        Stored under ``DR_CLASS`` / ``DR_ROUTING`` / ``DR_ENCODING`` GeoTIFF
        metadata keys so ``FlowDirection.open(path)`` can recover them.
        """
        self.meta_data = {
            META_CLASS: type(self).__name__,
            META_ROUTING: self.routing,
            META_ENCODING: self.encoding,
        }

    @classmethod
    def open(
        cls,
        path: str,
        *,
        routing: str | None = None,
        encoding: str | None = None,
    ) -> FlowDirection:
        """Open a ``FlowDirection`` GeoTIFF.

        Resolution order for the routing/encoding tags:

        1. Explicit ``routing=`` / ``encoding=`` kwargs win unconditionally
           (caller knows what the file is).
        2. Otherwise, ``DR_ROUTING`` / ``DR_ENCODING`` metadata tags are used
           if present.
        3. Otherwise, raise ``ValueError``. There is no silent fallback to
           ``"d8"`` — a D∞ raster on disk is float32 in ``[0, 2π]`` and
           reinterpreting it as int D8 codes silently corrupts every
           downstream computation.

        Args:
            path: Path to the GeoTIFF.
            routing: Explicit routing override. If ``None``, falls back to
                the ``DR_ROUTING`` tag.
            encoding: Explicit encoding override. If ``None``, falls back to
                the ``DR_ENCODING`` tag, then to ``"digitalrivers"``.

        Returns:
            A ``FlowDirection`` wrapping the opened raster.

        Raises:
            ValueError: If neither ``routing=`` nor a ``DR_ROUTING`` tag is
                available.
        """
        ds = Dataset.read_file(path)
        md = ds.meta_data or {}
        resolved_routing = routing or md.get(META_ROUTING)
        resolved_encoding = encoding or md.get(META_ENCODING) or "digitalrivers"
        if resolved_routing is None:
            raise ValueError(
                f"{path!r} carries no DR_ROUTING tag and no routing= was passed. "
                f"Pass routing= explicitly (one of {sorted(VALID_ROUTING)}) to "
                f"avoid silent misinterpretation of cell values."
            )
        return cls(ds.raster, routing=resolved_routing, encoding=resolved_encoding)

    def accumulate(self, weights: Dataset | None = None) -> "Accumulation":  # noqa: F821
        """Run flow accumulation over this raster's routing scheme.

        Implements a Kahn topological-sort sweep that handles all five routing
        schemes (D8, Rho8, D∞, MFD-Quinn, MFD-Holmgren) via a single algorithm,
        dispatched by ``self.routing``.

        Output semantics: ``out[cell] = sum of weights over strictly-upstream
        cells`` — the cell's own weight does not contribute to its own count.
        This matches the legacy ``DEM.flow_accumulation`` convention.

        Args:
            weights: Per-cell weight raster (rainfall, runoff coefficient,
                whatever). Must align with this FlowDirection's shape. ``None``
                means unit weights (cell-count accumulation).

        Returns:
            Accumulation carrying this object's ``routing`` for provenance.
        """
        import numpy as np

        from digitalrivers._accumulation import accumulate as _accumulate_array
        from digitalrivers.accumulation import Accumulation

        fd_arr = self.read_array()
        valid_mask = self._valid_mask_from_array(fd_arr)
        if weights is not None:
            w_arr = weights.read_array()
            if w_arr.shape != valid_mask.shape:
                raise ValueError(
                    f"weights shape {w_arr.shape} does not match flow_direction "
                    f"shape {valid_mask.shape}"
                )
        else:
            w_arr = None
        acc = _accumulate_array(fd_arr, self.routing, valid_mask, weights=w_arr)
        acc_f32 = acc.astype(np.float32, copy=False)
        plain = Dataset.create_from_array(
            acc_f32,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=self.default_no_data_value,
        )
        return Accumulation.from_dataset(plain, routing=self.routing)

    def _valid_mask_from_array(self, arr) -> "np.ndarray":  # noqa: F821
        """Compute the (rows, cols) bool mask of valid-data cells from the raster.

        For accumulation purposes ``valid`` means "this cell can hold and receive a
        contribution". For D8/Rho8 we cannot distinguish a sink (cell with no
        outgoing direction but still in the data envelope) from a truly-outside
        cell at the flow-direction level — both share the no-data sentinel. We
        treat all in-bounds cells as valid; truly-outside cells naturally end up
        with accumulation 0 because no valid direction points at them, and
        callers that want to mask them in the output do so against the original
        DEM (this is what ``DEM.flow_accumulation`` does).

        Multi-band MFD/D∞ rasters use band 0 as the routing-specific validity
        indicator (angle ``>= 0`` for D∞, any non-zero fraction for MFD).
        """
        import numpy as np

        if arr.ndim == 2:
            # D8 / Rho8: treat every in-bounds cell as a valid receiver. Sinks
            # (direction == no_data) are kept in the graph so they accumulate.
            return np.ones(arr.shape, dtype=bool)
        # Multi-band routings.
        band0 = arr[0]
        if self.routing == "dinf":
            return band0 >= 0
        no_val = self.no_data_value[0] if self.no_data_value else None
        if no_val is None:
            return np.ones(band0.shape, dtype=bool)
        return band0 != no_val

    def __repr__(self) -> str:
        return (
            f"<FlowDirection rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r} encoding={self.encoding!r}>"
        )
