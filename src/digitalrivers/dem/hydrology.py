"""Stream-relative DEM products and the end-to-end pipeline.

Two things that need a DEM *and* a stream network, so they sit above conditioning and
routing rather than inside either:

* `hand` -- Height Above Nearest Drainage (Renno 2008 / Nobre 2011). Every cell is
  re-expressed as its vertical distance above the channel it drains to, which turns an
  elevation surface into a first-order flood-susceptibility surface. Two methods: the
  D8 walk down the flow path, and a Euclidean nearest-stream fallback for when no flow
  grid is available.
* `full_hydro_pipeline` -- fill, flow direction, accumulation and stream extraction in
  one call, for the common case where the caller wants the whole chain rather than the
  intermediates.
"""

from __future__ import annotations

import warnings
import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.flow.direction import FlowDirection
from digitalrivers.streams._kernels.hand import hand_d8

__all__ = ["HydrologyMixin"]


class HydrologyMixin:
    """HAND and the end-to-end pipeline, mixed into :class:`DEM`."""

    def hand(
        self,
        streams,
        flow_direction=None,
        *,
        method: str = "d8",
    ) -> Dataset:
        """Compute Height Above Nearest Drainage (Rennó 2008 / Nobre 2011).

        Two methods are supported:

        * **`"d8"` (default)** — follows the D8 / Rho8 flow-direction raster
          downstream from every cell until it reaches a stream cell, and
          assigns `elev[cell] - elev[stream_cell]`. Orphans / sinks / no-data
          cells whose flow path never reaches a stream are NaN.
        * **`"euclidean"`** — for every cell, the nearest stream cell in 2-D
          space (Euclidean distance) is used as the reference. Cheaper than
          D8-HAND because there is no path tracing, but it does the wrong
          thing across ridges (a cell can be 2-D-closer to a stream in a
          different basin). Requires `scipy.ndimage`.

        Args:
            streams: `StreamRaster` aligned to this DEM. Only the underlying
                stream mask is read.
            flow_direction: Single-direction `FlowDirection` (`d8` /
                `rho8`) aligned to this DEM. Required for `method="d8"`;
                ignored for `method="euclidean"`.
            method: `"d8"` (default) or `"euclidean"`.

        Returns:
            `Dataset` containing the float32 HAND raster. No-data cells use
            this DEM's no-data sentinel.

        Raises:
            ValueError: If `method` is unknown, shapes do not match, or
                `flow_direction` is missing / multi-direction for the D8
                method.
        """
        from digitalrivers.streams.raster import StreamRaster

        if method not in ("d8", "euclidean"):
            raise ValueError(f"method must be 'd8' or 'euclidean'; got {method!r}")
        if not isinstance(streams, StreamRaster):
            raise ValueError("streams must be a StreamRaster instance")

        if method == "d8":
            return self._hand_d8(streams, flow_direction)
        return self._hand_euclidean(streams)

    def _hand_d8(self, streams, flow_direction) -> Dataset:
        """D8-traced HAND — original Rennó-style implementation."""
        if not isinstance(flow_direction, FlowDirection):
            raise ValueError(
                "flow_direction must be a FlowDirection instance for method='d8'"
            )
        if flow_direction.routing not in ("d8", "rho8"):
            raise ValueError(
                f"hand currently supports single-direction routing only; "
                f"got {flow_direction.routing!r}"
            )

        elev = self.values
        fdir = flow_direction.read_array().astype(np.int32, copy=False)
        stream_arr = streams.read_array().astype(bool, copy=False)
        if not (elev.shape == fdir.shape == stream_arr.shape):
            raise ValueError(
                f"Shape mismatch: dem={elev.shape}, flow_direction="
                f"{fdir.shape}, streams={stream_arr.shape}"
            )

        hand_arr = hand_d8(elev, fdir, stream_arr).astype(np.float32, copy=False)
        no_val = float(self.no_data_value[0])
        hand_arr = np.where(np.isnan(hand_arr), no_val, hand_arr)
        return Dataset.from_array(
            hand_arr,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def _hand_euclidean(self, streams) -> Dataset:
        """Euclidean-nearest-stream HAND — no flow direction required."""
        from scipy.ndimage import distance_transform_edt

        elev = self.values
        stream_arr = streams.read_array().astype(bool, copy=False)
        if elev.shape != stream_arr.shape:
            raise ValueError(
                f"Shape mismatch: dem={elev.shape}, streams={stream_arr.shape}"
            )
        # Drop stream cells that sit on no-data DEM positions. If we don't,
        # `distance_transform_edt` will happily point at them and the
        # resulting per-cell elevation lookup returns NaN — silently
        # corrupting HAND over a swath around the bad pixel.
        valid_elev = ~np.isnan(elev)
        dropped = int((stream_arr & ~valid_elev).sum())
        if dropped:
            warnings.warn(
                f"{dropped} stream cell(s) coincide with DEM no-data; "
                f"dropping them before the Euclidean nearest-stream lookup.",
                UserWarning,
                stacklevel=3,
            )
            stream_arr = stream_arr & valid_elev
        if not stream_arr.any():
            raise ValueError(
                "streams raster contains no stream cells with valid elevation "
                "— HAND is undefined"
            )
        # distance_transform_edt with return_indices returns (distance, indices),
        # where indices[*, r, c] is the (row, col) of the nearest True (stream)
        # cell from (r, c). We want the *complement* of stream_arr because
        # the EDT measures distance to the nearest False cell.
        _, (ri, ci) = distance_transform_edt(~stream_arr, return_indices=True)
        nearest_elev = elev[ri, ci]
        hand_arr = (elev - nearest_elev).astype(np.float32, copy=False)
        no_val = float(self.no_data_value[0])
        hand_arr = np.where(np.isnan(hand_arr), no_val, hand_arr)
        return Dataset.from_array(
            hand_arr,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
            no_data_value=no_val,
        )

    def full_hydro_pipeline(
        self,
        *,
        fill_method: str = "priority_flood",
        flow_method: str = "d8",
        stream_threshold_cells: int | None = None,
    ) -> dict:
        """Composite: fill → flow_direction → accumulate (→ optional streams).

        Convenience entry point that chains the four most common steps of a
        DEM-hydrology pre-processing pipeline. Equivalent to:

        ```python
        filled = dem.fill_depressions(method=fill_method)
        fdir = filled.flow_direction(method=flow_method)
        acc = fdir.accumulate()
        streams = acc.streams(threshold=stream_threshold_cells)  # if provided
        ```

        Args:
            fill_method: Argument forwarded to `fill_depressions`. Defaults
                to `"priority_flood"`.
            flow_method: Argument forwarded to `flow_direction`. Defaults to
                `"d8"`.
            stream_threshold_cells: Optional accumulation threshold (in
                cells). When supplied, a `StreamRaster` is also returned in
                the result dict under the `"streams"` key. When None, the
                streams step is skipped.

        Returns:
            `dict` with keys `"filled_dem"` (DEM), `"flow_direction"`
            (FlowDirection), and `"accumulation"` (Accumulation); plus an
            optional `"streams"` (StreamRaster) when
            `stream_threshold_cells` is supplied.
        """
        filled = self.fill_depressions(method=fill_method)
        fdir = filled.flow_direction(method=flow_method)
        acc = fdir.accumulate()
        out: dict = {
            "filled_dem": filled,
            "flow_direction": fdir,
            "accumulation": acc,
        }
        if stream_threshold_cells is not None:
            out["streams"] = acc.streams(threshold=stream_threshold_cells)
        return out
