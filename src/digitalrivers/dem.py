"""DEM processing module.

This module provides the `DEM` class for digital elevation model analysis,
including depression filling, slope calculation, D8 flow direction, and flow
accumulation.
"""
from __future__ import annotations

import warnings

import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset

from digitalrivers._conditioning.breach import breach_depressions as _breach_depressions_array
from digitalrivers._conditioning.flats import resolve_flats as _resolve_flats_array
from digitalrivers._flow.routing import (
    dinf_flow_direction as _dinf_flow_direction,
    mfd_flow_direction as _mfd_flow_direction,
    rho8_flow_direction as _rho8_flow_direction,
)
from digitalrivers._conditioning.pitremoval import fill_depressions as _fill_depressions_array
from digitalrivers.flow_direction import FlowDirection

#: D8 direction offsets mapping direction index to (column_offset, row_offset).
#:
#: Directions follow the convention:
#:   0 = South (bottom), 1 = Southwest (bottom-left), 2 = West (left),
#:   3 = Northwest (top-left), 4 = North (top), 5 = Northeast (top-right),
#:   6 = East (right), 7 = Southeast (bottom-right).
DIR_OFFSETS = {
    0: (0, 1),  # bottom
    1: (-1, 1),  # bottom left
    2: (-1, 0),  # left
    3: (-1, -1),  # top left
    4: (0, -1),  # top
    5: (1, -1),  # top right
    6: (1, 0),  # right
    7: (1, 1),  # bottom right
}


def _reproject_if_needed(layer, target_epsg: int | None):
    """Return `layer` reprojected to `target_epsg` when its CRS differs.

    Uses CRS object equality rather than `to_epsg()` integer comparison so
    custom projections (where `to_epsg()` returns `None`) are not falsely
    flagged as mismatched — see the Phase-3 N7 review note.

    Args:
        layer: `GeoDataFrame` (or any object with a `crs` attribute and a
            `to_crs(epsg)` method). When the attribute is missing or its
            value is `None` the layer is returned unchanged.
        target_epsg: EPSG code of the destination CRS, or `None` to skip
            reprojection entirely.

    Returns:
        Either the original `layer` (when CRSes already match, or when
        `target_epsg` is `None`, or when `layer` carries no CRS) or a
        new `GeoDataFrame` reprojected to `target_epsg`.

    Examples:
        - Same CRS short-circuit returns the original layer unchanged:

            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalrivers.dem import _reproject_if_needed
            >>> layer = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
            >>> _reproject_if_needed(layer, 4326) is layer
            True

        - Different CRS triggers an actual reprojection:

            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalrivers.dem import _reproject_if_needed
            >>> layer = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
            >>> reprojected = _reproject_if_needed(layer, 3857)
            >>> int(reprojected.crs.to_epsg())
            3857

        - `target_epsg=None` skips reprojection entirely:

            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalrivers.dem import _reproject_if_needed
            >>> layer = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=4326)
            >>> _reproject_if_needed(layer, None) is layer
            True

        - A layer with no `crs` attribute is returned untouched:

            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalrivers.dem import _reproject_if_needed
            >>> layer = gpd.GeoDataFrame(geometry=[Point(0, 0)])
            >>> _reproject_if_needed(layer, 4326) is layer
            True
    """
    if target_epsg is None:
        return layer
    layer_crs = getattr(layer, "crs", None)
    if layer_crs is None:
        return layer
    try:
        from pyproj import CRS

        target_crs = CRS.from_epsg(target_epsg)
        if layer_crs.equals(target_crs):
            return layer
    except Exception:
        # Fall back to the integer compare if pyproj/CRS isn't cooperating;
        # we'd rather do a no-op to_crs round-trip than crash.
        if layer_crs.to_epsg() == target_epsg:
            return layer
    return layer.to_crs(target_epsg)


class DEM(Dataset):
    """Digital Elevation Model processor.

    Wraps a GDAL raster dataset and adds hydrological analysis methods:
    sink filling, D8 flow direction, flow accumulation, and slope
    computation.

    Args:
        src: GDAL dataset containing a single-band elevation raster.
        access: `"read_only"` (default) or `"write"`.
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        super().__init__(src, access)

    @property
    def values(self):
        """Elevation array with no-data cells replaced by `np.nan`.

        Reads band 0 as `float32` and masks every cell whose value is
        close to the raster's no-data value (relative tolerance 1e-5).

        Returns:
            np.ndarray: 2-D `float32` array of shape `(rows, columns)`.
        """
        values = self.read_array(band=0).astype(np.float32)
        # get the value stores in no data value cells
        no_val = self.no_data_value[0]
        values[np.isclose(values, no_val, rtol=0.00001)] = np.nan
        return values

    def fill_depressions(
        self,
        method: str = "priority_flood",
        epsilon: float = 0.0,
        inplace: bool = False,
    ) -> DEM | None:
        """Fill closed depressions in the DEM.

        Three algorithms are available via the `method` argument:

        * `"priority_flood"` (default) — Barnes, Lehman & Mulla (2014) Priority-Flood
          with the two-queue plateau optimisation. With `epsilon == 0` it produces flat
          fills; with `epsilon > 0` it produces a strictly monotonic surface (every cell
          has at least one strictly lower neighbour along the flood path) at the cost of
          a small elevation inflation proportional to plateau width.
        * `"wang_liu"` — Wang & Liu (2006). Flat fill, no epsilon. Equivalent in output
          to `priority_flood` with `epsilon == 0`; kept as a named alternative for
          callers who plan to resolve flats explicitly afterwards (P4).
        * `"planchon_darboux"` — Planchon & Darboux (2002). Iterative directional-sweep
          algorithm. Slower than Priority-Flood on large DEMs; kept as a low-relief
          reference. Requires `epsilon > 0`.

        No-data handling is uniform across methods: cells flagged no-data act as outlets
        (they cannot be filled, and data cells adjacent to them are seeded as drainage
        sources alongside the true raster boundary).

        **Precision note.** Priority-flood / planchon-darboux compute the cumulative lift
        in float64 but the output is cast back to the input dtype. For `float32` DEMs
        with `epsilon` in the `0.1`-class on wide plateaus, the accumulated lift can
        approach float32's relative precision near the spill elevation and very long
        plateaus may underflow to identical filled values. Prefer `float64` inputs
        when running with `epsilon > 0` and large depressions; `wang_liu` /
        `epsilon=0` are immune.

        Args:
            method: One of `"priority_flood"`, `"wang_liu"`, `"planchon_darboux"`.
            epsilon: Per-step elevation lift inside depressions. `0.0` (default for
                `priority_flood`) returns a non-strictly-decreasing surface — flats
                remain flat. Positive values guarantee a unique downhill path at the
                cost of slight elevation inflation. `planchon_darboux` requires
                `epsilon > 0`.
            inplace: If `True` the current instance is updated in place and `None`
                is returned. If `False` (default) a new `DEM` is returned.

        Returns:
            DEM | None: A new `DEM` containing the filled elevation, or `None` when
            `inplace` is `True`.

        Raises:
            ValueError: If `method` is unknown, or `planchon_darboux` is requested
                with `epsilon <= 0`.
        """
        elev = self.values
        nodata_mask = np.isnan(elev)
        z_fill = _fill_depressions_array(
            elev.astype(np.float64, copy=False),
            nodata_mask=nodata_mask,
            method=method,
            epsilon=epsilon,
        )
        # Restore the original raster's no-data sentinel (the array carries NaN; the
        # GeoTIFF needs the numeric sentinel).
        no_val = self.no_data_value[0]
        z_fill[nodata_mask] = no_val

        # Build a plain Dataset (cls=Dataset so we don't get a DEM via cls(...)), then
        # wrap with the typed DEM. This mirrors the pattern used in flow_direction().
        plain_ds = Dataset.dataset_like(self, z_fill.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def breach_depressions(
        self,
        method: str = "least_cost",
        max_depth: float | None = None,
        max_length: int | None = None,
        fill_remaining: bool = True,
        inplace: bool = False,
    ) -> DEM | None:
        """Breach depressions in the DEM (Lindsay 2016 family).

        Breaching is the structural alternative to filling: instead of raising the pit
        floor, it cuts a channel through the lowest barrier between the pit and an
        outlet. On LiDAR DEMs this is usually more realistic — most internal pits are
        data artefacts and the natural drainage path is preserved by cutting the artefact
        away rather than inflating the surrounding terrain.

        Three methods are available via the `method` argument:

        * `"single_cell"` — cheap O(n) preprocessing pass that resolves isolated 1-cell
          pits by lowering an intermediate first-order neighbour to the midpoint of the
          pit and a lower second-order cell. Does nothing if no such configuration exists.
        * `"least_cost"` (default) — Lindsay 2016 Dijkstra-from-each-pit. Carves a
          strictly monotonic channel from the pit to the nearest outlet. Optional
          `max_depth` and `max_length` constraints abort the breach for any pit whose
          channel would exceed them; aborted pits are left unresolved.
        * `"hybrid"` — try `least_cost` first; pits that fail their constraint fall
          back to the Priority-Flood depression fill (P2). The breach phase has already
          lowered parts of the DEM where partial breaching occurred, so the fill operates
          on a modified surface and produces less overall lift than fill-only.

        No-data cells act as free outlets — any Dijkstra path that reaches a no-data cell
        terminates the search.

        Args:
            method: One of `"single_cell"`, `"least_cost"`, `"hybrid"`.
            max_depth: Maximum cumulative `|Δz|` for a single breach path. `None`
                disables the constraint.
            max_length: Maximum path length in cells. `None` disables.
            fill_remaining: Only meaningful when `method="hybrid"`. If `True`
                (default), unresolved pits are passed to Priority-Flood with
                `epsilon=0`. If `False`, they are left as pits in the output.
            inplace: If `True` the current instance is updated in place and `None` is
                returned. If `False` (default) a new `DEM` is returned.

        Returns:
            DEM | None: A new `DEM` containing the breached elevation, or `None` when
            `inplace` is `True`.

        Raises:
            ValueError: If `method` is unknown.
        """
        elev = self.values
        nodata_mask = np.isnan(elev)
        z_out = _breach_depressions_array(
            elev.astype(np.float64, copy=False),
            nodata_mask=nodata_mask,
            method=method,
            max_depth=max_depth,
            max_length=max_length,
            fill_remaining=fill_remaining,
        )
        no_val = self.no_data_value[0]
        z_out[np.isnan(z_out)] = no_val
        plain_ds = Dataset.dataset_like(self, z_out.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def resolve_flats(
        self,
        max_iter: int = 1000,
        epsilon: float = 1e-5,
        connectivity: int = 8,
        inplace: bool = False,
    ) -> DEM | None:
        """Impose a deterministic gradient on every flat plateau in the DEM.

        After `fill_depressions(method="wang_liu")` (or `"priority_flood"` with
        `epsilon=0`), every closed depression is filled to its spill elevation — but the
        interior of each filled depression is a flat plateau with no defined steepest
        descent, so D8 flow direction over the result has `NO_FLOW` cells across every
        plateau. `resolve_flats` nudges those cells so each has a unique downhill
        neighbour: combined Garbrecht & Martz (1997) gradient — drain *towards* the
        nearest outlet (LEC) with a tiebreak that drains *away from* the nearest rim
        (HEC). The towards-lower gradient is weighted `2x` so it dominates and the
        away-from-higher gradient acts as a deterministic tiebreaker.

        Plateaus without a low-edge cell (closed depressions that survived the fill — they
        should not exist if you ran `fill_depressions` first) are left untouched.

        Args:
            max_iter: Safety cap on BFS levels per plateau. Real plateaus rarely exceed
                `max(rows, cols)`; the default `1000` is essentially unbounded.
            epsilon: Per-BFS-step elevation lift. Total lift over a plateau is at most
                `(2 * max_high_dist + max_low_dist) * epsilon`; choose small enough
                that this stays well below the minimum elevation step between adjacent
                non-plateau cells. Default `1e-5` is safe for ~1000-cell-wide plateaus.
            connectivity: 4 or 8. Controls plateau-labelling and BFS step direction;
                LEC/HEC classification always uses 8-connectivity (Garbrecht-Martz
                convention). Default is 8.
            inplace: If `True` the current instance is updated in place and `None` is
                returned. If `False` (default) a new `DEM` is returned.

        Returns:
            DEM | None: A new `DEM` with flat plateaus resolved, or `None` when
            `inplace` is `True`.

        Raises:
            ValueError: If `connectivity` is not 4 or 8.
        """
        elev = self.values
        nodata_mask = np.isnan(elev)
        z_out = _resolve_flats_array(
            elev.astype(np.float64, copy=False),
            nodata_mask=nodata_mask,
            epsilon=epsilon,
            connectivity=connectivity,
            max_iter=max_iter,
        )
        no_val = self.no_data_value[0]
        z_out[np.isnan(z_out)] = no_val
        plain_ds = Dataset.dataset_like(self, z_out.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

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
        from digitalrivers.stream_raster import StreamRaster

        if method not in ("d8", "euclidean"):
            raise ValueError(
                f"method must be 'd8' or 'euclidean'; got {method!r}"
            )
        if not isinstance(streams, StreamRaster):
            raise ValueError("streams must be a StreamRaster instance")

        if method == "d8":
            return self._hand_d8(streams, flow_direction)
        return self._hand_euclidean(streams)

    def _hand_d8(self, streams, flow_direction) -> Dataset:
        """D8-traced HAND — original Rennó-style implementation."""
        from digitalrivers._streams.hand import hand_d8
        from digitalrivers.flow_direction import FlowDirection

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
        return Dataset.create_from_array(
            hand_arr,
            geo=self.geotransform,
            epsg=self.epsg,
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
        if not stream_arr.any():
            raise ValueError(
                "streams raster contains no stream cells — HAND is undefined"
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
        return Dataset.create_from_array(
            hand_arr,
            geo=self.geotransform,
            epsg=self.epsg,
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

    def stochastic_depressions(
        self,
        sigma: float,
        n_runs: int = 100,
        *,
        seed: int | None = None,
        method: str = "priority_flood",
    ) -> Dataset:
        """Per-cell depression-occurrence probability via Monte-Carlo.

        Adds Gaussian noise (zero-mean, supplied `sigma`) to the DEM, runs
        depression detection on each noisy realisation, and aggregates the
        per-cell probability across `n_runs` realisations. Cells with high
        probability are robust depressions; low probability cells are likely
        noise artefacts of a noisy DEM.

        Args:
            sigma: Standard deviation of the Gaussian noise in DEM elevation
                units. Must be non-negative. A reasonable choice is the DEM's
                stated vertical error.
            n_runs: Number of Monte-Carlo realisations. Must be positive.
                Defaults to 100.
            seed: Optional seed for the random number generator. Pass an
                integer for reproducible results.
            method: Fill-depressions algorithm passed through to
                `fill_depressions` for each realisation. Defaults to
                `"priority_flood"`.

        Returns:
            `Dataset` of float32 occurrence probabilities in `[0.0, 1.0]`,
            aligned to this DEM. No-data sentinel `-1.0`.

        Raises:
            ValueError: If `sigma` is negative or `n_runs` is not positive.
        """
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative; got {sigma!r}")
        if n_runs <= 0:
            raise ValueError(f"n_runs must be positive; got {n_runs!r}")

        rng = np.random.default_rng(seed)
        elev = self.values
        no_val = float(self.no_data_value[0])
        prob = np.zeros(elev.shape, dtype=np.float32)
        for _ in range(int(n_runs)):
            noise = rng.normal(0.0, sigma, size=elev.shape).astype(
                elev.dtype, copy=False
            )
            noisy = elev + noise
            # Wrap the noisy elevation grid back in a DEM so we can reuse the
            # existing fill_depressions kernel and its method dispatcher.
            noisy_disk = np.where(np.isnan(noisy), no_val, noisy)
            noisy_ds = Dataset.create_from_array(
                noisy_disk,
                geo=self.geotransform,
                epsg=self.epsg,
                no_data_value=no_val,
            )
            noisy_dem = DEM(noisy_ds.raster)
            filled = noisy_dem.fill_depressions(method=method).values
            # A cell is part of a depression in this realisation iff the fill
            # lifted it above the noisy elevation.
            depr = (filled - noisy) > 0
            prob += depr.astype(np.float32)
        prob /= float(n_runs)
        return Dataset.create_from_array(
            prob,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=-1.0,
        )

    def burn_streams(
        self,
        streams,
        method: str = "fill_burn",
        *,
        sharp: float = 10.0,
        smooth: float = 2.0,
        buffer_cells: int = 5,
        constant_drop: float = 1.0,
        max_breach_depth: float | None = None,
        max_breach_length: int | None = None,
        inplace: bool = False,
    ) -> DEM | None:
        """Condition the DEM by burning a vector stream network into it.

        Three methods are specified by P20; this implementation ships
        `"fill_burn"` (Lindsay 2018 — used by WhiteboxTools' FillBurn) as
        the default. `"agree"` (Hellweger 1997) and
        `"topological_breach"` (Lindsay 2016) raise `NotImplementedError`.

        Fill-burn algorithm:

        1. Rasterise every LineString in `streams` onto a stream mask.
        2. Lower every stream cell's elevation by `constant_drop`.
        3. Run `fill_depressions(method="priority_flood")` so the
           surrounding cells drain naturally into the channel.

        Args:
            streams: `GeoDataFrame` of LineString geometries.
            method: `"fill_burn"` (default); `"agree"` and
                `"topological_breach"` raise `NotImplementedError`.
            sharp / smooth / buffer_cells: AGREE parameters (unused for
                fill_burn).
            constant_drop: Elevation drop applied to every stream cell
                in fill_burn (default 1.0 map unit).
            max_breach_depth / max_breach_length: topological_breach
                parameters (unused for fill_burn).
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: New DEM with the conditioned surface, or None when
            `inplace=True`.

        Examples:
            - Fill-burn lowers the stream-row of a flat DEM:

                >>> import numpy as np
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> # Horizontal stream down row 2 (y = -2.5).
                >>> line = LineString([(0.5, -2.5), (4.5, -2.5)])
                >>> layer = gpd.GeoDataFrame(geometry=[line], crs=4326)
                >>> burnt = dem.burn_streams(layer, constant_drop=2.0)
                >>> bool(float(burnt.values[2, 2]) < float(burnt.values[0, 0]))
                True
        """
        if method == "agree":
            # Rasterise the stream lines, then apply a gradient buffer:
            # stream cells drop by `sharp`, buffer cells drop linearly from
            # `sharp` at the stream to `0` at buffer_cells radius. The
            # cumulative drop is then offset by `smooth` so the buffer
            # perimeter sits `smooth` units lower than the original DEM.
            elev = self.values
            rows, cols = elev.shape
            gt = self.geotransform
            stream_mask = np.zeros((rows, cols), dtype=bool)
            streams = _reproject_if_needed(streams, self.epsg)
            for geom in streams.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "MultiLineString":
                    for sub in geom.geoms:
                        self._rasterise_line(sub, stream_mask, gt)
                else:
                    self._rasterise_line(geom, stream_mask, gt)
            # Distance-from-stream within the buffer (cell-step BFS).
            dist = np.full((rows, cols), np.inf, dtype=np.float64)
            dist[stream_mask] = 0.0
            from collections import deque
            frontier: deque[tuple[int, int, int]] = deque(
                (int(r), int(c), 0) for r, c in zip(*np.where(stream_mask))
            )
            while frontier:
                r, c, d = frontier.popleft()
                if d >= buffer_cells:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if dist[nr, nc] > d + 1:
                            dist[nr, nc] = d + 1
                            frontier.append((nr, nc, d + 1))
            z = elev.astype(np.float64, copy=True)
            within_buffer = dist <= buffer_cells
            # Linear gradient: sharp at stream (dist=0) → 0 at perimeter.
            drop = np.where(
                within_buffer,
                sharp * (1.0 - dist / max(buffer_cells, 1)) + smooth,
                0.0,
            )
            z = z - drop
            no_val = self.no_data_value[0]
            z[np.isnan(z)] = no_val
            plain_ds = Dataset.dataset_like(
                self, z.astype(elev.dtype, copy=False)
            )
            if inplace:
                self._update_inplace(plain_ds.raster)
                return None
            return DEM(plain_ds.raster)

        if method == "topological_breach":
            # Lindsay 2016: rasterise the stream network onto the DEM
            # (like fill_burn but without the final priority-flood), then
            # invoke the Phase 1 least-cost breach engine so every internal
            # pit Dijkstras outward toward a stream cell. The burned stream
            # cells sit max_breach_depth-or-equivalent below their
            # surroundings, so the breach paths follow the vector topology
            # by construction.
            elev = self.values
            rows, cols = elev.shape
            gt = self.geotransform
            stream_mask = np.zeros((rows, cols), dtype=bool)
            streams = _reproject_if_needed(streams, self.epsg)
            for geom in streams.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "MultiLineString":
                    for sub in geom.geoms:
                        self._rasterise_line(sub, stream_mask, gt)
                else:
                    self._rasterise_line(geom, stream_mask, gt)
            z = elev.astype(np.float64, copy=True)
            z[stream_mask] = z[stream_mask] - constant_drop
            nodata_mask = np.isnan(z)
            z = _breach_depressions_array(
                z, nodata_mask=nodata_mask, method="hybrid",
                max_depth=max_breach_depth,
                max_length=max_breach_length,
                fill_remaining=True,
            )
            no_val = self.no_data_value[0]
            z[np.isnan(z)] = no_val
            plain_ds = Dataset.dataset_like(
                self, z.astype(elev.dtype, copy=False)
            )
            if inplace:
                self._update_inplace(plain_ds.raster)
                return None
            return DEM(plain_ds.raster)

        if method != "fill_burn":
            raise NotImplementedError(
                f"method={method!r} not yet implemented (supported: "
                "'fill_burn', 'agree', 'topological_breach')"
            )

        elev = self.values
        rows, cols = elev.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        stream_mask = np.zeros((rows, cols), dtype=bool)

        streams = _reproject_if_needed(streams, self.epsg)

        for geom in streams.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "MultiLineString":
                for sub in geom.geoms:
                    self._rasterise_line(sub, stream_mask, gt)
            else:
                # Delegate to the shared 2× oversampled floor-rasteriser so
                # burn_streams, enforce_breaklines, and enforce_culverts all
                # snap line samples to cells identically (I1 fix).
                self._rasterise_line(geom, stream_mask, gt)

        z = elev.astype(np.float64, copy=True)
        z[stream_mask] = z[stream_mask] - constant_drop
        nodata_mask = np.isnan(z)
        z = _fill_depressions_array(
            z, nodata_mask=nodata_mask, method="priority_flood", epsilon=0.0,
        )
        no_val = self.no_data_value[0]
        z[np.isnan(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def _rasterise_line(self, geom, mask, gt):
        """Rasterise a single LineString into `mask` using the supplied
        geotransform. Helper for `burn_streams` MultiLineString handling.
        """
        x0, dx, _, y0, _, dy = gt
        rows, cols = mask.shape
        coords = list(geom.coords)
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            c1 = (x1 - x0) / dx
            r1 = (y1 - y0) / dy
            c2 = (x2 - x0) / dx
            r2 = (y2 - y0) / dy
            # 2x oversampling avoids skipping cells when the line crosses
            # cell boundaries exactly between samples.
            steps = max(int(abs(r2 - r1)), int(abs(c2 - c1)), 1) * 2
            for i in range(steps + 1):
                t = i / steps
                r = int(np.floor(r1 + t * (r2 - r1)))
                c = int(np.floor(c1 + t * (c2 - c1)))
                if 0 <= r < rows and 0 <= c < cols:
                    mask[r, c] = True

    def enforce_culverts(
        self,
        roads,
        streams,
        culvert_drop: float = 0.5,
        inplace: bool = False,
    ) -> DEM | None:
        """Lower DEM cells at every stream-road intersection by
        `culvert_drop` so subsequent flow routing crosses roads instead of
        dead-ending against them. Simplified version of WhiteboxTools'
        `BurnStreamsAtRoads`.

        Args:
            roads: `GeoDataFrame` of LineString road geometries.
            streams: `GeoDataFrame` of LineString stream geometries.
            culvert_drop: Elevation drop applied to each intersection cell.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: New DEM with culverts enforced, or None when
            `inplace=True`.
        """
        elev = self.values
        rows, cols = elev.shape
        gt = self.geotransform
        road_mask = np.zeros((rows, cols), dtype=bool)
        stream_mask = np.zeros((rows, cols), dtype=bool)
        for layer, mask in ((roads, road_mask), (streams, stream_mask)):
            layer = _reproject_if_needed(layer, self.epsg)
            for geom in layer.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "MultiLineString":
                    for sub in geom.geoms:
                        self._rasterise_line(sub, mask, gt)
                else:
                    self._rasterise_line(geom, mask, gt)

        crossings = road_mask & stream_mask
        z = elev.astype(np.float64, copy=True)
        z[crossings] = z[crossings] - culvert_drop
        no_val = self.no_data_value[0]
        z[np.isnan(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def _polygon_cell_indices(self, geom, gt, rows, cols):
        """Return `(rows_idx, cols_idx)` of cells whose centre is inside `geom`.

        Uses `shapely.contains_xy` for one batched point-in-polygon test
        per polygon — orders of magnitude faster than the per-cell
        `geom.intersects(Point)` loop that this helper replaces.

        Args:
            geom: Shapely Polygon / MultiPolygon. The bounding box is used to
                clip the candidate cell range; the polygon itself decides
                which of those candidates are kept.
            gt: Six-element GDAL geotransform of this raster.
            rows: Number of rows in the raster.
            cols: Number of columns in the raster.

        Returns:
            Tuple `(rs, cs)` of int ndarrays giving the row / column
            indices of every cell whose centre lies inside `geom`. Empty
            arrays when the polygon's bounding box does not overlap the
            raster envelope.

        Examples:
            - A polygon entirely inside a single cell returns one index:

                >>> import numpy as np
                >>> from shapely.geometry import Polygon
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> # Tight polygon around cell-centre (col=2, row=2) at (2.5, -2.5).
                >>> poly = Polygon(
                ...     [(2.4, -2.6), (2.6, -2.6), (2.6, -2.4), (2.4, -2.4)]
                ... )
                >>> rs, cs = dem._polygon_cell_indices(poly, dem.geotransform, 5, 5)
                >>> rs.tolist(), cs.tolist()
                ([2], [2])

            - A polygon entirely outside the raster envelope returns empty
              arrays:

                >>> import numpy as np
                >>> from shapely.geometry import Polygon
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> far = Polygon(
                ...     [(100, 100), (101, 100), (101, 101), (100, 101)]
                ... )
                >>> rs, cs = dem._polygon_cell_indices(far, dem.geotransform, 5, 5)
                >>> rs.size, cs.size
                (0, 0)

            - Returned indices are integer ndarrays suitable for fancy
              indexing into the raster:

                >>> import numpy as np
                >>> from shapely.geometry import Polygon
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> poly = Polygon([(0, 0), (3, 0), (3, -3), (0, -3)])
                >>> rs, cs = dem._polygon_cell_indices(poly, dem.geotransform, 5, 5)
                >>> np.issubdtype(rs.dtype, np.integer)
                True
        """
        import numpy as np
        import shapely

        x0, dx, _, y0, _, dy = gt
        minx, miny, maxx, maxy = geom.bounds
        c_lo = max(0, int((minx - x0) / dx))
        c_hi = min(cols, int((maxx - x0) / dx) + 1)
        r_lo = max(0, int((maxy - y0) / dy))
        r_hi = min(rows, int((miny - y0) / dy) + 1)
        if c_lo >= c_hi or r_lo >= r_hi:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        rs_idx, cs_idx = np.meshgrid(
            np.arange(r_lo, r_hi), np.arange(c_lo, c_hi), indexing="ij",
        )
        xs = x0 + (cs_idx + 0.5) * dx
        ys = y0 + (rs_idx + 0.5) * dy
        inside = shapely.contains_xy(geom, xs, ys)
        return rs_idx[inside], cs_idx[inside]

    def hydroflatten(
        self,
        water_polygons,
        method: str = "min",
        inplace: bool = False,
    ) -> DEM | None:
        """Flatten lake / pond surfaces to a single elevation per polygon.

        For each input polygon, sample the DEM cells the polygon covers
        and assign every cell in the polygon the per-polygon statistic
        (`"min"` by default — the most defensive choice for hydrology;
        `"mean"` and `"median"` are also supported).

        Args:
            water_polygons: `GeoDataFrame` of Polygon / MultiPolygon
                geometries.
            method: `"min"` (default), `"mean"`, or `"median"`.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: Hydroflattened DEM.
        """
        if method not in ("min", "mean", "median"):
            raise ValueError(
                f"method must be 'min', 'mean', or 'median'; got {method!r}"
            )

        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape

        water_polygons = _reproject_if_needed(water_polygons, self.epsg)

        z = elev.astype(np.float64, copy=True)
        for geom in water_polygons.geometry:
            if geom is None or geom.is_empty:
                continue
            rs, cs = self._polygon_cell_indices(geom, gt, rows, cols)
            if rs.size == 0:
                continue
            vals = z[rs, cs]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if method == "min":
                target = float(vals.min())
            elif method == "mean":
                target = float(vals.mean())
            else:
                target = float(np.median(vals))
            z[rs, cs] = target

        no_val = self.no_data_value[0]
        z[np.isnan(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def burn_buildings(
        self,
        building_polygons,
        lift: float = 50.0,
        inplace: bool = False,
    ) -> DEM | None:
        """Lift building footprints above the DEM by `lift` map units so
        2D flood routing flows around them.

        Args:
            building_polygons: `GeoDataFrame` of Polygon geometries.
            lift: Elevation added to every cell whose centre falls inside a
                polygon.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: DEM with buildings raised.
        """
        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape

        building_polygons = _reproject_if_needed(building_polygons, self.epsg)

        z = elev.astype(np.float64, copy=True)
        for geom in building_polygons.geometry:
            if geom is None or geom.is_empty:
                continue
            rs, cs = self._polygon_cell_indices(geom, gt, rows, cols)
            if rs.size:
                z[rs, cs] = z[rs, cs] + lift

        no_val = self.no_data_value[0]
        z[np.isnan(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def enforce_breaklines(
        self,
        breaklines,
        lift: float = 5.0,
        inplace: bool = False,
    ) -> DEM | None:
        """Raise linear barriers (levees, walls, kerbs) above the surrounding DEM.

        Args:
            breaklines: `GeoDataFrame` of LineString geometries.
            lift: Elevation added at each rasterised cell along the lines.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: DEM with breaklines enforced.
        """
        elev = self.values
        gt = self.geotransform
        rows, cols = elev.shape
        mask = np.zeros((rows, cols), dtype=bool)

        breaklines = _reproject_if_needed(breaklines, self.epsg)

        for geom in breaklines.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "MultiLineString":
                for sub in geom.geoms:
                    self._rasterise_line(sub, mask, gt)
            else:
                self._rasterise_line(geom, mask, gt)

        z = elev.astype(np.float64, copy=True)
        z[mask] = z[mask] + lift
        no_val = self.no_data_value[0]
        z[np.isnan(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def subgrid_bathymetry(
        self,
        scale_factor: int,
        n_bins: int = 10,
    ) -> "pd.DataFrame":
        """Build per-coarse-cell bathymetry tables (SFINCS-style).

        For each coarse cell (`scale_factor × scale_factor` block of fine
        cells), compute a histogram-like table mapping a coarsened water-
        depth level to the wetted area within the block. This is the
        sub-grid representation SFINCS and similar reduced-order 2D models
        use to recover small-scale topography without resolving it on the
        coarse grid.

        Args:
            scale_factor: Integer aggregation factor (>= 2).
            n_bins: Number of depth bins per coarse cell.

        Returns:
            `pandas.DataFrame` indexed by coarse-cell `(row, col)` with
            `n_bins + 2` columns: `z_min`, `z_max`, plus
            `frac_below_<k>` for `k` in `[1, n_bins]` giving the
            fraction of fine cells at or below the `k`-th depth bin.
            For flat blocks (`z_max == z_min`) every `frac_below_<k>`
            is `1.0`.

        Raises:
            ValueError: For `scale_factor < 2` or `n_bins < 1`.

        Examples:
            - A flat block produces `frac_below_<k> == 1.0` for every bin
              (B1 regression — the columns are always present):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> ds = Dataset.create_from_array(
                ...     np.full((4, 4), 5.0, dtype=np.float32),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> df = DEM(ds.raster).subgrid_bathymetry(scale_factor=2, n_bins=3)
                >>> sorted(df.columns.tolist())
                ['frac_below_1', 'frac_below_2', 'frac_below_3', 'z_max', 'z_min']
                >>> float(df["frac_below_1"].iloc[0])
                1.0
        """
        import numpy as np
        import pandas as pd

        if scale_factor < 2:
            raise ValueError(
                f"scale_factor must be >= 2; got {scale_factor}"
            )
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1; got {n_bins}")

        elev = self.values
        rows, cols = elev.shape
        out_rows = rows // scale_factor
        out_cols = cols // scale_factor

        records: list[dict] = []
        for br in range(out_rows):
            for bc in range(out_cols):
                block = elev[
                    br * scale_factor : (br + 1) * scale_factor,
                    bc * scale_factor : (bc + 1) * scale_factor,
                ].ravel()
                valid = block[np.isfinite(block)]
                if valid.size == 0:
                    continue
                z_min = float(valid.min())
                z_max = float(valid.max())
                rec = {"row": br, "col": bc, "z_min": z_min, "z_max": z_max}
                if z_max == z_min:
                    # Flat block: every bin is "below" the single value, so
                    # frac_below_k == 1.0 for every k. (The B1 review found
                    # that the original code computed this list but never
                    # wrote it into the record, dropping the frac columns
                    # entirely when every block was flat.)
                    for k in range(1, n_bins + 1):
                        rec[f"frac_below_{k}"] = 1.0
                else:
                    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
                    for k, edge in enumerate(bin_edges[1:], start=1):
                        rec[f"frac_below_{k}"] = float(
                            (valid <= edge).sum() / valid.size
                        )
                records.append(rec)
        df = pd.DataFrame(records).set_index(["row", "col"])
        return df

    def export(
        self,
        path: str,
        target: str,
        *,
        breaklines=None,
        walls=None,
        buildings=None,
        manning_n=None,
        boundary_conditions=None,
        validate: bool = True,
        **kwargs,
    ) -> dict:
        """Export the DEM to a hydrodynamic-model format.

        Every target listed below ships with a working writer. Most were
        backfilled after the initial Phase-3 cut; `lisflood_fp` is the
        canonical Arc-ASCII writer and remains the only target that
        actually requires a sinks-free input.

        Args:
            path: Output file path.
            target: One of `"hec_ras"`, `"tuflow"`, `"sfincs"`,
                `"lisflood_fp"`, `"iber"`, `"gmsh"`.
            breaklines / walls / buildings / manning_n / boundary_conditions:
                Reserved for target-specific bundles. Currently ignored by
                the LISFLOOD-FP writer.
            validate: When `True` (default), refuse to export a DEM with
                internal sinks. **Only applied for** `target="lisflood_fp"`
                — the Arc-ASCII writer is the only target where downstream
                tooling actually requires sinks-free input. Other writers
                skip the sink scan even when `validate=True` so the
                `local_minima_8` pass does not run unnecessarily (I4
                fixup). Pass `validate=False` to also disable the
                lisflood_fp guard.
            **kwargs: Target-specific options.

        Returns:
            `dict` mapping artefact label → file path written.

        Raises:
            ValueError: For unknown `target`.
            RuntimeError: When `target == "lisflood_fp"`, `validate=True`,
                and the DEM has internal sinks.
        """
        valid_targets = {
            "hec_ras", "tuflow", "sfincs", "lisflood_fp", "iber", "gmsh",
        }
        if target not in valid_targets:
            raise ValueError(
                f"target must be one of {sorted(valid_targets)}; got {target!r}"
            )

        # Only run the (expensive) sink scan when we'll actually export to
        # an implemented target. The unimplemented targets raise
        # NotImplementedError further down and would otherwise pay the full
        # validation cost for nothing.
        if validate and target == "lisflood_fp":
            from digitalrivers._conditioning.pitremoval import local_minima_8
            sinks = local_minima_8(self.values)
            if int(sinks.sum()) > 0:
                raise RuntimeError(
                    f"DEM has {int(sinks.sum())} internal sinks; either fix "
                    "them (DEM.fill_depressions) or pass validate=False"
                )

        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape
        nodata = float(self.no_data_value[0])
        out = np.where(np.isnan(elev), nodata, elev)
        cell_size = abs(dx)
        yllcorner = y0 + rows * dy

        if target == "lisflood_fp":
            with open(path, "w", encoding="ascii", newline="\n") as fh:
                fh.write(f"ncols {cols}\n")
                fh.write(f"nrows {rows}\n")
                fh.write(f"xllcorner {x0}\n")
                fh.write(f"yllcorner {yllcorner}\n")
                fh.write(f"cellsize {cell_size}\n")
                fh.write(f"NODATA_value {nodata}\n")
                for r in range(rows):
                    fh.write(" ".join(f"{out[r, c]:.6f}" for c in range(cols)))
                    fh.write("\n")
            return {"dem_asc": path}

        if target == "hec_ras":
            # HEC-RAS Mapper expects a single-band float32 GeoTIFF in the
            # dataset CRS with consistent geotransform — exactly what
            # Dataset.create_from_array(driver_type="GTiff", path=...) writes.
            Dataset.create_from_array(
                out.astype(np.float32, copy=False),
                geo=gt, epsg=self.epsg,
                no_data_value=nodata,
                driver_type="GTiff", path=path,
            )
            return {"dem_tif": path}

        if target == "tuflow":
            # ESRI floating-point grid (.flt binary, row-major little-endian
            # float32, top-left first) + .hdr text header.
            flt_path = path if path.endswith(".flt") else path + ".flt"
            hdr_path = flt_path[:-4] + ".hdr"
            out.astype(np.float32, copy=False).tofile(flt_path)
            with open(hdr_path, "w") as fh:
                fh.write(f"ncols {cols}\n")
                fh.write(f"nrows {rows}\n")
                fh.write(f"xllcorner {x0}\n")
                fh.write(f"yllcorner {yllcorner}\n")
                fh.write(f"cellsize {cell_size}\n")
                fh.write(f"NODATA_value {nodata}\n")
                fh.write("byteorder LSBFIRST\n")
            return {"dem_flt": flt_path, "dem_hdr": hdr_path}

        if target == "sfincs":
            # SFINCS .dep: row-major little-endian float32, no header.
            # Companion .msk: 0 where no-data, 1 elsewhere.
            dep_path = path if path.endswith(".dep") else path + ".dep"
            msk_path = dep_path[:-4] + ".msk"
            out.astype(np.float32, copy=False).tofile(dep_path)
            mask = np.where(np.isnan(elev), 0, 1).astype(np.uint8)
            mask.tofile(msk_path)
            return {"dem_dep": dep_path, "dem_msk": msk_path}

        if target == "gmsh":
            # Minimal .geo script: define the DEM bounds as a rectangle
            # with a uniform characteristic length. Downstream meshers can
            # be run via `gmsh -2 <path>`.
            geo_path = path if path.endswith(".geo") else path + ".geo"
            ext_x_lo = x0
            ext_x_hi = x0 + cols * dx
            ext_y_hi = y0
            ext_y_lo = y0 + rows * dy
            cl = cell_size
            with open(geo_path, "w") as fh:
                fh.write(f"cl = {cl};\n")
                fh.write(f"Point(1) = {{{ext_x_lo}, {ext_y_lo}, 0, cl}};\n")
                fh.write(f"Point(2) = {{{ext_x_hi}, {ext_y_lo}, 0, cl}};\n")
                fh.write(f"Point(3) = {{{ext_x_hi}, {ext_y_hi}, 0, cl}};\n")
                fh.write(f"Point(4) = {{{ext_x_lo}, {ext_y_hi}, 0, cl}};\n")
                fh.write("Line(1) = {1, 2};\n")
                fh.write("Line(2) = {2, 3};\n")
                fh.write("Line(3) = {3, 4};\n")
                fh.write("Line(4) = {4, 1};\n")
                fh.write("Line Loop(1) = {1, 2, 3, 4};\n")
                fh.write("Plane Surface(1) = {1};\n")
            return {"geo": geo_path}

        if target == "iber":
            # Iber expects a .dat ascii mesh; pending mesh generation
            # (Phase 4 P33) we write a placeholder boundary file that the
            # user can refine in Iber's pre-processor.
            dat_path = path if path.endswith(".dat") else path + ".dat"
            with open(dat_path, "w") as fh:
                fh.write("# Iber mesh boundary (auto-generated)\n")
                fh.write(f"NCOLS {cols}\nNROWS {rows}\n")
                fh.write(f"XLLCORNER {x0}\nYLLCORNER {yllcorner}\n")
                fh.write(f"CELLSIZE {cell_size}\nNODATA {nodata}\n")
                for r in range(rows):
                    fh.write(" ".join(f"{out[r, c]:.6f}" for c in range(cols)))
                    fh.write("\n")
            return {"dem_dat": dat_path}

        # Unreachable — guarded by valid_targets check above.
        raise NotImplementedError(target)

    def anudem_interpolate(
        self,
        mask=None,
        max_iter: int = 200,
        tol: float = 1e-3,
        method: str = "laplacian",
        inplace: bool = False,
    ) -> DEM | None:
        """ANUDEM-lite: Laplacian-relaxation gap fill (P25).

        A pragmatic subset of Hutchinson 1989 ANUDEM that handles the
        common gap-filling case: a DEM with NaN holes (cloud shadows,
        survey gaps, vegetation occlusion) is filled by Gauss-Seidel
        Laplacian relaxation, holding the known cells fixed. Each
        iteration replaces every unknown cell with the mean of its four
        4-connected neighbours; iteration stops when the maximum change
        in a sweep drops below `tol` or after `max_iter` sweeps.

        Two solver methods are available:

        - `"laplacian"` (default): solves Δz = 0 via the 4-neighbour
          mean iteration. Fast, smooth interior, but only C⁰ continuity
          at the anchor cells — the surface has visible "kinks" at
          known points.
        - `"biharmonic"`: solves Δ²z = 0 by alternating two Laplacian
          sweeps (relax `u = Δz`, then relax `z` so `Δz = u`).
          C¹ continuity at anchors; closer to Hutchinson 1989 ANUDEM's
          tension-spline objective but still without multigrid
          acceleration or drainage enforcement.

        Limitations vs full ANUDEM:

        - No multigrid acceleration; iteration cost is O(N · max_iter).
        - No drainage enforcement. For stream-conditioned DEMs, combine
          with :meth:`burn_streams` before or after.
        - No tension parameter (Hutchinson's λ); the biharmonic mode
          is a fixed-λ approximation.

        Args:
            mask: Optional bool array same shape as the DEM. `True`
                marks cells whose values must be preserved (in addition
                to the existing finite cells). `None` keeps every
                finite cell fixed.
            max_iter: Maximum relaxation sweeps.
            tol: Convergence tolerance — stop when `max |Δz| < tol`.
            method: `"laplacian"` (default) or `"biharmonic"`.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: Filled DEM, or None when `inplace=True`.

        Raises:
            ValueError: If the input DEM has no finite cells or `method`
                is not `"laplacian"` / `"biharmonic"`.

        Examples:
            - Fill a single-cell NaN hole with the default Laplacian solver;
              the filled value sits inside the bracketing range:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.create_from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="laplacian", max_iter=200, tol=1e-6,
                ... )
                >>> out = filled.values
                >>> bool(1.0 <= out[1, 1] <= 9.0)
                True

            - The biharmonic mode (P32 backfill) approximates Hutchinson
              1989's Delta^2 z = 0 by alternating Laplacian sweeps:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.create_from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="biharmonic", max_iter=200, tol=1e-5,
                ... )
                >>> bool(np.isfinite(filled.values).all())
                True

            - Unknown method raises ValueError:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> ds = Dataset.create_from_array(
                ...     np.ones((2, 2), dtype=np.float32),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> DEM(ds.raster).anudem_interpolate(method="bogus")
                Traceback (most recent call last):
                    ...
                ValueError: method must be 'laplacian' or 'biharmonic'; got 'bogus'

            - The 5-point stencil uses edge-replication (Neumann) boundary
              padding, so a NaN cell adjacent to the raster boundary is
              filled from its in-bounds neighbours only — no periodic-wrap
              contamination from the opposite edge:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> # Top-left NaN with very different anchor at bottom-right.
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> z[-1, -1] = -100.0
                >>> z[0, 0] = np.nan
                >>> ds = Dataset.create_from_array(
                ...     np.where(np.isnan(z), -9999.0, z),
                ...     top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).anudem_interpolate(
                ...     method="laplacian", max_iter=300, tol=1e-6,
                ... )
                >>> bool(abs(float(filled.values[0, 0]) - 10.0) < 5.0)
                True

        See Also:
            DEM.fill_depressions: hydrologic conditioning that removes sinks.
            DEM.burn_streams: stream-network drainage enforcement.
        """
        if method not in ("laplacian", "biharmonic"):
            raise ValueError(
                f"method must be 'laplacian' or 'biharmonic'; got {method!r}"
            )

        elev = self.values
        rows, cols = elev.shape
        z = elev.astype(np.float64, copy=True)
        fixed = np.isfinite(z)
        if mask is not None:
            fixed = fixed | mask.astype(bool, copy=False)
        if not fixed.any():
            raise ValueError(
                "anudem_interpolate needs at least one finite anchor cell"
            )
        # Seed unknown cells to the mean of known values to speed convergence.
        z = np.where(fixed, z, z[fixed].mean())

        def _edge_shifts(arr):
            """Return (north, south, east, west) views of `arr` with
            edge-replication boundary handling (no periodic wrap).

            Using `np.pad(..., mode="edge")` matches a Neumann (zero
            normal-derivative) boundary, which is the natural choice for
            an interpolation kernel — the original `np.roll` formed a
            torus and injected far-edge values into near-edge cells,
            corrupting anchors near the DEM boundary.
            """
            padded = np.pad(arr, 1, mode="edge")
            return (
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, 2:],
                padded[1:-1, :-2],
            )

        if method == "laplacian":
            for _ in range(max_iter):
                north, south, east, west = _edge_shifts(z)
                new_z = (north + south + east + west) / 4.0
                new_z[fixed] = z[fixed]
                diff = float(np.max(np.abs(new_z - z)))
                z = new_z
                if diff < tol:
                    break
        else:  # biharmonic
            # Alternate two Laplacian sweeps to approximate Δ²z = 0.
            # Step A: compute u = Δz on the current z.
            # Step B: relax z so Δz ≈ smoothed u (mean of neighbour u's).
            # Composed, this approximates a biharmonic relaxation with C¹
            # continuity at the anchors.
            for _ in range(max_iter):
                north, south, east, west = _edge_shifts(z)
                u = north + south + east + west - 4.0 * z
                un, us, ue, uw = _edge_shifts(u)
                u_smooth = (un + us + ue + uw) / 4.0
                # Solve Δz = u_smooth → new z[i,j] =
                # (sum of neighbours - u_smooth) / 4.
                new_z = (north + south + east + west - u_smooth) / 4.0
                new_z[fixed] = z[fixed]
                diff = float(np.max(np.abs(new_z - z)))
                z = new_z
                if diff < tol:
                    break

        no_val = self.no_data_value[0]
        z[~np.isfinite(z)] = no_val
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return DEM(plain_ds.raster)

    def fill_sinks(self, inplace: bool = False) -> DEM | None:
        """Deprecated alias for `fill_depressions(method="priority_flood", epsilon=0.1)`.

        The original implementation was a single-pass, single-cell sink fill that did
        not cascade through nested pits. Calls now route through the Priority-Flood +
        epsilon algorithm, which is correct on cascading depressions. The output
        differs from the historical algorithm in two ways:

        1. Cascading pits are fully resolved (each pit fills to the rim of its enclosing
           pit, not just to its immediate-neighbour minimum).
        2. Drainage paths within filled depressions inherit a 0.1-unit gradient — so
           D8 routing on the result avoids `NO_FLOW` cells inside the fill.

        Args:
            inplace: If `True` the instance is updated in place; otherwise a new
                `DEM` is returned.

        Returns:
            DEM | None: New `DEM` with the sink-free elevation, or `None` when
            `inplace` is `True`.
        """
        warnings.warn(
            "DEM.fill_sinks is deprecated; use DEM.fill_depressions(method='priority_flood', "
            "epsilon=0.1) for equivalent behaviour or method='wang_liu' for a flat fill.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fill_depressions(method="priority_flood", epsilon=0.1, inplace=inplace)

    def _get_8_direction_slopes(self) -> np.ndarray:
        """Compute slopes to all eight neighbours for every cell.

        Uses a padded elevation array and vectorised NumPy slicing to
        calculate the elevation difference divided by the inter-cell
        distance (cell size for cardinal, cell size × √2 for diagonal)
        in each of the eight D8 directions.

        Returns:
            np.ndarray: 3-D `float32` array of shape
                `(rows, columns, 8)` where the third axis corresponds
                to the direction indices defined in `DIR_OFFSETS`.
        """
        elev = self.values
        cell_size = self.cell_size
        dist2 = cell_size * np.sqrt(2)
        distances = [
            cell_size,
            dist2,
            cell_size,
            dist2,
            cell_size,
            dist2,
            cell_size,
            dist2,
        ]
        rows, cols = elev.shape
        slopes = np.full((rows, cols, 8), np.nan, dtype=np.float32)

        # padding = 2
        # pad_1 = padding - 1
        # Create a padded elevation array for boundary conditions
        padded_elev = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
        padded_elev[1:-1, 1:-1] = elev

        # Calculate elevation differences using slicing
        diff_right = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, 2:]
        diff_top_right = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 2:]
        diff_top = padded_elev[1:-1, 1:-1] - padded_elev[:-2, 1:-1]
        diff_top_left = padded_elev[1:-1, 1:-1] - padded_elev[:-2, :-2]
        diff_left = padded_elev[1:-1, 1:-1] - padded_elev[1:-1, :-2]
        diff_bottom_left = padded_elev[1:-1, 1:-1] - padded_elev[2:, :-2]
        diff_bottom = padded_elev[1:-1, 1:-1] - padded_elev[2:, 1:-1]
        diff_bottom_right = padded_elev[1:-1, 1:-1] - padded_elev[2:, 2:]

        # Calculate slopes
        slopes[:, :, 0] = diff_bottom / distances[0]
        slopes[:, :, 1] = diff_bottom_left / distances[1]
        slopes[:, :, 2] = diff_left / distances[2]
        slopes[:, :, 3] = diff_top_left / distances[3]
        slopes[:, :, 4] = diff_top / distances[4]
        slopes[:, :, 5] = diff_top_right / distances[5]
        slopes[:, :, 6] = diff_right / distances[6]
        slopes[:, :, 7] = diff_bottom_right / distances[7]

        return slopes

    def twi(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Topographic Wetness Index (Beven & Kirkby 1979).

        TWI = ln(SCA / tan(slope)) where SCA = specific catchment area =
        `accumulation_cells * cell_size`. High TWI values mark cells likely
        to be wet (low slope and / or large upstream area). Slope is treated
        in radians internally; values below a small floor (≈ 0.06°) are
        clamped to avoid log-singularity at flat cells.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees. If
                None, `Terrain.slope`-equivalent is computed on the fly via
                `arctan` of `DEM.slope()`'s rise/run output.

        Returns:
            `Dataset` of float32 TWI values. No-data sentinel `-9999.0`.

        References:
            Beven, K. J. & Kirkby, M. J. (1979). "A physically based,
            variable contributing area model of basin hydrology."
            *Hydrological Sciences Bulletin* 24(1): 43-69.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="twi")

    def spi(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Stream Power Index (Moore et al. 1991).

        SPI = SCA * tan(slope). Proportional to the rate at which overland
        flow does work at a cell; useful as a proxy for erosion risk.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees. If
                None, computed on the fly.

        Returns:
            `Dataset` of float32 SPI values. No-data sentinel `-9999.0`.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="spi")

    def sti(
        self,
        accumulation,
        slope_deg: Dataset | None = None,
    ) -> Dataset:
        """Sediment Transport Index (Moore & Burch 1986).

        STI = (SCA / 22.13)^0.6 * (sin(slope) / 0.0896)^1.3. The 22.13 m and
        0.0896 m/m constants come from the original USLE plot length and
        slope; STI predicts where overland flow will transport sediment
        rather than deposit it.

        Args:
            accumulation: `Accumulation` raster aligned to this DEM.
            slope_deg: Optional pre-computed slope raster in degrees.

        Returns:
            `Dataset` of float32 STI values. No-data sentinel `-9999.0`.
        """
        return self._area_slope_index(accumulation, slope_deg, kind="sti")

    def _area_slope_index(
        self,
        accumulation,
        slope_deg,
        *,
        kind: str,
    ) -> Dataset:
        """Shared kernel for the TWI / SPI / STI family.

        All three indices need `(SCA, slope_rad)`; the difference is the
        functional form applied to them.
        """
        if slope_deg is None:
            slope_ratio = self.slope().read_array().astype(np.float64, copy=False)
            slope_deg_arr = np.degrees(np.arctan(np.where(
                np.isfinite(slope_ratio), slope_ratio, 0.0
            )))
        else:
            slope_deg_arr = slope_deg.read_array().astype(np.float64, copy=False)
            no_val = (
                slope_deg.no_data_value[0] if slope_deg.no_data_value else None
            )
            if no_val is not None:
                slope_deg_arr = np.where(
                    slope_deg_arr == no_val, np.nan, slope_deg_arr
                )

        acc = accumulation.read_array().astype(np.float64, copy=False)
        if slope_deg_arr.shape != acc.shape:
            raise ValueError(
                f"slope shape {slope_deg_arr.shape} != accumulation shape "
                f"{acc.shape}"
            )

        slope_rad = np.deg2rad(slope_deg_arr)
        # Floor at ~0.001 rad (≈ 0.06°) to keep tan() / sin() bounded away
        # from zero on flats.
        slope_rad = np.where(
            np.isfinite(slope_rad), np.maximum(slope_rad, 1.0e-3), np.nan,
        )

        cs = float(abs(self.geotransform[1]))
        sca = acc * cs  # specific catchment area (m, since width ≈ cell_size)

        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "twi":
                arr = np.log(sca / np.tan(slope_rad))
            elif kind == "spi":
                arr = sca * np.tan(slope_rad)
            elif kind == "sti":
                arr = ((sca / 22.13) ** 0.6) * (
                    (np.sin(slope_rad) / 0.0896) ** 1.3
                )
            else:  # pragma: no cover — defensive
                raise ValueError(f"unknown kind {kind!r}")

        no_val = -9999.0
        arr = np.where(np.isfinite(arr), arr, no_val).astype(np.float32)
        return Dataset.create_from_array(
            arr,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=no_val,
        )

    def _focal_window_stats(
        self,
        window: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared focal-window kernel for the terrain-index family.

        Returns `(z, focal_mean, focal_sd)` where the focal stats are
        rectangular `window×window` reductions computed by
        `scipy.ndimage.uniform_filter`. No-data cells contribute zero to the
        running sums; `valid` is recovered from `np.isnan(z)`.

        Used by `tpi`, `deviation_from_mean`, `elev_std`, `ruggedness` —
        every entry point applies the same NaN-handling and return-type
        wrapping, so the shared helper centralises the messy parts.
        """
        from scipy.ndimage import uniform_filter
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window!r}")
        z = self.values.astype(np.float64, copy=False)
        valid = ~np.isnan(z)
        z_filled = np.where(valid, z, 0.0)
        m = uniform_filter(z_filled, size=int(window), mode="reflect")
        sq = uniform_filter(z_filled * z_filled, size=int(window), mode="reflect")
        sd = np.sqrt(np.maximum(sq - m * m, 0.0))
        return z, m, sd

    def tpi(self, window: int = 3) -> Dataset:
        """Topographic Position Index (Guisan 1999).

        TPI = `z - focal_mean(z, window)`. Positive values mark ridges and
        upland positions; negative values mark valleys and depressions.

        Args:
            window: Side length of the focal window in cells (must be ≥ 1).
                Defaults to 3 (a 3×3 neighbourhood). Larger windows pick up
                regional / catchment-scale topography; smaller windows pick
                up local relief.

        Returns:
            `Dataset` of float32 TPI values. No-data cells use this DEM's
            no-data sentinel.

        References:
            Guisan, A., Weiss, S. B., & Weiss, A. D. (1999). "GLM versus
            CCA spatial modeling of plant species distribution." *Plant
            Ecology* 143(1): 107-122.

        Examples:
            - A flat surface has every cell at its focal mean → TPI = 0:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> tpi = DEM(ds.raster).tpi(window=3).read_array()
                >>> bool(np.allclose(tpi, 0.0))
                True

            - A single ridge cell on flat terrain reports positive TPI; a
              pit reports negative:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 9.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(DEM(ds.raster).tpi(window=3).read_array()[2, 2] > 0)
                True
        """
        z, m, _sd = self._focal_window_stats(window)
        out = (z - m).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def deviation_from_mean(self, window: int = 3) -> Dataset:
        """Deviation from mean elevation — standardised TPI.

        `(z - focal_mean) / focal_sd`. Dimensionless ridge / valley index;
        because it normalises by local roughness it allows comparing
        positions across regimes with very different relief.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 deviation values. Flat cells (focal_sd ≈ 0)
            yield 0.0 by definition. No-data cells use this DEM's no-data
            sentinel.

        Examples:
            - Flat terrain yields zero deviation everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).deviation_from_mean(window=3).read_array(), 0.0
                ... ))
                True

            - A peak reports positive standardised deviation:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 10.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> out = DEM(ds.raster).deviation_from_mean(window=3).read_array()
                >>> bool(out[2, 2] > 0)
                True
        """
        z, m, sd = self._focal_window_stats(window)
        out = (z - m) / np.where(sd == 0.0, 1.0, sd)
        out = out.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def elev_std(self, window: int = 3) -> Dataset:
        """Standard deviation of elevation in a focal window.

        Pure focal-window SD on the elevation raster. A roughness proxy:
        high values mark varied terrain, low values mark smooth terrain.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 SD values. No-data cells use this DEM's
            no-data sentinel.

        Examples:
            - Flat terrain reports zero SD everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).elev_std(window=3).read_array(), 0.0
                ... ))
                True

            - A step in elevation produces non-zero SD along the boundary:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[:, 3:] = 10.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> sd = DEM(ds.raster).elev_std(window=3).read_array()
                >>> bool((sd[:, 2] > 0).all())
                True
        """
        _z, _m, sd = self._focal_window_stats(window)
        out = sd.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(out), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def curvature(self, kind: str = "profile") -> Dataset:
        """Surface curvature (Zevenbergen & Thorne 1987).

        Fits a partial quartic polynomial `z = Ax²y² + Bx²y + Cxy² + Dx² +
        Ey² + Fxy + Gx + Hy + I` to the 3×3 neighbourhood of each cell and
        evaluates one of the five canonical curvature variants from the
        coefficient grid:

        * `"plan"` — curvature perpendicular to the slope direction;
          positive on diverging slopes, negative on converging ones.
        * `"profile"` — curvature parallel to the slope direction; positive
          on convex (decelerating) slopes, negative on concave (accelerating)
          ones.
        * `"total"` — `2 * (D + E)`; sign-independent total relief.
        * `"mean"` — average of the two principal curvatures.
        * `"gaussian"` — product of the two principal curvatures.

        Args:
            kind: One of `"plan"`, `"profile"`, `"total"`, `"mean"`,
                `"gaussian"`. Defaults to `"profile"`.

        Returns:
            `Dataset` of float32 curvature values. No-data cells use this
            DEM's no-data sentinel.

        Raises:
            ValueError: If `kind` is not one of the five recognised
                variants.

        References:
            Zevenbergen, L. W. & Thorne, C. R. (1987). "Quantitative
            analysis of land surface topography." *Earth Surface Processes
            and Landforms* 12(1): 47-56.

        Examples:
            - Every curvature variant is zero on a flat DEM:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).curvature(kind="total").read_array(), 0.0
                ... ))
                True

            - Mean curvature equals total / 2 on a paraboloid (interior):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> x, y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
                >>> z = (-(x * x + y * y)).astype(np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> total = dem.curvature(kind="total").read_array()[2:-2, 2:-2]
                >>> mean = dem.curvature(kind="mean").read_array()[2:-2, 2:-2]
                >>> bool(np.allclose(mean, total / 2.0, atol=1e-5))
                True
        """
        if kind not in ("plan", "profile", "total", "mean", "gaussian"):
            raise ValueError(
                f"kind must be one of 'plan', 'profile', 'total', 'mean', "
                f"'gaussian'; got {kind!r}"
            )
        z = self.values.astype(np.float64, copy=False)
        L = float(abs(self.geotransform[1]))
        # 3×3 stencil — `np.pad(mode="edge")` mirrors the boundary value so
        # finite differences stay defined at the raster edge.
        zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
        z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
        z4, z5, z6 = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
        z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
        # Zevenbergen-Thorne polynomial coefficients.
        D = ((z4 + z6) / 2.0 - z5) / (L * L)
        E = ((z2 + z8) / 2.0 - z5) / (L * L)
        F = (-z1 + z3 + z7 - z9) / (4.0 * L * L)
        G = (-z4 + z6) / (2.0 * L)
        H = (z2 - z8) / (2.0 * L)
        denom = G * G + H * H + 1.0e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "plan":
                arr = -2.0 * (D * H * H + E * G * G - F * G * H) / denom
            elif kind == "profile":
                arr = 2.0 * (D * G * G + E * H * H + F * G * H) / denom
            elif kind == "total":
                arr = 2.0 * (D + E)
            elif kind == "mean":
                arr = D + E
            else:  # gaussian
                arr = 4.0 * D * E - F * F
        out = arr.astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z) | ~np.isfinite(out), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def normal_vector_deviation(self, window: int = 3) -> Dataset:
        """Per-cell angular deviation of the surface normal from its focal mean.

        Computes each cell's outward-pointing surface normal from finite
        differences of the elevation grid, then takes the focal-mean of the
        unit-normal components in a `window×window` neighbourhood. The
        result at each cell is the angle (in radians) between the local
        normal and the focal-mean normal — a roughness metric that grows
        with how strongly the surface bends within the window.

        Args:
            window: Side length of the focal window in cells (≥ 1).

        Returns:
            `Dataset` of float32 angular deviations in radians,
            `[0, π/2]`. No-data cells use this DEM's no-data sentinel.

        Examples:
            - Flat terrain yields zero angular deviation:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).normal_vector_deviation(window=3).read_array(),
                ...     0.0, atol=1e-5,
                ... ))
                True

            - A constant-slope plane has identical normals in its deep
              interior, so deviation there is ~0:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> x, y = np.meshgrid(np.arange(7), np.arange(7))
                >>> z = (2.0 * x + y).astype(np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).normal_vector_deviation(window=3).read_array()
                >>> bool(np.allclose(arr[2:-2, 2:-2], 0.0, atol=1e-4))
                True
        """
        from scipy.ndimage import uniform_filter
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window!r}")
        z = self.values.astype(np.float64, copy=False)
        L = float(abs(self.geotransform[1]))
        # Finite-difference partials with reflective edge padding.
        zp = np.pad(np.where(np.isnan(z), 0.0, z), 1, mode="edge")
        dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2.0 * L)
        dzdy = (zp[2:, 1:-1] - zp[:-2, 1:-1]) / (2.0 * L)
        # Outward-pointing normal `(-dz/dx, -dz/dy, 1)`; renormalise to unit.
        nx_raw = -dzdx
        ny_raw = -dzdy
        nz_raw = np.ones_like(z)
        nm = np.sqrt(nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw)
        nx = nx_raw / nm
        ny = ny_raw / nm
        nz = nz_raw / nm
        # Focal mean of each component, then renormalise to keep the mean
        # vector unit-length.
        mnx = uniform_filter(nx, size=int(window), mode="reflect")
        mny = uniform_filter(ny, size=int(window), mode="reflect")
        mnz = uniform_filter(nz, size=int(window), mode="reflect")
        mm = np.sqrt(mnx * mnx + mny * mny + mnz * mnz) + 1.0e-12
        mnx /= mm
        mny /= mm
        mnz /= mm
        cos_theta = np.clip(nx * mnx + ny * mny + nz * mnz, -1.0, 1.0)
        out = np.arccos(cos_theta).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def openness(
        self,
        *,
        search_radius: int = 10,
        kind: str = "positive",
    ) -> Dataset:
        """Topographic openness (Yokoyama 2002).

        For each cell, walks outward along 8 azimuths up to `search_radius`
        cells and records the maximum elevation angle (positive openness)
        or the minimum (negative openness) along each walk. The per-cell
        output is the mean of `(π/2 - horizon_angle)` across the 8
        directions, in radians.

        High positive openness marks exposed / high-relief locations; high
        negative openness marks deep depressions / valley floors.

        Args:
            search_radius: Maximum walk distance in cells. Must be ≥ 1.
                Defaults to 10.
            kind: `"positive"` (default) or `"negative"`. Negative openness
                flips the sign of the elevation difference internally —
                effectively measuring the local pit / depression depth.

        Returns:
            `Dataset` of float32 openness values in radians. No-data cells
            use this DEM's no-data sentinel.

        Raises:
            ValueError: If `kind` is not one of `"positive"` / `"negative"`
                or `search_radius < 1`.

        References:
            Yokoyama, R., Shirasawa, M., & Pike, R. J. (2002). "Visualizing
            topography by openness: A new application of image processing
            to digital elevation models." *Photogrammetric Engineering and
            Remote Sensing* 68(3): 257-265.

        Examples:
            - Flat terrain: every horizon angle is 0, so positive openness
              is `π/2` at every cell:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).openness(search_radius=2).read_array()
                >>> bool(np.allclose(arr, np.pi / 2.0, atol=1e-5))
                True

            - A peak on flat terrain has strictly larger positive openness
              than its neighbours (which look up at it):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 10.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).openness(search_radius=3).read_array()
                >>> bool(arr[2, 2] > arr[1, 2])
                True
        """
        from digitalrivers._numba import horizon_walk_kernel
        if kind not in ("positive", "negative"):
            raise ValueError(
                f"kind must be 'positive' or 'negative'; got {kind!r}"
            )
        if search_radius < 1:
            raise ValueError(
                f"search_radius must be >= 1; got {search_radius!r}"
            )
        z = self.values.astype(np.float64, copy=False)
        # For negative openness, flip the elevation so the kernel's
        # "maximum upward angle" becomes "maximum downward angle" relative
        # to the original surface.
        z_in = (-z if kind == "negative" else z).astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z_in), 0.0, z_in)
        out = horizon_walk_kernel(
            z_filled, float(abs(self.geotransform[1])), int(search_radius), 0,
        ).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def sky_view_factor(
        self,
        *,
        search_radius: int = 10,
    ) -> Dataset:
        """Sky-view factor (Zakšek et al. 2011).

        For each cell, the fraction of the upper hemisphere that is visible
        from that cell. Walks along 8 azimuths up to `search_radius`,
        records the maximum elevation angle along each walk, and returns
        the mean of `(1 - sin(horizon_angle))` across the 8 directions —
        equivalent to the fraction of an isotropic sky dome not occluded
        by surrounding terrain.

        Shares the horizon-walk kernel with `openness` (W-27); the two
        differ only in the per-direction aggregation function.

        Args:
            search_radius: Maximum walk distance in cells. Must be ≥ 1.
                Defaults to 10.

        Returns:
            `Dataset` of float32 SVF values in `[0, 1]`. No-data cells use
            this DEM's no-data sentinel.

        Raises:
            ValueError: If `search_radius < 1`.

        References:
            Zakšek, K., Oštir, K., & Kokalj, Ž. (2011). "Sky-view factor as
            a relief visualization technique." *Remote Sensing* 3(2):
            398-415.

        Examples:
            - Flat terrain: nothing occludes the sky, SVF = 1 everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).sky_view_factor(search_radius=2).read_array()
                >>> bool(np.allclose(arr, 1.0, atol=1e-5))
                True

            - A pit surrounded by higher cells reports SVF strictly less
              than 1 (the walls occlude part of the sky):

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
                >>> z[2, 2] = 0.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> arr = DEM(ds.raster).sky_view_factor(search_radius=2).read_array()
                >>> bool(arr[2, 2] < 1.0)
                True
        """
        from digitalrivers._numba import horizon_walk_kernel
        if search_radius < 1:
            raise ValueError(
                f"search_radius must be >= 1; got {search_radius!r}"
            )
        z = self.values.astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z), 0.0, z)
        out = horizon_walk_kernel(
            z_filled, float(abs(self.geotransform[1])), int(search_radius), 1,
        ).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def ruggedness(self, window: int = 3) -> Dataset:
        """Terrain Ruggedness Index (Riley et al. 1999).

        Per-cell mean of absolute elevation differences to every other cell
        in a `window×window` neighbourhood. Output unit is the DEM elevation
        unit (metres). Higher values mark rougher terrain; flat terrain is
        zero.

        Args:
            window: Side length of the focal window in cells (≥ 1).
                Defaults to 3 (Riley's original 3×3 neighbourhood).

        Returns:
            `Dataset` of float32 ruggedness values. No-data cells use this
            DEM's no-data sentinel.

        References:
            Riley, S. J., DeGloria, S. D., & Elliot, R. (1999). "A terrain
            ruggedness index that quantifies topographic heterogeneity."
            *Intermountain Journal of Sciences* 5(1-4): 23-27.

        Examples:
            - Flat terrain has zero ruggedness everywhere:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.full((4, 4), 5.0, dtype=np.float32)
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> bool(np.allclose(
                ...     DEM(ds.raster).ruggedness(window=3).read_array(), 0.0
                ... ))
                True

            - A peak surrounded by flat terrain contributes positive
              ruggedness at the peak and its 8-neighbours:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.zeros((5, 5), dtype=np.float32)
                >>> z[2, 2] = 9.0
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> tri = DEM(ds.raster).ruggedness(window=3).read_array()
                >>> bool(tri[2, 2] > 0 and tri[1, 2] > 0 and tri[3, 2] > 0)
                True
        """
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window!r}")
        z = self.values.astype(np.float64, copy=False)
        z_filled = np.where(np.isnan(z), 0.0, z)
        total = np.zeros_like(z_filled)
        count = 0
        half = int(window) // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(z_filled, shift=(dr, dc), axis=(0, 1))
                total += np.abs(z_filled - shifted)
                count += 1
        out = (total / float(count)).astype(np.float32)
        no_val = float(self.no_data_value[0])
        out = np.where(np.isnan(z), no_val, out)
        return Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=no_val,
        )

    def slope(self) -> Dataset:
        """Compute the maximum downhill slope at every cell.

        Calculates slopes in all eight D8 directions via
        `_get_8_direction_slopes` and returns a raster whose cell
        values are the maximum slope across the eight neighbours.

        Returns:
            Dataset: Single-band raster with the same geometry as the
                DEM, containing the maximum slope value per cell.

        See Also:
            Terrain.slope: GDAL-based slope using Horn or
                Zevenbergen-Thorne algorithms.
        """
        slope = self._get_8_direction_slopes()
        max_slope = np.nanmax(slope, axis=2)

        src = self.dataset_like(self, max_slope)
        return src

    def set_outflow(
        self, outflow: GeoDataFrame, direction: int, inplace: bool = False
    ) -> Dataset:
        """Assign a fixed flow direction at the basin outfall cell.

        Args:
            outflow: GeoDataFrame with point geometry marking the
                outfall location.
            direction: D8 direction code (0–7) to force at the outfall.
            inplace: If `True` modify the current instance in place;
                otherwise return a new `Dataset`.

        Returns:
            Dataset with the outfall direction applied, or `None` when
            *inplace* is `True`.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("set_outflow is not yet implemented.")

    def flow_direction(
        self,
        method: str = "d8",
        exponent: float = 1.0,
        forced: GeoDataFrame | None = None,
        seed: int | None = None,
        forced_direction: GeoDataFrame | None = None,
    ) -> FlowDirection:
        """Derive a flow-direction raster from the DEM under one of five routing schemes.

        Schemes:

        * `"d8"` (default) — O'Callaghan & Mark (1984). Single-direction steepest
          descent. Output: 1-band `int32` raster of direction codes 0–7 following
          `DIR_OFFSETS`.
        * `"dinf"` — Tarboton (1997). Output: 2-band `float32` raster. Band 0 is
          the aspect angle in radians CCW from east in `[0, 2π)`; band 1 is the
          slope magnitude along the chosen facet. `-1.0` in band 0 marks sinks /
          no-data.
        * `"mfd_quinn"` — Quinn et al. (1991). Multi-direction with contour-length
          weighting. Output: 8-band `float32` raster of partition fractions,
          ordered by `DIR_OFFSETS`. Per-cell fractions sum to 1.0 (or all zero
          for sinks).
        * `"mfd_holmgren"` — Holmgren (1994). Same family as Quinn but tunable
          `exponent` (default 1.0 mimics Quinn; 4–6 mimics D8). 8-band output.
        * `"rho8"` — Fairfield & Leymarie (1991). Stochastic single-direction;
          cardinal slopes are perturbed before the steepest-neighbour pick. Pass
          `seed` for reproducibility. 1-band `int32` output like D8.

        Args:
            method: Routing scheme — one of `"d8"`, `"dinf"`, `"mfd_quinn"`,
                `"mfd_holmgren"`, `"rho8"`.
            exponent: `p` for `mfd_holmgren` and `mfd_quinn`. Ignored otherwise.
            forced: Optional GeoDataFrame with columns `geometry` (point) and
                `direction` (int 0–7) — cells at the given locations are forced
                to that D8 direction code regardless of the computed slope. Only
                meaningful for `"d8"` and `"rho8"`.
            seed: Random seed for `"rho8"` reproducibility.
            forced_direction: Deprecated alias for `forced`. If both are given,
                `forced` wins.

        Returns:
            FlowDirection: typed wrapper carrying the routing scheme and encoding.

        Raises:
            ValueError: If `method` is unknown.
        """
        if forced is None and forced_direction is not None:
            forced = forced_direction

        valid_methods = {"d8", "dinf", "mfd_quinn", "mfd_holmgren", "rho8"}
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {sorted(valid_methods)}; got {method!r}"
            )

        elev = self.values
        valid_mask = ~np.isnan(elev)

        if method == "d8":
            slopes = self._get_8_direction_slopes()
            slope_valid = ~np.all(np.isnan(slopes), axis=2)
            valid_cells_mask = valid_mask & slope_valid
            arr = np.full(elev.shape, Dataset.default_no_data_value, dtype=np.int32)
            if valid_cells_mask.any():
                best_dir = np.nanargmax(slopes[valid_cells_mask], axis=1)
                # Only commit a direction where the steepest slope is strictly downhill;
                # cells whose best 8-neighbour is at equal or higher elevation are sinks
                # and stay at the no-data sentinel (spec P5: "max(s_k) ≤ 0 → sink").
                rr, cc = np.where(valid_cells_mask)
                max_slope = slopes[rr, cc, best_dir]
                downhill = max_slope > 0
                arr[rr[downhill], cc[downhill]] = best_dir[downhill]
            if forced is not None:
                indices = self.map_to_array_coordinates(forced)
                for i, ind in enumerate(indices):
                    arr[tuple(ind)] = forced.loc[i, "direction"]
            plain_ds = Dataset.create_from_array(
                arr, geo=self.geotransform, epsg=self.epsg,
                no_data_value=self.default_no_data_value,
            )
            return FlowDirection.from_dataset(plain_ds, routing="d8")

        if method == "rho8":
            slopes = self._get_8_direction_slopes()
            rng = np.random.default_rng(seed)
            arr = _rho8_flow_direction(slopes, valid_mask, rng=rng)
            # Replace -1 (sentinel from rho8 helper) with the dataset no-data value.
            arr[arr < 0] = Dataset.default_no_data_value
            if forced is not None:
                indices = self.map_to_array_coordinates(forced)
                for i, ind in enumerate(indices):
                    arr[tuple(ind)] = forced.loc[i, "direction"]
            plain_ds = Dataset.create_from_array(
                arr.astype(np.int32, copy=False),
                geo=self.geotransform, epsg=self.epsg,
                no_data_value=self.default_no_data_value,
            )
            return FlowDirection.from_dataset(plain_ds, routing="rho8")

        if method == "dinf":
            angle, magnitude = _dinf_flow_direction(elev, self.cell_size)
            stacked = np.stack([angle, magnitude], axis=0).astype(np.float32, copy=False)
            plain_ds = Dataset.create_from_array(
                stacked, geo=self.geotransform, epsg=self.epsg,
                no_data_value=self.default_no_data_value,
            )
            return FlowDirection.from_dataset(plain_ds, routing="dinf")

        # mfd_quinn or mfd_holmgren
        slopes = self._get_8_direction_slopes()
        weighting = "quinn" if method == "mfd_quinn" else "holmgren"
        fractions = _mfd_flow_direction(
            slopes, valid_mask, weighting=weighting, exponent=exponent,
        )
        # Transpose (rows, cols, 8) -> (8, rows, cols) for pyramids's band-first layout.
        bands = np.transpose(fractions, (2, 0, 1)).astype(np.float32, copy=False)
        plain_ds = Dataset.create_from_array(
            bands, geo=self.geotransform, epsg=self.epsg,
            no_data_value=self.default_no_data_value,
        )
        return FlowDirection.from_dataset(plain_ds, routing=method)

    def accumulate_flow(self, r, c, flow_dir, acc, dir_offsets) -> int:
        """Count upstream cells that drain into `(r, c)` (iterative).

        Uses an explicit stack to perform a depth-first traversal of the
        flow-direction grid backwards.  For every neighbour whose flow
        direction points toward the current cell, the neighbour is pushed
        onto the stack.  Results are cached in *acc* so each cell is
        computed at most once.

        Args:
            r: Row index of the target cell.
            c: Column index of the target cell.
            flow_dir: 2-D `int` array of D8 direction codes (0–7).
            acc: 2-D `int32` accumulation array.  Cells initialised to
                `-1` are unprocessed; non-negative values are cached
                results.
            dir_offsets: Direction-offset mapping (see `DIR_OFFSETS`).

        Returns:
            Number of upstream cells that drain into `(r, c)`
            (excluding the cell itself).
        """
        rows, cols = flow_dir.shape

        if not (0 <= r < rows and 0 <= c < cols):
            return 0
        if acc[r, c] >= 0:
            return acc[r, c]

        offsets_list = [
            (d_col, d_row, self.opposite_direction(d_row, d_col, dir_offsets))
            for d_col, d_row in dir_offsets.values()
        ]

        stack = [(r, c, 0, 0)]

        while stack:
            cr, cc, idx, total = stack[-1]

            if acc[cr, cc] >= 0:
                stack.pop()
                if stack:
                    pr, pc, pidx, ptotal = stack[-1]
                    stack[-1] = (pr, pc, pidx, ptotal + acc[cr, cc] + 1)
                continue

            # Advance through remaining neighbours.
            found_unprocessed = False
            while idx < len(offsets_list):
                d_col, d_row, opp = offsets_list[idx]
                idx += 1
                rr, rc = cr + d_row, cc + d_col
                if not (0 <= rr < rows and 0 <= rc < cols):
                    continue
                if flow_dir[rr, rc] != opp:
                    continue
                if opp is None:
                    continue
                # Neighbour already computed — just add its count.
                if acc[rr, rc] >= 0:
                    total += acc[rr, rc] + 1
                    continue
                # Neighbour needs processing — save our state and push it.
                stack[-1] = (cr, cc, idx, total)
                stack.append((rr, rc, 0, 0))
                found_unprocessed = True
                break

            if not found_unprocessed:
                # All neighbours processed — finalise this cell.
                acc[cr, cc] = total
                stack.pop()
                if stack:
                    pr, pc, pidx, ptotal = stack[-1]
                    stack[-1] = (pr, pc, pidx, ptotal + total + 1)

        return acc[r, c]

    @staticmethod
    def opposite_direction(dr, dc, dir_offsets):
        """Return the D8 direction code opposite to the given offset.

        Args:
            dr: Row offset component.
            dc: Column offset component.
            dir_offsets: Direction-offset mapping (see `DIR_OFFSETS`).

        Returns:
            int or None: Direction code whose offset is `(-dr, -dc)`,
            or `None` if no match is found.
        """
        for d, (d_col, d_row) in dir_offsets.items():
            if d_row == -dr and d_col == -dc:
                return d
        return None

    def flow_accumulation(
        self,
        flow_direction,
        weights: Dataset | None = None,
        dir_offsets: dict = None,
    ) -> Dataset:
        """Compute flow accumulation under the given routing scheme.

        Generalised dispatcher that delegates to `FlowDirection.accumulate(...)`
        and returns an `int32` cast for backwards compatibility with the
        previous D8-only output. For weighted or fractional accumulation, call
        `flow_direction.accumulate(weights)` directly to get the underlying
        `Accumulation` (float32) instead.

        Args:
            flow_direction: A `FlowDirection` (preferred — its routing tag
                dispatches the algorithm) or a bare `Dataset` (assumed to be
                a D8 direction-code raster for back-compat).
            weights: Optional per-cell weight raster aligned to the DEM.
            dir_offsets: Deprecated/ignored. Kept for signature compatibility.

        Returns:
            Dataset: `int32` accumulation raster. No-data cells retain
            `Dataset.default_no_data_value`. Cell values are the count of
            (or weighted sum over) strictly-upstream cells — the cell's own
            weight does not contribute to its own value.

        Warns:
            UserWarning: When `flow_direction.routing` produces fractional
                accumulations (`"dinf"`, `"mfd_quinn"`, `"mfd_holmgren"`).
                The legacy `int32` cast truncates these toward zero, which is
                almost always wrong; call `flow_direction.accumulate(...)`
                directly to get the fractional `Accumulation` raster.

        Examples:
            - Compute D8 cell-count accumulation on a small east-flowing DEM
              and inspect the outlet value:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> fd = dem.flow_direction(method="d8")
                >>> acc = dem.flow_accumulation(fd)
                >>> int(acc.read_array().max()) > 0
                True

            - A D∞ `FlowDirection` triggers the truncation warning:

                >>> import warnings
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.create_from_array(
                ...     z, top_left_corner=(0.0, 0.0), cell_size=1.0,
                ...     epsg=4326, no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> fd = dem.flow_direction(method="dinf")
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     _ = dem.flow_accumulation(fd)
                >>> any("int32" in str(w.message) for w in caught)
                True
        """
        del dir_offsets  # legacy positional kwarg, no longer used
        import warnings

        if not isinstance(flow_direction, FlowDirection):
            # Wrap a bare Dataset as D8 for back-compat callers.
            flow_direction = FlowDirection.from_dataset(flow_direction, routing="d8")

        if flow_direction.routing not in ("d8", "rho8"):
            warnings.warn(
                f"DEM.flow_accumulation casts to int32 and truncates fractional "
                f"accumulations for routing={flow_direction.routing!r}. Call "
                f"flow_direction.accumulate(...) directly to get the float32 "
                f"Accumulation raster.",
                UserWarning,
                stacklevel=2,
            )

        acc = flow_direction.accumulate(weights=weights)
        arr = acc.read_array().astype(np.int32, copy=False)
        # Restore the dataset no-data sentinel where the original DEM is no-data.
        elev = self.values
        nodata_mask = np.isnan(elev)
        arr[nodata_mask] = Dataset.default_no_data_value
        return Dataset.create_from_array(
            arr,
            geo=self.geotransform,
            epsg=self.epsg,
            no_data_value=self.default_no_data_value,
        )

    def convert_flow_direction_to_cell_indices(self) -> np.ndarray:
        """Convert D8 direction codes to downstream cell row/column indices.

        Computes the flow direction from the DEM and translates each
        direction code into the absolute row and column index of the
        downstream neighbour.

        Returns:
            np.ndarray: 3-D `float64` array of shape
                `(rows, columns, 2)`.  Layer 0 holds the downstream
                row index; layer 1 holds the downstream column index.
                Cells with no valid direction contain `np.nan`.
        """
        flow_direction = self.flow_direction()
        flow_dir = flow_direction.read_array(band=0).astype(np.float32)
        no_val = flow_direction.no_data_value[0]
        flow_dir[np.isclose(flow_dir, no_val, rtol=0.00001)] = np.nan

        rows, cols = flow_dir.shape
        valid = ~np.isnan(flow_dir)

        # Build lookup arrays from DIR_OFFSETS (index 0 = first tuple
        # element, index 1 = second tuple element, matching the
        # original loop: cell[i,j,0] = i + offset[0]).
        offset_0 = np.array([DIR_OFFSETS[d][0] for d in range(8)], dtype=np.float64)
        offset_1 = np.array([DIR_OFFSETS[d][1] for d in range(8)], dtype=np.float64)

        flow_direction_cell = np.full((rows, cols, 2), np.nan, dtype=np.float64)

        dir_idx = flow_dir[valid].astype(int)
        row_idx, col_idx = np.where(valid)
        flow_direction_cell[valid, 0] = row_idx + offset_0[dir_idx]
        flow_direction_cell[valid, 1] = col_idx + offset_1[dir_idx]

        return flow_direction_cell


    @staticmethod
    def delete_basins(basins: Dataset, path: str):
        """Keep only the basin with the lowest ID and discard the rest.

        Reads a basin-ID raster produced during catchment delineation,
        replaces every cell that does not belong to the lowest basin ID
        with the no-data value, and writes the result to *path*.

        Args:
            basins: Dataset whose cell values are basin IDs (integers).
                The lowest unique basin ID (excluding no-data) is
                retained.
            path: Output GeoTIFF file path (must end with `".tif"`).

        Raises:
            TypeError: If *path* is not a string.
        """
        if not isinstance(path, str):
            raise TypeError(f"path: {path} input should be string type")

        basins_a = basins.read_array()
        no_val = np.float32(basins.no_data_value[0])

        valid_mask = basins_a != no_val
        unique_basins = np.unique(basins_a[valid_mask]).astype(int)

        if len(unique_basins) > 0:
            keep = unique_basins[0]
            basins_a[valid_mask & (basins_a != keep)] = no_val

        Dataset.dataset_like(basins, basins_a, path)
