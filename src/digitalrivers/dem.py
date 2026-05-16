"""DEM processing module.

This module provides the ``DEM`` class for digital elevation model analysis,
including depression filling, slope calculation, D8 flow direction, and flow
accumulation.
"""
from __future__ import annotations

import warnings

import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset

from shapely.geometry import Point as _make_point  # noqa: N812
from digitalrivers._breach import breach_depressions as _breach_depressions_array
from digitalrivers._flats import resolve_flats as _resolve_flats_array
from digitalrivers._flow_routing import (
    dinf_flow_direction as _dinf_flow_direction,
    mfd_flow_direction as _mfd_flow_direction,
    rho8_flow_direction as _rho8_flow_direction,
)
from digitalrivers._pitremoval import fill_depressions as _fill_depressions_array
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


class DEM(Dataset):
    """Digital Elevation Model processor.

    Wraps a GDAL raster dataset and adds hydrological analysis methods:
    sink filling, D8 flow direction, flow accumulation, and slope
    computation.

    Args:
        src: GDAL dataset containing a single-band elevation raster.
        access: ``"read_only"`` (default) or ``"write"``.
    """

    def __init__(self, src: gdal.Dataset, access: str = "read_only"):
        super().__init__(src, access)

    @property
    def values(self):
        """Elevation array with no-data cells replaced by ``np.nan``.

        Reads band 0 as ``float32`` and masks every cell whose value is
        close to the raster's no-data value (relative tolerance 1e-5).

        Returns:
            np.ndarray: 2-D ``float32`` array of shape ``(rows, columns)``.
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

        Three algorithms are available via the ``method`` argument:

        * ``"priority_flood"`` (default) — Barnes, Lehman & Mulla (2014) Priority-Flood
          with the two-queue plateau optimisation. With ``epsilon == 0`` it produces flat
          fills; with ``epsilon > 0`` it produces a strictly monotonic surface (every cell
          has at least one strictly lower neighbour along the flood path) at the cost of
          a small elevation inflation proportional to plateau width.
        * ``"wang_liu"`` — Wang & Liu (2006). Flat fill, no epsilon. Equivalent in output
          to ``priority_flood`` with ``epsilon == 0``; kept as a named alternative for
          callers who plan to resolve flats explicitly afterwards (P4).
        * ``"planchon_darboux"`` — Planchon & Darboux (2002). Iterative directional-sweep
          algorithm. Slower than Priority-Flood on large DEMs; kept as a low-relief
          reference. Requires ``epsilon > 0``.

        No-data handling is uniform across methods: cells flagged no-data act as outlets
        (they cannot be filled, and data cells adjacent to them are seeded as drainage
        sources alongside the true raster boundary).

        **Precision note.** Priority-flood / planchon-darboux compute the cumulative lift
        in float64 but the output is cast back to the input dtype. For ``float32`` DEMs
        with ``epsilon`` in the ``0.1``-class on wide plateaus, the accumulated lift can
        approach float32's relative precision near the spill elevation and very long
        plateaus may underflow to identical filled values. Prefer ``float64`` inputs
        when running with ``epsilon > 0`` and large depressions; ``wang_liu`` /
        ``epsilon=0`` are immune.

        Args:
            method: One of ``"priority_flood"``, ``"wang_liu"``, ``"planchon_darboux"``.
            epsilon: Per-step elevation lift inside depressions. ``0.0`` (default for
                ``priority_flood``) returns a non-strictly-decreasing surface — flats
                remain flat. Positive values guarantee a unique downhill path at the
                cost of slight elevation inflation. ``planchon_darboux`` requires
                ``epsilon > 0``.
            inplace: If ``True`` the current instance is updated in place and ``None``
                is returned. If ``False`` (default) a new ``DEM`` is returned.

        Returns:
            DEM | None: A new ``DEM`` containing the filled elevation, or ``None`` when
            ``inplace`` is ``True``.

        Raises:
            ValueError: If ``method`` is unknown, or ``planchon_darboux`` is requested
                with ``epsilon <= 0``.
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

        Three methods are available via the ``method`` argument:

        * ``"single_cell"`` — cheap O(n) preprocessing pass that resolves isolated 1-cell
          pits by lowering an intermediate first-order neighbour to the midpoint of the
          pit and a lower second-order cell. Does nothing if no such configuration exists.
        * ``"least_cost"`` (default) — Lindsay 2016 Dijkstra-from-each-pit. Carves a
          strictly monotonic channel from the pit to the nearest outlet. Optional
          ``max_depth`` and ``max_length`` constraints abort the breach for any pit whose
          channel would exceed them; aborted pits are left unresolved.
        * ``"hybrid"`` — try ``least_cost`` first; pits that fail their constraint fall
          back to the Priority-Flood depression fill (P2). The breach phase has already
          lowered parts of the DEM where partial breaching occurred, so the fill operates
          on a modified surface and produces less overall lift than fill-only.

        No-data cells act as free outlets — any Dijkstra path that reaches a no-data cell
        terminates the search.

        Args:
            method: One of ``"single_cell"``, ``"least_cost"``, ``"hybrid"``.
            max_depth: Maximum cumulative ``|Δz|`` for a single breach path. ``None``
                disables the constraint.
            max_length: Maximum path length in cells. ``None`` disables.
            fill_remaining: Only meaningful when ``method="hybrid"``. If ``True``
                (default), unresolved pits are passed to Priority-Flood with
                ``epsilon=0``. If ``False``, they are left as pits in the output.
            inplace: If ``True`` the current instance is updated in place and ``None`` is
                returned. If ``False`` (default) a new ``DEM`` is returned.

        Returns:
            DEM | None: A new ``DEM`` containing the breached elevation, or ``None`` when
            ``inplace`` is ``True``.

        Raises:
            ValueError: If ``method`` is unknown.
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

        After ``fill_depressions(method="wang_liu")`` (or ``"priority_flood"`` with
        ``epsilon=0``), every closed depression is filled to its spill elevation — but the
        interior of each filled depression is a flat plateau with no defined steepest
        descent, so D8 flow direction over the result has ``NO_FLOW`` cells across every
        plateau. ``resolve_flats`` nudges those cells so each has a unique downhill
        neighbour: combined Garbrecht & Martz (1997) gradient — drain *towards* the
        nearest outlet (LEC) with a tiebreak that drains *away from* the nearest rim
        (HEC). The towards-lower gradient is weighted ``2x`` so it dominates and the
        away-from-higher gradient acts as a deterministic tiebreaker.

        Plateaus without a low-edge cell (closed depressions that survived the fill — they
        should not exist if you ran ``fill_depressions`` first) are left untouched.

        Args:
            max_iter: Safety cap on BFS levels per plateau. Real plateaus rarely exceed
                ``max(rows, cols)``; the default ``1000`` is essentially unbounded.
            epsilon: Per-BFS-step elevation lift. Total lift over a plateau is at most
                ``(2 * max_high_dist + max_low_dist) * epsilon``; choose small enough
                that this stays well below the minimum elevation step between adjacent
                non-plateau cells. Default ``1e-5`` is safe for ~1000-cell-wide plateaus.
            connectivity: 4 or 8. Controls plateau-labelling and BFS step direction;
                LEC/HEC classification always uses 8-connectivity (Garbrecht-Martz
                convention). Default is 8.
            inplace: If ``True`` the current instance is updated in place and ``None`` is
                returned. If ``False`` (default) a new ``DEM`` is returned.

        Returns:
            DEM | None: A new ``DEM`` with flat plateaus resolved, or ``None`` when
            ``inplace`` is ``True``.

        Raises:
            ValueError: If ``connectivity`` is not 4 or 8.
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
        flow_direction,
    ) -> Dataset:
        """Compute Height Above Nearest Drainage (Rennó 2008 / Nobre 2011).

        For every cell, follows the flow-direction raster downstream until it
        reaches a stream cell, and assigns ``elev[cell] - elev[stream_cell]``
        as the cell's HAND value. Stream cells themselves are 0; cells whose
        flow path does not reach a stream (orphans, sinks, no-data) are NaN.

        Args:
            streams: ``StreamRaster`` aligned to this DEM. Only the underlying
                stream mask is read.
            flow_direction: Single-direction ``FlowDirection`` (``d8`` /
                ``rho8``) aligned to this DEM.

        Returns:
            ``Dataset`` containing the float32 HAND raster. No-data cells use
            this DEM's no-data sentinel (NaN in the underlying values
            property; the on-disk sentinel restored before write-back).

        Raises:
            ValueError: If shapes do not match or ``flow_direction`` is
                multi-direction.
        """
        from digitalrivers._hand import hand_d8
        from digitalrivers.flow_direction import FlowDirection
        from digitalrivers.stream_raster import StreamRaster

        if not isinstance(flow_direction, FlowDirection):
            raise ValueError(
                "flow_direction must be a FlowDirection instance"
            )
        if flow_direction.routing not in ("d8", "rho8"):
            raise ValueError(
                f"hand currently supports single-direction routing only; "
                f"got {flow_direction.routing!r}"
            )
        if not isinstance(streams, StreamRaster):
            raise ValueError("streams must be a StreamRaster instance")

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
        ``"fill_burn"`` (Saunders 1999 — used by WhiteboxTools' FillBurn) as
        the default. ``"agree"`` (Hellweger 1997) and
        ``"topological_breach"`` (Lindsay 2016) raise ``NotImplementedError``.

        Fill-burn algorithm:

        1. Rasterise every LineString in ``streams`` onto a stream mask.
        2. Lower every stream cell's elevation by ``constant_drop``.
        3. Run ``fill_depressions(method="priority_flood")`` so the
           surrounding cells drain naturally into the channel.

        Args:
            streams: ``GeoDataFrame`` of LineString geometries.
            method: ``"fill_burn"`` (default); ``"agree"`` and
                ``"topological_breach"`` raise ``NotImplementedError``.
            sharp / smooth / buffer_cells: AGREE parameters (unused for
                fill_burn).
            constant_drop: Elevation drop applied to every stream cell
                in fill_burn (default 1.0 map unit).
            max_breach_depth / max_breach_length: topological_breach
                parameters (unused for fill_burn).
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: New DEM with the conditioned surface, or None when
            ``inplace=True``.
        """
        import numpy as np

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
            target_epsg = self.epsg
            if (
                getattr(streams, "crs", None) is not None
                and target_epsg is not None
                and streams.crs.to_epsg() != target_epsg
            ):
                streams = streams.to_crs(target_epsg)
            for geom in streams.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    _ = list(geom.coords)
                    self._rasterise_line(geom, stream_mask, gt)
                except NotImplementedError:
                    for sub in geom.geoms:
                        self._rasterise_line(sub, stream_mask, gt)
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
            target_epsg = self.epsg
            if (
                getattr(streams, "crs", None) is not None
                and target_epsg is not None
                and streams.crs.to_epsg() != target_epsg
            ):
                streams = streams.to_crs(target_epsg)
            for geom in streams.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    _ = list(geom.coords)
                    self._rasterise_line(geom, stream_mask, gt)
                except NotImplementedError:
                    for sub in geom.geoms:
                        self._rasterise_line(sub, stream_mask, gt)
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
                f"method={method!r} not yet implemented (only 'fill_burn', "
                "'agree')"
            )

        elev = self.values
        rows, cols = elev.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        stream_mask = np.zeros((rows, cols), dtype=bool)

        target_epsg = self.epsg
        if (
            getattr(streams, "crs", None) is not None
            and target_epsg is not None
            and streams.crs.to_epsg() != target_epsg
        ):
            streams = streams.to_crs(target_epsg)

        for geom in streams.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                coords = list(geom.coords)
            except NotImplementedError:
                # MultiLineString — iterate each segment.
                for sub in geom.geoms:
                    self._rasterise_line(sub, stream_mask, gt)
                continue
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                c1 = (x1 - x0) / dx
                r1 = (y1 - y0) / dy
                c2 = (x2 - x0) / dx
                r2 = (y2 - y0) / dy
                steps = max(int(abs(r2 - r1)), int(abs(c2 - c1)), 1)
                for i in range(steps + 1):
                    t = i / steps
                    r = int(round(r1 + t * (r2 - r1)))
                    c = int(round(c1 + t * (c2 - c1)))
                    if 0 <= r < rows and 0 <= c < cols:
                        stream_mask[r, c] = True

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
        """Rasterise a single LineString into ``mask`` using the supplied
        geotransform. Helper for ``burn_streams`` MultiLineString handling.
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
        ``culvert_drop`` so subsequent flow routing crosses roads instead of
        dead-ending against them. Simplified version of WhiteboxTools'
        ``BurnStreamsAtRoads``.

        Args:
            roads: ``GeoDataFrame`` of LineString road geometries.
            streams: ``GeoDataFrame`` of LineString stream geometries.
            culvert_drop: Elevation drop applied to each intersection cell.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: New DEM with culverts enforced, or None when
            ``inplace=True``.
        """
        import numpy as np

        elev = self.values
        rows, cols = elev.shape
        gt = self.geotransform
        road_mask = np.zeros((rows, cols), dtype=bool)
        stream_mask = np.zeros((rows, cols), dtype=bool)
        for layer, mask in ((roads, road_mask), (streams, stream_mask)):
            target_epsg = self.epsg
            if (
                getattr(layer, "crs", None) is not None
                and target_epsg is not None
                and layer.crs.to_epsg() != target_epsg
            ):
                layer = layer.to_crs(target_epsg)
            for geom in layer.geometry:
                if geom is None or geom.is_empty:
                    continue
                try:
                    coords = list(geom.coords)
                except NotImplementedError:
                    for sub in geom.geoms:
                        self._rasterise_line(sub, mask, gt)
                    continue
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

    def hydroflatten(
        self,
        water_polygons,
        method: str = "min",
        inplace: bool = False,
    ) -> DEM | None:
        """Flatten lake / pond surfaces to a single elevation per polygon.

        For each input polygon, sample the DEM cells the polygon covers
        and assign every cell in the polygon the per-polygon statistic
        (``"min"`` by default — the most defensive choice for hydrology;
        ``"mean"`` and ``"median"`` are also supported).

        Args:
            water_polygons: ``GeoDataFrame`` of Polygon / MultiPolygon
                geometries.
            method: ``"min"`` (default), ``"mean"``, or ``"median"``.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: Hydroflattened DEM.
        """
        import numpy as np

        if method not in ("min", "mean", "median"):
            raise ValueError(
                f"method must be 'min', 'mean', or 'median'; got {method!r}"
            )

        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape

        target_epsg = self.epsg
        if (
            getattr(water_polygons, "crs", None) is not None
            and target_epsg is not None
            and water_polygons.crs.to_epsg() != target_epsg
        ):
            water_polygons = water_polygons.to_crs(target_epsg)

        z = elev.astype(np.float64, copy=True)
        for geom in water_polygons.geometry:
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            c_lo = max(0, int((minx - x0) / dx))
            c_hi = min(cols, int((maxx - x0) / dx) + 1)
            r_lo = max(0, int((maxy - y0) / dy))
            r_hi = min(rows, int((miny - y0) / dy) + 1)
            in_poly: list[tuple[int, int]] = []
            for r in range(r_lo, r_hi):
                for c in range(c_lo, c_hi):
                    cx = x0 + (c + 0.5) * dx
                    cy = y0 + (r + 0.5) * dy
                    if geom.intersects(_make_point(cx, cy)):
                        in_poly.append((r, c))
            if not in_poly:
                continue
            vals = np.array([z[r, c] for r, c in in_poly], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if method == "min":
                target = float(vals.min())
            elif method == "mean":
                target = float(vals.mean())
            else:
                target = float(np.median(vals))
            for r, c in in_poly:
                z[r, c] = target

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
        """Lift building footprints above the DEM by ``lift`` map units so
        2D flood routing flows around them.

        Args:
            building_polygons: ``GeoDataFrame`` of Polygon geometries.
            lift: Elevation added to every cell whose centre falls inside a
                polygon.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: DEM with buildings raised.
        """
        import numpy as np

        elev = self.values
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt
        rows, cols = elev.shape

        target_epsg = self.epsg
        if (
            getattr(building_polygons, "crs", None) is not None
            and target_epsg is not None
            and building_polygons.crs.to_epsg() != target_epsg
        ):
            building_polygons = building_polygons.to_crs(target_epsg)

        z = elev.astype(np.float64, copy=True)
        for geom in building_polygons.geometry:
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            c_lo = max(0, int((minx - x0) / dx))
            c_hi = min(cols, int((maxx - x0) / dx) + 1)
            r_lo = max(0, int((maxy - y0) / dy))
            r_hi = min(rows, int((miny - y0) / dy) + 1)
            for r in range(r_lo, r_hi):
                for c in range(c_lo, c_hi):
                    cx = x0 + (c + 0.5) * dx
                    cy = y0 + (r + 0.5) * dy
                    if geom.intersects(_make_point(cx, cy)):
                        z[r, c] = z[r, c] + lift

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
            breaklines: ``GeoDataFrame`` of LineString geometries.
            lift: Elevation added at each rasterised cell along the lines.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: DEM with breaklines enforced.
        """
        import numpy as np

        elev = self.values
        gt = self.geotransform
        rows, cols = elev.shape
        mask = np.zeros((rows, cols), dtype=bool)

        target_epsg = self.epsg
        if (
            getattr(breaklines, "crs", None) is not None
            and target_epsg is not None
            and breaklines.crs.to_epsg() != target_epsg
        ):
            breaklines = breaklines.to_crs(target_epsg)

        for geom in breaklines.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                _ = list(geom.coords)
                self._rasterise_line(geom, mask, gt)
            except NotImplementedError:
                for sub in geom.geoms:
                    self._rasterise_line(sub, mask, gt)

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
    ):
        """Build per-coarse-cell bathymetry tables (SFINCS-style).

        For each coarse cell (``scale_factor × scale_factor`` block of fine
        cells), compute a histogram-like table mapping a coarsened water-
        depth level to the wetted area within the block. This is the
        sub-grid representation SFINCS and similar reduced-order 2D models
        use to recover small-scale topography without resolving it on the
        coarse grid.

        Args:
            scale_factor: Integer aggregation factor (>= 2).
            n_bins: Number of depth bins per coarse cell.

        Returns:
            ``pandas.DataFrame`` indexed by coarse-cell (row, col) with
            ``n_bins + 1`` columns: ``z_min``, ``z_max``, plus
            ``frac_below_<k>`` for ``k`` in ``[1, n_bins]`` giving the
            fraction of fine cells at or below the k-th depth bin.

        Raises:
            ValueError: For ``scale_factor < 2`` or ``n_bins < 1``.
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
                    fracs = [1.0] * n_bins
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

        v1 status: only ``target="lisflood_fp"`` is fully implemented (writes
        an Arc-ASCII ``.dem.asc``). The other targets (``hec_ras``,
        ``tuflow``, ``sfincs``, ``iber``, ``gmsh``) ship as
        ``NotImplementedError`` pointing at the spec; native writers will
        land in follow-up commits.

        Args:
            path: Output file path.
            target: One of ``"hec_ras"``, ``"tuflow"``, ``"sfincs"``,
                ``"lisflood_fp"``, ``"iber"``, ``"gmsh"``.
            breaklines / walls / buildings / manning_n / boundary_conditions:
                Reserved for target-specific bundles. Currently ignored by
                the LISFLOOD-FP writer.
            validate: When True (default), refuse to export a DEM with
                internal sinks to targets that require sinks-free input.
            **kwargs: Target-specific options.

        Returns:
            ``dict`` mapping artefact label → file path written.

        Raises:
            ValueError: For unknown ``target``.
            NotImplementedError: For targets other than ``lisflood_fp``.
            RuntimeError: When ``validate=True`` and the DEM has internal
                sinks.
        """
        import numpy as np

        valid_targets = {
            "hec_ras", "tuflow", "sfincs", "lisflood_fp", "iber", "gmsh",
        }
        if target not in valid_targets:
            raise ValueError(
                f"target must be one of {sorted(valid_targets)}; got {target!r}"
            )

        if validate:
            from digitalrivers._pitremoval import local_minima_8
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
            with open(path, "w") as fh:
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
                fh.write(f"# Iber mesh boundary (auto-generated)\n")
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
        in a sweep drops below ``tol`` or after ``max_iter`` sweeps.

        Two solver methods are available:

        - ``"laplacian"`` (default): solves Δz = 0 via the 4-neighbour
          mean iteration. Fast, smooth interior, but only C⁰ continuity
          at the anchor cells — the surface has visible "kinks" at
          known points.
        - ``"biharmonic"``: solves Δ²z = 0 by alternating two Laplacian
          sweeps (relax ``u = Δz``, then relax ``z`` so ``Δz = u``).
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
            mask: Optional bool array same shape as the DEM. ``True``
                marks cells whose values must be preserved (in addition
                to the existing finite cells). ``None`` keeps every
                finite cell fixed.
            max_iter: Maximum relaxation sweeps.
            tol: Convergence tolerance — stop when ``max |Δz| < tol``.
            method: ``"laplacian"`` (default) or ``"biharmonic"``.
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: Filled DEM, or None when ``inplace=True``.

        Raises:
            ValueError: If the input DEM has no finite cells or ``method``
                is not ``"laplacian"`` / ``"biharmonic"``.

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

        See Also:
            DEM.fill_depressions: hydrologic conditioning that removes sinks.
            DEM.burn_streams: stream-network drainage enforcement.
        """
        import numpy as np

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

        if method == "laplacian":
            for _ in range(max_iter):
                north = np.roll(z, 1, axis=0)
                south = np.roll(z, -1, axis=0)
                east = np.roll(z, -1, axis=1)
                west = np.roll(z, 1, axis=1)
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
                # Laplacian of z (5-point stencil).
                north = np.roll(z, 1, axis=0)
                south = np.roll(z, -1, axis=0)
                east = np.roll(z, -1, axis=1)
                west = np.roll(z, 1, axis=1)
                u = north + south + east + west - 4.0 * z
                # Smooth u (one Laplacian sweep on u).
                un = np.roll(u, 1, axis=0)
                us = np.roll(u, -1, axis=0)
                ue = np.roll(u, -1, axis=1)
                uw = np.roll(u, 1, axis=1)
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
        """Deprecated alias for ``fill_depressions(method="priority_flood", epsilon=0.1)``.

        The original implementation was a single-pass, single-cell sink fill that did
        not cascade through nested pits. Calls now route through the Priority-Flood +
        epsilon algorithm, which is correct on cascading depressions. The output
        differs from the historical algorithm in two ways:

        1. Cascading pits are fully resolved (each pit fills to the rim of its enclosing
           pit, not just to its immediate-neighbour minimum).
        2. Drainage paths within filled depressions inherit a 0.1-unit gradient — so
           D8 routing on the result avoids ``NO_FLOW`` cells inside the fill.

        Args:
            inplace: If ``True`` the instance is updated in place; otherwise a new
                ``DEM`` is returned.

        Returns:
            DEM | None: New ``DEM`` with the sink-free elevation, or ``None`` when
            ``inplace`` is ``True``.
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
            np.ndarray: 3-D ``float32`` array of shape
                ``(rows, columns, 8)`` where the third axis corresponds
                to the direction indices defined in ``DIR_OFFSETS``.
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

    def slope(self) -> Dataset:
        """Compute the maximum downhill slope at every cell.

        Calculates slopes in all eight D8 directions via
        ``_get_8_direction_slopes`` and returns a raster whose cell
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
            inplace: If ``True`` modify the current instance in place;
                otherwise return a new ``Dataset``.

        Returns:
            Dataset with the outfall direction applied, or ``None`` when
            *inplace* is ``True``.

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

        * ``"d8"`` (default) — O'Callaghan & Mark (1984). Single-direction steepest
          descent. Output: 1-band ``int32`` raster of direction codes 0–7 following
          ``DIR_OFFSETS``.
        * ``"dinf"`` — Tarboton (1997). Output: 2-band ``float32`` raster. Band 0 is
          the aspect angle in radians CCW from east in ``[0, 2π)``; band 1 is the
          slope magnitude along the chosen facet. ``-1.0`` in band 0 marks sinks /
          no-data.
        * ``"mfd_quinn"`` — Quinn et al. (1991). Multi-direction with contour-length
          weighting. Output: 8-band ``float32`` raster of partition fractions,
          ordered by ``DIR_OFFSETS``. Per-cell fractions sum to 1.0 (or all zero
          for sinks).
        * ``"mfd_holmgren"`` — Holmgren (1994). Same family as Quinn but tunable
          ``exponent`` (default 1.0 mimics Quinn; 4–6 mimics D8). 8-band output.
        * ``"rho8"`` — Fairfield & Leymarie (1991). Stochastic single-direction;
          cardinal slopes are perturbed before the steepest-neighbour pick. Pass
          ``seed`` for reproducibility. 1-band ``int32`` output like D8.

        Args:
            method: Routing scheme — one of ``"d8"``, ``"dinf"``, ``"mfd_quinn"``,
                ``"mfd_holmgren"``, ``"rho8"``.
            exponent: ``p`` for ``mfd_holmgren`` and ``mfd_quinn``. Ignored otherwise.
            forced: Optional GeoDataFrame with columns ``geometry`` (point) and
                ``direction`` (int 0–7) — cells at the given locations are forced
                to that D8 direction code regardless of the computed slope. Only
                meaningful for ``"d8"`` and ``"rho8"``.
            seed: Random seed for ``"rho8"`` reproducibility.
            forced_direction: Deprecated alias for ``forced``. If both are given,
                ``forced`` wins.

        Returns:
            FlowDirection: typed wrapper carrying the routing scheme and encoding.

        Raises:
            ValueError: If ``method`` is unknown.
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
        """Count upstream cells that drain into ``(r, c)`` (iterative).

        Uses an explicit stack to perform a depth-first traversal of the
        flow-direction grid backwards.  For every neighbour whose flow
        direction points toward the current cell, the neighbour is pushed
        onto the stack.  Results are cached in *acc* so each cell is
        computed at most once.

        Args:
            r: Row index of the target cell.
            c: Column index of the target cell.
            flow_dir: 2-D ``int`` array of D8 direction codes (0–7).
            acc: 2-D ``int32`` accumulation array.  Cells initialised to
                ``-1`` are unprocessed; non-negative values are cached
                results.
            dir_offsets: Direction-offset mapping (see ``DIR_OFFSETS``).

        Returns:
            Number of upstream cells that drain into ``(r, c)``
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
            dir_offsets: Direction-offset mapping (see ``DIR_OFFSETS``).

        Returns:
            int or None: Direction code whose offset is ``(-dr, -dc)``,
            or ``None`` if no match is found.
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

        Generalised dispatcher that delegates to ``FlowDirection.accumulate(...)``
        and returns an ``int32`` cast for backwards compatibility with the
        previous D8-only output. For weighted or fractional accumulation, call
        ``flow_direction.accumulate(weights)`` directly to get the underlying
        ``Accumulation`` (float32) instead.

        Args:
            flow_direction: A ``FlowDirection`` (preferred — its routing tag
                dispatches the algorithm) or a bare ``Dataset`` (assumed to be
                a D8 direction-code raster for back-compat).
            weights: Optional per-cell weight raster aligned to the DEM.
            dir_offsets: Deprecated/ignored. Kept for signature compatibility.

        Returns:
            Dataset: ``int32`` accumulation raster. No-data cells retain
            ``Dataset.default_no_data_value``. Cell values are the count of
            (or weighted sum over) strictly-upstream cells — the cell's own
            weight does not contribute to its own value.

        Warns:
            UserWarning: When ``flow_direction.routing`` produces fractional
                accumulations (``"dinf"``, ``"mfd_quinn"``, ``"mfd_holmgren"``).
                The legacy ``int32`` cast truncates these toward zero, which is
                almost always wrong; call ``flow_direction.accumulate(...)``
                directly to get the fractional ``Accumulation`` raster.

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

            - A D∞ ``FlowDirection`` triggers the truncation warning:

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
            np.ndarray: 3-D ``float64`` array of shape
                ``(rows, columns, 2)``.  Layer 0 holds the downstream
                row index; layer 1 holds the downstream column index.
                Cells with no valid direction contain ``np.nan``.
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
            path: Output GeoTIFF file path (must end with ``".tif"``).

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
