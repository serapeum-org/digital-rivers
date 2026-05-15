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
        """
        del dir_offsets  # legacy positional kwarg, no longer used

        if not isinstance(flow_direction, FlowDirection):
            # Wrap a bare Dataset as D8 for back-compat callers.
            flow_direction = FlowDirection.from_dataset(flow_direction, routing="d8")

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
