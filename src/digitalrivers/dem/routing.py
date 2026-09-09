"""D8 and multi-direction flow routing off a conditioned surface.

`flow_direction` is the hinge of the package: it turns an elevation grid into a
direction grid, after which everything downstream -- accumulation, streams, basins,
HAND -- is a walk over that grid rather than over elevations. Five schemes are offered
(`d8`, `dinf`, `mfd_quinn`, `mfd_holmgren`, `rho8`), and the result is a typed
`FlowDirection` that records which one produced it.

`flow_accumulation` counts upstream contributing cells. `accumulate_flow` is the legacy
recursive per-cell walk, kept for compatibility; the iterative kernels in
`flow._kernels.accumulation` are what the typed classes use.

Direction codes follow the `DIR_OFFSETS` convention from
`digitalrivers.core.directions`: `0=S, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE`.
"""

from __future__ import annotations

import warnings
import numpy as np
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.core.directions import DIR_OFFSETS
from digitalrivers.flow._kernels.routing import (
    dinf_flow_direction as _dinf_flow_direction,
    mfd_flow_direction as _mfd_flow_direction,
    rho8_flow_direction as _rho8_flow_direction,
)
from digitalrivers.flow.direction import FlowDirection

__all__ = ["RoutingMixin"]


class RoutingMixin:
    """Flow-direction and accumulation methods, mixed into :class:`DEM`."""

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
            plain_ds = Dataset.from_array(
                arr,
                geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
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
            plain_ds = Dataset.from_array(
                arr.astype(np.int32, copy=False),
                geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
                no_data_value=self.default_no_data_value,
            )
            return FlowDirection.from_dataset(plain_ds, routing="rho8")

        if method == "dinf":
            angle, magnitude = _dinf_flow_direction(elev, self.cell_size)
            stacked = np.stack([angle, magnitude], axis=0).astype(
                np.float32, copy=False
            )
            plain_ds = Dataset.from_array(
                stacked,
                geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
                no_data_value=self.default_no_data_value,
            )
            return FlowDirection.from_dataset(plain_ds, routing="dinf")

        # mfd_quinn or mfd_holmgren
        slopes = self._get_8_direction_slopes()
        weighting = "quinn" if method == "mfd_quinn" else "holmgren"
        fractions = _mfd_flow_direction(
            slopes,
            valid_mask,
            weighting=weighting,
            exponent=exponent,
        )
        # Transpose (rows, cols, 8) -> (8, rows, cols) for pyramids's band-first layout.
        bands = np.transpose(fractions, (2, 0, 1)).astype(np.float32, copy=False)
        plain_ds = Dataset.from_array(
            bands,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
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
        *,
        engine: str = "auto",
        out_path: str | None = None,
        tile_size: int | tuple[int, int] = 2048,
        cache: str = "evict",
        workers: int = 1,
        scheduler: str = "threads",
        client=None,
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> fd = dem.flow_direction(method="d8")
                >>> acc = dem.flow_accumulation(fd)
                >>> int(acc.read_array().max()) > 0
                True

            - A D∞ `FlowDirection` triggers the truncation warning:

                >>> import warnings
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.array(
                ...     [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9]],
                ...     dtype=np.float32,
                ... )
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
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
        if not isinstance(flow_direction, FlowDirection):
            # Wrap a bare Dataset as D8 for back-compat callers.
            flow_direction = FlowDirection.from_dataset(flow_direction, routing="d8")

        from digitalrivers._outofcore.engine import resolve_engine  # lazy

        if resolve_engine(engine, self.rows, self.columns, k=6) == "tiled":
            # Out-of-core path: stream the float32 Accumulation to disk; no whole-array int32 cast (that would
            # defeat the larger-than-RAM goal). Routing / out_path guards live in FlowDirection.accumulate.
            return flow_direction.accumulate(
                weights=weights,
                engine="tiled",
                out_path=out_path,
                tile_size=tile_size,
                cache=cache,
                workers=workers,
                scheduler=scheduler,
                client=client,
            )

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
        return Dataset.from_array(
            arr,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
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
