"""DEM conditioning: everything that edits the surface before it is routed.

Raw elevation is rarely ready for hydrology. Depressions trap flow, flat plateaus leave
direction undefined, and mapped drainage the DEM never resolved -- culverts under
embankments, streams narrower than a cell -- has to be forced into the surface. The
methods here are the edits that make a DEM routable:

* **Depression removal** -- `fill_depressions` raises pits to their spill elevation;
  `breach_depressions` cuts a channel through the barrier instead, preserving storage
  volume at the cost of carving the DEM; `resolve_flats` gives the resulting plateaus a
  gradient to drain along.
* **Feature enforcement** -- `burn_streams`, `enforce_culverts`, `enforce_breaklines`,
  `burn_buildings` and `hydroflatten` push known drainage or blockage into the surface.
* **Uncertainty** -- `stochastic_depressions` perturbs the surface repeatedly to
  separate depressions that are real from ones that are noise artefacts.

Every method that edits the surface appends its name to the semicolon-separated
`DR_CONDITIONING` GeoTIFF tag, so a conditioned DEM carries its own history and
downstream code can decline to re-condition it.

The array algorithms live in :mod:`digitalrivers.dem._kernels`; what is here is the
`Dataset` plumbing around them.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import shapely
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.dem._kernels.breach import (
    breach_depressions as _breach_depressions_array,
)
from digitalrivers.dem._kernels.flats import resolve_flats as _resolve_flats_array
from digitalrivers.dem._kernels.pitremoval import (
    fill_depressions as _fill_depressions_array,
)

if TYPE_CHECKING:
    from digitalrivers.dem.dem import DEM

__all__ = ["ConditioningMixin", "_reproject_if_needed"]


def _dem_cls():
    """Return the `DEM` class.

    Circular import: `DEM` mixes this class in, so it cannot be named at module scope.
    The conditioning methods hand back a new `DEM` built from a conditioned raster, so
    the lookup is deferred to call time.
    """
    from digitalrivers.dem.dem import DEM

    return DEM


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


class ConditioningMixin:
    """Surface-editing methods, mixed into :class:`DEM`.

    The mixin declares no state: `values`, `geotransform`, `epsg`, `no_data_value` and
    `raster` all come from `Dataset`.
    """

    def fill_depressions(
        self,
        method: str = "priority_flood",
        epsilon: float = 0.0,
        inplace: bool = False,
        *,
        engine: str = "auto",
        out_path: str | None = None,
        tile_size: int | tuple[int, int] = 2048,
        cache: str = "evict",
        workers: int = 1,
        scratch_dir: str | None = None,
        eps_fill: str = "exact",
    ) -> DEM | None:
        """Fill closed depressions in the DEM.

        Three algorithms are available via the `method` argument:

        * `"priority_flood"` (default) — Barnes, Lehman & Mulla (2014) Priority-Flood
          with the two-queue plateau optimisation. With `epsilon == 0` it produces flat
          fills; with `epsilon > 0` it produces a monotonic surface (every cell has at
          least one lower neighbour along the drainage path) at the cost of a small
          elevation inflation proportional to plateau width. The `epsilon > 0` gradient is
          selected by `eps_fill` (see below).
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
                remain flat. Positive values impose a downhill path at the cost of
                slight elevation inflation. `planchon_darboux` requires `epsilon > 0`.
            inplace: If `True` the current instance is updated in place and `None`
                is returned. If `False` (default) a new `DEM` is returned.
            engine: `"auto"` (default) runs the in-memory algorithm and switches to the
                tiled one only for rasters large enough to risk exhausting RAM;
                `"tiled"` forces the streaming path and requires `out_path`;
                `"in_memory"` forces the whole-array one.
            out_path: Destination GeoTIFF the tiled path streams its result into.
                Required whenever the tiled engine runs, ignored otherwise.
            tile_size: Core tile shape for the tiled path, as an edge length or an
                explicit `(rows, columns)`. Defaults to 2048; keep it at `>= 512`,
                because the master graph's label count scales with total tile perimeter.
            cache: Tile-store mode for the tiled path. `"evict"` (default) keeps no
                per-tile interiors and recomputes them in the finalize stage, `"retain"`
                holds them in RAM, and `"cache"` spills them to `scratch_dir`. Ignored by
                the in-memory path.
            workers: Tiled path only. `> 1` runs the per-tile passes through the dask
                backend; only the `epsilon == 0` path is parallelised, `epsilon > 0` runs
                serially either way. Defaults to 1.
            scratch_dir: Directory the tiled path spills `.npy` tiles into. Required by
                (and only used for) `cache="cache"`; it is created if it does not exist.
            eps_fill: Gradient for `priority_flood` with `epsilon > 0` (ignored otherwise).
                `"exact"` (default) / `"monotone"` use the deterministic exit-distance
                ramp that is **identical in-memory and tiled** (so `engine="auto"` is
                consistent); it is flat-free for small `epsilon`. `"barnes"` uses the
                classic Priority-Flood step-count — flat-free for any `epsilon` but
                in-memory only (it is not tileable; `engine="tiled"` rejects it).

        Returns:
            DEM | None: A new `DEM` containing the filled elevation, or `None` when
            `inplace` is `True`.

        Raises:
            ValueError: If `method` is unknown, if `planchon_darboux` is requested with
                `epsilon <= 0`, if the tiled engine runs without an `out_path`, if
                `engine="tiled"` is combined with `inplace=True`, or if `cache="cache"`
                is used without a `scratch_dir`.
            NotImplementedError: If the tiled engine is asked for `eps_fill="barnes"`
                with `epsilon > 0`, which is not tileable.

        Out-of-core:
            `engine="auto"` (default) runs the in-memory algorithm unless the DEM is large enough to risk
            exhausting RAM, in which case it streams a tiled Barnes-2016 fill to `out_path`. Force the path with
            `engine="in_memory"` / `engine="tiled"`. `engine="tiled"` requires `out_path` and does not support
            `inplace`. For `epsilon>0`, `eps_fill="exact"` (default, alias `"monotone"`) produces a tiled fill
            **byte-for-byte identical** to the in-memory result; `eps_fill="barnes"` (the classic step-count) is
            in-memory only and is rejected by `engine="tiled"`.

        Examples:
            - The default priority-flood fill raises a one-cell pit to its rim, so the
              filled depression is flat:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((3, 3), 5.0, dtype=np.float64)
                >>> z[1, 1] = 1.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).fill_depressions()
                >>> float(filled.values[1, 1])
                5.0

            - A positive `epsilon` lifts the pit above its rim instead, leaving D8
              routing a downhill path out of the fill:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((3, 3), 5.0, dtype=np.float64)
                >>> z[1, 1] = 1.0
                >>> ds = Dataset.from_array(
                ...     z,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> filled = DEM(ds.raster).fill_depressions(epsilon=0.5)
                >>> float(filled.values[1, 1])
                5.5

            - `inplace=True` rewrites the instance and returns nothing:

                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((3, 3), 5.0, dtype=np.float64)
                >>> z[1, 1] = 1.0
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
                >>> dem.fill_depressions(epsilon=0.5, inplace=True) is None
                True
                >>> float(dem.values[1, 1])
                5.5

        See Also:
            DEM.breach_depressions: Cuts a channel through the barrier instead of raising
                the pit floor.
            DEM.fill_sinks: Deprecated alias that routes here with
                `method="priority_flood", epsilon=0.1`.
        """
        from digitalrivers._outofcore.engine import (  # lazy: keeps import digitalrivers light
            require_out_path,
            resolve_engine,
        )

        resolved = resolve_engine(engine, self.rows, self.columns, k=8)
        if resolved == "tiled":
            require_out_path(engine, out_path)
            if inplace:
                raise ValueError(
                    "engine='tiled' streams to disk; inplace=True is not supported"
                )
            # lazy import: pulls the numba kernels only when the tiled path is actually used
            from digitalrivers._outofcore.fill import fill_depressions_tiled

            tile_rows, tile_cols = (
                (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
            )
            out = fill_depressions_tiled(
                self,
                out_path,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                epsilon=epsilon,
                cache=cache,
                workers=workers,
                scratch_dir=scratch_dir,
                eps_fill=eps_fill,
            )
            return _dem_cls()(out.raster)

        # Read at native precision via read_array, not self.values: the latter downcasts float64 rasters to
        # float32, which would make the in-memory fill disagree with the native-precision tiled engine (the
        # epsilon>0 ramp is shared, so engine="auto" must give identical results). Build the no-data mask from the
        # sentinel here (read_array returns the raw sentinel, not NaN) to match the tiled path.
        native = np.asarray(self.read_array())
        no_val = self.no_data_value[0]
        nodata_mask = np.isnan(native)
        if no_val is not None and not np.isnan(no_val):
            nodata_mask = nodata_mask | (native == no_val)
        elev = native.astype(np.float64, copy=True)
        elev[nodata_mask] = np.nan
        z_fill = _fill_depressions_array(
            elev,
            nodata_mask=nodata_mask,
            method=method,
            epsilon=epsilon,
            eps_fill=eps_fill,
        )
        # Restore the original raster's no-data sentinel (the array carries NaN; the GeoTIFF needs the sentinel).
        if no_val is not None:
            z_fill[nodata_mask] = no_val

        # For epsilon>0 the fill is a fractional ramp, so never emit an integer output dtype (it would truncate
        # the gradient and silently collapse the fill to a flat). Floating DEMs keep their native precision;
        # integer DEMs (e.g. int16 SRTM) are promoted to float64. epsilon=0 fills keep the native dtype.
        out_dt = native.dtype
        if epsilon > 0.0 and not np.issubdtype(native.dtype, np.floating):
            out_dt = np.float64
        # Build a plain Dataset (cls=Dataset so we don't get a DEM via cls(...)), then
        # wrap with the typed DEM. This mirrors the pattern used in flow_direction().
        plain_ds = Dataset.dataset_like(self, z_fill.astype(out_dt, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return _dem_cls()(plain_ds.raster)

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
        return _dem_cls()(plain_ds.raster)

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
        return _dem_cls()(plain_ds.raster)

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
        # Pre-compute the no-data mask once — it doesn't change between
        # realisations since the noise only perturbs valid cells.
        nodata_mask = np.isnan(elev)
        prob = np.zeros(elev.shape, dtype=np.float32)
        for _ in range(int(n_runs)):
            noise = rng.normal(0.0, sigma, size=elev.shape).astype(
                elev.dtype, copy=False
            )
            noisy = elev + noise
            # Call the kernel directly — no GDAL Dataset wrapping inside the
            # Monte-Carlo loop. The kernel accepts a plain ndarray and an
            # optional nodata_mask.
            filled = _fill_depressions_array(
                noisy,
                nodata_mask=nodata_mask,
                method=method,
            )
            depr = (filled - noisy) > 0
            prob += depr.astype(np.float32)
        prob /= float(n_runs)
        return Dataset.from_array(
            prob,
            geo_ref=GeoReference(geo=self.geotransform, epsg=self.epsg),
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

        Implements all three P20 methods. `"fill_burn"` (Lindsay 2018 —
        used by WhiteboxTools' FillBurn) is the default; `"agree"`
        (Hellweger 1997) and `"topological_breach"` (Lindsay 2016) are
        also available.

        Fill-burn algorithm:

        1. Rasterise every LineString in `streams` onto a stream mask.
        2. Lower every stream cell's elevation by `constant_drop`.
        3. Run `fill_depressions(method="priority_flood")` so the
           surrounding cells drain naturally into the channel.

        AGREE algorithm: lower stream cells by `sharp`, then ramp the drop
        linearly from `sharp` at the channel to `0` at a `buffer_cells`-wide
        perimeter (offset by `smooth`), producing a smooth trench.

        Topological-breach algorithm: rasterise and lower the stream cells,
        then run the Phase 1 least-cost breach so every interior pit carves
        outward to a stream cell, honouring `max_breach_depth` /
        `max_breach_length`.

        Args:
            streams: `GeoDataFrame` of LineString geometries.
            method: `"fill_burn"` (default), `"agree"`, or
                `"topological_breach"`. Any other value raises
                `NotImplementedError`.
            sharp / smooth / buffer_cells: AGREE parameters (unused for
                fill_burn and topological_breach).
            constant_drop: Elevation drop applied to every stream cell in
                fill_burn and topological_breach (default 1.0 map unit;
                unused for agree).
            max_breach_depth / max_breach_length: topological_breach
                parameters (unused for fill_burn and agree).
            inplace: If True, update the instance; else return a new DEM.

        Returns:
            DEM | None: New DEM with the conditioned surface, or None when
            `inplace=True`.

        Examples:
            - Fill-burn lowers the stream-row of a flat DEM:

                >>> import numpy as np
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> z = np.full((5, 5), 10.0, dtype=np.float32)
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
            frontier: deque[tuple[int, int, int]] = deque(
                (int(r), int(c), 0) for r, c in zip(*np.nonzero(stream_mask))
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
            return self._conditioned_result(elev, z, inplace)

        if method == "topological_breach":
            # Lindsay 2016: rasterise the stream network onto the DEM
            # (like fill_burn but without the final priority-flood), then
            # invoke the Phase 1 least-cost breach engine so every internal
            # pit Dijkstras outward toward a stream cell. The burned stream
            # cells sit max_breach_depth-or-equivalent below their
            # surroundings, so the breach paths follow the vector topology
            # by construction.
            elev, z, nodata_mask = self._burn_stream_cells(streams, constant_drop)
            z = _breach_depressions_array(
                z,
                nodata_mask=nodata_mask,
                method="hybrid",
                max_depth=max_breach_depth,
                max_length=max_breach_length,
                fill_remaining=True,
            )
            return self._conditioned_result(elev, z, inplace)

        if method != "fill_burn":
            raise NotImplementedError(
                f"method={method!r} not yet implemented (supported: "
                "'fill_burn', 'agree', 'topological_breach')"
            )

        elev, z, nodata_mask = self._burn_stream_cells(streams, constant_drop)
        z = _fill_depressions_array(
            z,
            nodata_mask=nodata_mask,
            method="priority_flood",
            epsilon=0.0,
        )
        return self._conditioned_result(elev, z, inplace)

    def _burn_stream_cells(self, streams, constant_drop: float):
        """Rasterise a stream network and lower the cells it covers.

        The step `burn_streams` shares across its methods: reproject the network
        to the DEM's CRS when needed, rasterise it through the same 2x
        oversampled floor-rasteriser `enforce_breaklines` and `enforce_culverts`
        use — so all three snap line samples to cells identically — then subtract
        `constant_drop` from every cell the network touches.

        Args:
            streams: `GeoDataFrame` of LineString geometries.
            constant_drop: Elevation subtracted from each stream cell, in map
                units.

        Returns:
            Tuple `(elev, z, nodata_mask)` — the original elevations, the lowered
            copy as `float64`, and the mask of its no-data cells, which the
            conditioning kernels take as their gap mask.
        """
        elev = self.values
        gt = self.geotransform
        stream_mask = np.zeros(elev.shape, dtype=bool)
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
        return elev, z, np.isnan(z)

    def _conditioned_result(self, elev, z, inplace: bool, *, gaps=None):
        """Return a conditioned surface as a `DEM`, or apply it in place.

        The tail every conditioning operation shares: restore the raster's own
        no-data sentinel over the cells the kernel left empty and cast back to
        the source dtype, so conditioning never silently widens the band.

        Args:
            elev: The original elevation array, read for its dtype.
            z: The conditioned surface, modified in place.
            inplace: Update this instance instead of returning a new `DEM`.
            gaps: Boolean mask of the cells to stamp as no-data. Defaults to
                `np.isnan(z)`; pass `~np.isfinite(z)` for a kernel that can also
                produce infinities.

        Returns:
            A new `DEM`, or `None` when `inplace` is True.
        """
        z[np.isnan(z) if gaps is None else gaps] = self.no_data_value[0]
        plain_ds = Dataset.dataset_like(self, z.astype(elev.dtype, copy=False))
        if inplace:
            self._update_inplace(plain_ds.raster)
            return None
        return _dem_cls()(plain_ds.raster)

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
        return self._conditioned_result(elev, z, inplace)

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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> ds = Dataset.from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> ds = Dataset.from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
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
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalrivers import DEM
                >>> ds = Dataset.from_array(
                ...     np.zeros((5, 5), dtype=np.float32),
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0.0, 0.0),
                ...         cell_size=1.0,
                ...         epsg=4326,
                ...     ),
                ...     no_data_value=-9999.0,
                ... )
                >>> dem = DEM(ds.raster)
                >>> poly = Polygon([(0, 0), (3, 0), (3, -3), (0, -3)])
                >>> rs, cs = dem._polygon_cell_indices(poly, dem.geotransform, 5, 5)
                >>> np.issubdtype(rs.dtype, np.integer)
                True
        """
        x0, dx, _, y0, _, dy = gt
        minx, miny, maxx, maxy = geom.bounds
        c_lo = max(0, int((minx - x0) / dx))
        c_hi = min(cols, int((maxx - x0) / dx) + 1)
        r_lo = max(0, int((maxy - y0) / dy))
        r_hi = min(rows, int((miny - y0) / dy) + 1)
        if c_lo >= c_hi or r_lo >= r_hi:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        rs_idx, cs_idx = np.meshgrid(
            np.arange(r_lo, r_hi),
            np.arange(c_lo, c_hi),
            indexing="ij",
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

        return self._conditioned_result(elev, z, inplace)

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
        rows, cols = elev.shape

        building_polygons = _reproject_if_needed(building_polygons, self.epsg)

        z = elev.astype(np.float64, copy=True)
        for geom in building_polygons.geometry:
            if geom is None or geom.is_empty:
                continue
            rs, cs = self._polygon_cell_indices(geom, gt, rows, cols)
            if rs.size:
                z[rs, cs] = z[rs, cs] + lift

        return self._conditioned_result(elev, z, inplace)

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
        return self._conditioned_result(elev, z, inplace)

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
        return self.fill_depressions(
            method="priority_flood", epsilon=0.1, inplace=inplace
        )
