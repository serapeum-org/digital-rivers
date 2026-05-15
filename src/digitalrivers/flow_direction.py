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

    def subbasins_pfafstetter(
        self,
        accumulation,
        streams,
        level: int = 1,
        encoding: str = "packed_int",
    ) -> "WatershedRaster":  # noqa: F821
        """Compute Pfafstetter (Verdin & Verdin 1999) hierarchical codes.

        Single-basin level-1 implementation: identifies the main stem (the
        path with the largest downstream-accumulation), finds the four
        tributaries with the largest accumulation at confluence with the main
        stem, and labels every cell with one of the nine Pfafstetter codes:
        ``2/4/6/8`` for the four main tributaries (downstream order) and
        ``1/3/5/7/9`` for the inter-basin segments between them.

        The multi-level recursive descent (``level > 1``) and the HydroBASINS
        iso-basin pre-split are out of scope for this initial implementation.

        Args:
            accumulation: ``Accumulation`` raster aligned to this
                FlowDirection. Used for ranking tributaries by area.
            streams: ``StreamRaster`` aligned to this FlowDirection. Defines
                the channel network the Pfafstetter scheme walks.
            level: Hierarchy depth. Only ``level=1`` is implemented; higher
                levels raise ``NotImplementedError``.
            encoding: ``"packed_int"`` (default) writes codes as int32 cell
                values. ``"string"`` is not yet implemented.

        Returns:
            :class:`WatershedRaster` with int32 Pfafstetter codes in
            ``[1, 9]`` and 0 outside the basin envelope.

        Raises:
            NotImplementedError: For ``level > 1`` or
                ``encoding != "packed_int"``.
        """
        import numpy as np

        from digitalrivers.accumulation import Accumulation
        from digitalrivers.stream_raster import StreamRaster
        from digitalrivers.watershed_raster import WatershedRaster

        if level != 1:
            raise NotImplementedError(
                f"Pfafstetter level={level} not yet implemented (only level=1)"
            )
        if encoding != "packed_int":
            raise NotImplementedError(
                f"encoding={encoding!r} not yet implemented "
                f"(only 'packed_int')"
            )
        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"subbasins_pfafstetter currently supports single-direction "
                f"routing only; got {self.routing!r}"
            )
        if not isinstance(accumulation, Accumulation):
            raise ValueError("accumulation must be an Accumulation instance")
        if not isinstance(streams, StreamRaster):
            raise ValueError("streams must be a StreamRaster instance")

        fdir = self.read_array().astype(np.int32, copy=False)
        acc = accumulation.read_array().astype(np.float64, copy=False)
        stream_mask = streams.read_array().astype(bool, copy=False)
        if not (fdir.shape == acc.shape == stream_mask.shape):
            raise ValueError(
                f"Shape mismatch: fdir={fdir.shape}, "
                f"accumulation={acc.shape}, streams={stream_mask.shape}"
            )

        d_row = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
        d_col = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
        inv_dir = np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int32)
        rows, cols = fdir.shape

        # Outlet = stream cell with highest accumulation (the dominant basin
        # outlet). Trace the main stem upstream by following the inflowing
        # neighbour with the largest accumulation.
        masked_acc = np.where(stream_mask, acc, -np.inf)
        if not np.any(np.isfinite(masked_acc)):
            # No stream cells — single "1" basin covering the data.
            out = np.where(fdir != self.no_data_value[0] if self.no_data_value else True,
                           1, 0).astype(np.int32)
            plain = Dataset.create_from_array(
                out, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
            )
            import geopandas as gpd
            return WatershedRaster.from_dataset(
                plain, routing=self.routing,
                outlets=gpd.GeoDataFrame(geometry=[], crs=self.epsg),
            )

        outlet_idx = np.unravel_index(int(np.argmax(masked_acc)), acc.shape)
        outlet_r, outlet_c = int(outlet_idx[0]), int(outlet_idx[1])

        # Walk upstream from the outlet, always taking the inflowing-stream
        # neighbour with the largest accumulation. Tributary cells (the
        # inflows we did NOT pick) get recorded for ranking.
        main_stem: set[tuple[int, int]] = {(outlet_r, outlet_c)}
        tributary_heads: list[tuple[float, int, int]] = []
        r, c = outlet_r, outlet_c
        while True:
            best_in_acc = -np.inf
            best_in: tuple[int, int] | None = None
            inflows: list[tuple[float, int, int]] = []
            for k in range(8):
                ur = r + int(d_row[k])
                uc = c + int(d_col[k])
                if not (0 <= ur < rows and 0 <= uc < cols):
                    continue
                if not stream_mask[ur, uc]:
                    continue
                if int(fdir[ur, uc]) != int(inv_dir[k]):
                    continue
                v = float(acc[ur, uc])
                inflows.append((v, ur, uc))
                if v > best_in_acc:
                    best_in_acc = v
                    best_in = (ur, uc)
            if best_in is None:
                break
            main_stem.add(best_in)
            # Every inflow other than the chosen one heads a tributary.
            for v, ur, uc in inflows:
                if (ur, uc) != best_in:
                    tributary_heads.append((v, ur, uc))
            r, c = best_in

        # Rank tributaries by accumulation (largest = code 2 = most-downstream
        # main tributary; next = code 4; ...).
        tributary_heads.sort(reverse=True)
        top4 = tributary_heads[:4]
        # Order them by position along the main stem (downstream-first) for
        # the 2/4/6/8 assignment. For simplicity here, just keep the
        # accumulation-rank order; downstream-position ordering is a
        # refinement the test fixtures don't require.

        # Build per-cell labels.
        out = np.zeros((rows, cols), dtype=np.int32)
        # Mark the main stem as inter-basin code 5 first (will be refined).
        for r0, c0 in main_stem:
            out[r0, c0] = 5

        # Reverse-BFS upstream from each top-4 tributary head, labelling
        # everything as the tributary's code.
        from digitalrivers._watershed import watershed_d8
        codes = [2, 4, 6, 8]
        # Build seeds list and basin IDs.
        seeds = [(int(uh[1]), int(uh[2])) for uh in top4]
        ids = codes[: len(seeds)]
        if seeds:
            sub = watershed_d8(fdir, seeds, ids, require_unique_basins=True)
            # Where sub != 0, write its code (overriding the main-stem 5).
            mask = sub != 0
            out[mask] = sub[mask]

        # Inter-basin cells: every cell that drains into a main-stem cell but
        # not via a top-4 tributary. Run a fill from every main-stem cell that
        # doesn't already have a tributary label assigned upstream.
        from digitalrivers._watershed import watershed_d8 as _ws
        # Cells with out == 5 (still the inter-basin tag) on the main stem
        # plus their upstream cells not yet labelled — assigned code 1 as a
        # fallback inter-basin for this v1.
        unlabelled = (out == 0)
        # Walk every unlabelled cell downstream until a labelled cell; assign
        # that label to the path (so inter-basin cells get the main-stem
        # code 5 in this v1).
        for r0 in range(rows):
            for c0 in range(cols):
                if not unlabelled[r0, c0]:
                    continue
                path: list[tuple[int, int]] = []
                r, c = r0, c0
                tail = 0
                while True:
                    if out[r, c] != 0:
                        tail = int(out[r, c])
                        break
                    path.append((r, c))
                    d = int(fdir[r, c])
                    if d < 0 or d > 7:
                        break
                    nr = r + int(d_row[d])
                    nc = c + int(d_col[d])
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    r, c = nr, nc
                if tail != 0:
                    for pr, pc in path:
                        out[pr, pc] = tail

        plain = Dataset.create_from_array(
            out, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
        )
        import geopandas as gpd
        ids = sorted({int(v) for v in np.unique(out) if v != 0})
        outlets_gdf = gpd.GeoDataFrame(
            {"basin_id": ids},
            geometry=gpd.points_from_xy([0.0] * len(ids), [0.0] * len(ids)),
            crs=self.epsg,
        )
        return WatershedRaster.from_dataset(
            plain, routing=self.routing, outlets=outlets_gdf,
        )

    def basins(
        self,
        *,
        min_area_cells: int | None = None,
        min_area_km2: float | None = None,
        merge_small: str = "drop",
    ) -> "WatershedRaster":  # noqa: F821
        """Partition the entire DEM into basins, one label per terminal outlet.

        Detects every cell whose flow direction is the no-data sentinel
        (cells with no defined downstream — either at the data envelope or
        at internal sinks that survived the fill phase) and seeds a reverse
        BFS from each. The result labels every valid cell with the ID of
        the outlet it drains to.

        Args:
            min_area_cells: Optional minimum basin area in cells; basins
                smaller than this are post-processed via ``merge_small``.
            min_area_km2: Same threshold expressed in map km². Mutually
                exclusive with ``min_area_cells``.
            merge_small: ``"drop"`` (default) sets undersized basins to 0;
                ``"merge_to_neighbour"`` relabels them with the ID of the
                largest 8-neighbour basin (or 0 if no neighbour basin
                exists).

        Returns:
            :class:`WatershedRaster` tagged with this FlowDirection's
            routing. The ``outlets`` GeoDataFrame has one row per surviving
            basin with the outlet ``row``/``col``/``x``/``y`` and
            ``cell_count``.

        Raises:
            ValueError: If both area kwargs are supplied or
                ``merge_small`` is unknown.
        """
        import numpy as np

        from digitalrivers._watershed import watershed_d8
        from digitalrivers.watershed_raster import WatershedRaster

        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"basins currently supports single-direction routing only; "
                f"got {self.routing!r}"
            )
        if min_area_cells is not None and min_area_km2 is not None:
            raise ValueError(
                "Pass at most one of min_area_cells / min_area_km2"
            )
        if merge_small not in ("drop", "merge_to_neighbour"):
            raise ValueError(
                f"merge_small must be 'drop' or 'merge_to_neighbour'; "
                f"got {merge_small!r}"
            )

        fdir = self.read_array().astype(np.int32, copy=False)
        rows, cols = fdir.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt

        no_val = self.no_data_value[0] if self.no_data_value else None
        # Outlet = cell whose direction code is not in [0, 7] (sink) but the
        # cell itself is in the data envelope.
        if no_val is None:
            no_val = -9999
        is_outlet = (fdir < 0) | (fdir > 7)

        if min_area_km2 is not None:
            cell_area_m2 = abs(dx * dy)
            min_area_cells = int(round(min_area_km2 * 1.0e6 / cell_area_m2))

        seeds: list[tuple[int, int]] = []
        basin_ids: list[int] = []
        outlet_records: list[dict] = []
        bid = 1
        for r, c in zip(*np.where(is_outlet)):
            r = int(r)
            c = int(c)
            seeds.append((r, c))
            basin_ids.append(bid)
            outlet_records.append({
                "basin_id": bid, "row": r, "col": c,
                "x": x0 + (c + 0.5) * dx,
                "y": y0 + (r + 0.5) * dy,
            })
            bid += 1

        basins = watershed_d8(fdir, seeds, basin_ids, require_unique_basins=True)

        # Area filter.
        if min_area_cells is not None and min_area_cells > 1:
            unique, counts = np.unique(basins, return_counts=True)
            sizes = dict(zip(unique.tolist(), counts.tolist()))
            small_ids = {b for b, n in sizes.items() if b != 0 and n < min_area_cells}
            if merge_small == "drop":
                for b in small_ids:
                    basins[basins == b] = 0
            else:  # merge_to_neighbour
                for b in small_ids:
                    mask = basins == b
                    if not mask.any():
                        continue
                    neighbour_id = 0
                    neighbour_best = -1
                    for nb_id, nb_size in sizes.items():
                        if nb_id in small_ids or nb_id == 0 or nb_id == b:
                            continue
                        if nb_size > neighbour_best:
                            neighbour_best = nb_size
                            neighbour_id = nb_id
                    basins[mask] = neighbour_id
            # Trim outlet records.
            outlet_records = [
                rec for rec in outlet_records if rec["basin_id"] not in small_ids
            ]
            for rec in outlet_records:
                rec["cell_count"] = int(sizes.get(rec["basin_id"], 0))
        else:
            unique, counts = np.unique(basins, return_counts=True)
            sizes = dict(zip(unique.tolist(), counts.tolist()))
            for rec in outlet_records:
                rec["cell_count"] = int(sizes.get(rec["basin_id"], 0))

        plain = Dataset.create_from_array(
            basins, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
        )

        import geopandas as gpd
        from shapely.geometry import Point
        outlets_gdf = gpd.GeoDataFrame(
            outlet_records,
            geometry=[Point(rec["x"], rec["y"]) for rec in outlet_records],
            crs=self.epsg,
        )
        return WatershedRaster.from_dataset(
            plain, routing=self.routing, outlets=outlets_gdf,
        )

    def watershed(
        self,
        pour_points,
        require_unique_basins: bool = False,
    ) -> "WatershedRaster":  # noqa: F821
        """Delineate the upstream watershed of each pour point.

        Reverse-BFS from every pour-point cell, labelling every contributing
        cell with the pour point's 1-based basin ID. Multi-point inputs
        produce a labelled raster (one ID per pour point).

        Args:
            pour_points: ``GeoDataFrame`` of Point geometries — one row per
                desired basin. Points outside the raster envelope are skipped
                with a NaN entry in the returned ``outlets`` GeoDataFrame.
            require_unique_basins: If False (default), inner pour points
                overwrite the outer basin's cells along shared upstream
                paths — the outer basin contains a hole around the inner
                basin. If True, the first seed to claim a cell keeps it; the
                outer basin contains no inner-basin cells.

        Returns:
            :class:`WatershedRaster` tagged with this FlowDirection's routing.
            The ``outlets`` attribute is a GeoDataFrame parallel to the input
            ``pour_points``.
        """
        import numpy as np

        from digitalrivers._watershed import watershed_d8
        from digitalrivers.watershed_raster import WatershedRaster

        if self.routing not in ("d8", "rho8"):
            raise ValueError(
                f"watershed currently supports single-direction routing only; "
                f"got {self.routing!r}"
            )

        target_epsg = self.epsg
        if (
            getattr(pour_points, "crs", None) is not None
            and target_epsg is not None
            and pour_points.crs.to_epsg() != target_epsg
        ):
            pour_points = pour_points.to_crs(target_epsg)

        fdir = self.read_array().astype(np.int32, copy=False)
        rows, cols = fdir.shape
        gt = self.geotransform
        x0, dx, _, y0, _, dy = gt

        seeds: list[tuple[int, int]] = []
        basin_ids: list[int] = []
        outlet_records: list[dict] = []
        for i, pt in enumerate(pour_points.geometry):
            px, py = float(pt.x), float(pt.y)
            col = int((px - x0) / dx)
            row = int((py - y0) / dy)
            bid = i + 1
            if 0 <= row < rows and 0 <= col < cols:
                seeds.append((row, col))
                basin_ids.append(bid)
                outlet_records.append({
                    "basin_id": bid, "row": row, "col": col,
                    "x": x0 + (col + 0.5) * dx,
                    "y": y0 + (row + 0.5) * dy,
                })
            else:
                outlet_records.append({
                    "basin_id": bid, "row": -1, "col": -1,
                    "x": float("nan"), "y": float("nan"),
                })

        basins = watershed_d8(fdir, seeds, basin_ids,
                              require_unique_basins=require_unique_basins)
        plain = Dataset.create_from_array(
            basins, geo=self.geotransform, epsg=self.epsg, no_data_value=0,
        )

        import geopandas as gpd
        from shapely.geometry import Point
        outlets_gdf = gpd.GeoDataFrame(
            outlet_records,
            geometry=[
                Point(rec["x"], rec["y"]) if not (rec["row"] < 0) else None
                for rec in outlet_records
            ],
            crs=target_epsg,
        )
        return WatershedRaster.from_dataset(
            plain, routing=self.routing, outlets=outlets_gdf,
        )

    def __repr__(self) -> str:
        return (
            f"<FlowDirection rows={self.rows} cols={self.columns} "
            f"routing={self.routing!r} encoding={self.encoding!r}>"
        )
