"""DEM processing module.

This module provides the ``DEM`` class for digital elevation model analysis,
including sink filling, slope calculation, D8 flow direction, and flow
accumulation.
"""
from __future__ import annotations
import numpy as np
from osgeo import gdal
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset

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

    def fill_sinks(self, inplace: bool = False) -> Dataset | None:
        """Fill single-cell sinks in the elevation surface.

        A cell is considered a sink when its elevation is lower than all
        eight surrounding cells.  Each sink is raised to the minimum
        neighbour elevation plus 0.1 (in map units).

        Note:
            This is a single-pass algorithm.  Cascading sinks (a sink
            whose fill creates a new sink) may not be fully resolved.

        Args:
            inplace: If ``True`` the current instance is modified in
                place and ``None`` is returned.  If ``False`` (default) a
                new ``Dataset`` is returned.

        Returns:
            Dataset containing the sink-free elevation, or ``None`` when
            *inplace* is ``True``.
        """
        elev = self.values

        elev_sinkless = np.copy(elev)
        for i in range(1, self.rows - 1):
            for j in range(1, self.columns - 1):
                # Get elevation of surrounding cells
                f = elev[i - 1 : i + 2, j - 1 : j + 2].flatten()
                # Exclude the center cell
                f[4] = np.nan
                min_f = np.nanmin(f)
                if elev_sinkless[i, j] < min_f:
                    elev_sinkless[i, j] = min_f + 0.1

        src = self.dataset_like(self, elev_sinkless)
        if inplace:
            self._update_inplace(src.raster)
        else:
            return src

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

    def flow_direction(self, forced_direction: GeoDataFrame = None) -> "Dataset":
        """Derive the D8 flow-direction raster from the DEM.

        For each cell the direction with the steepest downhill slope is
        selected (``nanargmax`` over the eight neighbours).  Cells that
        are entirely ``NaN`` or have no valid slope receive the default
        no-data value.

        Args:
            forced_direction: Optional GeoDataFrame with columns
                ``geometry`` (point) and ``direction`` (int 0–7).  Cells
                at the given locations are overridden with the supplied
                direction regardless of the computed slope.

        Returns:
            Dataset: ``int32`` raster with cell values in ``{0 .. 7}``
                following the ``DIR_OFFSETS`` convention.  No-data cells
                are filled with ``Dataset.default_no_data_value``.
        """
        elev = self.values
        slopes = self._get_8_direction_slopes()
        # Create a mask for non-NaN cells in the elevation array
        mask = ~np.isnan(elev)

        # Create a mask for cells with at least one non-NaN slope
        valid_mask = ~np.all(np.isnan(slopes), axis=2)

        # Combine masks to identify cells where calculations should be done
        valid_cells_mask = mask & valid_mask

        # Initialize the flow_direction array with NaN values
        flow_direction = np.full(
            elev.shape, Dataset.default_no_data_value, dtype=np.int32
        )

        # Apply np.nanargmax only where the mask is True to get the index of the maximum slope
        # hence, the flow direction.
        flow_direction[valid_cells_mask] = np.nanargmax(
            slopes[valid_cells_mask], axis=1
        )

        if forced_direction is not None:
            indices = self.map_to_array_coordinates(forced_direction)
            for i, ind in enumerate(indices):
                flow_direction[tuple(ind)] = forced_direction.loc[i, "direction"]

        src = self.create_from_array(
            flow_direction, geo=self.geotransform, epsg=self.epsg,
            no_data_value=self.default_no_data_value,
        )
        return src

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

        # Pre-compute the opposite direction for each offset so we don't
        # re-derive it on every iteration.
        opposites = {}
        for d, (d_col, d_row) in dir_offsets.items():
            opp = self.opposite_direction(d_row, d_col, dir_offsets)
            opposites[d] = (d_col, d_row, opp)

        # Each stack frame: (row, col, neighbour_iterator_index, running_total)
        stack = [(r, c, 0, 0)]

        while stack:
            cr, cc, idx, total = stack[-1]

            # Already resolved — return cached value to caller.
            if acc[cr, cc] >= 0:
                stack.pop()
                if stack:
                    pr, pc, pidx, ptotal = stack[-1]
                    stack[-1] = (pr, pc, pidx, ptotal + acc[cr, cc] + 1)
                continue

            offsets_list = list(opposites.values())

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
        self, flow_direction: "DEM", dir_offsets: dict = None
    ) -> "Dataset":
        """Compute the flow-accumulation raster from a flow-direction grid.

        Each cell's value represents the number of upstream cells that
        drain into it (not counting itself).  The algorithm iterates
        over every valid cell and calls ``accumulate_flow``.

        Args:
            flow_direction: A ``DEM`` (or ``Dataset``) containing the D8
                flow-direction raster with values in ``{0 .. 7}``.
            dir_offsets: Direction-offset mapping.  Defaults to the
                module-level ``DIR_OFFSETS``.

        Returns:
            Dataset: ``int32`` raster where each cell holds its upstream
                count.  No-data cells retain
                ``Dataset.default_no_data_value``.
        """
        if dir_offsets is None:
            dir_offsets = DIR_OFFSETS

        fd_array = flow_direction.read_array()
        rows, cols = fd_array.shape
        acc = np.full((rows, cols), Dataset.default_no_data_value, dtype=np.int32)
        elev = self.values
        # Initialize with -1 to indicate unprocessed cells
        acc[~np.isnan(elev)] = -1

        for i in range(rows):
            for j in range(cols):
                if acc[i, j] == -1:  # Only process unprocessed cells
                    self.accumulate_flow(i, j, fd_array, acc, dir_offsets)

        src = self.create_from_array(
            acc, geo=self.geotransform, epsg=self.epsg,
            no_data_value=self.default_no_data_value,
        )
        return src

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
        no_columns = self.columns
        no_rows = self.rows

        flow_direction = self.flow_direction()
        flow_dir = flow_direction.read_array(band=0).astype(np.float32)
        no_val = flow_direction.no_data_value[0]
        flow_dir[np.isclose(flow_dir, no_val, rtol=0.00001)] = np.nan
        # convert index of the flow direction to the index of the cell
        flow_direction_cell = np.ones((no_rows, no_columns, 2)) * np.nan

        for i in range(no_rows):
            for j in range(no_columns):
                if not np.isnan(flow_dir[i, j]):
                    ind = int(flow_dir[i, j])
                    indices = DIR_OFFSETS[ind]
                    flow_direction_cell[i, j, 0] = i + indices[0]
                    flow_direction_cell[i, j, 1] = j + indices[1]

        return flow_direction_cell


    @staticmethod
    def delete_basins(basins: gdal.Dataset, path: str):
        """Keep only the first (largest) basin and discard the rest.

        Reads a basin-ID raster produced during catchment delineation,
        replaces every cell that does not belong to the first basin with
        the no-data value, and writes the result to *path*.

        Args:
            basins: GDAL dataset whose cell values are basin IDs
                (integers).  The first unique basin ID found (excluding
                no-data) is retained.
            path: Output GeoTIFF file path (must end with ``".tif"``).

        Raises:
            TypeError: If *path* is not a string or *basins* is not a
                ``gdal.Dataset``.
        """
        if not isinstance(path, str):
            raise TypeError(f"path: {path} input should be string type")
        if not isinstance(basins, gdal.Dataset):
            raise TypeError(
                "basins raster should be read using gdal (gdal dataset please read it using gdal library)"
            )

        # get number of rows
        rows = basins.RasterYSize
        # get number of columns
        cols = basins.RasterXSize
        # array
        basins_a = basins.ReadAsArray()
        # no data value
        no_val = np.float32(basins.GetRasterBand(1).GetNoDataValue())
        # get number of basins and there names
        basins_val = list(
            set(
                [
                    int(basins_a[i, j])
                    for i in range(rows)
                    for j in range(cols)
                    if basins_a[i, j] != no_val
                ]
            )
        )

        # keep the first basin and delete the others by filling their cells by nodata value
        for i in range(rows):
            for j in range(cols):
                if basins_a[i, j] != no_val and basins_a[i, j] != basins_val[0]:
                    basins_a[i, j] = no_val

        Dataset.dataset_like(basins, basins_a, path)
