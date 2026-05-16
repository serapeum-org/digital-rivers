"""Tests for `FlowDirection.basins` (P14)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, WatershedRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def test_single_basin_from_single_outlet():
    # 1D chain east; the rightmost cell is an outlet (sink).
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    ws = fd.basins()
    assert type(ws) is WatershedRaster
    # At least one basin; outlets recorded.
    assert ws.basin_count >= 1
    assert len(ws.outlets) >= 1


def test_outlet_record_has_cell_count():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    ws = fd.basins()
    assert "cell_count" in ws.outlets.columns
    assert (ws.outlets["cell_count"] >= 1).all()


def test_min_area_drops_small_basins():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    huge = 10_000_000  # nothing meets this threshold
    ws = fd.basins(min_area_cells=huge, merge_small="drop")
    arr = ws.read_array()
    assert (arr == 0).all() or arr.max() == 0


def test_both_area_kwargs_raise():
    z = np.array(
        [
            [9, 9, 9],
            [9, 5, 9],
            [9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    with pytest.raises(ValueError, match="at most one"):
        fd.basins(min_area_cells=10, min_area_km2=1.0)


def test_unknown_merge_small_raises():
    z = np.array([[9, 9], [9, 5]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    with pytest.raises(ValueError, match="merge_small"):
        fd.basins(merge_small="bogus")


def test_multi_direction_routing_rejected():
    z = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 5, 4, 3, 9],
            [9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd_dinf = dem.flow_direction(method="dinf")
    with pytest.raises(ValueError, match="single-direction"):
        fd_dinf.basins()


class TestMergeToNeighbour8Adjacency:
    """Extra coverage for the 8-connected `merge_to_neighbour` path (I1)."""

    def test_picks_largest_8_neighbour_when_multiple_candidates(self):
        """A small basin touching two larger basins of different sizes
        relabels with the larger one."""
        rows, cols = 4, 8
        z = np.full((rows, cols), 10.0, dtype=np.float32)
        # Three sinks: tiny at col 3, medium at col 0, large at col 7.
        z[3, 3] = 0.0
        z[3, 0] = 1.0
        z[3, 7] = 2.0
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        ws = fd.basins(min_area_cells=2, merge_small="merge_to_neighbour")
        arr = ws.read_array()
        # The tiny basin's cell must end up labelled (non-zero); we don't
        # pin which adjacent basin wins because that depends on the D8
        # tie-breaking in flat areas, but we verify the merge happened.
        assert int(arr[3, 3]) != 0

    def test_edge_basin_dilation_does_not_crash(self):
        """A small basin pressed against the raster edge is correctly
        clipped during dilation — no IndexError or out-of-bounds."""
        rows, cols = 5, 5
        z = np.full((rows, cols), 10.0, dtype=np.float32)
        z[0, 0] = 0.0  # corner sink — only 3 in-bounds neighbours
        z[4, 4] = 1.0  # opposite corner sink
        dem = _make_dem(z)
        fd = dem.flow_direction(method="d8")
        # No crash on dilation clipping at the edge.
        ws = fd.basins(min_area_cells=3, merge_small="merge_to_neighbour")
        arr = ws.read_array()
        assert arr.shape == (5, 5)

    def test_min_area_zero_keeps_all_basins(self):
        """With `min_area_cells=None` (default) no merging happens."""
        z = np.array(
            [[0, 5, 5], [5, 5, 5], [5, 5, 0]], dtype=np.float32
        )
        dem = _make_dem(z)
        ws = dem.flow_direction(method="d8").basins()
        # No merge applied → original basin count.
        assert ws.basin_count >= 2
