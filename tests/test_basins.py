"""Tests for ``FlowDirection.basins`` (P14)."""
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
