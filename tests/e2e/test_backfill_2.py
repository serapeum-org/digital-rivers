"""Second backfill pass: P20 topological_breach + P28 native Numba COTAT."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset
from shapely.geometry import LineString

from digitalrivers import DEM, FlowDirection
from digitalrivers._numba import (
    _DIR_DR_I32,
    _DIR_DC_I32,
    cotat_upscale_numba,
)


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _line_world(rows_cols: list[tuple[int, int]]) -> LineString:
    return LineString([(c + 0.5, -(r + 0.5)) for r, c in rows_cols])


# ----- P20 topological_breach -----------------------------------------------


def test_topological_breach_now_implemented():
    """topological_breach (Lindsay 2016) is now wired up via the
    rasterise-streams + Phase 1 breach composition."""
    z = np.full((7, 7), 10.0, dtype=np.float32)
    z[3, 3] = 0.0  # add a pit to force breach behaviour
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(3, 0), (3, 6)])], crs=4326,
    )
    out = dem.burn_streams(streams, method="topological_breach")
    assert isinstance(out, DEM)


def test_topological_breach_returns_dem_with_finite_values():
    z = np.full((5, 5), 10.0, dtype=np.float32)
    dem = _make_dem(z)
    streams = gpd.GeoDataFrame(
        geometry=[_line_world([(2, 0), (2, 4)])], crs=4326,
    )
    out = dem.burn_streams(streams, method="topological_breach")
    assert np.all(np.isfinite(out.values))


# ----- P28 native COTAT (Numba) ----------------------------------------------


def test_native_cotat_matches_pure_python():
    """The Numba COTAT kernel produces the same output as the pure-Python
    P18 loop on a small fixture."""
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    fdir_arr = fd.read_array().astype(np.int32)
    acc_arr = acc.read_array().astype(np.float64)
    native = cotat_upscale_numba(
        fdir_arr, acc_arr, 2, _DIR_DR_I32, _DIR_DC_I32, np.int32(-9999),
    )
    _, py_fd = fd.upscale(scale_factor=2, accumulation=acc)
    np.testing.assert_array_equal(native, py_fd.read_array())


def test_native_cotat_via_upscale_dispatcher():
    """FlowDirection.upscale should pick up the native fast path when
    Numba is enabled."""
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    up_dem, up_fd = fd.upscale(scale_factor=2, accumulation=acc, dem=dem)
    assert isinstance(up_fd, FlowDirection)
    assert up_dem is not None
