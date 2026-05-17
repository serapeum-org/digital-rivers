"""Tests for `FlowDirection.upscale` (P18 — COTAT only)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def test_scale_factor_one_is_noop():
    z = np.array(
        [
            [9, 9, 9, 9],
            [9, 5, 4, 1],
            [9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    _, up = fd.upscale(scale_factor=1, accumulation=acc)
    assert type(up) is FlowDirection
    np.testing.assert_array_equal(up.read_array(), fd.read_array())


def test_scale_factor_two_halves_dimensions():
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
    _, up = fd.upscale(scale_factor=2, accumulation=acc)
    assert up.shape == (1, 3, 3)


def test_cotat_returns_dem_when_input_dem_supplied():
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
    up_dem, up_fd = fd.upscale(scale_factor=2, accumulation=acc, dem=dem)
    assert isinstance(up_dem, DEM)
    assert up_dem.shape == up_fd.shape


def test_eam_now_implemented():
    """EAM shipped in the backfill — verify the accumulation-weighted
    voting kernel returns a coarse FlowDirection."""
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
    _, up = fd.upscale(scale_factor=2, method="eam", accumulation=acc)
    assert isinstance(up, FlowDirection)


def test_dmm_now_implemented():
    """DMM shipped in the backfill — verify uniform-weight voting works."""
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
    _, up = fd.upscale(scale_factor=2, method="dmm")
    assert isinstance(up, FlowDirection)


def test_eam_without_accumulation_raises():
    z = np.array(
        [[9, 9, 9, 9], [9, 5, 4, 1], [9, 9, 9, 9], [9, 9, 9, 9]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    with pytest.raises(ValueError, match="EAM"):
        fd.upscale(scale_factor=2, method="eam")


def test_missing_accumulation_for_cotat_raises():
    z = np.array(
        [
            [9, 9, 9, 9],
            [9, 5, 4, 1],
            [9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    with pytest.raises(ValueError, match="Accumulation"):
        fd.upscale(scale_factor=2)


def test_scale_factor_zero_raises():
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    with pytest.raises(ValueError, match="scale_factor"):
        fd.upscale(scale_factor=0)


def test_cotat_6x6_round_trip_sf2_no_runtime_error():
    """I5 regression: COTAT on a 6x6 east-flowing DEM with sf=2 must
    complete without the defensive `RuntimeError` ever firing."""
    z = np.array(
        [[float(c) for c in range(6, 0, -1)] for _ in range(6)],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    result = fd.upscale(scale_factor=2, accumulation=acc, dem=dem)
    coarse_fd = result[1]
    arr = coarse_fd.read_array()
    # Coarse output is 3x3 (6/2).
    assert arr.shape[-2:] == (3, 3)
    assert coarse_fd.routing == "d8"
