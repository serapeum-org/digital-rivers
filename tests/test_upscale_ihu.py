"""Tests for `FlowDirection.upscale_ihu` (P19)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def test_ihu_scale_one_is_noop():
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
    up_dem, up_fd, metrics = fd.upscale_ihu(
        scale_factor=1, accumulation=acc, dem=dem
    )
    assert isinstance(up_fd, FlowDirection)
    assert metrics == {}


def test_ihu_higher_scale_now_implemented():
    """IHU hill-climbing shipped — verify it produces a coarse FlowDirection."""
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
    up_dem, up_fd, metrics = fd.upscale_ihu(
        scale_factor=2, accumulation=acc, dem=dem, report=True
    )
    assert isinstance(up_fd, FlowDirection)
    assert up_dem is not None
    # report=True returns the metrics dict with the documented keys.
    for key in ("final_error", "iterations", "swaps", "converged"):
        assert key in metrics


def test_ihu_metrics_empty_when_report_false():
    """Default report=False returns an empty metrics dict."""
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
    _, _, metrics = fd.upscale_ihu(
        scale_factor=2, accumulation=acc, dem=dem
    )
    assert metrics == {}


def test_ihu_converges_with_swap_count():
    """The hill-climbing engine sets converged=True when no swap improves."""
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
    _, _, metrics = fd.upscale_ihu(
        scale_factor=2, accumulation=acc, dem=dem, report=True,
        max_iter=50,
    )
    assert metrics["converged"] is True


def test_upscale_dispatch_ihu_routes_to_ihu():
    """The unified dispatcher's method='ihu' branch wires through to
    upscale_ihu and returns the (dem, fdir) tuple."""
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
    up_dem, up_fd = fd.upscale(
        scale_factor=2, method="ihu", accumulation=acc, dem=dem
    )
    assert isinstance(up_fd, FlowDirection)


def test_ihu_scale_zero_raises():
    z = np.array([[1.0, 2.0]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    with pytest.raises(ValueError, match="scale_factor"):
        fd.upscale_ihu(scale_factor=0, accumulation=acc, dem=dem)
