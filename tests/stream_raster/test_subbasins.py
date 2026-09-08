"""Tests for `StreamRaster.subbasins` (P15)."""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers import WatershedRaster
from tests.helpers import make_dem as _make_dem


def _build(z: np.ndarray, threshold: int = 1):
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=threshold)
    return dem, fd, sr


def test_returns_watershed_raster():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build(z)
    sb = sr.subbasins(fd)
    assert type(sb) is WatershedRaster


def test_every_drained_cell_is_labelled():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build(z, threshold=1)
    sb = sr.subbasins(fd)
    arr = sb.read_array()
    # No-data cells are 0; in this fixture the entire surface drains through
    # the streamed chain, so at most a few orphan cells remain unlabelled.
    sr_mask = sr.read_array().astype(bool)
    # Every stream cell has a sub-basin label.
    assert (arr[sr_mask] > 0).all()


def test_multi_direction_routing_rejected():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd_d8, sr = _build(z)
    fd_dinf = dem.flow_direction(method="dinf")
    with pytest.raises(ValueError, match="single-direction"):
        sr.subbasins(fd_dinf)


def test_unsupported_method_raises():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build(z)
    with pytest.raises(ValueError, match="method must be 'link'"):
        sr.subbasins(fd, method="min_order")


def test_shape_mismatch_rejected():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, sr = _build(z)
    z2 = np.zeros((2, 2), dtype=np.float32)
    dem2 = _make_dem(z2)
    fd2 = dem2.flow_direction(method="d8")
    with pytest.raises(ValueError, match="shape"):
        sr.subbasins(fd2)
