"""Tests for ``FlowDirection.subbasins_pfafstetter`` (P16)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, WatershedRaster


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def _build(z: np.ndarray, threshold: int = 1):
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=threshold)
    return dem, fd, acc, sr


def test_returns_watershed_raster():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    assert type(ws) is WatershedRaster


def test_codes_in_pfafstetter_range():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    arr = ws.read_array()
    # All non-zero codes are within {1, 2, ..., 9}.
    nonzero = arr[arr != 0]
    if nonzero.size:
        assert set(np.unique(nonzero).tolist()).issubset(set(range(1, 10)))


def test_level_2_produces_two_digit_codes():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    ws = fd.subbasins_pfafstetter(acc, sr, level=2)
    nonzero = ws.read_array()[ws.read_array() != 0]
    if nonzero.size:
        codes = {int(v) for v in np.unique(nonzero)}
        # Level-2 codes are two-digit (11..99): both digits in {1..9}.
        for code in codes:
            assert 11 <= code <= 99
            parent, child = code // 10, code % 10
            assert 1 <= parent <= 9
            assert 1 <= child <= 9


def test_level_below_one_rejected():
    z = np.array(
        [[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32
    )
    dem, fd, acc, sr = _build(z)
    with pytest.raises(ValueError, match="level"):
        fd.subbasins_pfafstetter(acc, sr, level=0)


def test_unsupported_encoding_raises():
    z = np.array(
        [[9, 9, 9], [9, 5, 9], [9, 9, 9]], dtype=np.float32
    )
    dem, fd, acc, sr = _build(z)
    with pytest.raises(NotImplementedError, match="encoding"):
        fd.subbasins_pfafstetter(acc, sr, encoding="string")


def test_multi_direction_routing_rejected():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd_d8, acc, sr = _build(z)
    fd_dinf = dem.flow_direction(method="dinf")
    with pytest.raises(ValueError, match="single-direction"):
        fd_dinf.subbasins_pfafstetter(acc, sr)


def test_accumulation_type_validated():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    with pytest.raises(ValueError, match="Accumulation"):
        fd.subbasins_pfafstetter(sr, sr)


def test_streams_type_validated():
    z = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 5, 4, 3, 2, 1],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    with pytest.raises(ValueError, match="StreamRaster"):
        fd.subbasins_pfafstetter(acc, acc)
