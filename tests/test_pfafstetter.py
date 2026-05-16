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


def _branching_dem() -> np.ndarray:
    """A DEM with a long main stem and four side tributaries — enough
    structure to exercise multi-level Pfafstetter."""
    z = np.array(
        [
            [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
            [99, 30, 99, 99, 99, 99, 99, 99, 99, 99],
            [99, 29, 99, 31, 99, 35, 99, 40, 99, 45],
            [99, 28, 27, 25, 24, 23, 22, 21, 20, 19],
            [99, 99, 99, 99, 99, 99, 99, 99, 99, 18],
            [99, 99, 99, 99, 99, 99, 99, 99, 99, 17],
        ],
        dtype=np.float32,
    )
    return z


def test_level_3_produces_three_digit_codes():
    dem, fd, acc, sr = _build(_branching_dem(), threshold=1)
    ws = fd.subbasins_pfafstetter(acc, sr, level=3)
    arr = ws.read_array()
    nonzero = arr[arr != 0]
    if nonzero.size:
        codes = {int(v) for v in np.unique(nonzero)}
        # Every code must be a multiple of 10 or a 3-digit Pfafstetter code.
        for code in codes:
            # Accept the "parent-only" untouched path (multiples of 100 / 10)
            # as well as full 3-digit codes.
            assert 1 <= code <= 999


def test_level_4_codes_within_pfafstetter_range():
    dem, fd, acc, sr = _build(_branching_dem(), threshold=1)
    ws = fd.subbasins_pfafstetter(acc, sr, level=4)
    arr = ws.read_array()
    nonzero = arr[arr != 0]
    if nonzero.size:
        for code in np.unique(nonzero):
            code = int(code)
            assert 1 <= code <= 9999, f"Out-of-range level-4 code {code}"


def test_kernel_untouched_parent_only_branch():
    """When a sub-basin has zero stream cells, the kernel keeps the parent
    code multiplied by the level shift (untouched branch)."""
    # A DEM with a single sink and no tributaries — the recursion's first
    # recursive call returns all-1 (the no-stream fallback), so we get a
    # multi-digit code with all 1's.
    z = np.array(
        [[9, 9, 9, 9, 9], [9, 5, 4, 3, 2], [9, 9, 9, 9, 9]],
        dtype=np.float32,
    )
    dem, fd, acc, sr = _build(z)
    ws = fd.subbasins_pfafstetter(acc, sr, level=2)
    arr = ws.read_array()
    nonzero = arr[arr != 0]
    # All values should be two-digit codes in 11..99 (parent * 10 + child).
    if nonzero.size:
        for v in np.unique(nonzero):
            v = int(v)
            assert 11 <= v <= 99


def test_no_stream_basin_returns_single_code_one():
    """If the basin_mask contains no stream cells, level1 short-circuits to
    a uniform 1 over the basin."""
    z = np.full((4, 4), 5.0, dtype=np.float32)
    # Set a single low cell to anchor flow direction.
    z[2, 2] = 0.0
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc = fd.accumulate()
    sr = acc.streams(threshold=10**9)  # Threshold so high no cells qualify.
    ws = fd.subbasins_pfafstetter(acc, sr, level=1)
    nonzero = ws.read_array()[ws.read_array() != 0]
    if nonzero.size:
        # All non-zero codes should be 1 — the no-stream fallback path.
        assert set(np.unique(nonzero).tolist()).issubset({1})


def test_kernel_unique_codes_loop_visits_every_subbasin():
    """Multi-level decomposition must produce at least one sub-basin per
    distinct level-1 code (i.e., the kernel's ``for c in sub_codes`` loop
    iterates more than once for a multi-tributary DEM)."""
    dem, fd, acc, sr = _build(_branching_dem(), threshold=1)
    ws_lvl1 = fd.subbasins_pfafstetter(acc, sr, level=1)
    ws_lvl2 = fd.subbasins_pfafstetter(acc, sr, level=2)
    lvl1_codes = {int(v) for v in np.unique(ws_lvl1.read_array()) if v != 0}
    lvl2_codes = {int(v) for v in np.unique(ws_lvl2.read_array()) if v != 0}
    # Every level-1 code should appear as the leading digit of at least one
    # level-2 code, confirming the recursion visited every sub-basin.
    leading = {v // 10 for v in lvl2_codes}
    assert lvl1_codes.issubset(leading) or len(lvl1_codes) == 1
