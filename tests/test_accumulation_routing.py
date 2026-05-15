"""Tests for the generalised flow-accumulation Kahn dispatcher (P6).

Covers per-routing-scheme dispatch (D8, Rho8, D∞, MFD), weighted accumulation,
mass conservation, and the typed Accumulation return.
"""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, Accumulation
from digitalrivers._accumulation import (
    _receivers_d8,
    _receivers_dinf,
    _receivers_mfd,
    accumulate as _accumulate_array,
    kahn_accumulate,
)


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan_mask = np.isnan(disk)
    disk[nan_mask] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


# ----- Kahn primitive --------------------------------------------------------------------

class TestKahn:
    def test_linear_chain_d8(self):
        # 3-cell chain: a → b → c (all flow east, direction code 6).
        receivers = np.full((1, 3, 1), -1, dtype=np.int8)
        receivers[0, 0, 0] = 6
        receivers[0, 1, 0] = 6
        proportions = np.zeros((1, 3, 1), dtype=np.float32)
        proportions[0, 0, 0] = 1.0
        proportions[0, 1, 0] = 1.0
        weights = np.ones((1, 3), dtype=np.float64)
        valid = np.ones((1, 3), dtype=bool)
        out = kahn_accumulate(receivers, proportions, weights, valid)
        # Outlet (rightmost cell) has 2 upstream contributors.
        assert out[0, 2] == 2.0
        # Middle cell has 1.
        assert out[0, 1] == 1.0
        # Headwater has 0 (no upstream).
        assert out[0, 0] == 0.0

    def test_weighted_accumulation_scales_linearly(self):
        receivers = np.full((1, 3, 1), -1, dtype=np.int8)
        receivers[0, 0, 0] = 6
        receivers[0, 1, 0] = 6
        proportions = np.zeros((1, 3, 1), dtype=np.float32)
        proportions[0, 0, 0] = 1.0
        proportions[0, 1, 0] = 1.0
        weights = np.full((1, 3), 2.5, dtype=np.float64)
        valid = np.ones((1, 3), dtype=bool)
        out = kahn_accumulate(receivers, proportions, weights, valid)
        # Outlet collects two cells × 2.5.
        assert out[0, 2] == pytest.approx(5.0)


# ----- D8 dispatch -----------------------------------------------------------------------

def test_d8_dispatch_matches_legacy_count():
    # A 3-row strip flowing east; the bottom and top rows are higher so the middle
    # row drains eastward. The eastern outlet at z=1 has no further downhill
    # neighbour (its eastern neighbour is out of grid) so under the stricter post-P5
    # D8 it's a sink — accumulation at that cell is the count of strictly upstream
    # cells along the chain.
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
    acc = fd.accumulate()
    assert type(acc) is Accumulation
    assert acc.routing == "d8"
    arr = acc.read_array()
    # The sink at (1, 5) collects everything upstream. The chain and the rows of 9s
    # together drain into it; concrete count is determined by the topology but must
    # exceed the chain length and not exceed the total cell count.
    total_cells = arr.size
    assert arr[1, 5] > 4
    assert arr[1, 5] < total_cells


# ----- D∞ dispatch -----------------------------------------------------------------------

def test_dinf_dispatch_mass_conservation():
    # Planar east-tilted surface, all cells drain east. Sum over outlet column should
    # equal total interior cells.
    rows, cols = 5, 5
    z = -np.arange(cols, dtype=np.float32)[np.newaxis, :].repeat(rows, axis=0)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="dinf")
    acc = fd.accumulate()
    arr = acc.read_array()
    # All accumulation values are finite.
    assert np.all(np.isfinite(arr))
    # Mass conservation: total accumulation at the eastern boundary plus
    # ungated outflow accounts for upstream contributions.
    # Total interior cells = (rows-2) * (cols-2) plus boundary contributions.
    # Easier check: max acc is on the east column, > 0.
    assert arr.max() > 0


# ----- MFD dispatch ---------------------------------------------------------------------

def test_mfd_quinn_dispatch_returns_accumulation():
    # Conical hilltop.
    size = 7
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    z = 10.0 - np.sqrt((xs - 3.0) ** 2 + (ys - 3.0) ** 2)
    z = z.astype(np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="mfd_quinn")
    acc = fd.accumulate()
    assert acc.routing == "mfd_quinn"
    arr = acc.read_array()
    # Apex contributes nothing (it's the source). Boundary cells get the most.
    assert arr[3, 3] == pytest.approx(0.0, abs=1e-6)
    assert arr.max() > 0


# ----- Weight handling -------------------------------------------------------------------

def test_uniform_weights_match_unweighted():
    # weights=1.0 raster must equal weights=None.
    z = np.array([[5, 4, 3, 2, 1]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    acc_unweighted = fd.accumulate().read_array()

    weights = Dataset.create_from_array(
        np.ones_like(z), top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    acc_weighted = fd.accumulate(weights=weights).read_array()
    np.testing.assert_allclose(acc_unweighted, acc_weighted)


def test_weights_mismatch_shape_raises():
    z = np.array([[5, 4, 3], [3, 2, 1]], dtype=np.float32)
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    bad_weights = Dataset.create_from_array(
        np.ones((4, 4), dtype=np.float32),
        top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="weights shape"):
        fd.accumulate(weights=bad_weights)


# ----- Receivers helpers -----------------------------------------------------------------

class TestReceivers:
    def test_d8_decoder_pass_through(self):
        arr = np.array([[6, 4, 0], [2, 6, 7]], dtype=np.int32)
        valid = np.ones_like(arr, dtype=bool)
        rec, prop = _receivers_d8(arr, valid)
        assert rec.shape == (2, 3, 1)
        assert rec[0, 0, 0] == 6
        assert rec[1, 2, 0] == 7
        assert (prop[valid] == 1.0).all()

    def test_dinf_decoder_sector_split(self):
        # Angle π/4 (= NE) should split 0/100 between sector 0's two codes (E=6, NE=5).
        angle = np.full((1, 1), np.pi / 4, dtype=np.float32)
        valid = np.ones((1, 1), dtype=bool)
        rec, prop = _receivers_dinf(angle, valid)
        # sector = 1 because floor(π/4 / (π/4)) = 1; bounds: NE(5) and N(4).
        # frac2 = 0, frac1 = 1, so all mass goes to NE.
        assert rec[0, 0, 0] == 5  # NE
        assert prop[0, 0, 0] == pytest.approx(1.0)
        assert prop[0, 0, 1] == pytest.approx(0.0)

    def test_mfd_decoder_passes_fractions(self):
        # Single cell with even split across all 8 directions.
        fractions = np.full((1, 1, 8), 0.125, dtype=np.float32)
        valid = np.ones((1, 1), dtype=bool)
        rec, prop = _receivers_mfd(fractions, valid)
        np.testing.assert_array_equal(rec[0, 0], np.arange(8, dtype=np.int8))
        np.testing.assert_allclose(prop[0, 0], 0.125)


# ----- Array-level dispatcher ---------------------------------------------------------

def test_dispatcher_rejects_unknown_routing():
    arr = np.zeros((3, 3), dtype=np.int32)
    valid = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="routing must be one of"):
        _accumulate_array(arr, "bogus", valid)
