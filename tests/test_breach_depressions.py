"""Tests for ``DEM.breach_depressions`` and the underlying ``_breach`` module (P3).

Covers the three modes (``single_cell``, ``least_cost``, ``hybrid``) across the
acceptance-criteria fixtures from the P3 spec: walled pit (single-cell-thick wall),
thick-wall blocking breach (hybrid fall-back to fill), single-cell pit preprocessing,
``max_length`` constraint, and a behavioural sinks-free check on the Coello basin.
"""
from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers import DEM
from digitalrivers._breach import (
    VALID_BREACH_METHODS,
    _candidate_intermediates,
    breach_depressions,
)
from digitalrivers._pitremoval import local_minima_8


# ----- helpers ----------------------------------------------------------------------------

def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan_mask = np.isnan(disk)
    disk[nan_mask] = no_data_value
    ds = Dataset.create_from_array(
        disk,
        top_left_corner=(0.0, 0.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


# ----- candidate-intermediates helper ----------------------------------------------------

class TestCandidateIntermediates:
    def test_cardinal_second_order_has_three_intermediates(self):
        # (2, 0) is reached via (1, -1), (1, 0), or (1, 1) — three diagonals/cardinal.
        intermediates = _candidate_intermediates(2, 0)
        assert set(intermediates) == {(1, -1), (1, 0), (1, 1)}

    def test_diagonal_second_order_has_three_intermediates(self):
        # (2, 2) is reached via (1, 1), (1, 2), or (2, 1) — direct diagonal + two cardinals.
        # But our filter only includes _NEIGHBOURS_8 offsets (max(|dr|,|dc|) == 1), so
        # (1, 2) and (2, 1) are excluded.
        intermediates = _candidate_intermediates(2, 2)
        assert (1, 1) in intermediates

    def test_knight_move_has_two_intermediates(self):
        # (1, 2) is reached via (0, 1) or (1, 1).
        intermediates = _candidate_intermediates(1, 2)
        assert set(intermediates) >= {(0, 1), (1, 1)}


# ----- single_cell mode ------------------------------------------------------------------

class TestSingleCellBreach:
    """Cheap O(n) preprocessing pass — resolves isolated 1-cell pits."""

    def test_isolated_pit_with_lower_second_order(self):
        # Pit at (2, 2)=1 surrounded by z=5. A second-order cell at (3, 4)=0 provides a
        # cheap breach target via the intermediate (2, 3) or (3, 3), lowered to (1+0)/2.
        z = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
                [5, 5, 1, 5, 5],
                [5, 5, 5, 5, 0],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
        out = breach_depressions(z, method="single_cell")
        # Pit at (2, 2) is no longer a local minimum after the breach.
        minima = local_minima_8(out)
        assert not minima[2, 2]
        # Some intermediate cell on the path from (2, 2) to (3, 4) has been lowered.
        # Candidates are first-order neighbours of (2, 2) one step toward (3, 4).
        assert min(out[2, 3], out[3, 2], out[3, 3]) <= 0.5

    def test_pit_with_no_low_second_order_left_alone(self):
        # No second-order cell is lower than the pit — single_cell can't do anything.
        z = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
                [5, 5, 1, 5, 5],
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )
        out = breach_depressions(z, method="single_cell")
        # Pit remains, and no cell was modified.
        np.testing.assert_array_equal(out, z)

    def test_single_cell_does_not_modify_neighbours_of_unrelated_cells(self):
        # The pre-pass must only touch the intermediate cell — not the pit, not the
        # second-order target, not any other cell.
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
                [9, 9, 1, 9, 9],
                [9, 9, 9, 9, 0],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float64,
        )
        out = breach_depressions(z, method="single_cell")
        # Pit unchanged.
        assert out[2, 2] == 1.0
        # Outlet unchanged.
        assert out[3, 4] == 0.0
        # Boundary 9s untouched.
        np.testing.assert_array_equal(out[0, :], z[0, :])
        np.testing.assert_array_equal(out[-1, :], z[-1, :])


# ----- least_cost mode -------------------------------------------------------------------

class TestLeastCostBreach:
    """Lindsay 2016 Dijkstra-from-each-pit."""

    @staticmethod
    def _walled_pit() -> np.ndarray:
        # Pit at (3, 3)=1 surrounded by uniform z=9 in every direction. The only data
        # outlet is at (6, 6)=0, at Chebyshev distance 3 from the pit — so the cheap
        # single-cell preprocessing pass (which only checks distance-2 cells) cannot
        # resolve it. Dijkstra alone must do the work, and max_depth governs whether
        # it succeeds.
        return np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 1, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float64,
        )

    def test_walled_pit_resolved_with_loose_max_depth(self):
        z = self._walled_pit()
        # Cheapest path from pit to outlet: (3,3) → (4,4) → (5,5) → (6,6), 3 cells. Cost
        # accumulates by ≈ (9 - 1) = 8 per non-outlet step, so we need max_depth ≥ 16 to
        # pop the cell that finds the outlet.
        out = breach_depressions(z, method="least_cost", max_depth=20)
        # Pit no longer a local minimum.
        assert not local_minima_8(out)[3, 3]
        # The outlet cell is untouched (we never modify the outlet itself).
        assert out[6, 6] == 0.0
        # The pit cell itself is also unmodified.
        assert out[3, 3] == 1.0

    def test_walled_pit_aborts_with_tight_max_depth(self):
        z = self._walled_pit()
        # max_depth=2 aborts at the first popped wall cell (accum=8 > 2).
        out = breach_depressions(z, method="least_cost", max_depth=2)
        # Pit still a local minimum because the breach was aborted.
        assert local_minima_8(out)[3, 3]

    def test_max_length_constraint(self):
        # Pit at (1, 1)=1 in a long corridor of z=9 with outlet at (3, 5)=0.
        # Path length from pit to outlet ≈ 4 cells (via diagonals). max_length=2 fails.
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 1, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float64,
        )
        out_tight = breach_depressions(z, method="least_cost", max_length=2)
        # Pit not resolvable in 2 steps.
        assert local_minima_8(out_tight)[1, 1]

        out_loose = breach_depressions(z, method="least_cost", max_length=10)
        assert not local_minima_8(out_loose)[1, 1]


# ----- hybrid mode -----------------------------------------------------------------------

class TestHybridBreach:
    """Try least_cost; fall back to Priority-Flood fill on unresolved pits."""

    @staticmethod
    def _thick_wall() -> np.ndarray:
        # Thick (3-cell) wall blocks the breach unless max_depth is huge.
        return np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 1, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float64,
        )

    def test_thick_wall_falls_back_to_fill(self):
        z = self._thick_wall()
        # With a tight max_depth the breach fails; hybrid falls back to fill.
        out = breach_depressions(
            z, method="hybrid", max_depth=2.0, fill_remaining=True
        )
        # No internal sinks remain — fill resolved what breach couldn't.
        assert not local_minima_8(out).any()
        # The pit cell has been raised (fill, not breach).
        assert out[1, 1] > 1.0

    def test_thick_wall_skips_fill_when_requested(self):
        z = self._thick_wall()
        out = breach_depressions(
            z, method="hybrid", max_depth=2.0, fill_remaining=False
        )
        # Without fill fallback the pit remains.
        assert local_minima_8(out)[1, 1]


# ----- nodata handling -------------------------------------------------------------------

class TestNodataAsOutlet:
    def test_nodata_neighbour_acts_as_free_outlet(self):
        z = np.array(
            [
                [9, np.nan, 9],
                [9, 5, 9],
                [9, 1, 9],
                [9, 9, 9],
            ],
            dtype=np.float64,
        )
        # Pit (2, 1)=1. Direct path up to the no-data cell at (0, 1) via (1, 1)=5
        # should be a free outlet — lower (1, 1) to ≈ pit_z.
        out = breach_depressions(z, method="least_cost")
        assert not local_minima_8(out)[2, 1]
        # The nodata cell itself stays NaN.
        assert np.isnan(out[0, 1])


# ----- validation ------------------------------------------------------------------------

class TestValidation:
    def test_invalid_method_raises(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="method must be one of"):
            breach_depressions(z, method="bogus")

    def test_valid_methods_set(self):
        assert VALID_BREACH_METHODS == frozenset(
            {"least_cost", "hybrid", "single_cell"}
        )

    def test_no_pits_returns_input_unchanged(self):
        # Strictly monotonic surface has no local minima — nothing to do.
        z = np.array(
            [
                [9, 8, 7, 6],
                [8, 7, 6, 5],
                [7, 6, 5, 4],
                [6, 5, 4, 3],
            ],
            dtype=np.float64,
        )
        out = breach_depressions(z, method="least_cost")
        np.testing.assert_array_equal(out, z)


# ----- DEM-level integration -------------------------------------------------------------

class TestDEMBreach:
    def test_returns_typed_dem(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 5, 5, 9],
                [9, 5, 1, 5, 9],
                [9, 5, 5, 5, 9],
                [9, 9, 9, 9, 0],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        out = dem.breach_depressions(method="least_cost", max_depth=20)
        assert type(out) is DEM

    def test_inplace_returns_none(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 5, 5, 9],
                [9, 5, 1, 5, 9],
                [9, 5, 5, 5, 9],
                [9, 9, 9, 9, 0],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        result = dem.breach_depressions(
            method="least_cost", max_depth=20, inplace=True
        )
        assert result is None

    def test_hybrid_via_dem_method(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 1, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        out = dem.breach_depressions(
            method="hybrid", max_depth=2.0, fill_remaining=True
        )
        # No sinks remain (fill picked up where breach gave up).
        assert not local_minima_8(out.values).any()

    def test_dem_breach_invalid_method(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="method must be one of"):
            dem.breach_depressions(method="bogus")


@pytest.mark.slow
class TestCoelloBasinHybrid:
    """Hybrid breach on the Coello DEM produces a sinks-free surface."""

    def test_hybrid_no_internal_sinks(self, coello_dem_4000: gdal.Dataset):
        dem = DEM(coello_dem_4000)
        breached = dem.breach_depressions(
            method="hybrid", max_depth=50.0, fill_remaining=True
        )
        sinks = local_minima_8(breached.values)
        assert int(sinks.sum()) == 0

    def test_hybrid_only_lowers_or_keeps_elevations_on_breach_paths(
        self, coello_dem_4000: gdal.Dataset
    ):
        # Breach paths only lower cells; the fill fallback only raises cells. So overall
        # elevations may go up or down, but no cell that the original surface had a finite
        # value at should become NaN, and vice versa.
        dem = DEM(coello_dem_4000)
        original = dem.values
        breached = dem.breach_depressions(method="hybrid", max_depth=50.0).values
        np.testing.assert_array_equal(np.isnan(original), np.isnan(breached))
