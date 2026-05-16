"""Tests for ``DEM.fill_depressions`` and the underlying ``_pitremoval`` module (P2).

Covers the three algorithms (Priority-Flood + ε, Wang & Liu, Planchon-Darboux) across the
acceptance-criteria fixtures from the P2 spec: single pit, cascading pit, no-data as drain,
flat plateau with two outlets, and a behavioural sinks-free check on the Coello basin.
"""
from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers import DEM
from digitalrivers._conditioning.pitremoval import (
    VALID_METHODS,
    _nodata_adjacent,
    _seed_mask,
    fill_depressions,
    local_minima_8,
)


# ----- synthetic fixtures (from the P2 spec) --------------------------------------------------

SINGLE_PIT_5x5 = np.array(
    [
        [5, 5, 5, 5, 5],
        [5, 4, 4, 4, 5],
        [5, 4, 1, 4, 5],
        [5, 4, 4, 4, 5],
        [5, 5, 5, 5, 5],
    ],
    dtype=np.float64,
)

# Expected flat fill: the *only* drainage outlet is the outer 5-ring, so every cell strictly
# below 5 lifts to 5. The intermediate 4-ring is below the boundary spill and therefore also
# part of the depression — Priority-Flood lifts the whole interior, not just the central pit.
SINGLE_PIT_5x5_FLAT_FILL = np.full((5, 5), 5.0, dtype=np.float64)


PIT_IN_PIT_6x6 = np.array(
    [
        [9, 9, 9, 9, 9, 9],
        [9, 5, 5, 5, 5, 9],
        [9, 5, 3, 3, 5, 9],
        [9, 5, 3, 1, 5, 9],
        [9, 5, 5, 5, 5, 9],
        [9, 9, 9, 9, 9, 9],
    ],
    dtype=np.float64,
)

# Expected flat fill: the only outlet is the outer 9-ring; every interior cell (z < 9) lifts
# to 9. The two-level pit nesting matters for *the cascading property* (today's single-pass
# code stops at the inner 3-ring), not for the final analytical value of the deepest cell.
PIT_IN_PIT_6x6_FLAT_FILL = np.full((6, 6), 9.0, dtype=np.float64)


# ----- helpers --------------------------------------------------------------------------------

# Local-minima detection now lives in digitalrivers._conditioning.pitremoval.local_minima_8 (moved out
# of this test file in P3; previously inlined here as _internal_sinks_mask).
_internal_sinks_mask = local_minima_8


def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0) -> DEM:
    """Build a DEM wrapping a float32 GeoTIFF-shaped Dataset over ``arr``.

    ``NaN`` in ``arr`` is materialised as ``no_data_value`` on disk so the DEM's
    ``no_data_value`` round-trips correctly.
    """
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


# ----- pure-array unit tests ------------------------------------------------------------------


class TestSeedHelpers:
    def test_no_nodata_seeds_array_boundary_only(self):
        nodata = np.zeros((4, 5), dtype=bool)
        seed = _seed_mask(nodata)
        # All boundary cells set, no interior cell set.
        expected = np.ones((4, 5), dtype=bool)
        expected[1:-1, 1:-1] = False
        assert np.array_equal(seed, expected)

    def test_nodata_corner_extends_seed_inward(self):
        nodata = np.zeros((5, 5), dtype=bool)
        nodata[0, 0] = True
        adj = _nodata_adjacent(nodata)
        # 8-connected dilation of (0,0) minus the cell itself: (0,1), (1,0), (1,1).
        expected = np.zeros((5, 5), dtype=bool)
        expected[0, 1] = True
        expected[1, 0] = True
        expected[1, 1] = True
        assert np.array_equal(adj, expected)

    def test_nodata_adjacent_returns_zero_when_no_nodata(self):
        nodata = np.zeros((3, 3), dtype=bool)
        assert not _nodata_adjacent(nodata).any()


class TestPriorityFloodSinglePit:
    """All three methods must analytically fill the 5×5 single-pit fixture."""

    @pytest.mark.parametrize(
        "method, epsilon",
        [("priority_flood", 0.0), ("wang_liu", 0.0), ("planchon_darboux", 1e-6)],
    )
    def test_single_pit_fills_to_rim(self, method, epsilon):
        z_fill = fill_depressions(SINGLE_PIT_5x5, method=method, epsilon=epsilon)
        # The pit cell must rise to (or just above) the 4-rim. Epsilon variants add tiny lift.
        assert z_fill[2, 2] >= 4.0
        # Rim and outer cells must not change (epsilon is 0 for flat methods; for PD with
        # eps=1e-6 the rim cells are seeds so they stay at Z).
        assert np.allclose(z_fill[0, :], SINGLE_PIT_5x5[0, :])
        assert np.allclose(z_fill[-1, :], SINGLE_PIT_5x5[-1, :])

    def test_priority_flood_flat_matches_analytical(self):
        """Flat fill (epsilon=0) must equal the analytical 4-rim flat-fill exactly."""
        z_fill = fill_depressions(SINGLE_PIT_5x5, method="priority_flood", epsilon=0.0)
        assert np.array_equal(z_fill, SINGLE_PIT_5x5_FLAT_FILL)


class TestPriorityFloodCascadingPit:
    """Pit-inside-pit: today's single-pass code fails; the new algorithm must fill both levels."""

    def test_priority_flood_resolves_inner_and_outer(self):
        z_fill = fill_depressions(PIT_IN_PIT_6x6, method="priority_flood", epsilon=0.0)
        # Cascading property: the inner pit (z=1) lifts past the inner 3-ring it sits in.
        # Today's single-pass code stops at the inner ring; Priority-Flood continues until
        # the outer 9-ring is the spill height.
        assert z_fill[3, 3] >= 5.0
        # Specifically, with epsilon=0 the whole interior is flat at the outer rim (9).
        assert z_fill[3, 3] == 9.0

    def test_wang_liu_resolves_inner_and_outer(self):
        z_fill = fill_depressions(PIT_IN_PIT_6x6, method="wang_liu")
        assert z_fill[3, 3] == 9.0
        assert np.array_equal(z_fill, PIT_IN_PIT_6x6_FLAT_FILL)

    def test_planchon_darboux_resolves_inner_and_outer(self):
        # PD with a tiny epsilon — every interior cell must be at least the outer rim.
        z_fill = fill_depressions(PIT_IN_PIT_6x6, method="planchon_darboux", epsilon=1e-6)
        assert np.all(z_fill[1:-1, 1:-1] >= 5.0 - 1e-9)


class TestNodataAsDrain:
    def test_nodata_corner_drains_pit(self):
        """Single-pit fixture with a no-data corner — the pit must drain via that corner."""
        z = SINGLE_PIT_5x5.copy()
        z[0, 0] = np.nan
        z_fill = fill_depressions(z, method="priority_flood", epsilon=0.0)
        # The nodata position is preserved as NaN.
        assert np.isnan(z_fill[0, 0])
        # The pit still fills to the rim (4); the corner's absence does not break the fill.
        assert z_fill[2, 2] >= 4.0
        # No other cell becomes NaN.
        assert np.sum(np.isnan(z_fill)) == 1

    def test_nodata_cells_unmodified(self):
        z = SINGLE_PIT_5x5.copy()
        z[0, 0] = np.nan
        z[4, 4] = np.nan
        z_fill = fill_depressions(z, method="wang_liu")
        assert np.isnan(z_fill[0, 0])
        assert np.isnan(z_fill[4, 4])


class TestFlatPlateauTwoOutlets:
    """Plateau with two outlets — epsilon variant must propagate a gradient from each."""

    PLATEAU = np.array(
        [
            [9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 1],
            [9, 2, 2, 2, 2, 9],
            [1, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9],
        ],
        dtype=np.float64,
    )

    def test_flat_fill_keeps_plateau_flat(self):
        z_fill = fill_depressions(self.PLATEAU, method="wang_liu")
        # No cell on the plateau (originally z=2) drops below 2.
        plateau = self.PLATEAU == 2
        assert np.all(z_fill[plateau] >= 2.0)

    def test_epsilon_propagates_gradient(self):
        eps = 0.01
        z_fill = fill_depressions(self.PLATEAU, method="priority_flood", epsilon=eps)
        # Each plateau cell should now be strictly greater than at least one neighbour
        # (no internal sinks). This is the property that lets D8 route across the plateau.
        sinks = _internal_sinks_mask(z_fill)
        assert not sinks.any()


class TestValidationAndDispatch:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method must be one of"):
            fill_depressions(SINGLE_PIT_5x5, method="bogus")

    def test_planchon_darboux_rejects_zero_epsilon(self):
        with pytest.raises(ValueError, match="requires epsilon > 0"):
            fill_depressions(SINGLE_PIT_5x5, method="planchon_darboux", epsilon=0.0)

    def test_valid_methods_set(self):
        assert VALID_METHODS == frozenset(
            {"priority_flood", "wang_liu", "planchon_darboux"}
        )


# ----- DEM-level integration -----------------------------------------------------------------


class TestDEMFillDepressions:
    def test_returns_typed_dem(self):
        dem = _make_dem(SINGLE_PIT_5x5.astype(np.float32))
        out = dem.fill_depressions(method="priority_flood")
        assert type(out) is DEM

    def test_inplace_returns_none(self):
        dem = _make_dem(SINGLE_PIT_5x5.astype(np.float32))
        result = dem.fill_depressions(method="priority_flood", inplace=True)
        assert result is None
        # The instance now holds the filled surface.
        assert dem.values[2, 2] >= 4.0 - 1e-5

    def test_pit_in_pit_via_dem_method(self):
        dem = _make_dem(PIT_IN_PIT_6x6.astype(np.float32))
        out = dem.fill_depressions(method="priority_flood", epsilon=0.0)
        vals = out.values
        # Inner pit cell rose past the inner ring to the outer-rim spill height (= 9).
        assert vals[3, 3] == pytest.approx(9.0, abs=1e-4)

    def test_nodata_preserved_through_geotiff_roundtrip(self):
        arr = SINGLE_PIT_5x5.copy()
        arr[0, 0] = np.nan
        dem = _make_dem(arr.astype(np.float32))
        out = dem.fill_depressions(method="priority_flood")
        # NaN survives the GeoTIFF round-trip via the no-data sentinel.
        assert np.isnan(out.values[0, 0])


@pytest.mark.slow
class TestCoelloBasinSinksFree:
    """End-to-end: ``priority_flood`` on the Coello DEM yields zero internal sinks."""

    def test_no_internal_sinks_after_priority_flood(
        self, coello_dem_4000: gdal.Dataset
    ):
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="priority_flood", epsilon=0.1)
        sinks = _internal_sinks_mask(filled.values)
        assert int(sinks.sum()) == 0, (
            f"priority_flood left {int(sinks.sum())} internal sinks on the Coello DEM"
        )

    def test_fill_only_raises_elevations(self, coello_dem_4000: gdal.Dataset):
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="priority_flood", epsilon=0.0)
        original = dem.values
        new = filled.values
        valid = ~np.isnan(original) & ~np.isnan(new)
        assert np.all(new[valid] >= original[valid])

    def test_nodata_pattern_preserved(self, coello_dem_4000: gdal.Dataset):
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="priority_flood")
        assert np.array_equal(np.isnan(dem.values), np.isnan(filled.values))
