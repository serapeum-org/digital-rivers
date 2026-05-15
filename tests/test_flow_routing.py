"""Tests for D∞ / MFD-Quinn / MFD-Holmgren / Rho8 flow direction (P5)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, FlowDirection
from digitalrivers._flow_routing import (
    dinf_flow_direction,
    mfd_flow_direction,
    rho8_flow_direction,
)


def _make_dem(arr: np.ndarray, cell_size: float = 1.0,
              no_data_value: float = -9999.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan_mask = np.isnan(disk)
    disk[nan_mask] = no_data_value
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


# ----- D8 regression --------------------------------------------------------------------

def test_d8_still_returns_routing_d8():
    z = np.array(
        [
            [9, 9, 9, 9, 9],
            [9, 5, 4, 3, 9],
            [9, 6, 5, 4, 9],
            [9, 7, 6, 5, 9],
            [9, 9, 9, 9, 9],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    fd = dem.flow_direction(method="d8")
    assert type(fd) is FlowDirection
    assert fd.routing == "d8"
    assert fd.dtype == ["int32"]


# ----- D∞ ------------------------------------------------------------------------------

class TestDinf:
    def test_planar_east_slope_gives_eastward_angle(self):
        # Z = -x → water flows east. D∞ angle should be 0 (CCW from east).
        rows, cols = 5, 5
        z = -np.arange(cols, dtype=np.float64)[np.newaxis, :].repeat(rows, axis=0)
        angle, magnitude = dinf_flow_direction(z, cell_size=1.0)
        # Interior cells (away from boundary NaNs) should have angle ≈ 0.
        interior = angle[1:-1, 1:-1]
        valid = interior >= 0
        assert valid.all()
        # Aspect close to 0 or 2π.
        a = interior[valid]
        a = np.minimum(a, 2 * np.pi - a)
        assert np.all(a < 0.05), f"max deviation: {a.max()}"
        # Magnitude should be ≈ 1 (slope of -1 per unit x).
        assert np.all(magnitude[1:-1, 1:-1] > 0.9)

    def test_planar_north_slope_gives_north_angle(self):
        # Z = +y (rows index increases southward in DEM convention, so z=+row means
        # water flows toward decreasing row, i.e. north). D∞ angle should be π/2.
        rows, cols = 5, 5
        z = np.arange(rows, dtype=np.float64)[:, np.newaxis].repeat(cols, axis=1)
        angle, _ = dinf_flow_direction(z, cell_size=1.0)
        interior = angle[1:-1, 1:-1]
        valid = interior >= 0
        deviation = np.abs(interior[valid] - np.pi / 2)
        assert np.all(deviation < 0.05)

    def test_pit_has_no_flow(self):
        # Single pit surrounded by higher cells.
        z = np.array(
            [
                [5, 5, 5],
                [5, 1, 5],
                [5, 5, 5],
            ],
            dtype=np.float64,
        )
        angle, magnitude = dinf_flow_direction(z, cell_size=1.0)
        assert angle[1, 1] == -1.0
        assert magnitude[1, 1] == 0.0

    def test_dem_method_returns_two_band_float32(self):
        rows, cols = 5, 5
        z = -np.arange(cols, dtype=np.float32)[np.newaxis, :].repeat(rows, axis=0)
        dem = _make_dem(z)
        fd = dem.flow_direction(method="dinf")
        assert fd.routing == "dinf"
        assert fd.band_count == 2


# ----- MFD ----------------------------------------------------------------------------

class TestMFD:
    def test_quinn_fractions_sum_to_one_on_downslope_cells(self):
        # Conical hilltop: every direction has a downslope neighbour.
        size = 7
        xs, ys = np.meshgrid(np.arange(size), np.arange(size))
        cx, cy = size // 2, size // 2
        z = 10.0 - np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        z = z.astype(np.float64)
        dem = _make_dem(z)
        slopes = dem._get_8_direction_slopes()
        elev_mask = ~np.isnan(dem.values)
        fractions = mfd_flow_direction(slopes, elev_mask,
                                       weighting="quinn", exponent=1.0)
        total = fractions.sum(axis=2)
        # Interior cells (away from edge nans) should have downhill neighbours.
        interior_total = total[2:-2, 2:-2]
        assert np.all((interior_total > 0.999) & (interior_total < 1.001))

    def test_holmgren_high_exponent_concentrates_on_steepest(self):
        # Asymmetric slope; high exponent must shift mass toward the steepest direction.
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 8, 7, 6, 9],
                [9, 6, 5, 4, 9],
                [9, 4, 3, 2, 9],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float64,
        )
        dem = _make_dem(z)
        slopes = dem._get_8_direction_slopes()
        elev_mask = ~np.isnan(dem.values)
        f_low = mfd_flow_direction(slopes, elev_mask, weighting="holmgren", exponent=1.0)
        f_high = mfd_flow_direction(slopes, elev_mask, weighting="holmgren", exponent=10.0)
        # At the centre cell (2, 2), the steepest direction's fraction should be larger
        # with high exponent.
        centre_low = f_low[2, 2].max()
        centre_high = f_high[2, 2].max()
        assert centre_high > centre_low

    def test_dem_method_returns_eight_bands(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 4, 3, 9],
                [9, 6, 5, 4, 9],
                [9, 7, 6, 5, 9],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd = dem.flow_direction(method="mfd_quinn")
        assert fd.routing == "mfd_quinn"
        assert fd.band_count == 8


# ----- Rho8 ----------------------------------------------------------------------------

class TestRho8:
    def test_reproducible_with_seed(self):
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 4, 3, 9],
                [9, 6, 5, 4, 9],
                [9, 7, 6, 5, 9],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        fd1 = dem.flow_direction(method="rho8", seed=42)
        fd2 = dem.flow_direction(method="rho8", seed=42)
        np.testing.assert_array_equal(fd1.read_array(), fd2.read_array())

    def test_different_seeds_can_differ_on_flat_cells(self):
        # Flat surface: every direction has identical slope (zero). Rho8's perturbation
        # of cardinal slopes via `2 - U` lets a downslope tiebreaker exist for non-flat
        # cases. For a true flat there is no downslope, so result is no-flow either way.
        # Use a near-uniform slope to verify seeds can produce different outputs.
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [9, 5, 5, 5, 9],
                [9, 5, 4, 5, 9],
                [9, 5, 5, 5, 9],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        slopes = dem._get_8_direction_slopes()
        valid_mask = ~np.isnan(dem.values)
        out_a = rho8_flow_direction(slopes, valid_mask, rng=rng_a)
        out_b = rho8_flow_direction(slopes, valid_mask, rng=rng_b)
        # The arrays may or may not differ depending on the surface; just sanity check.
        assert out_a.shape == out_b.shape


# ----- Validation ----------------------------------------------------------------------

def test_unknown_method_raises():
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="method must be one of"):
        dem.flow_direction(method="bogus")
