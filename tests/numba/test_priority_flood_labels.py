"""Tests for the labelled Priority-Flood kernel (B2)."""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers._numba import (
    _DIR_DC_I32,
    _DIR_DR_I32,
    priority_flood_labels_numba,
    priority_flood_numba,
)


def _pit_dem() -> np.ndarray:
    """A 7x7 plateau with a deep interior pit (a classic depression)."""
    dem = np.full((7, 7), 10.0, dtype=np.float64)
    dem[2:5, 2:5] = 2.0
    dem[3, 3] = 0.0
    return dem


@pytest.fixture(scope="module")
def offsets():
    return _DIR_DR_I32, _DIR_DC_I32


class TestFillEquivalence:
    """filled must equal priority_flood_numba(epsilon=0) bit-for-bit."""

    def test_pit_dem(self, offsets):
        dr, dc = offsets
        dem = _pit_dem()
        mask = np.zeros(dem.shape, dtype=bool)
        baseline = priority_flood_numba(dem, mask, 0.0, dr, dc)
        filled, _ = priority_flood_labels_numba(dem, mask, dr, dc)
        np.testing.assert_array_equal(filled, baseline)

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    @pytest.mark.parametrize("shape", [(10, 10), (8, 13), (5, 5)])
    def test_random_dems(self, offsets, seed, shape):
        dr, dc = offsets
        rng = np.random.default_rng(seed)
        dem = rng.uniform(0.0, 100.0, size=shape).astype(np.float64)
        mask = np.zeros(shape, dtype=bool)
        baseline = priority_flood_numba(dem, mask, 0.0, dr, dc)
        filled, _ = priority_flood_labels_numba(dem, mask, dr, dc)
        np.testing.assert_array_equal(filled, baseline)

    def test_with_nodata_region(self, offsets):
        dr, dc = offsets
        dem = _pit_dem()
        mask = np.zeros(dem.shape, dtype=bool)
        mask[0, :2] = True  # a no-data patch on the edge
        baseline = priority_flood_numba(dem, mask, 0.0, dr, dc)
        filled, labels = priority_flood_labels_numba(dem, mask, dr, dc)
        # NaN positions match; finite positions equal
        np.testing.assert_array_equal(np.isnan(filled), np.isnan(baseline))
        finite = ~np.isnan(filled)
        np.testing.assert_array_equal(filled[finite], baseline[finite])
        # no-data cells carry label 0
        assert np.all(labels[mask] == 0)


class TestLabels:
    def test_every_data_cell_labelled(self, offsets):
        dr, dc = offsets
        dem = _pit_dem()
        mask = np.zeros(dem.shape, dtype=bool)
        _, labels = priority_flood_labels_numba(dem, mask, dr, dc)
        assert np.all(labels >= 1)
        assert labels.dtype == np.int32

    def test_nodata_cells_are_label_zero(self, offsets):
        dr, dc = offsets
        dem = _pit_dem()
        mask = np.zeros(dem.shape, dtype=bool)
        mask[3, 0] = True
        mask[6, 6] = True
        _, labels = priority_flood_labels_numba(dem, mask, dr, dc)
        assert labels[3, 0] == 0
        assert labels[6, 6] == 0
        assert np.all(labels[~mask] >= 1)

    def test_pit_interior_inherits_one_label(self, offsets):
        # The interior depression should all drain to one outlet -> single label.
        dr, dc = offsets
        dem = _pit_dem()
        mask = np.zeros(dem.shape, dtype=bool)
        _, labels = priority_flood_labels_numba(dem, mask, dr, dc)
        interior = labels[2:5, 2:5]
        assert len(np.unique(interior)) == 1

    def test_deterministic(self, offsets):
        dr, dc = offsets
        rng = np.random.default_rng(123)
        dem = rng.uniform(0.0, 50.0, size=(12, 9)).astype(np.float64)
        mask = np.zeros(dem.shape, dtype=bool)
        f1, l1 = priority_flood_labels_numba(dem, mask, dr, dc)
        f2, l2 = priority_flood_labels_numba(dem, mask, dr, dc)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(l1, l2)
