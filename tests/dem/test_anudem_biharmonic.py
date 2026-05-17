"""Tests for the biharmonic mode of `DEM.anudem_interpolate` (P32 backfill)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM


def _make_dem(arr: np.ndarray) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    disk[np.isnan(arr)] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


def test_biharmonic_fills_central_hole():
    """A single NaN at the centre of a smooth tilted plane is filled close
    to the analytic linear value."""
    n = 9
    xs, ys = np.meshgrid(np.arange(n, dtype=np.float32),
                         np.arange(n, dtype=np.float32))
    z = xs + 2.0 * ys
    truth_centre = float(z[n // 2, n // 2])
    z_holed = z.copy()
    z_holed[n // 2, n // 2] = np.nan
    dem = _make_dem(z_holed)
    filled = dem.anudem_interpolate(method="biharmonic", max_iter=500, tol=1e-6)
    out = filled.values
    assert np.isfinite(out[n // 2, n // 2])
    assert abs(out[n // 2, n // 2] - truth_centre) < 0.5


def test_biharmonic_preserves_known_cells():
    z = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(method="biharmonic", max_iter=200, tol=1e-5)
    out = filled.values
    known_mask = np.isfinite(z)
    np.testing.assert_allclose(out[known_mask], z[known_mask], atol=1e-4)


def test_biharmonic_invalid_method_rejected():
    z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="laplacian"):
        dem.anudem_interpolate(method="bogus")


def test_biharmonic_max_iter_one_returns_partial():
    """max_iter=1 stops after a single sweep — output should still be
    finite and the anchors preserved."""
    z = np.array(
        [
            [1.0, 2.0, 3.0, np.nan, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(method="biharmonic", max_iter=1, tol=0.0)
    out = filled.values
    assert np.all(np.isfinite(out))
    known = np.isfinite(z)
    np.testing.assert_allclose(out[known], z[known], atol=1e-4)


def test_biharmonic_high_tol_stops_immediately():
    """When tol is huge, the very first sweep converges and exits."""
    z = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(
        method="biharmonic", max_iter=200, tol=1e9
    )
    # The convergence-check branch fires; output should still be finite.
    assert np.all(np.isfinite(filled.values))


def test_biharmonic_with_explicit_mask_treats_cells_as_fixed():
    """The `mask` kwarg adds extra anchor cells to the relaxation."""
    z = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    # Pin every known cell explicitly via mask (redundant — already finite),
    # which exercises the `fixed = fixed | mask` branch.
    extra = np.array(
        [[True, True, True], [True, False, True], [True, True, True]],
        dtype=bool,
    )
    filled = dem.anudem_interpolate(
        method="biharmonic", max_iter=200, tol=1e-5, mask=extra,
    )
    out = filled.values
    known = np.isfinite(z)
    np.testing.assert_allclose(out[known], z[known], atol=1e-4)


def test_biharmonic_all_known_returns_input():
    """A DEM with no NaN holes should return the same surface (anchors
    cover every cell)."""
    z = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(method="biharmonic", max_iter=50)
    np.testing.assert_allclose(filled.values, z, atol=1e-6)


def test_biharmonic_no_anchors_raises():
    """All-NaN input has no anchor cells; should raise ValueError."""
    z = np.full((3, 3), np.nan, dtype=np.float32)
    dem = _make_dem(z)
    with pytest.raises(ValueError, match="anchor"):
        dem.anudem_interpolate(method="biharmonic")


def test_biharmonic_inplace_returns_none():
    """`inplace=True` updates the instance and returns None."""
    z = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    result = dem.anudem_interpolate(
        method="biharmonic", max_iter=50, inplace=True
    )
    assert result is None
    assert np.all(np.isfinite(dem.values))
