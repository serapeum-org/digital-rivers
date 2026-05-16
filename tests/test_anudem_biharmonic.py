"""Tests for the biharmonic mode of ``DEM.anudem_interpolate`` (P32 backfill)."""
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
