"""Tests for `DEM.stochastic_depressions` (W-11)."""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


class TestStochasticDepressions:
    """Tests for `DEM.stochastic_depressions`."""

    def test_zero_sigma_matches_deterministic_depressions(self):
        """Test sigma=0 reduces to deterministic depression detection.

        Test scenario:
            With sigma=0 every realisation is identical (no noise); a clear
            depression (single low cell surrounded by higher cells) must
            register with probability 1.0 in every run.
        """
        # Centre cell is a clear sink (elev 0) surrounded by elev 5.
        z = np.full((5, 5), 5.0, dtype=np.float32)
        z[2, 2] = 0.0
        dem = _make_dem(z)
        prob = dem.stochastic_depressions(sigma=0.0, n_runs=3, seed=42)
        arr = prob.read_array()
        # The depression cell has probability 1.0; surrounding cells are 0.0.
        assert arr[2, 2] == pytest.approx(1.0)
        # All non-sink cells should remain 0.0.
        assert (arr[z >= 5.0] == 0.0).all()

    def test_probability_in_unit_interval(self):
        """Test all per-cell probabilities lie in `[0, 1]`.

        Test scenario:
            Any number of runs / sigma combination must keep the per-cell
            probability ≥ 0 and ≤ 1 by construction.
        """
        z = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 1, 2, 1, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        prob = dem.stochastic_depressions(sigma=0.5, n_runs=10, seed=7)
        arr = prob.read_array()
        assert (arr >= 0.0).all()
        assert (arr <= 1.0).all()

    def test_reproducible_with_seed(self):
        """Test the same seed yields identical results across calls.

        Test scenario:
            Two calls with the same seed and parameters must produce
            bit-identical probability rasters.
        """
        z = np.array(
            [
                [5, 5, 5, 5],
                [5, 1, 2, 5],
                [5, 5, 5, 5],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        a = dem.stochastic_depressions(sigma=0.3, n_runs=5, seed=123).read_array()
        b = dem.stochastic_depressions(sigma=0.3, n_runs=5, seed=123).read_array()
        np.testing.assert_array_equal(a, b)

    def test_dtype_is_float32(self):
        """Test the output raster uses float32 storage.

        Test scenario:
            Probability data fits comfortably in float32; the API uses it
            consistently.
        """
        z = np.array(
            [
                [5, 5, 5],
                [5, 1, 5],
                [5, 5, 5],
            ],
            dtype=np.float32,
        )
        dem = _make_dem(z)
        prob = dem.stochastic_depressions(sigma=0.1, n_runs=3, seed=1)
        assert prob.read_array().dtype == np.float32

    def test_negative_sigma_raises(self):
        """Test sigma < 0 raises ValueError.

        Test scenario:
            Negative noise standard deviation is meaningless; the API rejects
            it with a clear error.
        """
        z = np.array([[5, 5], [5, 1]], dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="non-negative"):
            dem.stochastic_depressions(sigma=-0.1, n_runs=5)

    def test_zero_n_runs_raises(self):
        """Test n_runs <= 0 raises ValueError.

        Test scenario:
            Zero or negative run count yields no information; reject.
        """
        z = np.array([[5, 5], [5, 1]], dtype=np.float32)
        dem = _make_dem(z)
        with pytest.raises(ValueError, match="positive"):
            dem.stochastic_depressions(sigma=0.1, n_runs=0)

    def test_many_runs_finish_under_budget(self):
        """L1 regression: many Monte-Carlo iterations stay within a tight time budget.

        Test scenario:
            On a 32×32 DEM with `n_runs=50`, the pre-fix loop created and
            tore down a GDAL Dataset per iteration — at this size that
            still completes in well under 1 s, but ANY perf regression
            that reintroduces the wrapper churn would land here. Keep the
            budget loose enough to survive slow CI but tight enough to
            catch a Dataset-per-iteration regression.
        """
        import time
        rng = np.random.default_rng(123)
        z = rng.uniform(0, 100, size=(32, 32)).astype(np.float32)
        dem = _make_dem(z)
        start = time.perf_counter()
        out = dem.stochastic_depressions(sigma=1.0, n_runs=50, seed=0)
        elapsed = time.perf_counter() - start
        assert out.read_array().shape == z.shape
        assert elapsed < 5.0, f"50-run Monte-Carlo took {elapsed:.2f}s — perf regression?"
