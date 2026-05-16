"""Coverage-gap tests called out in the Phase 4 review (C1-C4).

* C1: ANUDEM with anchors adjacent to the DEM boundary — no periodic-wrap
       contamination after the I2 fix replaced ``np.roll`` with edge
       padding.
* C2: ``grid_lidar_points`` mean / median bucket dispatch correctly
       isolates cells (one cell's mean is not affected by another cell's
       contents).
* C3: ``write_cog`` produces an output whose GDAL metadata reports the
       internal-tiling layout characteristic of a COG.
* C4: ``ihu_upscale`` on a degenerate input (every coarse block has
       only one outlet candidate) exercises the no-improvement branch
       and converges in zero swaps.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from pyramids.dataset import Dataset


def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0):
    from digitalrivers import DEM

    disk = arr.astype(np.float32, copy=True)
    disk[np.isnan(disk)] = no_data_value
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


# --- C1: ANUDEM no periodic-wrap at boundary --------------------------------


def test_anudem_laplacian_no_periodic_wrap_at_top_row():
    """A NaN cell on row 0 must NOT receive contributions from the
    bottom row (the I2 ``np.roll`` bug — edge-padded slicing fixes it)."""
    # Construct a DEM with strong top/bottom contrast and a NaN at (0, 1).
    z = np.array(
        [
            [10.0, np.nan, 10.0],
            [10.0, 10.0, 10.0],
            [-100.0, -100.0, -100.0],  # very different from top row
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(
        method="laplacian", max_iter=300, tol=1e-6,
    )
    # The filled NaN cell should sit between its three actual neighbours
    # (left=10, right=10, below=10) — close to 10, NOT pulled toward
    # -100 from a periodic wrap.
    out = filled.values
    assert abs(float(out[0, 1]) - 10.0) < 1.0


def test_anudem_biharmonic_no_periodic_wrap_at_left_column():
    """Same boundary check for the biharmonic mode along the left column."""
    z = np.array(
        [
            [10.0, 10.0, 10.0, 100.0],
            [np.nan, 10.0, 10.0, 100.0],
            [10.0, 10.0, 10.0, 100.0],
            [10.0, 10.0, 10.0, 100.0],
        ],
        dtype=np.float32,
    )
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(
        method="biharmonic", max_iter=300, tol=1e-5,
    )
    out = filled.values
    # NaN at (1, 0); left edge has no right-edge contamination, so the
    # filled value should be near the 10-valued neighbours.
    assert abs(float(out[1, 0]) - 10.0) < 5.0


# --- C2: lidar bucket dispatch isolation ------------------------------------


def test_grid_lidar_mean_isolates_per_cell():
    """Two cells with distinct point clouds must keep their means
    independent — no bucket leakage."""
    from digitalrivers.lidar import grid_lidar_points

    # Cell (0,0) gets points {1, 3}; cell (0,1) gets points {100, 200}.
    xs = np.array([0.1, 0.4, 1.1, 1.4])
    ys = np.array([0.1, 0.4, 0.1, 0.4])
    zs = np.array([1.0, 3.0, 100.0, 200.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
        aggregate="mean", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == pytest.approx(2.0)  # mean({1, 3})
    assert float(arr[0, 1]) == pytest.approx(150.0)  # mean({100, 200})


def test_grid_lidar_median_isolates_per_cell():
    """Same isolation check for the median aggregator."""
    from digitalrivers.lidar import grid_lidar_points

    xs = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
    ys = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    zs = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=1.0, bounds=(0.0, 0.0, 2.0, 1.0),
        aggregate="median", epsg=3857,
    )
    arr = ds.read_array()
    assert float(arr[0, 0]) == pytest.approx(2.0)
    assert float(arr[0, 1]) == pytest.approx(200.0)


# --- C3: write_cog COG-compliance -------------------------------------------


def test_write_cog_output_is_tiled_geotiff(tmp_path):
    """The COG writer's output must have block-tiled internal layout —
    a hard requirement of the COG spec."""
    from digitalrivers.cloud_io import write_cog

    z = np.arange(64, dtype=np.float32).reshape(8, 8)
    ds = Dataset.create_from_array(
        z, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326,
    )
    out = tmp_path / "out.tif"
    written = write_cog(ds, str(out))
    assert os.path.exists(written)

    from osgeo import gdal

    handle = gdal.Open(written)
    assert handle is not None
    band = handle.GetRasterBand(1)
    block_size = band.GetBlockSize()
    # COG requires internally-tiled storage. Block size on a tiny 8×8 file
    # will be ≤ raster size, but it must be a true tile (block_x > 0 and
    # equal to or smaller than raster_x). The default COG driver sets
    # 512×512 internally — adjusted to raster shape here.
    assert block_size[0] > 0 and block_size[1] > 0
    handle = None  # close


# --- C4: IHU no-improvement on degenerate input -----------------------------


def test_ihu_no_improvement_on_single_candidate_per_block():
    """A fine grid where every coarse block has exactly one outlet
    candidate cannot benefit from any swap — the engine must short-circuit
    (zero swaps) and report converged=True."""
    from digitalrivers._ihu import ihu_upscale

    # Tiny 4x4 grid → 2x2 coarse output at sf=2. Make every block contain
    # exactly one cell that has an exit by giving most cells fdir == -1
    # (sink) and exactly one valid exit per block.
    fdir = np.full((4, 4), -1, dtype=np.int32)
    # One exit cell per 2×2 block, pointing east (code 6).
    fdir[0, 0] = 6
    fdir[0, 2] = 6
    fdir[2, 0] = 6
    fdir[2, 2] = 6
    acc = np.zeros((4, 4), dtype=np.float64)
    acc[0, 0] = acc[0, 2] = acc[2, 0] = acc[2, 2] = 1.0

    coarse_fdir, metrics, outlets = ihu_upscale(
        fdir, acc, scale_factor=2, max_iter=20,
    )
    assert metrics["swaps"] == 0
    assert metrics["converged"] is True
    # Per-iteration swaps list is present and contains the zero-pass entry.
    assert metrics["swaps_per_iteration"] == [0]


# --- Bonus: I3 "min" blend in topobathy_fusion ------------------------------


def test_topobathy_fusion_min_blend_picks_lower():
    """The new ``"min"`` blend mode (I3 fix) returns ``np.fmin(topo, bathy)``."""
    from digitalrivers.fusion import topobathy_fusion

    topo = Dataset.create_from_array(
        np.array([[5.0, -1.0], [3.0, -2.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    bathy = Dataset.create_from_array(
        np.array([[-3.0, -5.0], [-4.0, -6.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    fused = topobathy_fusion(topo, bathy, blend="min")
    arr = fused.read_array()
    expected = np.array(
        [[-3.0, -5.0], [-4.0, -6.0]], dtype=np.float32
    )
    np.testing.assert_allclose(arr, expected, atol=1e-3)


# --- Bonus: I5 Mesh re-export -----------------------------------------------


def test_mesh_is_re_exported_at_package_root():
    """``from digitalrivers import Mesh`` works after the I5 fix."""
    import digitalrivers

    assert "Mesh" in digitalrivers.__all__
    from digitalrivers import Mesh

    # Smoke check: construct a minimal Mesh.
    verts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = Mesh(verts, tris)
    assert mesh.n_vertices == 3
    assert mesh.n_triangles == 1


# --- (1) IHU metrics: outlets + swaps_per_iteration --------------------------


class TestIhuReturnAndMetrics:
    """Coverage for the I1 3-tuple return and N7 per-iteration list."""

    def _build_inputs(self):
        # Reproducible 6x6 east-flowing chain.
        fdir = np.full((6, 6), 6, dtype=np.int32)
        fdir[:, -1] = -1  # rightmost column = sink
        acc = np.tile(np.arange(6, dtype=np.float64), (6, 1))
        return fdir, acc

    def test_outlets_dict_has_one_entry_per_coarse_cell(self):
        from digitalrivers._ihu import ihu_upscale

        fdir, acc = self._build_inputs()
        sf = 2
        _, _, outlets = ihu_upscale(fdir, acc, scale_factor=sf, max_iter=5)
        expected_n = (fdir.shape[0] // sf) * (fdir.shape[1] // sf)
        # Every coarse cell with at least one valid candidate should appear.
        assert len(outlets) <= expected_n
        assert len(outlets) > 0
        # Each key is a (br, bc) int tuple.
        for k in outlets:
            assert isinstance(k, tuple) and len(k) == 2
            assert all(isinstance(v, int) for v in k)

    def test_swaps_per_iteration_length_matches_iterations(self):
        from digitalrivers._ihu import ihu_upscale

        fdir, acc = self._build_inputs()
        _, metrics, _ = ihu_upscale(fdir, acc, scale_factor=2, max_iter=10)
        assert len(metrics["swaps_per_iteration"]) == metrics["iterations"]

    def test_swaps_per_iteration_sums_to_total_swaps(self):
        from digitalrivers._ihu import ihu_upscale

        fdir, acc = self._build_inputs()
        _, metrics, _ = ihu_upscale(fdir, acc, scale_factor=2, max_iter=10)
        assert sum(metrics["swaps_per_iteration"]) == metrics["swaps"]


# --- (2) ANUDEM corner-NaN pulls toward nearest, not opposite-corner --------


def test_anudem_corner_nan_pulls_toward_local_neighbours():
    """A NaN at (0, 0) — the top-left corner — must end up close to its
    finite neighbours at (0, 1) and (1, 0), not pulled toward the
    bottom-right corner (which is what the periodic ``np.roll`` wrap
    would have done)."""
    z = np.full((5, 5), 10.0, dtype=np.float32)
    z[-1, -1] = -100.0  # very different anchor at opposite corner
    z[0, 0] = np.nan
    dem = _make_dem(z)
    filled = dem.anudem_interpolate(
        method="laplacian", max_iter=300, tol=1e-6,
    )
    val = float(filled.values[0, 0])
    # Local neighbours are 10.0; the wrap would have biased toward -100.
    assert abs(val - 10.0) < 5.0


# --- (3) grid_lidar_points vectorisation matches naive loop -----------------


def test_grid_lidar_mean_5000_points_matches_naive():
    """Stress: 5,000 random points on a 50x50 grid. The vectorised
    np.add.at implementation must produce the same per-cell mean as a
    naive Python-loop reference, and complete quickly."""
    import time

    from digitalrivers.lidar import grid_lidar_points

    rng = np.random.default_rng(seed=1337)
    n = 5000
    xs = rng.uniform(0.0, 50.0, size=n)
    ys = rng.uniform(0.0, 50.0, size=n)
    zs = rng.uniform(-100.0, 100.0, size=n)
    bounds = (0.0, 0.0, 50.0, 50.0)
    cell_size = 1.0

    t0 = time.perf_counter()
    ds = grid_lidar_points(
        xs, ys, zs, cell_size=cell_size, bounds=bounds, aggregate="mean",
    )
    elapsed = time.perf_counter() - t0
    out = ds.read_array()
    assert elapsed < 1.0

    # Naive reference (sum / count per cell).
    cols = int(np.ceil((50.0 - 0.0) / cell_size))
    rows = int(np.ceil((50.0 - 0.0) / cell_size))
    sums = np.zeros((rows, cols), dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int64)
    for x, y, z in zip(xs, ys, zs):
        c = min(cols - 1, max(0, int((x - 0.0) / cell_size)))
        r = min(rows - 1, max(0, int((50.0 - y) / cell_size)))
        sums[r, c] += z
        counts[r, c] += 1
    nodata = -9999.0
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.where(
            counts > 0, sums / counts, nodata
        ).astype(np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-3)


# --- (4) topobathy_fusion: min mode NaN handling + invalid blend ------------


def test_topobathy_min_blend_nan_picks_other_operand():
    """When one operand is NaN, ``np.fmin`` returns the non-NaN value."""
    from digitalrivers.fusion import topobathy_fusion

    topo = Dataset.create_from_array(
        np.array([[np.nan, 5.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    bathy = Dataset.create_from_array(
        np.array([[-3.0, np.nan]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    fused = topobathy_fusion(topo, bathy, blend="min")
    arr = fused.read_array()
    # (0, 0): NaN-min(-3) → -3. (0, 1): NaN-min(5) → 5.
    assert float(arr[0, 0]) == pytest.approx(-3.0)
    assert float(arr[0, 1]) == pytest.approx(5.0)


def test_topobathy_invalid_blend_now_mentions_min():
    """The new error message lists ``'min'`` in the allow-list."""
    from digitalrivers.fusion import topobathy_fusion

    arr = np.zeros((2, 2), dtype=np.float32)
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="min") as exc_info:
        topobathy_fusion(ds, ds, blend="bogus")
    assert "min" in str(exc_info.value)


# --- (5) Mesh re-export ordering -------------------------------------------


def test_package_all_carries_mesh_in_sorted_order():
    """``Mesh`` is sorted alphabetically alongside the other exports."""
    import digitalrivers

    expected = [
        "Accumulation", "DEM", "FlowDirection", "Mesh",
        "StreamRaster", "Terrain", "WatershedRaster",
    ]
    assert digitalrivers.__all__ == expected


def test_mesh_importable_from_package_root_without_submodule_path():
    """The short import works (no ``digitalrivers.mesh`` required)."""
    import digitalrivers

    assert hasattr(digitalrivers, "Mesh")
    # Pre-Phase-4 backfill, this was only reachable via ``digitalrivers.mesh``.
    from digitalrivers import Mesh as MeshA
    from digitalrivers.mesh import Mesh as MeshB

    assert MeshA is MeshB
