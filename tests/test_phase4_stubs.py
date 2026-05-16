"""Tests for Phase 4 API surfaces (P28-P35).

Phase 4 is the scalability / advanced phase; all eight tasks are L-effort
and shipped as ``NotImplementedError`` stubs in this initial cut. These
tests assert the stubs are present and raise the documented error so
downstream code can wire against the API today and pick up the
implementations when they land.
"""
from __future__ import annotations

import pytest

from digitalrivers import phase4
from digitalrivers import _phase4_stubs as stubs  # umbrella stubs only


def test_native_cotat_upscale_stub_raises():
    """P28 raises NotImplementedError with a P28-specific message."""
    with pytest.raises(NotImplementedError, match="P28"):
        stubs.native_cotat_upscale()


def test_native_ihu_upscale_stub_points_to_working_api():
    """P29 stub now points callers at the working IHU implementation
    on FlowDirection.upscale_ihu / .upscale(method='ihu')."""
    with pytest.raises(NotImplementedError, match="upscale_ihu"):
        stubs.native_ihu_upscale()


def test_dask_backend_stub_raises():
    """P30 raises with Dask backend deferral note."""
    with pytest.raises(NotImplementedError, match="P30"):
        stubs.dask_backend()


def test_cloud_io_stub_raises():
    """P31 raises with cloud-IO deferral note."""
    with pytest.raises(NotImplementedError, match="P31"):
        stubs.cloud_io()


def test_anudem_solver_stub_points_to_biharmonic():
    """P32 umbrella now points callers at the working biharmonic method
    on DEM.anudem_interpolate."""
    with pytest.raises(NotImplementedError, match="anudem_interpolate"):
        stubs.anudem_solver()


def test_mesh_quality_optimise_stub_points_to_mesh_module():
    """P33 umbrella now points callers at Mesh.laplacian_smooth."""
    with pytest.raises(NotImplementedError, match="laplacian_smooth"):
        stubs.mesh_quality_optimise()


def test_pdal_pipeline_stub_points_to_grid_lidar_points():
    """P34 umbrella now points callers at the working grid_lidar_points."""
    with pytest.raises(NotImplementedError, match="grid_lidar_points"):
        stubs.pdal_lidar_pipeline()


def test_topobathy_fusion_now_implemented_max_blend():
    """P35 implemented in the backfill commit: max-blend topo/bathy fuse."""
    import numpy as np
    from pyramids.dataset import Dataset

    topo = Dataset.create_from_array(
        np.array([[5.0, -1.0], [3.0, -2.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    bathy = Dataset.create_from_array(
        np.array([[-3.0, -5.0], [-4.0, -6.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    fused = phase4.topobathy_fusion(topo, bathy, blend="max")
    arr = fused.read_array()
    # Cell-by-cell max picks the higher value.
    expected = np.array([[5.0, -1.0], [3.0, -2.0]], dtype=np.float32)
    np.testing.assert_allclose(arr, expected, atol=1e-3)


def test_topobathy_fusion_topo_above_branch():
    """topo_above pulls topo above the shoreline, bathy below."""
    import numpy as np
    from pyramids.dataset import Dataset

    topo = Dataset.create_from_array(
        np.array([[5.0, -1.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    bathy = Dataset.create_from_array(
        np.array([[-3.0, -5.0]], dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    fused = phase4.topobathy_fusion(topo, bathy, blend="topo_above",
                                    shoreline_elev=0.0)
    arr = fused.read_array()
    # Cell 0: topo=5 >= 0 → topo wins (5). Cell 1: topo=-1 < 0 → bathy (-5).
    np.testing.assert_allclose(arr[0], [5.0, -5.0], atol=1e-3)


def test_topobathy_fusion_invalid_blend_raises():
    import numpy as np
    from pyramids.dataset import Dataset

    topo = Dataset.create_from_array(
        np.zeros((2, 2), dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="blend must be"):
        phase4.topobathy_fusion(topo, topo, blend="bogus")


def test_tile_windows_partitions_dataset_into_tiles():
    """tile_windows yields edge-clipped (row, col, h, w) windows."""
    import numpy as np
    from pyramids.dataset import Dataset

    ds = Dataset.create_from_array(
        np.zeros((10, 10), dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    wins = list(phase4.tile_windows(ds, tile_rows=4, tile_cols=4))
    # 10 / 4 = 3 row stripes (4, 4, 2) and 3 col stripes (4, 4, 2) = 9 tiles.
    assert len(wins) == 9
    # Edge tile clipped to remaining size.
    assert (8, 8, 2, 2) in wins


def test_tile_windows_invalid_sizes_raise():
    import numpy as np
    from pyramids.dataset import Dataset

    ds = Dataset.create_from_array(
        np.zeros((4, 4), dtype=np.float32),
        top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
    )
    with pytest.raises(ValueError, match="tile_rows"):
        list(phase4.tile_windows(ds, tile_rows=0, tile_cols=2))
    with pytest.raises(ValueError, match="overlap"):
        list(phase4.tile_windows(ds, tile_rows=2, tile_cols=2, overlap=-1))


def test_module_loads_and_exposes_all_eight_stubs():
    """The module wires up the full Phase 4 API surface so downstream
    callers can ``from digitalrivers._phase4_stubs import …``."""
    expected = {
        "native_cotat_upscale", "native_ihu_upscale", "dask_backend",
        "cloud_io", "anudem_solver", "mesh_quality_optimise",
        "pdal_lidar_pipeline", "topobathy_fusion",
    }
    public = {name for name in dir(stubs) if not name.startswith("_")}
    missing = expected - public
    assert not missing, f"Missing Phase 4 stubs: {missing}"
