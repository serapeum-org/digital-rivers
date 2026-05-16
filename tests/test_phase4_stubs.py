"""Tests for Phase 4 API surfaces (P28-P35).

Phase 4 is the scalability / advanced phase; all eight tasks are L-effort
and shipped as ``NotImplementedError`` stubs in this initial cut. These
tests assert the stubs are present and raise the documented error so
downstream code can wire against the API today and pick up the
implementations when they land.
"""
from __future__ import annotations

import pytest

from digitalrivers import _phase4_stubs as stubs


def test_native_cotat_upscale_stub_raises():
    """P28 raises NotImplementedError with a P28-specific message."""
    with pytest.raises(NotImplementedError, match="P28"):
        stubs.native_cotat_upscale()


def test_native_ihu_upscale_stub_raises():
    """P29 raises NotImplementedError citing Eilander 2021."""
    with pytest.raises(NotImplementedError, match="P29"):
        stubs.native_ihu_upscale()


def test_dask_backend_stub_raises():
    """P30 raises with Dask backend deferral note."""
    with pytest.raises(NotImplementedError, match="P30"):
        stubs.dask_backend()


def test_cloud_io_stub_raises():
    """P31 raises with cloud-IO deferral note."""
    with pytest.raises(NotImplementedError, match="P31"):
        stubs.cloud_io()


def test_anudem_solver_stub_raises():
    """P32 raises with the ANUDEM solver deferral note."""
    with pytest.raises(NotImplementedError, match="P32"):
        stubs.anudem_solver()


def test_mesh_quality_optimise_stub_raises():
    """P33 raises with the mesh-optimisation deferral note."""
    with pytest.raises(NotImplementedError, match="P33"):
        stubs.mesh_quality_optimise()


def test_pdal_pipeline_stub_raises():
    """P34 raises with the PDAL deferral note."""
    with pytest.raises(NotImplementedError, match="P34"):
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
    fused = stubs.topobathy_fusion(topo, bathy, blend="max")
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
    fused = stubs.topobathy_fusion(topo, bathy, blend="topo_above",
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
        stubs.topobathy_fusion(topo, topo, blend="bogus")


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
