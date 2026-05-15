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


def test_topobathy_fusion_stub_raises():
    """P35 raises with the topobathy-fusion deferral note."""
    with pytest.raises(NotImplementedError, match="P35"):
        stubs.topobathy_fusion()


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
