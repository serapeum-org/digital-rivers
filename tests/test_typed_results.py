"""Tests for the typed result classes introduced by P1.

Covers ``FlowDirection``, ``Accumulation``, and ``StreamRaster``: strict
type discipline, required-routing safety property, no-silent-fallback on
open, explicit-routing override, cross-type rejection at construction
(the ismulti guard), and metadata round-trip.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers import DEM, Accumulation, FlowDirection, StreamRaster
from digitalrivers._metadata import META_ENCODING, META_ROUTING, META_THRESHOLD


def _make_plain_dataset(arr: np.ndarray) -> Dataset:
    """Helper: build an in-memory Dataset from a small int array."""
    return Dataset.create_from_array(
        arr.astype(np.int32),
        top_left_corner=(0.0, 0.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=-9999,
    )


@pytest.fixture()
def fd_array() -> np.ndarray:
    return np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=np.int32)


@pytest.fixture()
def flow_direction(fd_array: np.ndarray) -> FlowDirection:
    return FlowDirection.from_dataset(_make_plain_dataset(fd_array), routing="d8")


class TestRequiredRouting:
    """`routing` is keyword-only with no default — construction without it fails."""

    def test_init_without_routing_raises(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(TypeError):
            FlowDirection(ds.raster)

    def test_from_dataset_without_routing_raises(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(TypeError):
            FlowDirection.from_dataset(ds)

    def test_inherited_create_from_array_raises(self, fd_array: np.ndarray):
        # Pyramids' classmethod calls cls(dst, access="write") with no routing,
        # so this must raise on the typed subclass. That is the safety property.
        with pytest.raises(TypeError):
            FlowDirection.create_from_array(
                fd_array, top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326
            )

    def test_inherited_dataset_like_raises(self, fd_array: np.ndarray):
        src = _make_plain_dataset(fd_array)
        with pytest.raises(TypeError):
            FlowDirection.dataset_like(src, fd_array)

    def test_inherited_read_file_raises(self, tmp_path, fd_array: np.ndarray):
        path = str(tmp_path / "fd.tif")
        Dataset.create_from_array(
            fd_array,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
            driver_type="GTiff",
            path=path,
        )
        with pytest.raises(TypeError):
            FlowDirection.read_file(path)

    def test_invalid_routing_value_raises(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(ValueError, match="routing must be one of"):
            FlowDirection(ds.raster, routing="bogus")

    def test_invalid_encoding_value_raises(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(ValueError, match="encoding must be one of"):
            FlowDirection(ds.raster, routing="d8", encoding="bogus")


class TestStrictType:
    """Returned types must be the typed subclass, not just a Dataset."""

    def test_from_dataset_returns_flow_direction(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        fd = FlowDirection.from_dataset(ds, routing="d8")
        assert type(fd) is FlowDirection

    def test_dem_flow_direction_returns_flow_direction(
        self, coello_dem_4000: gdal.Dataset
    ):
        # Today (pre-P1) this returns a DEM by accident because
        # pyramids' create_from_array uses cls(...). The strict-type
        # check is the regression test that locks the new behaviour in.
        dem = DEM(coello_dem_4000)
        fd = dem.flow_direction()
        assert type(fd) is FlowDirection
        assert fd.routing == "d8"
        assert fd.encoding == "digitalrivers"


class TestMetadataPersistence:
    """`persist_metadata` + `open` round-trips routing/encoding via tags."""

    def test_round_trip_via_geotiff(self, tmp_path, fd_array: np.ndarray):
        path = str(tmp_path / "fd.tif")
        # Materialise to GeoTIFF so the metadata setter has somewhere to live.
        plain = Dataset.create_from_array(
            fd_array,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
            driver_type="GTiff",
            path=path,
        )
        fd = FlowDirection.from_dataset(plain, routing="dinf", encoding="taudem")
        fd.persist_metadata()
        # Close and reopen via the typed factory.
        del fd, plain
        reopened = FlowDirection.open(path)
        assert reopened.routing == "dinf"
        assert reopened.encoding == "taudem"
        assert reopened.shape == (1, 3, 3)
        assert np.array_equal(reopened.read_array(), fd_array)


class TestOpenFallbackPolicy:
    """`FlowDirection.open` raises rather than silently defaulting to d8."""

    def test_open_without_tags_or_routing_raises(self, tmp_path, fd_array: np.ndarray):
        path = str(tmp_path / "no_tags.tif")
        Dataset.create_from_array(
            fd_array,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
            driver_type="GTiff",
            path=path,
        )
        # No persist_metadata call — file has no DR_ROUTING tag.
        with pytest.raises(ValueError, match="no DR_ROUTING tag"):
            FlowDirection.open(path)

    def test_open_explicit_routing_succeeds_without_tag(
        self, tmp_path, fd_array: np.ndarray
    ):
        path = str(tmp_path / "no_tags.tif")
        Dataset.create_from_array(
            fd_array,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
            driver_type="GTiff",
            path=path,
        )
        fd = FlowDirection.open(path, routing="d8")
        assert fd.routing == "d8"

    def test_open_explicit_routing_overrides_tag(self, tmp_path, fd_array: np.ndarray):
        path = str(tmp_path / "with_tags.tif")
        plain = Dataset.create_from_array(
            fd_array,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999,
            driver_type="GTiff",
            path=path,
        )
        fd_written = FlowDirection.from_dataset(plain, routing="d8")
        fd_written.persist_metadata()
        del fd_written, plain
        # Tag says d8; caller asserts dinf — caller wins.
        fd = FlowDirection.open(path, routing="dinf")
        assert fd.routing == "dinf"


class TestCrossTypeRejection:
    """StreamRaster constructor rejects multi-direction routing (ismulti guard)."""

    @pytest.mark.parametrize("multi_routing", ["dinf", "mfd_quinn", "mfd_holmgren"])
    def test_stream_raster_rejects_multi(
        self, fd_array: np.ndarray, multi_routing: str
    ):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(TypeError, match="single-direction routing"):
            StreamRaster(ds.raster, threshold=10, routing=multi_routing)

    def test_stream_raster_accepts_d8(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        sr = StreamRaster(ds.raster, threshold=10, routing="d8")
        assert type(sr) is StreamRaster
        assert sr.routing == "d8"
        assert sr.threshold == 10


class TestAccumulationProvenance:
    """`Accumulation.routing` exists for provenance and validates input."""

    def test_construction_requires_routing(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        with pytest.raises(TypeError):
            Accumulation(ds.raster)

    def test_accepts_any_valid_routing(self, fd_array: np.ndarray):
        # Unlike StreamRaster, Accumulation does not narrow the routing set —
        # the accumulation surface is scheme-agnostic; routing is provenance.
        for r in ["d8", "dinf", "mfd_quinn", "mfd_holmgren", "rho8"]:
            ds = _make_plain_dataset(fd_array)
            acc = Accumulation(ds.raster, routing=r)
            assert acc.routing == r


class TestStubMethods:
    """``Accumulation.streams`` remained stubbed until P8; ``FlowDirection.accumulate``
    was implemented in P6 (the Kahn topological-sort dispatcher)."""

    def test_accumulation_streams_is_stub(self, fd_array: np.ndarray):
        ds = _make_plain_dataset(fd_array)
        acc = Accumulation(ds.raster, routing="d8")
        with pytest.raises(NotImplementedError, match="P8"):
            acc.streams(threshold=10)


class TestToDataset:
    """`to_dataset` strips the typed wrapper but shares the underlying raster."""

    def test_to_dataset_returns_plain_dataset(self, flow_direction: FlowDirection):
        ds = flow_direction.to_dataset()
        assert type(ds) is Dataset
        # Underlying GDAL handle is the same.
        assert ds.raster is flow_direction.raster
