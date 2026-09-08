"""Tests for how the typed wrappers carry pyramids' handle configuration.

pyramids 0.60 added two keyword-only constructor arguments — `gdal_env` and
`open_options` — and `Dataset.read_file` passes both when it builds the
instance, so every subclass has to accept them. It also began refusing a
metadata write on a read-only *on-disk* handle, which turned a promotion that
dropped the source's access mode into a `persist_metadata()` failure.

These tests pin the three boundaries that config crosses: `read_file` into a
subclass, `from_dataset` / `to_dataset` across the typed wrappers, and the
`open()` classmethods.
"""

from __future__ import annotations

import gc

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from shapely.geometry import Point

from digitalrivers import (
    DEM,
    Accumulation,
    FlowDirection,
    StreamRaster,
    Terrain,
    WatershedRaster,
)

GEO_REF = GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326)


@pytest.fixture
def grid() -> np.ndarray:
    """A small deterministic float32 grid used for every raster here."""
    return np.arange(9, dtype=np.float32).reshape(3, 3)


@pytest.fixture
def tif_path(tmp_path, grid: np.ndarray) -> str:
    """Write `grid` to a GeoTIFF and return its path, with the handle flushed."""
    path = str(tmp_path / "raster.tif")
    handle = Dataset.from_array(grid, geo_ref=GEO_REF, no_data_value=-9999.0, path=path)
    del handle
    gc.collect()
    return path


def _plain(grid: np.ndarray, path: str | None = None) -> Dataset:
    """Build a plain `Dataset` over `grid`, on disk when `path` is given."""
    return Dataset.from_array(grid, geo_ref=GEO_REF, no_data_value=-9999.0, path=path)


def _outlets() -> gpd.GeoDataFrame:
    """A one-row pour-point frame, the minimum `WatershedRaster` accepts."""
    return gpd.GeoDataFrame({"geometry": [Point(0.5, -0.5)]}, crs="EPSG:4326")


def _promote(cls, ds: Dataset):
    """Promote `ds` through `cls.from_dataset`, supplying each class's own kwargs."""
    if cls is FlowDirection:
        return cls.from_dataset(ds, routing="d8")
    if cls is Accumulation:
        return cls.from_dataset(ds, routing="d8")
    if cls is StreamRaster:
        return cls.from_dataset(ds, threshold=10, routing="d8")
    return cls.from_dataset(ds, routing="d8", outlets=_outlets())


TYPED_CLASSES = [FlowDirection, Accumulation, StreamRaster, WatershedRaster]


class TestReadFileAcceptsHandleConfig:
    """`read_file` builds the subclass with pyramids' two new keyword arguments."""

    @pytest.mark.parametrize("cls", [DEM, Terrain])
    def test_read_file_constructs_subclass(self, cls, tif_path: str):
        """Test `read_file` returns the subclass rather than raising.

        Args:
            cls: The `Dataset` subclass under test.
            tif_path: Fixture path to a flushed GeoTIFF.

        Test scenario:
            pyramids' `read_file` calls `cls(src, access=..., gdal_env=...,
            open_options=...)`. A subclass whose `__init__` does not accept the
            two keyword-only arguments raises `TypeError` here.
        """
        result = cls.read_file(tif_path)
        assert (
            type(result) is cls
        ), f"Expected {cls.__name__}, got {type(result).__name__}"

    @pytest.mark.parametrize("cls", [DEM, Terrain])
    def test_read_file_captures_explicit_config(self, cls, tif_path: str):
        """Test an explicit `gdal_env` / `open_options` survives onto the instance.

        Args:
            cls: The `Dataset` subclass under test.
            tif_path: Fixture path to a flushed GeoTIFF.

        Test scenario:
            Config passed to `read_file` must be captured on the returned object,
            because that is what pyramids re-installs on the paths that reopen the
            file (threadsafe reads, lazy chunked reads, unpickling on a worker).
        """
        env = {"GDAL_HTTP_MAX_RETRY": "3"}
        result = cls.read_file(
            tif_path, gdal_env=env, open_options={"NUM_THREADS": "1"}
        )
        assert result.gdal_env == env, f"gdal_env not captured: {result.gdal_env}"
        assert result.open_options == [
            "NUM_THREADS=1"
        ], f"open_options not captured: {result.open_options}"


class TestFromDatasetCarriesConfig:
    """Promoting a plain `Dataset` preserves access and driver configuration."""

    @pytest.mark.parametrize("cls", TYPED_CLASSES)
    def test_from_dataset_preserves_write_access(self, cls, grid: np.ndarray, tmp_path):
        """Test a writable file-backed source stays writable after promotion.

        Args:
            cls: The typed wrapper under test.
            grid: Fixture array.
            tmp_path: pytest temporary directory.

        Test scenario:
            `from_array(path=...)` yields a `"write"` handle. Dropping that on
            promotion produced a wrapper pyramids refuses metadata writes on.
        """
        source = _plain(grid, path=str(tmp_path / f"{cls.__name__}.tif"))
        promoted = _promote(cls, source)
        assert (
            promoted.access == source.access
        ), f"{cls.__name__} changed access {source.access!r} -> {promoted.access!r}"

    @pytest.mark.parametrize("cls", TYPED_CLASSES)
    def test_from_dataset_preserves_driver_config(
        self, cls, grid: np.ndarray, tmp_path
    ):
        """Test the source's `gdal_env` and `open_options` reach the wrapper.

        Args:
            cls: The typed wrapper under test.
            grid: Fixture array.
            tmp_path: pytest temporary directory.

        Test scenario:
            A signed remote raster carries its credentials in `gdal_env` and its
            driver settings in `open_options`; losing either on promotion breaks
            every later reopen. Both are asserted, since only one of the pair was
            covered when these tests were first written.
        """
        path = str(tmp_path / f"{cls.__name__}_env.tif")
        _plain(grid, path=path)
        gc.collect()
        source = Dataset.read_file(
            path,
            gdal_env={"GDAL_HTTP_MAX_RETRY": "3"},
            open_options={"NUM_THREADS": "1"},
        )
        promoted = _promote(cls, source)
        assert (
            promoted.gdal_env == source.gdal_env
        ), f"{cls.__name__} dropped gdal_env: {promoted.gdal_env}"
        assert (
            promoted.open_options == source.open_options
        ), f"{cls.__name__} dropped open_options: {promoted.open_options}"

    def test_persist_metadata_succeeds_on_promoted_file_handle(
        self, grid: np.ndarray, tmp_path
    ):
        """Test `persist_metadata` works on a wrapper promoted from a writable file.

        Args:
            grid: Fixture array.
            tmp_path: pytest temporary directory.

        Test scenario:
            The regression this branch fixed. pyramids refuses a metadata write on
            a read-only on-disk handle, so a promotion that reset access to
            `"read_only"` made `persist_metadata()` raise `ReadOnlyError`.
        """
        source = _plain(grid, path=str(tmp_path / "persist.tif"))
        fd = FlowDirection.from_dataset(source, routing="mfd_quinn")
        fd.persist_metadata()
        assert fd.routing == "mfd_quinn", f"routing changed: {fd.routing}"


class TestToDatasetCarriesConfig:
    """Unwrapping is symmetric with promotion — it keeps access and config too."""

    @pytest.mark.parametrize("cls", [FlowDirection, Accumulation, StreamRaster])
    def test_to_dataset_preserves_write_access(self, cls, grid: np.ndarray, tmp_path):
        """Test `to_dataset` returns a handle with the wrapper's access mode.

        Args:
            cls: The typed wrapper under test.
            grid: Fixture array.
            tmp_path: pytest temporary directory.

        Test scenario:
            `to_dataset` used to build `Dataset(self.raster)`, defaulting to
            `"read_only"`, so a round trip silently downgraded a writable handle.
        """
        path = str(tmp_path / f"{cls.__name__}_out.tif")
        _plain(grid, path=path)
        gc.collect()
        source = Dataset.read_file(
            path, read_only=False, open_options={"NUM_THREADS": "1"}
        )
        wrapper = _promote(cls, source)
        unwrapped = wrapper.to_dataset()
        assert (
            unwrapped.access == wrapper.access
        ), f"{cls.__name__}.to_dataset dropped access {wrapper.access!r}"
        assert (
            unwrapped.open_options == wrapper.open_options
        ), f"{cls.__name__}.to_dataset dropped open_options"


class TestOpenAcceptsHandleConfig:
    """The `open()` classmethods expose the access and config knobs."""

    @pytest.fixture
    def tagged_tif(self, grid: np.ndarray, tmp_path) -> str:
        """A GeoTIFF carrying `DR_ROUTING`, flushed so a reopen sees the tags."""
        path = str(tmp_path / "tagged.tif")
        source = _plain(grid, path=path)
        fd = FlowDirection.from_dataset(source, routing="d8")
        fd.persist_metadata()
        del fd, source
        gc.collect()
        return path

    def test_open_defaults_to_read_only(self, tagged_tif: str):
        """Test `open()` without arguments yields a read-only handle.

        Args:
            tagged_tif: Fixture path to a tagged GeoTIFF.

        Test scenario:
            Reading is the default; the caller opts into writing explicitly.
        """
        opened = FlowDirection.open(tagged_tif)
        assert (
            opened.access == "read_only"
        ), f"Expected read_only, got {opened.access!r}"

    def test_open_read_only_false_yields_writable_handle(self, tagged_tif: str):
        """Test `open(read_only=False)` returns a handle that accepts metadata writes.

        Args:
            tagged_tif: Fixture path to a tagged GeoTIFF.

        Test scenario:
            Without this argument `persist_metadata()` was unreachable through the
            package's own entry point — the only way in was a hand-built subclass.
        """
        opened = FlowDirection.open(tagged_tif, read_only=False)
        assert opened.access == "write", f"Expected write, got {opened.access!r}"
        opened.persist_metadata()

    def test_open_captures_gdal_env(self, tagged_tif: str):
        """Test `open()` forwards and captures `gdal_env`.

        Args:
            tagged_tif: Fixture path to a tagged GeoTIFF.

        Test scenario:
            The config must reach `Dataset.read_file` and survive onto the result,
            otherwise a signed remote raster cannot be opened through `open()`.
        """
        env = {"GDAL_HTTP_MAX_RETRY": "3"}
        opened = FlowDirection.open(tagged_tif, gdal_env=env)
        assert opened.gdal_env == env, f"gdal_env not captured: {opened.gdal_env}"
