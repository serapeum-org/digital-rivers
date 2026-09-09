"""Tests for the hydrodynamic-model writers in `digitalrivers.interop.export`.

These writers used to be an if-chain inside `DEM.export`, reachable only with a raster on
disk. Extracting them is what makes this file possible: every test below hands a writer a
3x3 grid built in memory and checks the exact bytes it produces.

The headers are where these formats go wrong in practice. `yllcorner` is the *bottom*
edge, which is not the geotransform's y origin, and getting it wrong shifts a whole model
domain by its own height. Several tests pin that arithmetic directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.interop.export import (
    TARGETS,
    ExportGrid,
    write,
    write_gmsh,
    write_hec_ras,
    write_iber,
    write_lisflood_fp,
    write_sfincs,
    write_tuflow,
)

NO_DATA = -9999.0


@pytest.fixture(scope="function")
def grid() -> ExportGrid:
    """A 3x3 grid at 10 m cells with one no-data cell.

    The origin is `(100, 500)` at the top-left with a `-10` y pitch, so the bottom edge
    sits at `500 + 3 * -10 = 470`. Every header assertion below is against that number.

    Returns:
        ExportGrid: The fixture grid, no-data already substituted.
    """
    elev = np.array(
        [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    nan_mask = np.isnan(elev)
    return ExportGrid(
        values=np.where(nan_mask, NO_DATA, elev),
        nan_mask=nan_mask,
        geotransform=(100.0, 10.0, 0.0, 500.0, 0.0, -10.0),
        epsg=32636,
        no_data=NO_DATA,
    )


class TestExportGrid:
    """Tests for the ExportGrid value object."""

    def test_shape_is_the_array_shape(self, grid):
        """`shape` reports the underlying array's rows and columns."""
        assert grid.shape == (3, 3), f"Expected (3, 3), got {grid.shape}"

    def test_cell_size_is_the_absolute_x_pitch(self, grid):
        """`cell_size` takes the geotransform's x pitch, without its sign."""
        assert grid.cell_size == 10.0, f"Expected 10.0, got {grid.cell_size}"

    def test_cell_size_is_positive_for_a_negative_x_pitch(self):
        """A mirrored geotransform still reports a positive cell size."""
        g = ExportGrid(
            values=np.zeros((2, 2)),
            nan_mask=np.zeros((2, 2), dtype=bool),
            geotransform=(0.0, -5.0, 0.0, 0.0, 0.0, -5.0),
            epsg=4326,
            no_data=NO_DATA,
        )
        assert g.cell_size == 5.0, f"Expected 5.0, got {g.cell_size}"

    def test_xllcorner_is_the_geotransform_origin(self, grid):
        """X of the lower-left corner is the geotransform's x origin unchanged."""
        assert grid.xllcorner == 100.0, f"Expected 100.0, got {grid.xllcorner}"

    def test_yllcorner_is_the_bottom_edge_not_the_origin(self, grid):
        """Y of the lower-left corner is `y0 + rows * dy`, not `y0`.

        The geotransform's y origin is the top edge and its y pitch is negative, so the
        bottom edge is below it. Reporting `y0` here would shift the model domain up by
        its own height.
        """
        assert grid.yllcorner == 470.0, f"Expected 470.0, got {grid.yllcorner}"

    def test_grid_is_frozen(self, grid):
        """`ExportGrid` is immutable, so a writer cannot alter what the next one sees."""
        with pytest.raises(Exception) as exc_info:
            grid.no_data = 0.0
        assert "frozen" in str(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        ), f"Expected a frozen-dataclass error, got: {exc_info.value!r}"


class TestWriteLisfloodFp:
    """Tests for write_lisflood_fp."""

    def test_returns_the_path_under_its_label(self, grid, tmp_path):
        """The writer reports the file it wrote, under the `dem_asc` label."""
        path = str(tmp_path / "dem.asc")
        assert write_lisflood_fp(grid, path) == {
            "dem_asc": path
        }, "Label or path does not match what was written"

    def test_header_carries_the_six_arc_fields(self, grid, tmp_path):
        """The six-line Arc header states shape, corner, cell size and no-data."""
        path = str(tmp_path / "dem.asc")
        write_lisflood_fp(grid, path)
        header = (tmp_path / "dem.asc").read_text().splitlines()[:6]
        assert header == [
            "ncols 3",
            "nrows 3",
            "xllcorner 100.0",
            "yllcorner 470.0",
            "cellsize 10.0",
            "NODATA_value -9999.0",
        ], f"Unexpected header: {header}"

    def test_rows_are_written_at_six_decimals(self, grid, tmp_path):
        """Elevations are space-separated, six decimals, one line per raster row."""
        path = str(tmp_path / "dem.asc")
        write_lisflood_fp(grid, path)
        body = (tmp_path / "dem.asc").read_text().splitlines()[6:]
        assert body[0] == "1.000000 2.000000 3.000000", f"Row 0 wrong: {body[0]!r}"
        assert body[1] == "4.000000 -9999.000000 6.000000", f"Row 1 wrong: {body[1]!r}"

    def test_uses_unix_newlines_on_every_platform(self, grid, tmp_path):
        """The file is written `newline="\\n"`, so it does not gain CRLF on Windows."""
        path = str(tmp_path / "dem.asc")
        write_lisflood_fp(grid, path)
        raw = (tmp_path / "dem.asc").read_bytes()
        assert b"\r\n" not in raw, "Arc grid must not contain CRLF line endings"

    def test_path_is_used_exactly_as_given(self, grid, tmp_path):
        """No extension is appended; LISFLOOD-FP takes whatever name it was handed."""
        path = str(tmp_path / "no_extension_here")
        write_lisflood_fp(grid, path)
        assert (tmp_path / "no_extension_here").exists(), "Writer renamed the output"


class TestWriteHecRas:
    """Tests for write_hec_ras."""

    def test_returns_the_path_under_its_label(self, grid, tmp_path):
        """The writer reports the GeoTIFF it wrote, under the `dem_tif` label."""
        path = str(tmp_path / "dem.tif")
        assert write_hec_ras(grid, path) == {
            "dem_tif": path
        }, "Label or path does not match what was written"

    def test_writes_a_readable_raster_with_the_same_values(self, grid, tmp_path):
        """The GeoTIFF round-trips the grid's values, no-data substitution included."""
        from pyramids.dataset import Dataset

        path = str(tmp_path / "dem.tif")
        write_hec_ras(grid, path)
        arr = Dataset.read_file(path).read_array()
        assert np.allclose(
            arr, grid.values
        ), f"Raster values differ from the grid:\n{arr}\nvs\n{grid.values}"

    def test_writes_float32(self, grid, tmp_path):
        """HEC-RAS Mapper expects a single-band float32 raster."""
        from pyramids.dataset import Dataset

        path = str(tmp_path / "dem.tif")
        write_hec_ras(grid, path)
        arr = Dataset.read_file(path).read_array()
        assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"


class TestWriteTuflow:
    """Tests for write_tuflow."""

    def test_writes_both_the_binary_and_its_header(self, grid, tmp_path):
        """Both `.flt` and `.hdr` are written and reported."""
        path = str(tmp_path / "dem.flt")
        written = write_tuflow(grid, path)
        assert set(written) == {
            "dem_flt",
            "dem_hdr",
        }, f"Expected both artefacts, got {sorted(written)}"
        assert (tmp_path / "dem.flt").exists(), ".flt was not written"
        assert (tmp_path / "dem.hdr").exists(), ".hdr was not written"

    def test_appends_the_flt_extension_when_missing(self, grid, tmp_path):
        """A path without `.flt` gains it, and the header follows the binary's name."""
        written = write_tuflow(grid, str(tmp_path / "dem"))
        assert written["dem_flt"].endswith("dem.flt"), written["dem_flt"]
        assert written["dem_hdr"].endswith("dem.hdr"), written["dem_hdr"]

    def test_does_not_double_the_extension(self, grid, tmp_path):
        """A path that already ends `.flt` is left alone."""
        written = write_tuflow(grid, str(tmp_path / "dem.flt"))
        assert not written["dem_flt"].endswith(
            ".flt.flt"
        ), f"Extension doubled: {written['dem_flt']}"

    def test_binary_is_row_major_little_endian_float32(self, grid, tmp_path):
        """The `.flt` is a raw float32 dump in row-major order, top-left first."""
        write_tuflow(grid, str(tmp_path / "dem.flt"))
        raw = np.fromfile(tmp_path / "dem.flt", dtype="<f4")
        assert np.allclose(
            raw, grid.values.astype(np.float32).ravel()
        ), f"Binary does not match the grid:\n{raw}"

    def test_header_declares_the_byte_order(self, grid, tmp_path):
        """The header ends with the `byteorder` line that the six Arc fields lack."""
        write_tuflow(grid, str(tmp_path / "dem.flt"))
        lines = (tmp_path / "dem.hdr").read_text().splitlines()
        assert lines[:6] == [
            "ncols 3",
            "nrows 3",
            "xllcorner 100.0",
            "yllcorner 470.0",
            "cellsize 10.0",
            "NODATA_value -9999.0",
        ], f"Unexpected Arc header: {lines[:6]}"
        assert lines[6] == "byteorder LSBFIRST", f"Unexpected byte order: {lines[6]!r}"


class TestWriteSfincs:
    """Tests for write_sfincs."""

    def test_writes_both_the_depth_and_its_mask(self, grid, tmp_path):
        """Both `.dep` and `.msk` are written and reported."""
        written = write_sfincs(grid, str(tmp_path / "dem.dep"))
        assert set(written) == {
            "dem_dep",
            "dem_msk",
        }, f"Expected both artefacts, got {sorted(written)}"

    def test_appends_the_dep_extension_when_missing(self, grid, tmp_path):
        """A path without `.dep` gains it, and the mask follows its name."""
        written = write_sfincs(grid, str(tmp_path / "dem"))
        assert written["dem_dep"].endswith("dem.dep"), written["dem_dep"]
        assert written["dem_msk"].endswith("dem.msk"), written["dem_msk"]

    def test_depth_file_has_no_header(self, grid, tmp_path):
        """`.dep` is exactly `rows * cols` float32 values and nothing else.

        SFINCS takes the shape from its own control file, so a header here would be read
        as data.
        """
        write_sfincs(grid, str(tmp_path / "dem.dep"))
        size = (tmp_path / "dem.dep").stat().st_size
        assert size == 3 * 3 * 4, f"Expected 36 bytes of float32, got {size}"

    def test_mask_is_zero_at_no_data_and_one_elsewhere(self, grid, tmp_path):
        """The `.msk` companion marks the no-data cell `0` and every other cell `1`."""
        write_sfincs(grid, str(tmp_path / "dem.dep"))
        mask = np.fromfile(tmp_path / "dem.msk", dtype=np.uint8).reshape(3, 3)
        expected = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        assert np.array_equal(mask, expected), f"Mask wrong:\n{mask}"

    def test_mask_is_all_ones_when_nothing_is_no_data(self, tmp_path):
        """A grid with no `NaN` produces a mask of ones."""
        g = ExportGrid(
            values=np.ones((2, 2), dtype=np.float32),
            nan_mask=np.zeros((2, 2), dtype=bool),
            geotransform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
            epsg=4326,
            no_data=NO_DATA,
        )
        write_sfincs(g, str(tmp_path / "d.dep"))
        mask = np.fromfile(tmp_path / "d.msk", dtype=np.uint8)
        assert (mask == 1).all(), f"Expected every cell valid, got {mask}"


class TestWriteGmsh:
    """Tests for write_gmsh."""

    def test_returns_the_path_under_its_label(self, grid, tmp_path):
        """The writer reports the `.geo` script under the `geo` label."""
        written = write_gmsh(grid, str(tmp_path / "dem.geo"))
        assert set(written) == {"geo"}, f"Expected just 'geo', got {sorted(written)}"

    def test_appends_the_geo_extension_when_missing(self, grid, tmp_path):
        """A path without `.geo` gains it."""
        written = write_gmsh(grid, str(tmp_path / "dem"))
        assert written["geo"].endswith("dem.geo"), written["geo"]

    def test_script_bounds_the_full_raster_extent(self, grid, tmp_path):
        """The four corner points span the raster, x from 100 to 130 and y 470 to 500."""
        write_gmsh(grid, str(tmp_path / "dem.geo"))
        text = (tmp_path / "dem.geo").read_text()
        assert "Point(1) = {100.0, 470.0, 0, cl};" in text, text
        assert "Point(3) = {130.0, 500.0, 0, cl};" in text, text

    def test_characteristic_length_is_the_cell_size(self, grid, tmp_path):
        """`cl` is set to the cell size, so the mesh resolves at raster resolution."""
        write_gmsh(grid, str(tmp_path / "dem.geo"))
        first = (tmp_path / "dem.geo").read_text().splitlines()[0]
        assert first == "cl = 10.0;", f"Unexpected characteristic length: {first!r}"

    def test_script_closes_the_loop_into_a_surface(self, grid, tmp_path):
        """Four lines form a loop, and the loop forms a plane surface Gmsh can mesh."""
        text = (tmp_path / "dem.geo").read_text() if False else None
        write_gmsh(grid, str(tmp_path / "dem.geo"))
        text = (tmp_path / "dem.geo").read_text()
        assert "Line Loop(1) = {1, 2, 3, 4};" in text, "Loop missing"
        assert "Plane Surface(1) = {1};" in text, "Surface missing"

    def test_script_carries_no_elevations(self, grid, tmp_path):
        """Only the footprint is exported; the surface itself is not in the `.geo`."""
        write_gmsh(grid, str(tmp_path / "dem.geo"))
        text = (tmp_path / "dem.geo").read_text()
        assert "1.000000" not in text, "Elevations leaked into the geometry script"


class TestWriteIber:
    """Tests for write_iber."""

    def test_returns_the_path_under_its_label(self, grid, tmp_path):
        """The writer reports the `.dat` file under the `dem_dat` label."""
        written = write_iber(grid, str(tmp_path / "dem.dat"))
        assert set(written) == {
            "dem_dat"
        }, f"Expected just 'dem_dat', got {sorted(written)}"

    def test_appends_the_dat_extension_when_missing(self, grid, tmp_path):
        """A path without `.dat` gains it."""
        written = write_iber(grid, str(tmp_path / "dem"))
        assert written["dem_dat"].endswith("dem.dat"), written["dem_dat"]

    def test_header_is_uppercase_and_commented(self, grid, tmp_path):
        """Iber's header differs from Arc's: a comment line, then uppercase keys."""
        write_iber(grid, str(tmp_path / "dem.dat"))
        lines = (tmp_path / "dem.dat").read_text().splitlines()
        assert lines[0] == "# Iber mesh boundary (auto-generated)", lines[0]
        assert lines[1:5] == [
            "NCOLS 3",
            "NROWS 3",
            "XLLCORNER 100.0",
            "YLLCORNER 470.0",
        ], f"Unexpected header: {lines[1:5]}"

    def test_body_matches_the_lisflood_row_format(self, grid, tmp_path):
        """The elevation block is the same six-decimal row format the Arc writer uses."""
        write_iber(grid, str(tmp_path / "dem.dat"))
        lines = (tmp_path / "dem.dat").read_text().splitlines()
        assert lines[7] == "1.000000 2.000000 3.000000", f"Row 0 wrong: {lines[7]!r}"


class TestWrite:
    """Tests for the `write` dispatcher."""

    @pytest.mark.parametrize("target", sorted(TARGETS))
    def test_every_registered_target_writes_something(self, grid, tmp_path, target):
        """Each registry entry dispatches to a writer that produces real files.

        Args:
            target: The registry key under test.

        Test scenario:
            Every target is given a path carrying the extension it expects, since the
            writers disagree about supplying their own — see
            `test_only_four_writers_append_their_own_extension`.
        """
        import os

        suffix = {"hec_ras": ".tif", "lisflood_fp": ".asc"}.get(target, "")
        written = write(target, grid, str(tmp_path / (target + suffix)))
        assert written, f"{target} reported no artefacts"
        for label, path in written.items():
            assert os.path.exists(path), f"{target}::{label} claims {path}, missing"

    def test_only_four_writers_append_their_own_extension(self, grid, tmp_path):
        """Pins an inconsistency that predates the extraction, rather than hiding it.

        Test scenario:
            `tuflow`, `sfincs`, `gmsh` and `iber` append their extension to a bare path.
            `lisflood_fp` uses the path verbatim, which is deliberate — an Arc grid has
            no single conventional suffix. `hec_ras` is the odd one: it *requires* an
            extension, because the GeoTIFF driver is resolved from it, but does not
            supply one, so a bare path fails inside pyramids with a driver error rather
            than a message naming the export target.

            Changing that is a behaviour change and belongs in its own commit; this test
            states what the code does today so the change is visible when it happens.
        """
        for target, expected in (
            ("tuflow", ".flt"),
            ("sfincs", ".dep"),
            ("gmsh", ".geo"),
            ("iber", ".dat"),
        ):
            written = write(target, grid, str(tmp_path / ("bare_" + target)))
            first = sorted(written.values())[0]
            assert any(
                p.endswith(expected) for p in written.values()
            ), f"{target} did not append {expected}: {first}"

        written = write_lisflood_fp(grid, str(tmp_path / "bare_lisflood"))
        assert written["dem_asc"].endswith(
            "bare_lisflood"
        ), "lisflood_fp is documented to use the path verbatim"

        with pytest.raises(Exception) as exc_info:
            write("hec_ras", grid, str(tmp_path / "bare_hecras"))
        assert "extension" in str(
            exc_info.value
        ), f"Expected an extension-related failure, got: {exc_info.value!r}"

    def test_registry_holds_exactly_the_six_documented_targets(self):
        """The registry is the contract `DEM.export` validates against."""
        assert sorted(TARGETS) == [
            "gmsh",
            "hec_ras",
            "iber",
            "lisflood_fp",
            "sfincs",
            "tuflow",
        ], f"Registry changed: {sorted(TARGETS)}"

    def test_unknown_target_raises_value_error_naming_the_valid_ones(
        self, grid, tmp_path
    ):
        """An unknown target fails with a message that lists what is accepted."""
        with pytest.raises(ValueError, match="target must be one of") as exc_info:
            write("bogus", grid, str(tmp_path / "x"))
        message = str(exc_info.value)
        assert "'bogus'" in message, f"Message omits the bad value: {message}"
        assert "lisflood_fp" in message, f"Message omits the valid targets: {message}"

    def test_dispatch_matches_calling_the_writer_directly(self, grid, tmp_path):
        """`write(target, ...)` and the writer itself produce identical bytes."""
        via_dispatch = write("lisflood_fp", grid, str(tmp_path / "a.asc"))
        direct = write_lisflood_fp(grid, str(tmp_path / "b.asc"))
        assert (
            open(via_dispatch["dem_asc"], "rb").read()
            == open(direct["dem_asc"], "rb").read()
        ), "Dispatching changed the output"
