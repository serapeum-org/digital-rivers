"""Tests for ``DEM.resolve_flats`` and the underlying ``_flats`` module (P4).

Covers plateau detection, LEC/HEC classification, BFS-level computation, and the combined
gradient lift on synthetic single-outlet, two-outlet, HEC-less, and LEC-less plateaus,
plus an end-to-end fill→resolve_flats→flow_direction check on the Coello basin.
"""
from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers import DEM
from digitalrivers._flats import (
    _bfs_levels,
    _classify_lec_hec,
    _invert_per_plateau,
    _label_plateaus,
    _NEIGHBOURS_8,
    resolve_flats,
)


def _make_dem(arr: np.ndarray, no_data_value: float = -9999.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan_mask = np.isnan(disk)
    disk[nan_mask] = no_data_value
    ds = Dataset.create_from_array(
        disk,
        top_left_corner=(0.0, 0.0),
        cell_size=1.0,
        epsg=4326,
        no_data_value=no_data_value,
    )
    return DEM(ds.raster)


# ----- helpers ----------------------------------------------------------------------------

SINGLE_OUTLET_PLATEAU = np.array(
    [
        [9, 9, 9, 9, 9],
        [9, 5, 5, 5, 9],
        [9, 5, 5, 5, 9],
        [9, 5, 5, 5, 1],
        [9, 9, 9, 9, 9],
    ],
    dtype=np.float64,
)


def _has_internal_flats(z: np.ndarray, nodata_mask: np.ndarray | None = None) -> bool:
    """True if any interior cell has no strictly lower 8-neighbour AND at least one
    equal-elevation 8-neighbour — i.e., the surface still has a flat that would yield a
    NO_FLOW cell under D8.
    """
    if nodata_mask is None:
        nodata_mask = np.isnan(z)
    rows, cols = z.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if nodata_mask[r, c]:
                continue
            z_c = z[r, c]
            has_equal = False
            has_lower = False
            for dr, dc in _NEIGHBOURS_8:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if nodata_mask[nr, nc]:
                    continue
                if z[nr, nc] == z_c:
                    has_equal = True
                elif z[nr, nc] < z_c:
                    has_lower = True
            if has_equal and not has_lower:
                return True
    return False


# ----- plateau labelling ------------------------------------------------------------------

class TestLabelPlateaus:
    def test_no_plateaus_in_strict_slope(self):
        # Every cell uniquely valued so no two 8-neighbours can share an elevation.
        z = np.arange(9, dtype=np.float64).reshape(3, 3)
        labels, n = _label_plateaus(
            z, np.zeros_like(z, dtype=bool), _NEIGHBOURS_8
        )
        assert n == 0
        assert not labels.any()

    def test_two_plateaus_including_boundary_ring(self):
        # Plateau labelling is purely geometric: it labels every connected component of
        # equal-elevation cells, including the 9-ring boundary that surrounds the inner
        # z=5 region. Whether each plateau actually needs resolving is decided later in
        # resolve_flats, not at labelling time.
        labels, n = _label_plateaus(
            SINGLE_OUTLET_PLATEAU,
            np.zeros_like(SINGLE_OUTLET_PLATEAU, dtype=bool),
            _NEIGHBOURS_8,
        )
        assert n == 2
        # One label spans the z=5 block (9 cells), the other spans the z=9 ring (15
        # cells — 16 boundary cells minus (3, 4) which is at z=1).
        sizes = sorted(((labels == lbl).sum() for lbl in (1, 2)))
        assert sizes == [9, 15]

    def test_two_disjoint_plateaus_with_unique_background(self):
        # Two disjoint plateaus at different elevations against a background of
        # uniquely-valued cells (no third plateau from the surrounding terrain).
        z = np.array(
            [
                [10, 11, 12, 13],
                [14,  5,  5, 15],
                [16,  5,  5, 17],
                [18, 19,  3,  3],
            ],
            dtype=np.float64,
        )
        labels, n = _label_plateaus(z, np.zeros_like(z, dtype=bool), _NEIGHBOURS_8)
        assert n == 2
        plateau_sizes = sorted(((labels == lbl).sum() for lbl in (1, 2)))
        assert plateau_sizes == [2, 4]  # 3-block has 2 cells, 5-block has 4

    def test_singleton_equal_elevation_is_not_a_plateau(self):
        # Cell at z=5 with no equal-elevation 8-neighbour. Surrounded by uniquely-valued
        # cells so no other plateau exists either.
        z = np.array(
            [
                [10, 11, 12],
                [13,  5, 14],
                [15, 16, 17],
            ],
            dtype=np.float64,
        )
        labels, n = _label_plateaus(z, np.zeros_like(z, dtype=bool), _NEIGHBOURS_8)
        assert n == 0


# ----- LEC/HEC classification ------------------------------------------------------------

class TestClassifyLecHec:
    def test_single_outlet_plateau_has_one_lec(self):
        labels, _ = _label_plateaus(
            SINGLE_OUTLET_PLATEAU,
            np.zeros_like(SINGLE_OUTLET_PLATEAU, dtype=bool),
            _NEIGHBOURS_8,
        )
        is_lec, is_hec = _classify_lec_hec(
            SINGLE_OUTLET_PLATEAU,
            labels,
            np.zeros_like(SINGLE_OUTLET_PLATEAU, dtype=bool),
        )
        # The outlet at (3, 4)=1 is reached from plateau cell (2, 3) (diagonal) or (3, 3)
        # (cardinal) — both should be LECs.
        assert is_lec[2, 3]
        assert is_lec[3, 3]
        # Cells far from the outlet should not be LEC.
        assert not is_lec[1, 1]
        # The outer ring of the inner z=5 plateau touches the z=9 boundary — those cells
        # are HECs. The geometric centre (2, 2) is surrounded entirely by other plateau
        # cells at z=5, so it has NO strictly higher neighbour and is NOT a HEC.
        for r, c in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            assert is_hec[r, c], f"plateau cell {(r, c)} should be HEC"
        assert not is_hec[2, 2]


# ----- BFS levels ------------------------------------------------------------------------

class TestBfsLevels:
    def test_bfs_assigns_increasing_levels(self):
        labels, _ = _label_plateaus(
            SINGLE_OUTLET_PLATEAU,
            np.zeros_like(SINGLE_OUTLET_PLATEAU, dtype=bool),
            _NEIGHBOURS_8,
        )
        is_lec, _ = _classify_lec_hec(
            SINGLE_OUTLET_PLATEAU,
            labels,
            np.zeros_like(SINGLE_OUTLET_PLATEAU, dtype=bool),
        )
        g_low = _bfs_levels(labels, is_lec, _NEIGHBOURS_8, max_iter=1000)
        # LEC cells are at level 1.
        assert g_low[2, 3] == 1
        assert g_low[3, 3] == 1
        # The farthest plateau cell (1, 1) is multi-hop away from the LECs.
        assert g_low[1, 1] > 1
        # The outlet at (3, 4)=1 is a singleton (no equal-elevation neighbour), so it is
        # not in any plateau and g_low is 0 there.
        assert g_low[3, 4] == 0


# ----- gradient inversion ----------------------------------------------------------------

class TestInvertPerPlateau:
    def test_max_becomes_zero_after_invert(self):
        # Use a plateau with a non-trivial internal structure so the BFS from HECs has
        # different levels at different cells. The 3-cell-wide plateau here has a centre
        # cell at higher BFS depth than its rim, so the inversion has something to do.
        z = np.array(
            [
                [10, 11, 12, 13, 14],
                [15,  5,  5,  5, 16],
                [17,  5,  5,  5, 18],
                [19,  5,  5,  5, 20],
                [21, 22, 23, 24,  1],
            ],
            dtype=np.float64,
        )
        labels, n = _label_plateaus(z, np.zeros_like(z, dtype=bool), _NEIGHBOURS_8)
        _, is_hec = _classify_lec_hec(z, labels, np.zeros_like(z, dtype=bool))
        g_high_raw = _bfs_levels(labels, is_hec, _NEIGHBOURS_8, max_iter=1000)
        g_high = _invert_per_plateau(g_high_raw, labels, n)
        # The raw BFS reaches at least level 2 somewhere (the centre of the plateau).
        plateau_mask = labels > 0
        assert g_high_raw[plateau_mask].max() >= 2
        # The cell at the deepest raw BFS level inverts to 0.
        max_position = np.argmax(g_high_raw * plateau_mask)
        max_r, max_c = np.unravel_index(max_position, g_high_raw.shape)
        assert g_high[max_r, max_c] == 0


# ----- end-to-end resolve_flats ---------------------------------------------------------

class TestResolveFlats:
    def test_single_outlet_plateau_is_resolved(self):
        out = resolve_flats(SINGLE_OUTLET_PLATEAU)
        # After resolve_flats, the plateau is no longer flat — every interior cell has at
        # least one strictly lower 8-neighbour.
        assert not _has_internal_flats(out)
        # Plateau cells were lifted (not lowered).
        plateau_mask = SINGLE_OUTLET_PLATEAU == 5
        assert (out[plateau_mask] >= 5.0).all()
        # The cell closest to the outlet (the LEC, e.g. (3, 3)) is the lowest lifted cell
        # in the plateau; the cell farthest (1, 1) is among the highest.
        assert out[3, 3] < out[1, 1]

    def test_two_outlet_plateau_drains_to_nearest(self):
        # Plateau with two outlets, one on each side. Each plateau cell should slope
        # towards its nearest outlet — neither outlet should attract cells from the wrong
        # side of the saddle.
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9],
                [9, 5, 5, 5, 5, 1],
                [9, 5, 5, 5, 5, 9],
                [1, 5, 5, 5, 5, 9],
                [9, 9, 9, 9, 9, 9],
            ],
            dtype=np.float64,
        )
        out = resolve_flats(z)
        assert not _has_internal_flats(out)
        # Cells nearest the east outlet should be lower than cells nearest the west
        # outlet's column.
        assert out[1, 4] < out[1, 1]  # east-row, near east outlet < far from outlet
        assert out[3, 1] < out[3, 4]  # west-row, near west outlet < far from outlet

    def test_lec_less_plateau_is_left_alone(self):
        # Closed depression — multi-cell plateau (2x2 at z=5) entirely surrounded by
        # higher terrain (9-ring). No cell has a strictly lower neighbour → no LEC. The
        # algorithm must leave it untouched. (In practice fill_depressions would have
        # already filled this; this test guards the graceful no-op if it sneaks through.)
        z = np.array(
            [
                [9, 9, 9, 9],
                [9, 5, 5, 9],
                [9, 5, 5, 9],
                [9, 9, 9, 9],
            ],
            dtype=np.float64,
        )
        out = resolve_flats(z)
        np.testing.assert_array_equal(out, z)

    def test_no_plateaus_returns_unchanged_surface(self):
        # Every cell uniquely valued so no two 8-neighbours share an elevation → no
        # plateaus.
        z = np.arange(16, dtype=np.float64).reshape(4, 4)
        out = resolve_flats(z)
        np.testing.assert_array_equal(out, z)


class TestNodataHandling:
    def test_plateau_adjacent_to_nodata_resolved(self):
        # Plateau where the western half is no-data — nodata-adjacent plateau cells
        # behave like LECs (water drains into the no-data void).
        z = np.array(
            [
                [9, 9, 9, 9, 9],
                [np.nan, 5, 5, 5, 9],
                [np.nan, 5, 5, 5, 9],
                [np.nan, 5, 5, 5, 1],
                [9, 9, 9, 9, 9],
            ],
            dtype=np.float64,
        )
        # The (3, 4)=1 cell is still a strictly-lower neighbour, so the plateau has a LEC
        # and gets resolved.
        out = resolve_flats(z)
        # No-data cells are NaN in the output.
        assert np.isnan(out[1, 0])
        # Internal cells have a strict downhill direction.
        assert not _has_internal_flats(out, nodata_mask=np.isnan(z))


# ----- validation -----------------------------------------------------------------------

class TestValidation:
    def test_invalid_connectivity_raises(self):
        z = SINGLE_OUTLET_PLATEAU.copy()
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            resolve_flats(z, connectivity=5)


# ----- DEM-level integration -----------------------------------------------------------

class TestDEMResolveFlats:
    def test_returns_typed_dem(self):
        dem = _make_dem(SINGLE_OUTLET_PLATEAU.astype(np.float32))
        out = dem.resolve_flats()
        assert type(out) is DEM

    def test_inplace_returns_none(self):
        dem = _make_dem(SINGLE_OUTLET_PLATEAU.astype(np.float32))
        assert dem.resolve_flats(inplace=True) is None
        # After the in-place call the plateau is resolved on the instance.
        assert not _has_internal_flats(dem.values)

    def test_invalid_connectivity_via_dem_method(self):
        dem = _make_dem(SINGLE_OUTLET_PLATEAU.astype(np.float32))
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            dem.resolve_flats(connectivity=5)


@pytest.mark.slow
class TestCoelloFillThenResolve:
    """End-to-end fill→resolve_flats pipeline on the Coello basin."""

    def test_no_internal_flats_after_pipeline(self, coello_dem_4000: gdal.Dataset):
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="wang_liu")
        resolved = filled.resolve_flats(epsilon=1e-4)
        # The pipeline produces a surface with no internal flats (every data interior
        # cell has a strictly lower 8-neighbour).
        assert not _has_internal_flats(resolved.values)

    def test_no_undefined_flow_directions_after_pipeline(
        self, coello_dem_4000: gdal.Dataset
    ):
        """D8 flow direction over the fill→resolve_flats pipeline has at most one
        undefined cell inside the data envelope — the basin outlet itself, which
        legitimately has no downhill neighbour. P5's stricter D8 marks this cell as
        a sink instead of assigning a spurious least-negative direction."""
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="wang_liu")
        resolved = filled.resolve_flats(epsilon=1e-4)
        fd = resolved.flow_direction()
        fd_arr = fd.read_array()
        no_data_value = Dataset.default_no_data_value
        nan_in_original = np.isnan(dem.values)
        undefined_in_fd = fd_arr == no_data_value
        spurious = undefined_in_fd & ~nan_in_original
        # At most one — the legitimate basin outlet.
        assert int(spurious.sum()) <= 1
