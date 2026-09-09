"""Pins the single D8 direction convention that every kernel now shares.

Before these tables were consolidated into `digitalrivers.core.directions`, ten sites
across the package declared their own copy of the same offsets. The values are asserted
here against the literals those sites used to hold, so a future edit to the shared table
fails loudly instead of silently re-routing every walk in the package.

The dtype and writability assertions are not pedantry: Numba compiles a separate
specialisation per argument type, and `ndarray.flags.writeable` is part of that type. The
`int8` variant exists because `flow._kernels.routing` broadcasts against `int8` rasters;
every `@njit` kernel takes the `int32` one. Swapping either, or flagging the arrays
read-only, changes the compiled signature of kernels these tests do not touch.
"""

from __future__ import annotations

import numpy as np
import pytest

from digitalrivers.core.directions import (
    DIR_DC_I8,
    DIR_DC_I32,
    DIR_DR_I8,
    DIR_DR_I32,
    DIR_OFFSETS,
    INV_DIR,
)

# The literals every consolidated site used to declare for itself.
_EXPECTED_DR = [1, 1, 0, -1, -1, -1, 0, 1]
_EXPECTED_DC = [0, -1, -1, -1, 0, 1, 1, 1]
_EXPECTED_INV = [4, 5, 6, 7, 0, 1, 2, 3]


class TestValuesMatchTheReplacedLiterals:
    """Each exported table holds exactly what the sites it replaced held."""

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            (DIR_DR_I8, _EXPECTED_DR),
            (DIR_DR_I32, _EXPECTED_DR),
            (DIR_DC_I8, _EXPECTED_DC),
            (DIR_DC_I32, _EXPECTED_DC),
            (INV_DIR, _EXPECTED_INV),
        ],
    )
    def test_table_equals_its_literal(self, table, expected):
        """The array equals the literal it replaced, element for element."""
        assert np.array_equal(table, np.array(expected, dtype=table.dtype))

    def test_dir_offsets_dict_matches_the_arrays(self):
        """`DIR_OFFSETS` is the (col, row) transpose of the (row, col) arrays."""
        pairs = list(zip(DIR_DC_I32.tolist(), DIR_DR_I32.tolist()))
        assert [DIR_OFFSETS[k] for k in range(8)] == pairs

    def test_dir_offsets_covers_exactly_the_eight_neighbours(self):
        """Direction codes are 0-7 and no offset is the centre cell or a repeat."""
        assert sorted(DIR_OFFSETS) == list(range(8))
        offsets = set(DIR_OFFSETS.values())
        assert len(offsets) == 8
        assert (0, 0) not in offsets


class TestConventionInvariants:
    """Properties every walk in the package relies on."""

    def test_direction_zero_is_south(self):
        """Row index grows downward, so direction 0 steps to `row + 1`."""
        assert (int(DIR_DR_I32[0]), int(DIR_DC_I32[0])) == (1, 0)

    def test_inverting_a_direction_twice_is_the_identity(self):
        """`INV_DIR` is an involution."""
        assert INV_DIR[INV_DIR].tolist() == list(range(8))

    def test_a_direction_and_its_inverse_cancel(self):
        """Stepping out along `k` and back along `INV_DIR[k]` returns to the centre."""
        for k in range(8):
            back = int(INV_DIR[k])
            assert int(DIR_DR_I32[k]) + int(DIR_DR_I32[back]) == 0
            assert int(DIR_DC_I32[k]) + int(DIR_DC_I32[back]) == 0

    def test_cardinals_and_diagonals_alternate(self):
        """Even codes are cardinal, odd codes diagonal — the slope divisors assume it."""
        for k in range(8):
            is_diagonal = bool(DIR_DR_I32[k]) and bool(DIR_DC_I32[k])
            assert is_diagonal == (k % 2 == 1)


class TestNumbaSignatureStability:
    """The array *types*, not just their values, are load-bearing."""

    @pytest.mark.parametrize(
        ("table", "dtype"),
        [
            (DIR_DR_I8, np.int8),
            (DIR_DC_I8, np.int8),
            (DIR_DR_I32, np.int32),
            (DIR_DC_I32, np.int32),
            (INV_DIR, np.int32),
        ],
    )
    def test_dtype_is_pinned(self, table, dtype):
        """Widening or narrowing a table recompiles every kernel that takes it."""
        assert table.dtype == dtype

    @pytest.mark.parametrize(
        "table", [DIR_DR_I8, DIR_DC_I8, DIR_DR_I32, DIR_DC_I32, INV_DIR]
    )
    def test_tables_stay_writable(self, table):
        """`writeable` is part of the Numba type; flagging these read-only changes it."""
        assert table.flags.writeable

    @pytest.mark.parametrize(
        "table", [DIR_DR_I8, DIR_DC_I8, DIR_DR_I32, DIR_DC_I32, INV_DIR]
    )
    def test_tables_are_contiguous_one_dimensional(self, table):
        """Kernels are typed on `C`-contiguous 1-D arrays of length 8."""
        assert table.shape == (8,)
        assert table.flags.c_contiguous


class TestConsumersSeeTheSharedTable:
    """The modules that used to declare their own copy now alias this one."""

    def test_flow_routing_uses_the_int8_variant(self):
        """The routing kernels broadcast against int8 rasters and must keep that dtype."""
        from digitalrivers.flow._kernels import routing

        assert routing._DIR_DR is DIR_DR_I8
        assert routing._DIR_DC is DIR_DC_I8

    @pytest.mark.parametrize(
        "module_path",
        [
            "digitalrivers.flow._kernels.ihu",
            "digitalrivers._flow.watershed",
            "digitalrivers._streams.hand",
            "digitalrivers._streams.order",
        ],
    )
    def test_int32_consumers_share_one_object(self, module_path):
        """Every int32 consumer aliases the same arrays, not a private copy."""
        import importlib

        module = importlib.import_module(module_path)
        assert module._DIR_DR is DIR_DR_I32
        assert module._DIR_DC is DIR_DC_I32

    def test_dem_still_re_exports_dir_offsets(self):
        """`from digitalrivers.dem import DIR_OFFSETS` was the historical path."""
        from digitalrivers.dem import DIR_OFFSETS as from_dem

        assert from_dem is DIR_OFFSETS
