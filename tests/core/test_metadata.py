"""Tests for `digitalrivers.core.metadata`.

These constants are the wire format. A typed raster records how it was made in GeoTIFF
tags, and the whole point of that provenance is that a downstream reader can refuse to
misinterpret it — a `dinf` direction grid read as `d8` produces plausible-looking garbage
rather than an error. If a tag key or a valid-value set drifts, rasters written by one
version stop being readable by the next, so the values are pinned here as literals rather
than by importing and comparing to themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalrivers.core.metadata import (
    META_CLASS,
    META_ENCODING,
    META_ROUTING,
    META_THRESHOLD,
    VALID_ENCODING,
    VALID_ROUTING,
    resolve_no_val,
)


class TestMetadataKeys:
    """Tests for the DR_* GeoTIFF tag keys."""

    @pytest.mark.parametrize(
        ("constant", "expected"),
        [
            (META_CLASS, "DR_CLASS"),
            (META_ROUTING, "DR_ROUTING"),
            (META_ENCODING, "DR_ENCODING"),
            (META_THRESHOLD, "DR_THRESHOLD"),
        ],
    )
    def test_key_literal_is_pinned(self, constant, expected):
        """Each tag key holds its exact on-disk string.

        Args:
            constant: The imported constant.
            expected: The literal it must equal.

        Test scenario:
            Changing one of these silently orphans every raster already written with the
            old key, so the literal is asserted rather than the identifier.
        """
        assert constant == expected, f"Expected {expected!r}, got {constant!r}"

    def test_keys_are_distinct(self):
        """Four tags, four names — a collision would overwrite provenance."""
        keys = [META_CLASS, META_ROUTING, META_ENCODING, META_THRESHOLD]
        assert len(set(keys)) == 4, f"Tag keys collide: {keys}"

    def test_keys_share_the_dr_prefix(self):
        """The `DR_` prefix is what separates our tags from GDAL's own."""
        for key in (META_CLASS, META_ROUTING, META_ENCODING, META_THRESHOLD):
            assert key.startswith("DR_"), f"{key!r} is missing the DR_ prefix"


class TestValidValueSets:
    """Tests for VALID_ROUTING and VALID_ENCODING."""

    def test_valid_routing_holds_the_five_schemes(self):
        """The routing vocabulary is the five schemes the package implements."""
        assert VALID_ROUTING == {
            "d8",
            "dinf",
            "mfd_quinn",
            "mfd_holmgren",
            "rho8",
        }, f"Routing vocabulary changed: {sorted(VALID_ROUTING)}"

    def test_valid_encoding_holds_the_four_conventions(self):
        """The encoding vocabulary is the four direction-code conventions understood."""
        assert VALID_ENCODING == {
            "digitalrivers",
            "taudem",
            "esri",
            "whitebox",
        }, f"Encoding vocabulary changed: {sorted(VALID_ENCODING)}"

    @pytest.mark.parametrize("vocabulary", [VALID_ROUTING, VALID_ENCODING])
    def test_vocabularies_are_immutable(self, vocabulary):
        """Both are `frozenset`, so a caller cannot widen what validates.

        Args:
            vocabulary: The set under test.
        """
        assert isinstance(
            vocabulary, frozenset
        ), f"Expected frozenset, got {type(vocabulary).__name__}"

    def test_digitalrivers_is_the_default_encoding_and_is_valid(self):
        """The default the typed classes fall back to must be in the vocabulary."""
        assert "digitalrivers" in VALID_ENCODING, sorted(VALID_ENCODING)

    def test_d8_is_valid_routing(self):
        """The most common scheme is accepted; a guard against an inverted check."""
        assert "d8" in VALID_ROUTING, sorted(VALID_ROUTING)


class TestResolveNoVal:
    """Tests for resolve_no_val."""

    def test_returns_the_band_zero_sentinel(self):
        """A dataset built with a sentinel reports it back as a scalar, not a tuple."""
        ds = Dataset.from_array(
            np.ones((2, 2), dtype=np.float32),
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
            no_data_value=-9999.0,
        )
        assert float(resolve_no_val(ds)) == -9999.0, f"Got {resolve_no_val(ds)!r}"

    def test_returns_none_when_the_attribute_is_none(self):
        """`no_data_value is None` means no sentinel, not a crash."""

        class _Ds:
            no_data_value = None

        assert resolve_no_val(_Ds()) is None, "Expected None for an unset sentinel"

    def test_returns_none_for_an_empty_tuple(self):
        """An empty band tuple is also "no sentinel set"."""

        class _Ds:
            no_data_value = ()

        assert resolve_no_val(_Ds()) is None, "Expected None for an empty band tuple"

    def test_reads_only_the_first_band(self):
        """Multi-band datasets resolve to band 0; the rest are not consulted."""

        class _Ds:
            no_data_value = (-1.0, -2.0, -3.0)

        assert resolve_no_val(_Ds()) == -1.0, f"Got {resolve_no_val(_Ds())!r}"

    def test_preserves_an_integer_sentinel_as_an_integer(self):
        """An `int` sentinel is not coerced to `float`; the dtype of the tag matters."""

        class _Ds:
            no_data_value = (255,)

        result = resolve_no_val(_Ds())
        assert isinstance(result, int), f"Expected int, got {type(result).__name__}"
        assert result == 255, f"Expected 255, got {result}"

    def test_zero_sentinel_is_returned_not_treated_as_missing(self):
        """`0` is falsy but is a legitimate sentinel.

        Test scenario:
            The guard is `if not nv`, which tests the *container*, not the value. A
            one-element tuple holding `0` is truthy, so the zero survives. Pinned
            because narrowing that check to the value would silently drop it.
        """

        class _Ds:
            no_data_value = (0,)

        assert resolve_no_val(_Ds()) == 0, "A zero sentinel was dropped as missing"

    def test_nan_sentinel_is_returned(self):
        """`NaN` is a valid sentinel and must survive the helper unchanged."""

        class _Ds:
            no_data_value = (float("nan"),)

        result = resolve_no_val(_Ds())
        assert np.isnan(result), f"Expected NaN, got {result!r}"
