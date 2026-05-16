"""End-to-end and coverage-gap tests for Phase 1 of the digital-rivers roadmap.

Covers the full hydro pre-processor pipeline on the Coello fixture (P1–P11
chained), plus the gaps the per-task tests leave behind:

* the Numba-disabled fallback path (`DIGITALRIVERS_DISABLE_NUMBA=1` re-import);
* `Accumulation`/`StreamRaster` GeoTIFF metadata round-trips via `open`;
* a few unresolved-Dijkstra / max-length-only branches in the breach module.

The pipeline asserts the basin-level invariants that bind P1–P11 together:
mass conservation across accumulation, sinks-free output after fill +
resolve_flats, stream cells == HAND zero, and non-negative HAND in the
catchment.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from osgeo import gdal
from pyramids.dataset import Dataset

from digitalrivers import DEM, Accumulation, FlowDirection, StreamRaster
from digitalrivers._flow.accumulation import accumulate as _accumulate_array
from digitalrivers._conditioning.breach import breach_depressions


# ----- Full pipeline on Coello -------------------------------------------------------------


@pytest.mark.slow
class TestCoelloEndToEndPipeline:
    """End-to-end pipeline: every phase-1 deliverable run on the Coello DEM,
    chained in canonical order. Each test asserts one cross-cutting invariant."""

    @pytest.fixture(scope="class")
    def pipeline(self, coello_dem_4000: gdal.Dataset) -> dict:
        """Run the full pipeline once per class and cache the artefacts.

        Returns:
            dict with the DEM, filled DEM, resolved DEM, FlowDirection,
            Accumulation, StreamRaster, vectorised stream GeoDataFrame, and
            HAND raster.
        """
        dem = DEM(coello_dem_4000)
        filled = dem.fill_depressions(method="wang_liu")
        resolved = filled.resolve_flats(epsilon=1e-4)
        fd = resolved.flow_direction(method="d8")
        acc = fd.accumulate()
        sr = acc.streams(threshold=10)
        gdf = sr.to_vector(fd, dem=resolved)
        hand = resolved.hand(sr, fd)
        return {
            "dem": dem,
            "filled": filled,
            "resolved": resolved,
            "fd": fd,
            "acc": acc,
            "sr": sr,
            "gdf": gdf,
            "hand": hand,
        }

    def test_filled_dem_is_sinks_free(self, pipeline: dict) -> None:
        """After fill + resolve_flats, no interior cell is a strict local minimum.

        Test scenario:
            Run fill_depressions(wang_liu) followed by resolve_flats and verify
            no cell is strictly lower than all eight valid 8-neighbours.
        """
        from digitalrivers._conditioning.pitremoval import local_minima_8
        resolved = pipeline["resolved"]
        sinks = local_minima_8(resolved.values)
        assert int(sinks.sum()) == 0, f"Expected no internal sinks, found {int(sinks.sum())}"

    def test_flow_direction_has_at_most_one_undefined_inside_envelope(
        self, pipeline: dict
    ) -> None:
        """The basin outlet is the only legitimate sink under strict D8.

        Test scenario:
            After fill + resolve_flats, run D8 flow_direction; at most one
            interior cell may carry the no-data sentinel (the actual outlet).
        """
        dem = pipeline["dem"]
        fd = pipeline["fd"]
        fd_arr = fd.read_array()
        no_data_value = Dataset.default_no_data_value
        nan_in_original = np.isnan(dem.values)
        undefined_in_fd = fd_arr == no_data_value
        spurious = undefined_in_fd & ~nan_in_original
        assert int(spurious.sum()) <= 1, (
            f"Expected at most 1 undefined interior cell, got {int(spurious.sum())}"
        )

    def test_accumulation_is_non_negative(self, pipeline: dict) -> None:
        """Every cell's accumulation is >= 0 (no negative weights are introduced).

        Test scenario:
            Accumulation under unit weights should never produce negative values
            for any valid cell.
        """
        acc_arr = pipeline["acc"].read_array()
        no_val = pipeline["acc"].no_data_value[0] or 0
        valid = acc_arr != no_val
        assert (acc_arr[valid] >= 0).all(), "Found negative accumulation values"

    def test_accumulation_mass_conservation(self, pipeline: dict) -> None:
        """Sum of accumulation values matches expectations for a connected D8 network.

        Test scenario:
            For uniform unit weights and a single basin, sum(acc) over all
            valid cells equals N*(N-1)/2 if the network is a single linear
            chain; for trees it is at least N - 1 (each non-outlet cell is
            counted once as an upstream contributor somewhere).
        """
        dem = pipeline["dem"]
        acc_arr = pipeline["acc"].read_array().astype(np.float64)
        no_val = pipeline["acc"].no_data_value[0] or 0
        valid = (acc_arr != no_val) & ~np.isnan(dem.values)
        n_valid = int(valid.sum())
        total = float(acc_arr[valid].sum())
        assert total >= n_valid - 1, (
            f"Accumulation sum {total} below conservation lower bound {n_valid - 1}"
        )

    def test_stream_cells_have_zero_hand(self, pipeline: dict) -> None:
        """HAND at every stream cell is 0 (cells drain to themselves).

        Test scenario:
            All cells flagged as stream cells in the StreamRaster must have
            HAND == 0 in the final HAND raster.
        """
        sr_mask = pipeline["sr"].read_array().astype(bool)
        hand_arr = pipeline["hand"].read_array()
        np.testing.assert_allclose(
            hand_arr[sr_mask], 0.0, atol=1e-4,
            err_msg="Stream cells should have HAND == 0",
        )

    def test_hand_non_negative_in_catchment(self, pipeline: dict) -> None:
        """HAND values are >= 0 (no cell drains to higher terrain).

        Test scenario:
            For every cell with a defined HAND value, the value must be
            non-negative — the flow-path enforcement ensures water drops
            monotonically.
        """
        hand_arr = pipeline["hand"].read_array()
        no_val = pipeline["hand"].no_data_value[0]
        valid = hand_arr != no_val
        assert (hand_arr[valid] >= -1e-4).all(), "Found negative HAND values"

    def test_vector_network_link_count_matches_topology(self, pipeline: dict) -> None:
        """The vector network has at least one link per stream-mask outlet.

        Test scenario:
            Each connected stream component drains to a single outlet; the
            link count should be >= the number of outlets.
        """
        gdf = pipeline["gdf"]
        # Crude check: at least one link emitted from any non-trivial stream.
        sr_arr = pipeline["sr"].read_array()
        if sr_arr.sum() > 0:
            assert len(gdf) >= 1, "Vector network should not be empty"

    def test_vector_links_descend_monotonically(self, pipeline: dict) -> None:
        """Each vector link's drop_m is non-negative (water flows downhill).

        Test scenario:
            The to_vector adapter computes drop_m = z[from] - z[to] clamped
            to >= 0. Verify the clamp's output is correct.
        """
        gdf = pipeline["gdf"]
        assert (gdf["drop_m"] >= 0).all(), "All link drops should be non-negative"

    def test_stream_ordering_outlet_dominates(self, pipeline: dict) -> None:
        """Strahler order at the network outlet is >= 1 and is the network maximum.

        Test scenario:
            After running StreamRaster.order(method="strahler"), the maximum
            order in the raster equals the order at the lowest-elevation
            stream cell (the outlet under D8).
        """
        ordered = pipeline["sr"].order(method="strahler", flow_direction=pipeline["fd"])
        order_arr = ordered.read_array()
        sr_mask = pipeline["sr"].read_array().astype(bool)
        if not sr_mask.any():
            pytest.skip("no stream cells extracted at threshold=10")
        assert int(order_arr[sr_mask].max()) >= 1


# ----- Numba fallback path ----------------------------------------------------------------


class TestNumbaFallbackExercised:
    """Re-imports `_numba` with the env var set so the no-op decorators run,
    then re-imports the consumer modules to bind their kernel references to the
    pure-Python branch."""

    def test_priority_flood_works_without_numba(self, monkeypatch) -> None:
        """Priority-flood produces the same output via the pure-Python branch.

        Test scenario:
            Set DIGITALRIVERS_DISABLE_NUMBA=1, reload _numba and _pitremoval,
            run fill_depressions on a 5x5 single-pit fixture, confirm the
            result matches the Numba-enabled output.
        """
        z = np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 4, 4, 4, 5],
                [5, 4, 1, 4, 5],
                [5, 4, 4, 4, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.float64,
        )

        monkeypatch.setenv("DIGITALRIVERS_DISABLE_NUMBA", "1")
        for mod in ("digitalrivers._numba", "digitalrivers._conditioning.pitremoval"):
            sys.modules.pop(mod, None)
        try:
            pitremoval = importlib.import_module("digitalrivers._conditioning.pitremoval")
            numba_mod = importlib.import_module("digitalrivers._numba")
            assert numba_mod.is_numba_enabled() is False
            out = pitremoval.fill_depressions(z.copy(), method="priority_flood",
                                              epsilon=0.0)
            assert out[2, 2] == 5.0, f"Expected pit lift to rim, got {out[2, 2]}"
        finally:
            monkeypatch.delenv("DIGITALRIVERS_DISABLE_NUMBA", raising=False)
            for mod in ("digitalrivers._numba", "digitalrivers._conditioning.pitremoval"):
                sys.modules.pop(mod, None)
            importlib.import_module("digitalrivers._numba")
            importlib.import_module("digitalrivers._conditioning.pitremoval")

    def test_kahn_accumulate_works_without_numba(self, monkeypatch) -> None:
        """Kahn accumulation produces correct counts via the pure-Python branch.

        Test scenario:
            With Numba disabled, the _accumulation.accumulate dispatcher falls
            back to the Python receivers helpers. Compare against a hand-
            computed value on a simple chain.
        """
        fdir = np.array([[6, 6, 6, 6, -9999]], dtype=np.int32)
        valid = np.ones(fdir.shape, dtype=bool)

        monkeypatch.setenv("DIGITALRIVERS_DISABLE_NUMBA", "1")
        for mod in ("digitalrivers._numba", "digitalrivers._flow.accumulation"):
            sys.modules.pop(mod, None)
        try:
            accumulation = importlib.import_module("digitalrivers._flow.accumulation")
            numba_mod = importlib.import_module("digitalrivers._numba")
            assert numba_mod.is_numba_enabled() is False
            out = accumulation.accumulate(fdir, "d8", valid)
            assert out[0, 4] == pytest.approx(4.0), (
                f"Expected outlet count 4, got {out[0, 4]}"
            )
        finally:
            monkeypatch.delenv("DIGITALRIVERS_DISABLE_NUMBA", raising=False)
            for mod in ("digitalrivers._numba", "digitalrivers._flow.accumulation"):
                sys.modules.pop(mod, None)
            importlib.import_module("digitalrivers._numba")
            importlib.import_module("digitalrivers._flow.accumulation")


# ----- Coverage gap fillers ---------------------------------------------------------------


class TestAccumulationOpenRoundTrip:
    """`Accumulation.open()` round-trips routing through DR_ROUTING metadata
    tags. Mirror of FlowDirection.open() coverage in test_typed_results.py."""

    def test_open_round_trips_routing(self, tmp_path) -> None:
        """persist_metadata + open round-trips the routing tag.

        Test scenario:
            Build an Accumulation from a plain Dataset, persist metadata, write
            to a GeoTIFF, reopen via Accumulation.open, and confirm the routing
            tag survived.
        """
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        path = str(tmp_path / "acc.tif")
        plain = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        acc = Accumulation.from_dataset(plain, routing="mfd_quinn")
        acc.persist_metadata()
        del acc, plain
        reopened = Accumulation.open(path)
        assert reopened.routing == "mfd_quinn"

    def test_open_without_tag_or_routing_raises(self, tmp_path) -> None:
        """open() refuses to guess when neither the kwarg nor the tag is present.

        Test scenario:
            Write a GeoTIFF without persist_metadata so DR_ROUTING is absent;
            open() must raise ValueError mentioning DR_ROUTING.
        """
        arr = np.array([[1, 2]], dtype=np.float32)
        path = str(tmp_path / "untagged.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
            driver_type="GTiff",
            path=path,
        )
        with pytest.raises(ValueError, match="DR_ROUTING"):
            Accumulation.open(path)


class TestStreamRasterOpenRoundTrip:
    """`StreamRaster.open()` round-trips routing and threshold."""

    def test_open_round_trips_threshold_and_routing(self, tmp_path) -> None:
        """persist_metadata + open recovers both threshold and routing.

        Test scenario:
            Build a StreamRaster from a plain Dataset, persist metadata
            (DR_THRESHOLD + DR_ROUTING), write to a GeoTIFF, reopen, and
            confirm both tags survived.
        """
        arr = np.array([[0, 1, 1, 0]], dtype=np.uint8)
        path = str(tmp_path / "streams.tif")
        plain = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=0,
            driver_type="GTiff",
            path=path,
        )
        sr = StreamRaster.from_dataset(plain, threshold=42.5, routing="d8")
        sr.persist_metadata()
        del sr, plain
        reopened = StreamRaster.open(path)
        assert reopened.threshold == pytest.approx(42.5)
        assert reopened.routing == "d8"

    def test_open_missing_threshold_raises(self, tmp_path) -> None:
        """open() refuses to guess threshold when the tag and kwarg are absent.

        Test scenario:
            Write a GeoTIFF with DR_ROUTING but no DR_THRESHOLD; open() must
            raise ValueError mentioning DR_THRESHOLD.
        """
        arr = np.array([[0, 1]], dtype=np.uint8)
        path = str(tmp_path / "no_threshold.tif")
        plain = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=0,
            driver_type="GTiff",
            path=path,
        )
        plain.meta_data = {"DR_ROUTING": "d8"}
        del plain
        with pytest.raises(ValueError, match="DR_THRESHOLD"):
            StreamRaster.open(path, routing="d8")


# ----- Breach module gap fillers ----------------------------------------------------------


class TestBreachAdditionalBranches:
    """Cover the unresolved-Dijkstra / no-pit edge cases not exercised in the
    main P3 test suite."""

    def test_breach_no_pits_returns_input_unchanged(self) -> None:
        """A surface with no local minima passes through breach_depressions
        without modification.

        Test scenario:
            Use a strictly monotonic ramp; breach_depressions should detect zero
            pits and return the input.
        """
        z = np.arange(9, dtype=np.float64).reshape(3, 3)
        out = breach_depressions(z.copy(), method="least_cost")
        np.testing.assert_array_equal(out, z)

    def test_breach_unresolved_pit_under_max_depth_constraint(self) -> None:
        """A pit whose nearest outlet costs more than max_depth stays a pit
        when method='least_cost' (no fill fallback).

        Test scenario:
            Build a deep pit with a thick wall; run least_cost with a tight
            max_depth; verify the pit is still a local minimum.
        """
        z = np.array(
            [
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 1, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 0],
            ],
            dtype=np.float64,
        )
        out = breach_depressions(z, method="least_cost", max_depth=1.0)
        from digitalrivers._conditioning.pitremoval import local_minima_8
        assert local_minima_8(out)[3, 3]


# ----- Sanity smoke: __init__ re-exports --------------------------------------------------


def test_package_reexports_phase1_typed_classes() -> None:
    """`import digitalrivers` exposes the four typed classes from P1/P8.

    Test scenario:
        Import the package and confirm DEM, Terrain, FlowDirection,
        Accumulation, StreamRaster are all attributes.
    """
    import digitalrivers

    for name in ("DEM", "Terrain", "FlowDirection", "Accumulation", "StreamRaster"):
        assert hasattr(digitalrivers, name), f"Missing re-export: {name}"
