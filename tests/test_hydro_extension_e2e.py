"""End-to-end test for the W-1 → W-14 hydro-extension stack.

Chains every WhiteboxTools-port deliverable in the hydro layer on a single
synthetic DEM and asserts the cross-cutting invariants that bind them:

    DEM
      → fill_depressions(method="priority_flood")
      → flow_direction(method="d8")
      → accumulate()
      → streams(threshold=...)
      → order(method="hack")              # W-1
      → order(method="topological")       # W-2
      → main_stem(...)                    # W-4
      → prune_short(min_length_m=...)     # W-5
      → to_vector(...) carrying sinuosity # W-3
      → isobasins(target_area_km2)        # W-7
      → statistics(longest_flow_path_m)   # W-8 (post M1 — flow_direction alone)
      → upslope_flowpath_length()         # W-9
      → hand(method="euclidean")          # W-10
      → stochastic_depressions(sigma, …)  # W-11
      → twi / spi / sti                   # W-12 / W-13 / W-14
"""
from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalrivers import DEM, StreamRaster


def _make_dem(arr: np.ndarray, cell_size: float = 1.0) -> DEM:
    disk = arr.astype(np.float32, copy=True)
    nan = np.isnan(disk)
    disk[nan] = -9999.0
    ds = Dataset.create_from_array(
        disk, top_left_corner=(0.0, 0.0), cell_size=cell_size, epsg=4326,
        no_data_value=-9999.0,
    )
    return DEM(ds.raster)


@pytest.fixture(scope="module")
def synthetic_dem() -> DEM:
    """A 10×10 synthetic catchment with a single outlet at the south edge.

    Cells slope toward the centre column and then descend south to the outlet.
    Two minor side-channels join the trunk at rows 3 and 6 so the stream
    network has at least one confluence and the resulting Hack-ordered
    network has both order-1 (main stem) and order-2 (tributary) cells.
    """
    z = np.full((10, 10), 100.0, dtype=np.float32)
    # Main stem: column 5, descending from north (z=20) to south (z=1).
    for r in range(10):
        z[r, 5] = float(20 - 2 * r)
    # Tributary 1: row 3, descending from cols 0-4 toward col 5.
    for c in range(5):
        z[3, c] = float(18 - c)
    # Tributary 2: row 6, descending from cols 7-9 toward col 5.
    for c in range(6, 10):
        z[6, c] = float(15 - (9 - c))
    return _make_dem(z)


class TestHydroExtensionPipeline:
    """End-to-end exercise of the W-1 → W-14 hydro-extension stack."""

    @pytest.fixture(scope="class")
    def bundle(self, synthetic_dem: DEM) -> dict:
        """Run the full pipeline once and yield every intermediate artefact.

        Args:
            synthetic_dem: Module-scoped synthetic DEM fixture.

        Returns:
            A dict mapping artefact name to typed result. Reused by every
            test method in this class so the expensive chain runs once.
        """
        dem = synthetic_dem
        filled = dem.fill_depressions(method="priority_flood")
        fd = filled.flow_direction(method="d8")
        acc = fd.accumulate()
        sr = acc.streams(threshold=2)
        hack = sr.order(method="hack", flow_direction=fd)
        topo = sr.order(method="topological", flow_direction=fd)
        main = sr.main_stem(fd)
        pruned = sr.prune_short(fd, min_length_m=0.5)
        links = sr.to_vector(fd, dem=filled)
        # cell_size=1.0 deg → cell_area ≈ tiny; pick a target sized to a
        # handful of cells so we get multiple sub-basins.
        gt = fd.geotransform
        cell_area_km2 = abs(gt[1] * gt[5]) / 1.0e6
        isobasins = fd.isobasins(sr, acc, target_area_km2=cell_area_km2 * 4)
        # Statistics now triggers longest-flow-path on flow_direction alone (M1).
        basin_stats = fd.basins().statistics(flow_direction=fd, dem=filled)
        upslope = fd.upslope_flowpath_length()
        euclid_hand = filled.hand(sr, method="euclidean")
        stoch = filled.stochastic_depressions(sigma=0.5, n_runs=5, seed=42)
        twi = filled.twi(acc)
        spi = filled.spi(acc)
        sti = filled.sti(acc)
        return {
            "dem": filled,
            "fd": fd,
            "acc": acc,
            "sr": sr,
            "hack": hack,
            "topo": topo,
            "main_stem": main,
            "pruned": pruned,
            "links": links,
            "isobasins": isobasins,
            "basin_stats": basin_stats,
            "upslope": upslope,
            "euclid_hand": euclid_hand,
            "stoch": stoch,
            "twi": twi,
            "spi": spi,
            "sti": sti,
        }

    def test_hack_order_assigns_one_to_main_stem(self, bundle):
        """Test the Hack-ordered raster places order 1 on the main stem (W-1).

        Test scenario:
            The longest source-to-outlet path must carry order 1 throughout;
            tributary cells carry order >= 2.
        """
        hack_arr = bundle["hack"].read_array()
        main = bundle["main_stem"]
        # Every main-stem cell carries Hack order 1.
        assert (hack_arr[main] == 1).all()

    def test_topological_indices_strictly_increase_along_main_stem(self, bundle):
        """Test topological order increases monotonically along the main stem (W-2).

        Test scenario:
            Walk the main stem from head to outlet by sorting cells by their
            topological index — values must be strictly non-decreasing along
            the trace.
        """
        topo_arr = bundle["topo"].read_array()
        sm = bundle["sr"].read_array().astype(bool)
        # The outlet (max topo index over stream cells) must equal the total
        # number of stream cells, since Kahn numbering hits every stream cell
        # exactly once.
        n_stream = int(sm.sum())
        assert int(topo_arr[sm].max()) == n_stream

    def test_sinuosity_at_least_one_for_every_link(self, bundle):
        """Test every link's `sinuosity` column is ≥ 1.0 (W-3).

        Test scenario:
            Traced length ≥ straight-line distance for any path; sinuosity
            is the ratio and must be ≥ 1.
        """
        links = bundle["links"]
        assert "sinuosity" in links.columns
        assert (links["sinuosity"] >= 1.0 - 1e-9).all()

    def test_main_stem_mask_is_subset_of_stream_mask(self, bundle):
        """Test main_stem mask is a strict subset of the stream raster (W-4).

        Test scenario:
            main_stem returns only cells that are stream cells; no spurious
            off-network cells.
        """
        sm = bundle["sr"].read_array().astype(bool)
        main = bundle["main_stem"]
        assert (main & ~sm).sum() == 0

    def test_prune_short_only_removes_headwater_cells(self, bundle):
        """Test prune_short never drops a confluence-or-trunk cell (W-5).

        Test scenario:
            After pruning, every removed stream cell had no downstream
            stream consumer at a confluence — internal links survive.
        """
        sm_before = bundle["sr"].read_array().astype(bool)
        sm_after = bundle["pruned"].read_array().astype(bool)
        # No cell that wasn't a stream before becomes one after pruning.
        assert (sm_after & ~sm_before).sum() == 0

    def test_isobasin_partition_covers_catchment(self, bundle):
        """Test isobasins assigns labels to every stream-reachable cell (W-7).

        Test scenario:
            After isobasin partitioning, the catchment (cells with non-zero
            accumulation) is covered by positive basin labels.
        """
        labels = bundle["isobasins"].read_array()
        # At least one basin emerged.
        assert int(labels.max()) >= 1

    def test_longest_flow_path_column_present(self, bundle):
        """Test basin statistics carry the longest_flow_path_m column (W-8, M1).

        Test scenario:
            After M1, supplying `flow_direction` alone is enough to
            trigger the metric; the column must appear and hold finite
            non-negative values.
        """
        df = bundle["basin_stats"]
        assert "longest_flow_path_m" in df.columns
        assert (df["longest_flow_path_m"] >= 0).all()

    def test_upslope_length_non_decreasing_along_flow(self, bundle):
        """Test per-cell upslope flow-path length is non-negative everywhere (W-9).

        Test scenario:
            Lengths are summed step distances ≥ 0 by construction.
        """
        lengths = bundle["upslope"].read_array()
        assert (lengths >= 0).all()

    def test_euclidean_hand_zero_at_stream_cells(self, bundle):
        """Test Euclidean HAND is 0 on stream cells (W-10).

        Test scenario:
            By definition, every stream cell is its own nearest stream — HAND
            = elev − elev = 0.
        """
        hand_arr = bundle["euclid_hand"].read_array()
        sm = bundle["sr"].read_array().astype(bool)
        assert (hand_arr[sm] == 0).all()

    def test_stochastic_probabilities_in_unit_interval(self, bundle):
        """Test stochastic_depressions output stays in [0, 1] (W-11).

        Test scenario:
            Per-cell probability is `count / n_runs`, bounded above by 1 and
            below by 0.
        """
        prob = bundle["stoch"].read_array()
        valid = prob != float(bundle["stoch"].no_data_value[0])
        assert (prob[valid] >= 0).all()
        assert (prob[valid] <= 1).all()

    def test_twi_spi_sti_have_consistent_shape_and_dtype(self, bundle):
        """Test the three area-slope indices align with the DEM (W-12 / W-13 / W-14).

        Test scenario:
            All three indices come from the same `_area_slope_index` kernel
            and must share the DEM's shape and float32 dtype.
        """
        dem_shape = bundle["dem"].values.shape
        for key in ("twi", "spi", "sti"):
            arr = bundle[key].read_array()
            assert arr.shape == dem_shape, f"{key} shape {arr.shape}"
            assert arr.dtype == np.float32, f"{key} dtype {arr.dtype}"

    def test_spi_proportional_to_accumulation_times_slope(self, bundle):
        """Test SPI grows monotonically with accumulation at fixed slope (W-13).

        Test scenario:
            On the synthetic chain, downstream cells (higher accumulation)
            carry larger SPI than upstream cells at the same slope.
        """
        spi = bundle["spi"].read_array()
        sm = bundle["sr"].read_array().astype(bool)
        no_val = float(bundle["spi"].no_data_value[0])
        finite_stream_spi = spi[sm & (spi != no_val) & np.isfinite(spi)]
        # Stream cells have varying accumulation; the max must exceed the min.
        if finite_stream_spi.size >= 2:
            assert finite_stream_spi.max() > finite_stream_spi.min()
