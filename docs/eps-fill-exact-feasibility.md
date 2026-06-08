# Feasibility of `eps_fill="exact"` for the tiled engine (issue #69)

**Resolution (Path B, implemented).** Rather than reproduce the in-memory Barnes step-count tile-by-tile (which
is ill-posed — see below), both engines were unified on a single deterministic, tile-reconstructible
`epsilon > 0` gradient: the **exit-distance ramp** (`fill_0 + epsilon * exit_distance`,
:func:`digitalrivers._outofcore.fill_ramp.ramp_fill_reference`). The in-memory engine now computes that
ramp directly and the tiled engine reproduces it bit-for-bit, so `eps_fill="exact"` is **byte-for-byte identical
across `engine="in_memory"` / `"tiled"` / `"auto"`** for every tile size and dtype (float32 and float64, with and
without no-data — covered by `tests/_outofcore/test_engine.py`). The classic Barnes step-count is preserved as
the in-memory-only `eps_fill="barnes"` (flat-free for any epsilon, but not tileable). `eps_fill="exact"` no
longer raises.

The original investigation below established **why** byte-identicality to the *classic Barnes kernel* was
ill-posed, which is what motivated redefining the shared gradient instead.

---

**Verdict (re: matching the classic Barnes kernel): ill-posed.** Byte-identical tiled `epsilon > 0` fill against
the *Barnes step-count* is not achievable by a per-tile + perimeter-graph reconstruction, because the quantity it
would have to reproduce depends on the in-memory kernel's **global, sequential traversal order** (a heap
tie-break), not on any tile-reconstructible distance. This is the outcome the issue's Definition-of-Done step 1
named as the gate: *"if tie-breaks turn out to matter, exact is essentially ill-posed for tiling and should stay
in-memory-only."* They do.

## What exact would have to reproduce

The in-memory ε-fill is `fill_eps = fill_0 + epsilon * g`. `fill_0` is already computed bit-for-bit by the tiled
engine (B3), so exact reduces to reproducing the integer field

```
g[cell] = (fill_eps[cell] - fill_0[cell]) / epsilon
```

`g` is the number of `epsilon` increments the Priority-Flood applied to that cell — the step-count along the
flood. The kernel (`priority_flood_numba`) lifts an unvisited neighbour `n` of a popped cell at value `e` to
`e + epsilon` whenever `out[n] <= e`, pushing it onto a shared FIFO pit-queue; the heap pops by
`(elevation, row-major linear index)`.

## Experiments

All runs use the real kernel `digitalrivers._numba.priority_flood_numba` with `epsilon = 1e-3`, deriving
`g = round((fill_eps - fill_0) / epsilon)`.

### 1. `g` is integer-valued — confirmed

Across every fixture (symmetric basins, nested depressions, trenches, plateaus, random terrain) `g` is a
nonnegative integer field. Good sanity check; consistent with "count of ε steps".

### 2. `g` is tie-break-dependent — the decisive result

Fixture `sym-two-outlet`: a flat basin (elevation 5.0) walled by 10.0, with two outlets at elevation 5.0 at
`(0,4)` and `(6,4)`. This DEM is **exactly** top–bottom symmetric (`elev == flipud(elev)`).

A tie-break-independent `g` would therefore satisfy `g == flipud(g)`. It does not:

```
[flipud] tie-break invariant: False
   differs at 28 cells, e.g. [[1,1],[1,2],[1,3]]  vals 3 vs 5
```

The kernel drains the whole basin from the **single first-popped** outlet (the one with the lower row-major
linear index, `(0,4)`, linear 4 < `(6,4)`, linear 58) via the FIFO pit-queue, *before* the equal-elevation
opposite outlet is ever popped from the heap. So `g` is a **single-source** distance from a globally-ordered
entry — and flipping the raster swaps which outlet wins, changing `g`. A symmetric input produces an asymmetric
output: proof that `g` encodes the global pop order, not an intrinsic property of the elevation field.

(Only `flipud` flips it here; `fliplr`/`transpose` happen to preserve the winning outlet for this particular
fixture. One counterexample is sufficient.)

### 3. `g` is not reproduced by tile-local models

Two natural tile-reconstructible candidates were implemented and compared to the real `g`:

- **Multi-source min-distance** (BFS from all spill-adjacent lifted cells): matched only 1 / 5 fixtures.
- **Single-source BFS from the min-(elevation, linear-index) entry per lifted component**: matched 0 / 7
  fixtures (e.g. `plateau` mismatched all 64 lifted cells, `nested` 48 / 49, `random20` 16).

Both diverge because (a) the entry is chosen by the *global* heap order, not a per-component rule, and (b) the
running ε-max lifts cells *above* the nominal spill, so the ramp region and its orientation are themselves
globally determined (a flat plateau becomes a one-sided ramp whose direction is set by the global order).

## Why this is intrinsic, not a missing trick

The `epsilon > 0` mode exists to produce a **strictly monotonic** surface (no remaining flats). Strict
monotonicity requires a **total order** over otherwise-equal-elevation cells. The in-memory kernel gets that
total order for free from its sequential heap (`linear index` breaks ties). Any tile-*local* gradient that drops
the global order — e.g. a symmetric min-distance — reintroduces flats exactly where two equal-distance fronts
meet. So a gradient cannot be simultaneously (a) byte-identical to the current kernel, (b) strictly monotonic,
and (c) reconstructible from local tiles + a perimeter graph. You can have any two.

Reproducing it exactly would mean running the global serial Priority-Flood with a disk-backed priority queue —
which abandons the constant-memory, embarrassingly-parallel tiled design that is the whole point of the
out-of-core engine.

## Recommendation

1. **Close #69 as resolved-without-implementation (ill-posed).** Keep the current behaviour: `eps_fill="monotone"`
   for a valid, flat-free-for-small-ε tiled fill, and `engine="in_memory"` for the exact result at any ε. The
   existing `NotImplementedError` message in `_outofcore/fill.py` already states this accurately.
2. **Optional future alternative (Path B), only if a real workload needs byte-identical ε>0 at >RAM scale.**
   Redefine the ε-gradient in *both* engines to a shared, tile-reconstructible function (e.g. the exit-distance
   ramp the `monotone` mode already uses). Then `tiled == in_memory` *by construction* — but it changes the
   in-memory ε>0 output (the decimals shift) and likely relaxes strict monotonicity for large ε, so it needs
   sign-off and a change-log entry. This is a design change, not a reconstruction of the current kernel.

## Reproduction

The two probe scripts used here construct the fixtures, call `priority_flood_numba` for `epsilon ∈ {0, 1e-3}`,
derive `g`, test `g == flipud(g)` on the symmetric fixture, and compare `g` against the two tile-local models.
They are self-contained (`numpy`, `scipy.ndimage`, `digitalrivers._numba`) and were run under `pixi run -e py311`.
