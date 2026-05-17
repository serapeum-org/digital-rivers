# Topobathy fusion

Blend a topographic DEM with a bathymetric DEM into a single seamless surface. Introduced as
Phase-4 backfill P31.

Four blend modes:

* **`max`** — per-cell maximum (canonical choice when bathy and topo overlap on a coastline and you
  want the higher of the two).
* **`min`** — per-cell minimum (post-fixup mode used by the four-phase review).
* **`topo_above`** — topo wins above sea-level; bathy fills below.
* **`bathy_below`** — bathy wins below sea-level; topo fills above.

::: digitalrivers.fusion.topobathy_fusion
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
