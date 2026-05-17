# Installation

This page covers installation of **digital-rivers** and its native dependencies.

| Item | Value |
|---|---|
| Distribution name (PyPI / conda-forge) | `digital-rivers` *(not yet published)* |
| Python import name | `digitalrivers` |
| Current version | `0.1.0` (pre-release) |
| Supported Python | **3.11 – 3.13** |
| License | GPL v3 |

> The package is not yet on PyPI or conda-forge. Install from source using the instructions below.

## Dependencies

### Runtime
- `numpy >= 2.0.0`
- `geopandas >= 1.0.0`
- `pyramids-gis >= 0.18.0` (provides the `pyramids` import; pulled from PyPI)
- `gdal >= 3.10, < 4` (best installed from conda-forge — pip wheels are platform-fragile)

### Optional extras
| Extra | Purpose | Pulls |
|---|---|---|
| `viz` | plotting / color tables | `pyramids-gis[viz]` → `cleopatra` |
| `dev` | tests, linting, build tooling | pytest, pre-commit, mypy, build, twine, … |
| `docs` | documentation toolchain | mkdocs, mkdocs-material, mkdocstrings, mike, … |
| `notebook` | Jupyter | jupyterlab, notebook, ipykernel |

## Recommended: Pixi

This repository ships a [Pixi](https://pixi.sh/) configuration that resolves GDAL from conda-forge and `pyramids-gis` from PyPI, avoiding the usual GDAL-wheel headaches.

Prerequisites: install [Pixi](https://pixi.sh/latest/#installation).

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers

# Solve and install the dev environment
pixi install -e dev

# Drop into a shell with everything available
pixi shell -e dev

# Or run a task directly
pixi run main          # main test suite
pixi run plot          # plot/visualization tests
pixi run notebooks     # validate example notebooks
```

### Available Pixi environments

| Environment | Features | Purpose |
|---|---|---|
| `default` | base runtime | minimal install |
| `dev` | `dev` extra | tests, linting, build tooling |
| `docs` | `docs` extra | docs site (`mkdocs serve`) |
| `py311` | `py311` + `dev` | pinned Python 3.11 |
| `py312` | `py312` + `dev` | pinned Python 3.12 |

## Alternative: conda + pip

If you'd rather use conda directly:

```bash
mamba create -n digital-rivers -c conda-forge \
    python=3.12 "gdal>=3.10,<4" libgdal-netcdf libgdal-hdf4
mamba activate digital-rivers
pip install git+https://github.com/serapeum-org/digital-rivers.git
```

## pip-only (advanced)

GDAL is hard to install via pip alone. If you must:

1. Make sure `gdal` and `osgeo` are importable in your environment (system package, prebuilt wheel, etc.).
2. Then:

   ```bash
   pip install git+https://github.com/serapeum-org/digital-rivers.git
   ```

With the `viz` extra:

```bash
pip install "digital-rivers[viz] @ git+https://github.com/serapeum-org/digital-rivers.git"
```

## Editable / development install

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers
pixi install -e dev          # or: pip install -e ".[dev]"
pre-commit install
```

## Quick check

```python
>>> import digitalrivers
>>> digitalrivers.__version__
'0.1.0'
>>> from digitalrivers import DEM, Terrain
```

## Notes

- `pyramids` (conda-forge name) and `pyramids-gis` (PyPI name) are the **same package**. digital-rivers depends on the PyPI distribution name (`pyramids-gis`) so it works regardless of how pyramids itself was installed.
- For very recent pyramids releases the conda-forge ↔ PyPI hash mapping pixi uses can lag by a day; if `pixi update` reports "No candidates were found for pyramids", either wait for the mapping to refresh or temporarily comment out the conda `pyramids` pin in `[tool.pixi.dependencies]`.
- Documentation: <https://serapeum-org.github.io/digital-rivers/latest>
- Source repository: <https://github.com/serapeum-org/digital-rivers>
