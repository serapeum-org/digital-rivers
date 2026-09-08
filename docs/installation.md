# Installation

This page covers installation of **digital-rivers** and its native dependencies.

| Item | Value |
|---|---|
| Distribution name (PyPI / conda-forge) | `digital-rivers` *(not yet published)* |
| Python import name | `digitalrivers` |
| Current version | `0.4.0` |
| Supported Python | **3.11 – 3.14** |
| License | GPL v3 |

> The package is not yet on PyPI or conda-forge. Install from source using the instructions below.

## Dependencies

### Runtime
- `numpy >= 2.0.0`
- `geopandas >= 1.0.0`
- `pyramids-gis >= 0.60.0` (provides the `pyramids` import; pulled from PyPI)
- `gdal >= 3.13.3, < 3.13.4` (best installed from conda-forge — pip wheels are
  platform-fragile). The narrow pin matches `pyramids-gis`, which builds against
  one GDAL minor at a time.

### Optional extras
| Extra | Purpose | Pulls |
|---|---|---|
| `viz` | plotting / color tables | `pyramids-gis[viz]`, `cleopatra` |
| `distributed` | out-of-core / Dask backend | `pyramids-gis[lazy]` |
| `all` | both of the above | — |

### Development dependency groups
`dev`, `docs` and `notebook` are [PEP 735](https://peps.python.org/pep-0735/) dependency
groups, not extras — they are local tooling and are deliberately not published in the
package metadata. Pixi maps each to an environment of the same name:

```bash
pixi install -e dev      # tests, linting, build tooling
pixi install -e docs     # mkdocs toolchain
```

## Recommended: Pixi

This repository ships a [Pixi](https://pixi.sh/) configuration that resolves GDAL from conda-forge and
`pyramids-gis` from PyPI, avoiding the usual GDAL-wheel headaches.

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
| `py313` | `py313` + `dev` | pinned Python 3.13 |
| `py314` | `py314` + `dev` | pinned Python 3.14 |

## Alternative: conda

If you'd rather manage the environment yourself, install the native stack from
conda-forge and add the package from source:

```bash
mamba create -n digital-rivers -c conda-forge     python=3.12 "gdal>=3.13.3,<3.13.4" libgdal-netcdf libgdal-hdf4
mamba activate digital-rivers
```

The repository's own environments are defined in `pyproject.toml` and resolved by
pixi; `pixi install -e dev` is the supported way to reproduce them exactly, and the
only one that honours `pixi.lock`.

## Editable / development install

```bash
git clone https://github.com/serapeum-org/digital-rivers.git
cd digital-rivers
pixi install -e dev
pixi run -e dev pre-commit install
```

The package itself is registered as an editable pixi pypi-dependency, so
`pixi install -e dev` already puts your checkout on the path — there is no separate
editable-install step.

## Quick check

```python
>>> import digitalrivers
>>> digitalrivers.__version__
'0.4.0'
>>> from digitalrivers import DEM, Terrain
```

## Notes

- `pyramids` (conda-forge name) and `pyramids-gis` (PyPI name) are the **same package**. digital-rivers
  depends on the PyPI distribution name (`pyramids-gis`) so it works regardless of how pyramids itself was
  installed.
- For very recent pyramids releases the conda-forge ↔ PyPI hash mapping pixi uses can lag by a day; if
  `pixi update` reports "No candidates were found for pyramids", either wait for the mapping to refresh or
  temporarily comment out the conda `pyramids` pin in `[tool.pixi.dependencies]`.
- Documentation: <https://serapeum-org.github.io/digital-rivers/latest>
- Source repository: <https://github.com/serapeum-org/digital-rivers>
