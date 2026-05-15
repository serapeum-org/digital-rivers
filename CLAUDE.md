# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**digital-rivers** is a GIS utility library for Digital Elevation Model (DEM) processing, terrain analysis, and raster data manipulation. It uses GDAL as its geospatial backend and builds on the `pyramids` library for dataset operations.

## Build & Development Commands

This project uses **Pixi** as its environment/package manager and **setuptools** as the build backend. Python `>=3.11, <4`.

```bash
# Run main test suite (excludes plot tests, with coverage)
pixi run main

# Run plot/visualization tests only
pixi run plot

# Run notebook validation tests (examples/notebooks)
pixi run notebooks

# Run a single test file or test
pytest -vvv -sv tests/test_dem.py
pytest -vvv -sv tests/test_dem.py::TestClassName::test_method_name

# Pre-commit hooks (also runs pytest, doctest on src/, nbval on notebooks)
pre-commit run --all-files
```

### Pixi environments

Defined in `pyproject.toml` under `[tool.pixi.environments]`: `default`, `dev`, `docs`, `py311`, `py312`. Use `pixi run -e docs <task>` (etc.) to target a specific env.

### Pytest markers

Configured in `pyproject.toml`: `vfs`, `slow`, `fast`, `plot`. The `main` task selects `-m 'not plot'`; the `plot` task selects `-m 'plot'`.

## Architecture

### Core Classes

Both core classes inherit from `pyramids.dataset.Dataset`, which wraps GDAL raster operations:

- **`DEM`** (`src/digitalrivers/dem.py`) — DEM processing: sink filling, D8 flow direction/accumulation, slope calculation. Uses a recursive D8 algorithm with `sys.setrecursionlimit(50000)`. Flow directions are encoded via `DIR_OFFSETS` (8-directional: S, SW, W, NW, N, NE, E, SE).

- **`Terrain`** (`src/digitalrivers/terrain.py`) — Color relief generation from raster bands. Accepts color tables as DataFrames (hex or RGBA format). Writes output with DEFLATE compression (`CREATION_OPTIONS = ["COMPRESS=DEFLATE", "PREDICTOR=2"]`). Heavy lifting is delegated to GDAL's `DEMProcessing`.

Both are re-exported from `digitalrivers/__init__.py`.

### Key Dependencies

`pyramids` and `gdal` are **conda-forge dependencies** managed via Pixi (`[tool.pixi.dependencies]`), not pip. The same library is published to PyPI as **`pyramids-gis`** (see `[project.dependencies]`) — be aware of the naming difference when reading either dependency block.

### Test Data

Tests use the Coello river basin dataset in `tests/data/coello/` (DEMs, flow direction/accumulation rasters, slope arrays as `.npy` files). Fixtures are defined in `tests/conftest.py`.

## Code Style

- **Line length**: Hard-wrap every line in any file you create or edit at **120 characters**. Applies to all file types — Python, Markdown, YAML, TOML, plain text. Break long sentences in prose, long imports, long function signatures, long table rows, etc., so no line exceeds 120 columns.
- **Formatter**: Black (line-length 88, skip-string-normalization). Note that Black will still reformat Python files to 88; the 120-column rule sets the absolute ceiling for any file you write before formatters run.
- **Linter**: Flake8 (excludes `examples/` and `tests/`; `E501` is ignored, so over-length lines won't fail lint but Black will still reformat).
- **Docstrings**: Google style. Doctests inside `src/` modules are executed by the pre-commit `doctest` hook — keep examples runnable.
- **Typing**: source files use `from __future__ import annotations`; prefer modern typing (`X | None`, `list[X]`, `dict[...]`) over `Optional`/`List`/`Dict`/`Union`.
- **Commit messages**: Enforced by pre-commit hooks — capitalized, imperative mood, no trailing punctuation, summary max length, empty second line.

## Git & GitHub — Commit & Push Policy

These are absolute rules for anything written to git history or to GitHub (commits, tags, PR titles, PR bodies,
issue comments, review comments, release notes).

### Never sign commits

- **Do not GPG-sign** any commit or tag. The user's global git config has signing enabled; override it per-command.
- Every `git commit` invocation **must** include `-c commit.gpgsign=false`. Example:
  `git -c commit.gpgsign=false commit -m "..."`.
- Every `git tag` invocation **must** include `-c tag.gpgsign=false` (and `-c commit.gpgsign=false` for safety).
- Never permanently change the signing config — do not run `git config commit.gpgsign false`. Per-command override
  only. (This is consistent with the global rule "NEVER update the git config".)
- If a commit fails with a GPG/signing error, retry with the override; do not prompt the user.

### No attribution to Claude, Anthropic, or any AI/LLM

- **Do not append** `Co-Authored-By: Claude …` or any `Co-Authored-By:` trailer that names an AI/LLM/assistant.
  Override the harness's default commit-message template — the only co-authors permitted are real humans the user
  explicitly names.
- **Do not append** `🤖 Generated with [Claude Code](…)` or any equivalent generated-by banner to PR bodies, issue
  bodies, or commit messages. Remove it from any template you would otherwise emit.
- **Do not mention** Claude, Claude Code, Anthropic, "AI", "LLM", "assistant", "agent", or this tool anywhere in
  commit messages, PR titles/bodies, issue titles/bodies, comments on PRs/issues, or release notes — not in prose,
  not in footers, not in metadata.
- **Do not modify** `git config user.name` / `git config user.email`. Commits are authored by the user; the
  identity stays whatever the local git config already has.
- Apply this to every git-touching action: `git commit`, `git tag`, `gh pr create`, `gh pr edit`, `gh issue create`,
  `gh issue comment`, `gh pr comment`, `gh pr review`, `gh release create`, etc.

The commit body should describe the change in the user's voice (what changed and why), nothing else. If a HEREDOC
template you are about to emit contains any of the forbidden strings, strip them before writing.

## Review Markdown Files

- Review/planning markdown files in `planning/` contain an **Issue Tracker** table at the bottom.
- The **State** column must be the **2nd column**.
- When you resolve an issue, update its **State** column to `Solved` (or `Closed`) and **re-sort the table rows by the State column** so that solved/closed issues appear at the top, followed by open issues. Preserve the original `#` identifiers.