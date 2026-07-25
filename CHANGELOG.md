# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `pytrale.algorithms.Interpolator`, an abstract interface for pluggable
  weight interpolation/extrapolation algorithms, so alternative algorithms
  can be implemented and passed to `Trale(..., algorithm=...)`.
- `pytrale.algorithms.GaussianKernelSmoother` — pytrale's original algorithm
  (denoise, then linearly interpolate/extrapolate, then smooth again),
  extracted out of `Trale` and now the default `algorithm`.
- `pytrale.algorithms.LinearInterpolator` — a plain linear-interpolation
  baseline with edge clamping, no smoothing.
- `pytrale.algorithms.GaussianProcess` — moved from `pytrale.utils.gp`,
  fixed to implement `Interpolator` (`predict` returns the posterior mean;
  `predict_with_uncertainty` returns mean and variance).
- `Trale.weights_predicted`, the generic output of whichever algorithm is
  configured.
- Test suite (`tests/`) and CI (GitHub Actions: ruff + pytest on
  3.11/3.12/3.13).
- `ruff` for linting.
- `LICENSE` (MIT, matching `pyproject.toml`'s existing license declaration).
- `CONTRIBUTING.md`.

### Changed

- `Trale` no longer implements the Gaussian-kernel pipeline directly;
  `weights_smoothed`, `weights_linear_extrapolated`,
  `weights_gaussian_extrapolated`, and `weights_linear_interpol` are replaced
  by `weights_predicted` (delegates to `algorithm`).
- Switched the build backend from `pdm-backend` to `hatchling`.
- `matplotlib`/`prettypyplot` moved to an optional `notebooks` extra;
  `python-dateutil` added as an explicit direct dependency (it was
  previously relied on transitively via `matplotlib`).

### Fixed

- `DefaultDataClass.__post_init__` used `setattr` on fields of a
  `frozen=True` dataclass, which raised `FrozenInstanceError`; now uses
  `object.__setattr__`.
- `GaussianKernelSmoother`'s single-measurement extrapolation branch read
  the wrong array index for the measurement's value, producing an
  all-zero (or crashing) result; fixed to look up the one non-zero entry
  directly.
- Fixed placeholder author email in `pyproject.toml`.
- Fixed the `GaussianProcess` demo bugs: mismatched method/parameter names
  (`optimize_hyperparameters`/`num_iters` vs. the actual
  `optimize_hyperparameters_log`/`n_iters`) and an invalid `tqdm` call.
- Fixed a stale `db.is_interpolation` reference in
  `notebooks/analyze.ipynb` (no such attribute existed) and updated the
  notebook to the new `algorithm`/`weights_predicted` API.
