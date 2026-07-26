# Plan: Preparing pytrale for public release

Goal: publish pytrale as an open-source package where others can plug in
alternative weight-interpolation/extrapolation algorithms. This plan is
ordered by priority — architecture first (since it drives the public API),
then hygiene/bugs, then packaging and community-facing docs.

## 1. Design an extensible algorithm interface

This is the main blocker for "others can implement alternative algorithms" —
today there isn't one. `Trale` (`src/pytrale/weightsDatabase.py`) hardcodes a
specific linear+Gaussian smoothing pipeline directly as methods
(`_linear_interpolation`, `_gaussian_interpolation`, `_linear_regression`,
...), and `GaussianProcess` (`src/pytrale/utils/gp.py`) is a second, unrelated
implementation that nothing else uses. Neither shares a contract.

- [ ] Define an abstract base (e.g. `pytrale.algorithms.base.Interpolator`)
      with a small, stable contract — something like
      `fit(times, weights) -> None` / `predict(times) -> weights` (+
      optionally `predict_with_uncertainty`), so algorithms are swappable.
- [ ] Refactor the existing linear/Gaussian pipeline in `Trale` into one
      implementation of that interface (e.g. `algorithms/gaussian_kernel.py`),
      and move `GaussianProcess` into another (`algorithms/gaussian_process.py`),
      fixing its bugs along the way (see §2).
- [ ] Make `Trale` accept an algorithm instance/class rather than hardcoding
      the pipeline, e.g. `Trale(times_measured, weights_measured, algorithm=GaussianKernelInterpolator())`.
- [ ] Document the interface and provide a minimal "write your own algorithm"
      example — this is the thing external contributors will actually read.
- [ ] Decide + document how a third-party algorithm is expected to be
      distributed/discovered (import and pass in directly vs. an entry-point
      registry). For a small library, "just pass an instance" is simplest —
      avoid building a plugin-discovery system unless you expect many
      independent packages.

## 2. Fix known bugs before anyone builds on top of them

- [ ] `DefaultDataClass.__post_init__` (`src/pytrale/utils/baseclass.py`) does
      `setattr` on fields, but `Trale` is a `frozen=True` dataclass — this
      will raise `FrozenInstanceError` the moment a default is actually
      `None` and needs filling. Either drop the frozen requirement or use
      `object.__setattr__`.
- [ ] `interpolate()` in `weightsDatabase.py` is documented as "linear
      interpolation of NaNs" but actually checks `y == 0` (used here as "no
      measurement" sentinel). Fix the docstring or, better, use `np.nan` as
      the actual sentinel so `0` isn't ambiguous with a real reading of zero.
- [ ] `gp.py`'s `__main__` block calls `gp.optimize_hyperparameters(...,
      num_iters=...)`, but the method is `optimize_hyperparameters_log(...,
      n_iters=...)` — the demo script currently crashes if run.
- [ ] `notebooks/analyze.ipynb` references `db.is_interpolation`, which
      doesn't exist on `Trale` (only `is_measurement`, `is_no_measurement`,
      `is_extrapolation`, `is_no_extrapolation`) — update or remove the stale
      cell.
- [ ] Add regression tests for the above before/while fixing so they don't
      resurface.

## 3. Testing & CI

- [ ] `tests/__init__.py` is empty and there are no actual tests. Add unit
      tests for `Trale` (interpolation/extrapolation correctness on small
      synthetic series, edge cases: 0 measurements, 1 measurement) and for
      each algorithm implementation.
- [ ] Add `pytest` (+ `pytest-cov`) as a dev dependency and a `tests` job.
- [ ] Add GitHub Actions CI: run tests + lint on push/PR across the Python
      versions you intend to support (`requires-python = ">=3.11"` currently).
- [ ] Add a lint/format step (e.g. `ruff`) and, ideally, a pre-commit config
      so external contributors get fast feedback.

## 4. Packaging correctness

Not publishing to PyPI, so this is about `pyproject.toml` being correct for
installing directly from GitHub (`pip install git+https://github.com/...` or
`uv add git+https://github.com/...`), not about a PyPI listing.

- [ ] `pyproject.toml` still declares `pdm-backend` as the build backend
      (`[build-system]`), but the project has migrated to `uv` — confirm the
      backend is intentional and update if it's stale (a git install still
      needs a working build backend).
- [ ] Fix placeholder author info: `authors = [{name = "braniii", email =
      "''"}]` — the empty-string email is broken metadata.
- [ ] Add a `repository`/`homepage` URL in `pyproject.toml` pointing at the
      GitHub repo, so it's discoverable from the installed package metadata.
- [ ] Confirm the dependency list is minimal for the *library* — `matplotlib`
      and `prettypyplot` are plotting conveniences and `tqdm` is only used by
      `GaussianProcess`'s optimizer; consider moving these to an `[extras]`
      group (e.g. `plotting`, `gp`) so installing the core package stays
      lightweight for consumers who bring their own algorithm.
- [ ] Verify a clean `uv add git+<repo-url>` (or `pip install
      git+<repo-url>`) into a fresh environment actually works end-to-end.
- [ ] Decide on a versioning scheme and tag releases (e.g. `v0.1.0`) so
      consumers can pin `git+<repo-url>@v0.1.0` instead of tracking `main`.

## 5. License & repo metadata

- [ ] `pyproject.toml` declares `license = {text = "MIT"}` but there is no
      `LICENSE` file in the repo — add one (MIT) so the claim is actually
      backed by a file, which GitHub/PyPI both expect.
- [ ] Double check the source data format this library parses
      (`load_backup`, trale export `.txt` files) doesn't require documenting
      any third-party format license/attribution.

## 6. Documentation

- [ ] `README.md` is currently just the title (`# pytrale`). Write a real
      one: what problem it solves (interpolating/extrapolating sparse,
      noisy body-weight measurements), install instructions (since there's
      no PyPI release, this means `pip install git+https://github.com/...`
      or `uv add git+https://github.com/...`, ideally pinned to a tag), a
      minimal usage example (ideally the notebook's synthetic-data example
      reduced to a snippet), and a link to the algorithm-interface docs for
      contributors.
- [ ] Add a "Custom algorithms" section (or separate `CONTRIBUTING.md`)
      showing how to implement and plug in a new `Interpolator`.
- [ ] Clean up `notebooks/analyze.ipynb` so it runs top-to-bottom without
      errors and reflects the post-refactor API — this is likely to be the
      first thing new users open.
- [ ] Add a `CHANGELOG.md` (even minimal, Keep-a-Changelog style) so future
      algorithm contributions and breaking API changes are tracked.

## 7. Community hygiene (lightweight, do last)

- [ ] `CONTRIBUTING.md` covering: dev setup with `uv`, running tests/lint,
      and — most importantly for this project's stated goal — how to add a
      new algorithm implementation and where its tests should live.
- [ ] Basic issue/PR templates (bug report, new-algorithm proposal).
- [ ] Confirm the GitHub repo (`QuantumPhysique/pytrale`) description/topics
      are set once public, and that no personal data files were ever
      committed (checked — repo history only ever contained source/config
      files, no real weight-export data).

## Suggested order of execution

1. §1 (interface) + §2 (bugs) together, since the refactor is the natural
   place to fix the bugs.
2. §3 (tests/CI) as the refactor lands, not after.
3. §4 + §5 (packaging/license) — mechanical, low-risk, can happen in parallel.
4. §6 (docs) once the API is stable, so you're not documenting something
   you'll immediately change.
5. §7 last, right before/after making the repo public.
