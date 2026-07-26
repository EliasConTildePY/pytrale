# Contributing

## Setup

```bash
git clone https://github.com/QuantumPhysique/pytrale.git
cd pytrale
uv sync --group dev
```

Run tests and lint before opening a PR:

```bash
uv run pytest
uv run ruff check .
```

## Adding a new algorithm

This is the main way to contribute to pytrale. Algorithms live in
`src/pytrale/algorithms/` and implement the `Interpolator` interface
(`src/pytrale/algorithms/base.py`):

```python
class Interpolator(ABC):
    def fit(self, times_measured, weights_measured) -> "Interpolator": ...
    def predict(self, times) -> NDArray: ...
```

- `fit` receives the raw, sparse measurements (`times_measured` in days,
  `weights_measured` in kg) and should store whatever state your algorithm
  needs.
- `predict` receives an arbitrary array of query times and must return an
  estimated weight for each one.

Steps:

1. Add `src/pytrale/algorithms/<your_algorithm>.py` with a class implementing
   `Interpolator`.
2. Export it from `src/pytrale/algorithms/__init__.py`.
3. Add tests under `tests/algorithms/test_<your_algorithm>.py` — see
   `tests/algorithms/test_linear.py` for a minimal example and
   `tests/algorithms/test_gaussian_process.py` for one with a more involved
   fit.
4. Verify it works through `Trale`, e.g.:

   ```python
   Trale(times_measured=..., weights_measured=..., algorithm=YourAlgorithm())
   ```

No need to touch `Trale` itself — `algorithm` accepts any `Interpolator`.
