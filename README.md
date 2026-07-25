# pytrale

Interpolate and extrapolate sparse, noisy body-weight measurements — the
kind you get from stepping on a scale a few times a week — into a smooth,
continuous curve. Built around data exported from trale, an Android
weight-tracking app.

The interpolation/extrapolation algorithm is pluggable: pytrale ships a
default and one alternative, and you can implement your own.

## Install

There's no PyPI release. Install directly from GitHub, ideally pinned to a
tag:

```bash
pip install git+https://github.com/QuantumPhysique/pytrale.git@v0.1.0
# or
uv add git+https://github.com/QuantumPhysique/pytrale.git@v0.1.0
```

Drop the `@v0.1.0` to track `main` instead.

## Usage

```python
import numpy as np
from pytrale import Trale

db = Trale(
    times_measured=np.array([0, 1, 3, 4, 8, 9, 10]),      # days
    weights_measured=np.array([70.2, 70.0, 69.8, 70.1, 69.5, 69.6, 69.4]),  # kg
)

db.times              # daily grid, padded by `extrapolation_range` on each side
db.weights_predicted   # smoothed/interpolated/extrapolated weight on that grid
db.is_measurement      # 1 where a real measurement exists on that day, else 0
```

Load directly from a trale export file instead of passing arrays by hand:

```python
db = Trale.fromFile("trale_export.txt")
```

### Choosing an algorithm

`Trale` accepts any `Interpolator` implementation via the `algorithm`
argument. Two are built in:

```python
from pytrale.algorithms import GaussianKernelSmoother, LinearInterpolator

# Default: denoises each measurement, then interpolates/extrapolates with a
# Gaussian kernel.
db = Trale(times_measured=..., weights_measured=..., algorithm=GaussianKernelSmoother())

# A plain linear-interpolation baseline, with no smoothing.
db = Trale(times_measured=..., weights_measured=..., algorithm=LinearInterpolator())
```

`GaussianProcess` (`pytrale.algorithms.GaussianProcess`) is also included, as
a from-scratch Gaussian Process regression with a trend + weekly/monthly/
annual periodic kernel.

### Implementing your own algorithm

Subclass `pytrale.algorithms.Interpolator` and implement `fit`/`predict`:

```python
from pytrale.algorithms import Interpolator

class MyAlgorithm(Interpolator):
    def fit(self, times_measured, weights_measured):
        # store whatever your algorithm needs from the sparse measurements
        return self

    def predict(self, times):
        # return an estimated weight for each entry in `times`
        ...

db = Trale(times_measured=..., weights_measured=..., algorithm=MyAlgorithm())
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more on contributing a new
algorithm.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

To run [notebooks/analyze.ipynb](notebooks/analyze.ipynb), also sync the
`notebooks` group and the `notebooks` extra (matplotlib/prettypyplot):

```bash
uv sync --group dev --group notebooks --extra notebooks
```

## License

MIT — see [LICENSE](LICENSE).
