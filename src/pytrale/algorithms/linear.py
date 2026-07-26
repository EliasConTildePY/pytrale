import numpy as np
from numpy.typing import NDArray

from pytrale.algorithms.base import Interpolator


class LinearInterpolator(Interpolator):
    """Plain linear interpolation between measurements.

    Query times before the first or after the last measurement are clamped
    to the nearest measured value (no extrapolation). This is the simplest
    possible `Interpolator` and is mainly useful as a baseline to compare
    against smoothing algorithms such as `GaussianKernelSmoother`.
    """

    def __init__(self) -> None:
        self._times_measured: NDArray | None = None
        self._weights_measured: NDArray | None = None

    def fit(
        self, times_measured: NDArray, weights_measured: NDArray
    ) -> "LinearInterpolator":
        self._times_measured = np.asarray(times_measured, dtype=float)
        self._weights_measured = np.asarray(weights_measured, dtype=float)
        return self

    def predict(self, times: NDArray) -> NDArray:
        if self._times_measured is None or self._weights_measured is None:
            raise RuntimeError(
                "LinearInterpolator must be fit before predict is called."
            )
        order = np.argsort(self._times_measured)
        return np.interp(
            times,
            self._times_measured[order],
            self._weights_measured[order],
        )
