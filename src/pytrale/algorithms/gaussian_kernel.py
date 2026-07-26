import numpy as np
from numpy.typing import NDArray

from pytrale.algorithms.base import Interpolator


def _fill_zero_gaps(weights: NDArray) -> NDArray:
    """Linearly interpolate zero-valued gaps in a dense raster.

    `weights` is a raster aligned to a regular daily time grid where `0`
    marks "no measurement on this day" (a real body weight is never exactly
    zero, so it doubles as an unambiguous missing-value sentinel here).
    Zero entries are filled by linear interpolation between their nearest
    non-zero neighbors.
    """
    is_gap = weights == 0
    indices = np.arange(len(weights))
    filled = weights.copy()
    filled[is_gap] = np.interp(indices[is_gap], indices[~is_gap], weights[~is_gap])
    return filled


class GaussianKernelSmoother(Interpolator):
    """Denoise sparse measurements, then interpolate/extrapolate a full curve.

    This is pytrale's original, default algorithm. It works in three stages:

    1. Each raw measurement is replaced by a Gaussian-kernel-weighted
       average of its (already interpolated/extrapolated) neighbors,
       denoising individual noisy readings.
    2. The denoised measurements are linearly interpolated between
       neighbors and linearly extrapolated at the edges (over
       `extrapolation_range` days).
    3. The result is smoothed once more with the same Gaussian kernel to
       produce the final continuous curve.

    Measurements are weighted more heavily than interpolated/extrapolated
    points when computing the kernel (`interpol_weight`), and the kernel
    bandwidth differs between measured and non-measured points
    (`interpol_strength_measurement` / `interpol_strength_interpol`).
    """

    def __init__(
        self,
        extrapolation_range: int = 7,
        interpol_strength_measurement: float = 4.0,
        interpol_strength_interpol: float = 2.0,
        interpol_weight: float = 2.0,
    ) -> None:
        self.extrapolation_range = extrapolation_range
        self.interpol_strength_measurement = interpol_strength_measurement
        self.interpol_strength_interpol = interpol_strength_interpol
        self.interpol_weight = interpol_weight

        self._n_measurements: int | None = None
        self._times: NDArray | None = None

    def fit(
        self, times_measured: NDArray, weights_measured: NDArray
    ) -> "GaussianKernelSmoother":
        times_measured = np.asarray(times_measured, dtype=float)
        weights_measured = np.asarray(weights_measured, dtype=float)
        self._n_measurements = len(times_measured)
        if not self._n_measurements:
            raise ValueError(
                "GaussianKernelSmoother requires at least one measurement."
            )

        self._times = np.arange(
            np.floor(times_measured.min()) - self.extrapolation_range,
            np.floor(times_measured.max()) + self.extrapolation_range + 1,
        )

        weights = np.zeros_like(self._times)
        for idx, t in enumerate(self._times):
            idx_measurements = np.abs(np.floor(times_measured) - t) < 0.1
            if np.any(idx_measurements):
                weights[idx] = np.mean(weights_measured[idx_measurements])

        self._is_measurement = (weights > 0).astype(int)
        self._is_no_measurement = 1 - self._is_measurement

        self._is_extrapolation = np.zeros_like(self._times, dtype=int)
        self._is_extrapolation[: self.extrapolation_range] = 1
        self._is_extrapolation[-self.extrapolation_range :] = 1
        self._is_no_extrapolation = 1 - self._is_extrapolation

        self._sigma = (
            self._is_measurement * self.interpol_strength_measurement
            + self._is_no_measurement * self.interpol_strength_interpol
        )

        self.smoothed_ = (
            self._gaussian_interpolation(
                self._linear_extrapolation(self._linear_interpolation(weights)),
            )
            * self._is_measurement
        )
        self.linear_extrapolated_ = self._linear_extrapolation(
            self._linear_interpolation(self.smoothed_),
        )
        self.gaussian_extrapolated_ = self._gaussian_interpolation(
            self.linear_extrapolated_
        )

        return self

    def predict(self, times: NDArray) -> NDArray:
        if self._times is None:
            raise RuntimeError(
                "GaussianKernelSmoother must be fit before predict is called."
            )
        return np.interp(times, self._times, self.gaussian_extrapolated_)

    def _gaussian_weights(self, t: float, ms: NDArray) -> NDArray:
        gaussian_weights = (
            1
            / (self._sigma * np.sqrt(2 * np.pi))
            * np.exp(-1 * (self._times - t) ** 2 / (2 * self._sigma**2))
            * (self._is_measurement * self.interpol_weight + self._is_no_measurement)
        )
        return np.where(
            ms > 0,
            gaussian_weights / gaussian_weights[ms > 0].sum(),
            0,
        )

    def _gaussian_mean(self, t: float, ms: NDArray) -> float:
        return np.dot(self._gaussian_weights(t, ms), ms)

    def _gaussian_interpolation(self, weights: NDArray) -> NDArray:
        return np.array(
            [
                self._gaussian_mean(t, weights) if weights[idx] else 0
                for idx, t in enumerate(self._times)
            ]
        )

    def _linear_extrapolation(self, weights: NDArray) -> NDArray:
        if self._n_measurements == 1:
            # Only one non-zero entry exists anywhere in `weights` (the
            # single real measurement) — its position depends on where the
            # measurement falls in the grid, so it can't be read via a fixed
            # index like `weights[0]`.
            (value,) = weights[weights != 0]
            return np.full_like(self._times, value)

        weights_extrapol = weights.copy()
        weights_extrapol[: self.extrapolation_range] = self._linear_regression(
            weights,
            self._times[self.extrapolation_range],
            self._times[: self.extrapolation_range],
        )
        weights_extrapol[-self.extrapolation_range :] = self._linear_regression(
            weights,
            self._times[-self.extrapolation_range - 1],
            self._times[-self.extrapolation_range :],
        )
        return weights_extrapol

    def _linear_interpolation(self, weights: NDArray) -> NDArray:
        return _fill_zero_gaps(weights) * self._is_no_extrapolation

    def _linear_regression(
        self, weights: NDArray, t_ref: float, times: NDArray
    ) -> NDArray:
        gs_weights = self._gaussian_weights(t_ref, weights)
        mean_weight = np.dot(gs_weights, weights)
        mean_time = np.dot(gs_weights, self._times)
        mean_change = np.dot(
            gs_weights, (weights - mean_weight) * self._times
        ) / np.dot(gs_weights, (self._times - mean_time) * self._times)

        intercept = mean_weight - mean_change * mean_time
        return mean_change * times + intercept
