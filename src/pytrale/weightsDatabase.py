from dataclasses import dataclass
from dateutil import parser
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from pytrale.utils.baseclass import DefaultDataClass


def load_backup(filename: str) -> list:
    def parse_line(date, weight):
        return parser.parse(date), float(weight)
    with open(filename, 'r') as reader:
        return [
            parse_line(*line.split())
            for line in reader.readlines() if not line.startswith('#')
        ]


def interpolate(weights):
    def zero_helper(y):
        """Helper to handle indices and logical indices of 0.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - zeros, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> zeros, x= zero_helper(y)
            >>> y[zeros]= np.interp(x(zeros), x(~zeros), y[~zeros])
        """
        return y == 0, lambda z: z.nonzero()[0]

    zeros, weights_idx = zero_helper(weights)
    weights_interpolated = weights.copy()
    weights_interpolated[zeros] = np.interp(
        weights_idx(zeros), weights_idx(~zeros), weights_interpolated[~zeros],
    )
    return weights_interpolated


@dataclass(frozen=True)
class Trale(DefaultDataClass):
    times_measured: NDArray  # [days]
    weights_measured: NDArray  # [kg]

    extrapolation_range: int = 7  # [days]

    interpol_strength_measurement: float = 4  # [days]
    interpol_strength_interpol: float = 2  # [days]
    interpol_weight: float = 2

    @cached_property
    def n_measurements(self) -> int:
        return len(self.times_measured)

    @cached_property
    def times(self) -> NDArray:
        if not self.n_measurements:
            return np.array([])

        return np.arange(
            np.floor(self.times_measured.min()) - self.extrapolation_range,
            np.floor(self.times_measured.max()) + self.extrapolation_range + 1 ,
        )

    @cached_property
    def weights(self) -> NDArray:
        _weights = np.zeros_like(self.times)

        for idx, time in enumerate(self.times):
            idx_measurements = np.abs(np.floor(self.times_measured) - time) < 0.1
            if np.any(idx_measurements):
                _weights[idx] = np.mean(self.weights_measured[idx_measurements])
        return _weights

    @cached_property
    def weights_smoothed(self) -> NDArray:
        return self._gaussian_interpolation(
            self._linear_extrapolation(
                self._linear_interpolation(self.weights),
            )
        ) * self.is_measurement

    @cached_property
    def weights_linear_extrapolated(self) -> NDArray:
        return self._linear_extrapolation(
            self._linear_interpolation(self.weights_smoothed)
        )

    @cached_property
    def weights_gaussian_extrapolated(self) -> NDArray:
        return self._gaussian_interpolation(
            self.weights_linear_extrapolated
        )

    @cached_property
    def weights_linear_interpol(self) -> NDArray:
        return interpolate(self.weights)

    @cached_property
    def is_measurement(self) -> NDArray:
        return (self.weights > 0).astype(int)

    @cached_property
    def is_no_measurement(self) -> NDArray:
        return 1 - self.is_measurement

    @cached_property
    def is_extrapolation(self) -> NDArray:
        _is_extrapolation = np.zeros_like(self.times)
        _is_extrapolation[:self.extrapolation_range] = 1
        _is_extrapolation[-self.extrapolation_range:] = 1
        return _is_extrapolation

    @cached_property
    def sigma(self) -> NDArray:
        return (
            self.is_measurement * self.interpol_strength_measurement + 
            self.is_no_measurement * self.interpol_strength_interpol
        )

    def _gaussian_weights(self, t: float, ms: NDArray) -> NDArray:
        gaussian_weights = 1 / (
            self.sigma * np.sqrt(2 * np.pi)
        ) * np.exp(
            -1 * (self.times - t) ** 2 / (2 * self.sigma ** 2)
        ) * (
            self.is_measurement * self.interpol_weight + self.is_no_measurement
        )
        return gaussian_weights / gaussian_weights[ms > 0].sum()

    def _gaussian_mean(self, t: float, ms: NDArray) -> float:
        return np.dot(self._gaussian_weights(t, ms), ms)

    def _gaussian_interpolation(self, weights: NDArray) -> NDArray:
        return np.array([
            self._gaussian_mean(t, weights) if weights[idx] else 0
            for idx, t in enumerate(self.times)
        ])

    def _linear_extrapolation(self, weights: NDArray) -> NDArray:
        if not self.n_measurements:
            return np.array([])
        elif self.n_measurements == 1:
            return np.full_like(self.times, weights[0])

        weights_extrapol = weights.copy()
        weights_extrapol[:self.extrapolation_range] = self._linear_regression(
            weights,
            self.times[self.extrapolation_range],
            self.times[:self.extrapolation_range],
        )
        weights_extrapol[-self.extrapolation_range:] = self._linear_regression(
            weights,
            self.times[-self.extrapolation_range - 1],
            self.times[-self.extrapolation_range:],
        )
        return weights_extrapol

    def _linear_interpolation(self, weights: NDArray) -> NDArray:
        return interpolate(weights)

    def _linear_regression(
        self, weights: NDArray, t_ref: float, times: NDArray
    ) -> NDArray:
        mean_weight = self._gaussian_mean(t_ref, weights)
        mean_time = self._gaussian_mean(t_ref, self.times)
        mean_change = self._gaussian_mean(
            t_ref, (weights - mean_weight) * self.times
        ) / self._gaussian_mean(
            t_ref, (self.times - mean_time) * self.times
        )

        intercept = mean_weight - mean_change * mean_time
        return mean_change * times + intercept

    @classmethod
    def fromFile(cls, filename, **kwargs):
        measurements = load_backup(filename)
        times = np.array([
            time.timestamp() / (24 * 3600) for time, _ in measurements
        ])
        weights = np.array([weight for _, weight in measurements])

        return cls(
            times_measured=times,
            weights_measured=weights,
            **kwargs,
        )
