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

    sigma: float = 3  # [days]
    extrapolation_range: int = 7  # [days]

    @cached_property
    def times(self) -> NDArray:
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
    def weights_linear_interpol(self) -> NDArray:
        return interpolate(self.weights)

    @cached_property
    def is_measurement(self) -> NDArray:
        return (self.weights > 0).astype(int)

    @cached_property
    def is_interpolation(self) -> NDArray:
        return 1 - self.is_measurement

    @cached_property
    def is_extrapolation(self) -> NDArray:
        _is_extrapolation = np.zeros_like(self.times)
        _is_extrapolation[:self.extrapolation_range] = 1
        _is_extrapolation[-self.extrapolation_range:] = 1
        return _is_extrapolation

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
        return cls(mlp_network=[128, 128, 128, 128, 1], **kwargs)
