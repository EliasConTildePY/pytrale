from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from dateutil import parser
from numpy.typing import NDArray

from pytrale.algorithms.base import Interpolator
from pytrale.algorithms.gaussian_kernel import GaussianKernelSmoother
from pytrale.utils.baseclass import DefaultDataClass


def load_backup(filename: str) -> list:
    def parse_line(date, weight):
        return parser.parse(date), float(weight)

    with open(filename) as reader:
        return [
            parse_line(*line.split())
            for line in reader.readlines()
            if not line.startswith("#")
        ]


@dataclass(frozen=True)
class Trale(DefaultDataClass):
    times_measured: NDArray  # [days]
    weights_measured: NDArray  # [kg]

    extrapolation_range: int = 7  # [days]

    interpol_strength_measurement: float = 4  # [days]
    interpol_strength_interpol: float = 2  # [days]
    interpol_weight: float = 2

    # Algorithm used to interpolate/extrapolate `weights_predicted`. Defaults
    # to `GaussianKernelSmoother`, configured from the fields above. Pass any
    # `Interpolator` implementation to use a different algorithm — see
    # `pytrale.algorithms`.
    algorithm: Interpolator | None = field(default=None)

    @cached_property
    def n_measurements(self) -> int:
        return len(self.times_measured)

    @cached_property
    def times(self) -> NDArray:
        if not self.n_measurements:
            return np.array([])

        return np.arange(
            np.floor(self.times_measured.min()) - self.extrapolation_range,
            np.floor(self.times_measured.max()) + self.extrapolation_range + 1,
        )

    @cached_property
    def weights(self) -> NDArray:
        _weights = np.zeros_like(self.times)

        for idx, time in enumerate(self.times):
            idx_measurements = np.abs(np.floor(self.times_measured) - time) < 0.1
            if np.any(idx_measurements):
                _weights[idx] = np.mean(
                    self.weights_measured[idx_measurements],
                )
        return _weights

    @cached_property
    def is_measurement(self) -> NDArray:
        return (self.weights > 0).astype(int)

    @cached_property
    def is_no_measurement(self) -> NDArray:
        return 1 - self.is_measurement

    @cached_property
    def is_extrapolation(self) -> NDArray:
        _is_extrapolation = np.zeros_like(self.times, dtype=int)
        _is_extrapolation[: self.extrapolation_range] = 1
        _is_extrapolation[-self.extrapolation_range :] = 1
        return _is_extrapolation

    @cached_property
    def is_no_extrapolation(self) -> NDArray:
        return 1 - self.is_extrapolation

    @cached_property
    def _resolved_algorithm(self) -> Interpolator:
        if self.algorithm is not None:
            return self.algorithm
        return GaussianKernelSmoother(
            extrapolation_range=self.extrapolation_range,
            interpol_strength_measurement=self.interpol_strength_measurement,
            interpol_strength_interpol=self.interpol_strength_interpol,
            interpol_weight=self.interpol_weight,
        )

    @cached_property
    def _fitted_algorithm(self) -> Interpolator:
        return self._resolved_algorithm.fit(self.times_measured, self.weights_measured)

    @cached_property
    def weights_predicted(self) -> NDArray:
        """Weights estimated by `algorithm` (or its default) on `times`."""
        if not self.n_measurements:
            return np.array([])
        return self._fitted_algorithm.predict(self.times)

    @classmethod
    def fromFile(cls, filename, **kwargs):
        measurements = load_backup(filename)
        times = np.array([time.timestamp() / (24 * 3600) for time, _ in measurements])
        weights = np.array([weight for _, weight in measurements])

        return cls(
            times_measured=times,
            weights_measured=weights,
            **kwargs,
        )
