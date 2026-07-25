from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Interpolator(ABC):
    """Contract for pluggable weight interpolation/extrapolation algorithms.

    An `Interpolator` is fit on sparse, irregularly-spaced measurements and
    must then be able to estimate the weight at arbitrary query times. This
    is the extension point for alternative algorithms — implement this
    interface and pass an instance to `Trale(..., algorithm=...)`.
    """

    @abstractmethod
    def fit(self, times_measured: NDArray, weights_measured: NDArray) -> "Interpolator":
        """Fit the algorithm to sparse measurements.

        Args:
            times_measured: 1D array of measurement times [days].
            weights_measured: 1D array of measured weights [kg], same shape
                as `times_measured`.

        Returns:
            self, so calls can be chained with `predict`.
        """

    @abstractmethod
    def predict(self, times: NDArray) -> NDArray:
        """Estimate the weight at the given query times.

        Args:
            times: 1D array of query times [days].

        Returns:
            1D array of estimated weights [kg], same shape as `times`.
        """
