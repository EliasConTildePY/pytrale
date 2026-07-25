import pytest

from pytrale.algorithms.base import Interpolator


def test_interpolator_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Interpolator()


def test_interpolator_requires_fit_and_predict():
    class Incomplete(Interpolator):
        def fit(self, times_measured, weights_measured):
            return self

    with pytest.raises(TypeError):
        Incomplete()
