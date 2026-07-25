import numpy as np
import pytest

from pytrale.algorithms.gaussian_kernel import GaussianKernelSmoother, _fill_zero_gaps


def test_fill_zero_gaps_interpolates_between_neighbors():
    weights = np.array([70.0, 0.0, 0.0, 73.0])

    filled = _fill_zero_gaps(weights)

    np.testing.assert_allclose(filled, [70.0, 71.0, 72.0, 73.0])


def test_fit_requires_at_least_one_measurement():
    with pytest.raises(ValueError):
        GaussianKernelSmoother().fit(np.array([]), np.array([]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GaussianKernelSmoother().predict(np.array([0.0]))


def test_fit_exposes_grid_and_stage_outputs():
    rng = np.random.default_rng(0)
    times_measured = np.arange(0, 30, 2, dtype=float)
    weights_measured = 70 + rng.normal(scale=0.2, size=times_measured.shape)

    model = GaussianKernelSmoother(extrapolation_range=5).fit(
        times_measured, weights_measured
    )

    assert model._times[0] == times_measured.min() - 5
    assert model._times[-1] == times_measured.max() + 5
    assert model.smoothed_.shape == model._times.shape
    assert model.gaussian_extrapolated_.shape == model._times.shape
    assert np.all(np.isfinite(model.gaussian_extrapolated_))


def test_predict_on_fitted_grid_matches_gaussian_extrapolated():
    times_measured = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    weights_measured = np.array([70.0, 70.5, 71.0, 70.5, 70.0])

    model = GaussianKernelSmoother().fit(times_measured, weights_measured)

    np.testing.assert_allclose(
        model.predict(model._times), model.gaussian_extrapolated_
    )


def test_single_measurement_predicts_constant_curve():
    times_measured = np.array([5.0])
    weights_measured = np.array([72.0])

    model = GaussianKernelSmoother(extrapolation_range=3).fit(
        times_measured, weights_measured
    )
    predicted = model.predict(model._times)

    np.testing.assert_allclose(predicted, 72.0)
