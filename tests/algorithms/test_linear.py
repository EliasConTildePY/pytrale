import numpy as np
import pytest

from pytrale.algorithms.linear import LinearInterpolator


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearInterpolator().predict(np.array([0.0]))


def test_predict_matches_measurements_at_measured_times():
    times_measured = np.array([0.0, 5.0, 10.0])
    weights_measured = np.array([70.0, 71.0, 69.0])

    model = LinearInterpolator().fit(times_measured, weights_measured)
    predicted = model.predict(times_measured)

    np.testing.assert_allclose(predicted, weights_measured)


def test_predict_interpolates_linearly_between_measurements():
    times_measured = np.array([0.0, 10.0])
    weights_measured = np.array([70.0, 80.0])

    model = LinearInterpolator().fit(times_measured, weights_measured)

    assert model.predict(np.array([5.0]))[0] == pytest.approx(75.0)


def test_predict_clamps_outside_measured_range():
    times_measured = np.array([0.0, 10.0])
    weights_measured = np.array([70.0, 80.0])

    model = LinearInterpolator().fit(times_measured, weights_measured)

    predicted = model.predict(np.array([-5.0, 15.0]))

    np.testing.assert_allclose(predicted, [70.0, 80.0])


def test_fit_handles_unsorted_input():
    times_measured = np.array([10.0, 0.0, 5.0])
    weights_measured = np.array([80.0, 70.0, 75.0])

    model = LinearInterpolator().fit(times_measured, weights_measured)

    assert model.predict(np.array([5.0]))[0] == pytest.approx(75.0)
