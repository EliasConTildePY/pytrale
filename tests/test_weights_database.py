import numpy as np
import pytest

from pytrale import Trale
from pytrale.algorithms.linear import LinearInterpolator


def _synthetic_measurements(seed=0, n=40):
    rng = np.random.default_rng(seed)
    times_measured = np.unique(rng.integers(0, 200, size=n).astype(float))
    weights_measured = 70 + rng.normal(scale=0.3, size=times_measured.shape)
    return times_measured, weights_measured


def test_empty_trale_has_empty_derived_arrays():
    db = Trale(times_measured=np.array([]), weights_measured=np.array([]))

    assert db.n_measurements == 0
    assert db.times.size == 0
    assert db.weights_predicted.size == 0


def test_times_grid_pads_by_extrapolation_range():
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(
        times_measured=times_measured,
        weights_measured=weights_measured,
        extrapolation_range=7,
    )

    assert db.times[0] == np.floor(times_measured.min()) - 7
    assert db.times[-1] == np.floor(times_measured.max()) + 7


def test_is_measurement_marks_exactly_the_measured_days():
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(times_measured=times_measured, weights_measured=weights_measured)

    measured_days = set(np.floor(times_measured).astype(int))
    grid_days = np.floor(db.times).astype(int)
    expected_mask = np.array([day in measured_days for day in grid_days], dtype=int)

    np.testing.assert_array_equal(db.is_measurement, expected_mask)
    np.testing.assert_array_equal(db.is_no_measurement, 1 - expected_mask)


def test_is_extrapolation_marks_edges_only():
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(
        times_measured=times_measured,
        weights_measured=weights_measured,
        extrapolation_range=5,
    )

    assert db.is_extrapolation[:5].sum() == 5
    assert db.is_extrapolation[-5:].sum() == 5
    assert db.is_extrapolation[5:-5].sum() == 0
    np.testing.assert_array_equal(db.is_no_extrapolation, 1 - db.is_extrapolation)


def test_weights_predicted_uses_default_gaussian_kernel_smoother():
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(times_measured=times_measured, weights_measured=weights_measured)

    assert db.weights_predicted.shape == db.times.shape
    assert np.all(np.isfinite(db.weights_predicted))


def test_custom_algorithm_is_used_instead_of_default():
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(
        times_measured=times_measured,
        weights_measured=weights_measured,
        algorithm=LinearInterpolator(),
    )

    expected = LinearInterpolator().fit(times_measured, weights_measured).predict(
        db.times
    )
    np.testing.assert_allclose(db.weights_predicted, expected)


def test_single_measurement_does_not_raise():
    db = Trale(times_measured=np.array([10.0]), weights_measured=np.array([72.0]))

    assert db.weights_predicted.shape == db.times.shape
    np.testing.assert_allclose(db.weights_predicted, 72.0)


def test_from_file(tmp_path):
    backup = tmp_path / "backup.txt"
    backup.write_text(
        "# comment line\n"
        "2024-01-01 70.0\n"
        "2024-01-02 70.5\n"
    )

    db = Trale.fromFile(str(backup))

    assert db.n_measurements == 2
    np.testing.assert_allclose(db.weights_measured, [70.0, 70.5])


@pytest.mark.parametrize("field_name", ["extrapolation_range", "interpol_weight"])
def test_explicit_none_falls_back_to_field_default(field_name):
    times_measured, weights_measured = _synthetic_measurements()
    db = Trale(
        times_measured=times_measured,
        weights_measured=weights_measured,
        **{field_name: None},
    )

    assert getattr(db, field_name) is not None
