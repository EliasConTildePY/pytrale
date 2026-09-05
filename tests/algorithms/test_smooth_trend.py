import numpy as np
import pytest

from pytrale import Trale
from pytrale.algorithms.smooth_trend import (
    SmoothTrend,
    _concentrated_objective,
    _filter_forward,
)
from tests.preview_data import preview_measurements


def _simulated_series(
    n_days: int = 500, diffusion: float = 2e-4, sigma_eps: float = 0.8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a trajectory from the model the smoother assumes."""
    rng = np.random.default_rng(seed)
    increment_cov = diffusion * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])

    level, rate = 80.0, 0.0
    levels = np.empty(n_days)
    for day in range(n_days):
        noise_level, noise_rate = rng.multivariate_normal([0.0, 0.0], increment_cov)
        level, rate = level + rate + noise_level, rate + noise_rate
        levels[day] = level

    times = np.arange(n_days, dtype=float)
    return times, levels + rng.normal(0.0, sigma_eps, n_days), levels


def _sparse_series(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A short, irregularly sampled series in the shape of real exports."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.choice(np.arange(120.0), size=60, replace=False))
    return times, 80.0 - 0.02 * times + rng.normal(0.0, 0.7, times.size)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SmoothTrend().predict(np.array([0.0]))


def test_fit_requires_at_least_one_measurement():
    with pytest.raises(ValueError):
        SmoothTrend().fit(np.array([]), np.array([]))


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        SmoothTrend().fit(np.array([0.0, 1.0]), np.array([70.0]))


def test_recovers_the_hyperparameters_of_a_simulated_trajectory():
    diffusion, sigma_eps = 2e-4, 0.8
    times, weights, _ = _simulated_series(diffusion=diffusion, sigma_eps=sigma_eps)

    model = SmoothTrend(huber_c=None).fit(times, weights)

    assert model.lam_ == pytest.approx(diffusion / sigma_eps**2, rel=0.5)
    assert model.sigma_eps_ == pytest.approx(sigma_eps, rel=0.1)
    assert model.bandwidth_ == pytest.approx(model.lam_**-0.25)


def test_smoothed_curve_tracks_the_latent_level():
    times, weights, levels = _simulated_series()

    predicted = SmoothTrend(huber_c=None).fit(times, weights).predict(times)

    assert np.sqrt(np.mean((predicted - levels) ** 2)) < 0.3


def test_smoothed_state_matches_the_filter_at_the_last_time():
    times, weights = _sparse_series()
    model = SmoothTrend(huber_c=None).fit(times, weights)
    posterior = model.smooth(times)

    noise_variance = model.sigma_eps_**2
    filtered = _filter_forward(
        times - times[0],
        weights,
        np.ones(times.size, dtype=bool),
        noise_variance=noise_variance,
        diffusion=noise_variance * model.lam_,
        trend_prior_variance=model.trend_prior_variance,
        huber_c=None,
    )

    assert posterior.mean[-1] == pytest.approx(filtered.mu[-1])
    assert posterior.trend[-1] == pytest.approx(filtered.nu[-1])
    assert posterior.std[-1] ** 2 == pytest.approx(filtered.p11[-1])
    assert posterior.trend_std[-1] ** 2 == pytest.approx(filtered.p22[-1])


def test_negligible_observation_noise_makes_the_curve_interpolate():
    times, weights = _sparse_series()
    sigma_eps, diffusion = 1e-3, 1e-2

    # Gating has to be off: at this noise level every reading is an outlier.
    model = SmoothTrend(lam=diffusion / sigma_eps**2, sigma_eps=sigma_eps, huber_c=None)

    predicted = model.fit(times, weights).predict(times)

    np.testing.assert_allclose(predicted, weights, atol=10 * sigma_eps)


def test_predictions_are_invariant_to_the_time_origin():
    times, weights = _sparse_series()
    query = np.arange(-7.0, 127.0)
    offset = 20_000.0  # trale exports days since the epoch

    here = SmoothTrend().fit(times, weights).predict(query)
    shifted = SmoothTrend().fit(times + offset, weights).predict(query + offset)

    np.testing.assert_allclose(shifted, here)


def test_predictions_are_invariant_to_the_query_grid():
    """A query grid must not change the answer at the times it shares.

    This is what forces the exact continuous-time discretisation of `Q(dt)`,
    cross term included: with a diagonal `Q` the two grids disagree.
    """
    times, weights = _sparse_series()
    model = SmoothTrend().fit(times, weights)

    fine = np.arange(-7.0, 127.0, 0.5)
    coarse = fine[::6]

    np.testing.assert_allclose(model.predict(fine)[::6], model.predict(coarse))


def test_uncertainty_widens_in_gaps_and_beyond_the_data():
    times = np.concatenate([np.arange(0.0, 30.0), np.arange(50.0, 80.0)])
    weights = 80.0 + 0.01 * times
    model = SmoothTrend().fit(times, weights)

    std = model.predict_std(np.array([15.0, 40.0, 95.0]))

    assert std[0] < std[1]
    assert std[1] < std[2]


def test_huber_gating_limits_the_pull_of_an_outlier():
    times, weights = _sparse_series()
    contaminated = weights.copy()
    contaminated[30] += 3.0

    clean = SmoothTrend(huber_c=None).fit(times, weights).predict(times)
    gaussian = SmoothTrend(huber_c=None).fit(times, contaminated).predict(times)
    gated = SmoothTrend(huber_c=2.5).fit(times, contaminated).predict(times)

    assert np.abs(gated - clean).max() < 0.5 * np.abs(gaussian - clean).max()


def test_concentrated_objective_is_unimodal_on_the_grid():
    times, weights, _ = _simulated_series()
    grid = np.linspace(-8.0, -2.0, 40)

    objective = np.array(
        [_concentrated_objective(x, times, weights, 0.035) for x in grid]
    )

    assert 0 < objective.argmin() < len(grid) - 1
    sign_changes = np.count_nonzero(np.diff(np.sign(np.diff(objective))))
    assert sign_changes == 1


def test_single_measurement_predicts_a_constant_curve():
    model = SmoothTrend().fit(np.array([10.0]), np.array([70.0]))

    np.testing.assert_allclose(model.predict(np.array([0.0, 10.0, 20.0])), 70.0)
    np.testing.assert_allclose(model.predict_trend(np.array([0.0, 20.0])), 0.0)


def test_two_measurements_predict_the_connecting_line():
    model = SmoothTrend().fit(np.array([0.0, 10.0]), np.array([70.0, 80.0]))

    np.testing.assert_allclose(model.predict(np.array([5.0, 20.0])), [75.0, 90.0])
    np.testing.assert_allclose(model.predict_trend(np.array([5.0])), 1.0)


@pytest.mark.parametrize("n_measurements", [1, 2])
def test_uncertainty_is_undefined_below_three_measurements(n_measurements):
    times, weights = _sparse_series()
    model = SmoothTrend().fit(times[:n_measurements], weights[:n_measurements])

    assert np.all(np.isnan(model.predict_std(times)))
    assert np.all(np.isnan(model.predict_trend_std(times)))
    assert model.lam_ is None


def test_repeated_measurements_on_the_same_day_are_handled_in_order():
    times = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([70.0, 70.2, 70.9, 70.1, 70.3, 70.4, 70.5])
    swapped = weights.copy()
    swapped[[2, 3]] = swapped[[3, 2]]

    model = SmoothTrend(lam=1e-3, sigma_eps=0.5).fit(times, weights)
    other = SmoothTrend(lam=1e-3, sigma_eps=0.5).fit(times, swapped)

    query = np.arange(-2.0, 8.0)
    np.testing.assert_allclose(model.predict(query), other.predict(query))
    assert np.all(np.isfinite(model.predict_std(query)))


def test_fit_handles_unsorted_input():
    times, weights = _sparse_series()
    order = np.random.default_rng(2).permutation(times.size)

    sorted_fit = SmoothTrend().fit(times, weights).predict(times)
    shuffled_fit = SmoothTrend().fit(times[order], weights[order]).predict(times)

    np.testing.assert_allclose(shuffled_fit, sorted_fit)


def test_explicit_hyperparameters_are_used_as_given():
    times, weights = _sparse_series()

    model = SmoothTrend(lam=1e-5, sigma_eps=0.9).fit(times, weights)

    assert model.lam_ == 1e-5
    assert model.sigma_eps_ == 0.9
    assert model.bandwidth_ == pytest.approx(1e-5**-0.25)


def test_few_measurements_skip_the_maximum_likelihood_search():
    times, weights = _sparse_series()
    model = SmoothTrend().fit(times[:5], weights[:5])

    assert model.bandwidth_ == pytest.approx(6.0)


def test_fit_is_deterministic():
    times, weights = _sparse_series()
    query = np.arange(-7.0, 127.0)

    first = SmoothTrend().fit(times, weights).smooth(query)
    second = SmoothTrend().fit(times, weights).smooth(query)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_smooths_the_trale_preview_series():
    times, weights = preview_measurements()

    model = SmoothTrend().fit(times, weights)
    predicted = model.predict(times)

    assert np.all(np.isfinite(predicted))
    assert predicted.min() > weights.min()
    assert predicted.max() < weights.max()


def test_works_through_trale():
    times, weights = preview_measurements()

    db = Trale(times_measured=times, weights_measured=weights, algorithm=SmoothTrend())

    assert db.weights_predicted.shape == db.times.shape
    assert np.all(np.isfinite(db.weights_predicted))
