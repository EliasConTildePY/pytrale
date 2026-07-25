import numpy as np

from pytrale.algorithms.gaussian_process import GaussianProcess


def _fitted_model():
    rng = np.random.default_rng(0)
    times_measured = np.arange(0, 60, 7, dtype=float)
    weights_measured = 70 + 0.05 * times_measured + rng.normal(
        scale=0.1, size=times_measured.shape
    )
    model = GaussianProcess(
        trend_length_scale=30,
        weekly_signal_variance=0.1,
        monthly_signal_variance=0.1,
        annual_signal_variance=0.1,
        noise_variance=0.05,
    )
    model.fit(times_measured, weights_measured)
    return model, times_measured, weights_measured


def test_predict_returns_array_matching_query_shape():
    model, times_measured, _ = _fitted_model()
    times_query = np.arange(0, 60, 1, dtype=float)

    predicted = model.predict(times_query)

    assert predicted.shape == times_query.shape
    assert np.all(np.isfinite(predicted))


def test_predict_with_uncertainty_returns_nonnegative_variance():
    model, times_measured, _ = _fitted_model()

    mean, variance = model.predict_with_uncertainty(times_measured)

    assert mean.shape == times_measured.shape
    assert np.all(variance >= 0)


def test_predict_at_training_points_is_close_to_observations():
    model, times_measured, weights_measured = _fitted_model()

    predicted = model.predict(times_measured)

    np.testing.assert_allclose(predicted, weights_measured, atol=3.0)


def test_log_marginal_likelihood_is_finite():
    model, times_measured, weights_measured = _fitted_model()

    lml = model.log_marginal_likelihood(times_measured, weights_measured)

    assert np.isfinite(lml)


def test_optimize_hyperparameters_log_updates_params_and_returns_lml():
    model, times_measured, weights_measured = _fitted_model()

    best_lml = model.optimize_hyperparameters_log(
        times_measured,
        weights_measured,
        n_iters=2,
        n_restarts=1,
    )

    assert np.isfinite(best_lml)
    assert np.isfinite(model.trend_length_scale)
