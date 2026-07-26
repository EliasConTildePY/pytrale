import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pytrale.algorithms.base import Interpolator


class GaussianProcess(Interpolator):
    """Gaussian Process regression with a trend + weekly/monthly/annual kernel.

    A from-scratch GP implementation (no scikit-learn dependency) intended as
    a second reference algorithm alongside `GaussianKernelSmoother`, showing
    that an `Interpolator` can be backed by a genuinely different model.
    `predict` returns the posterior mean only, to satisfy the `Interpolator`
    contract; use `predict_with_uncertainty` for mean and variance.
    """

    def __init__(
        self,
        trend_length_scale=1.0,
        trend_signal_variance=1.0,
        weekly_length_scale=1.0,
        weekly_signal_variance=1.0,
        monthly_length_scale=1.0,
        monthly_signal_variance=1.0,
        annual_length_scale=1.0,
        annual_signal_variance=1.0,
        noise_variance=1e-8,
    ):
        self.trend_length_scale = trend_length_scale
        self.trend_signal_variance = trend_signal_variance

        self.weekly_length_scale = weekly_length_scale
        self.weekly_signal_variance = weekly_signal_variance

        self.monthly_length_scale = monthly_length_scale
        self.monthly_signal_variance = monthly_signal_variance

        self.annual_length_scale = annual_length_scale
        self.annual_signal_variance = annual_signal_variance

        self.noise_variance = noise_variance

        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _prepare_X(self, X):
        """
        Ensure that X is a 2D array with shape (n_samples, n_features).
        If X is 1D, then convert it to a column vector.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def kernel(self, X1, X2):
        """
        Combined kernel: trend (RBF) + periodic (weekly, monthly, and annual).
        """
        X1 = self._prepare_X(X1)
        X2 = self._prepare_X(X2)

        # Use the first column as time
        t1 = X1[:, 0]
        t2 = X2[:, 0]

        # Trend Kernel: Squared Exponential (RBF)
        sqdist = (np.subtract.outer(t1, t2)) ** 2
        K_trend = self.trend_signal_variance * np.exp(
            -0.5 * sqdist / self.trend_length_scale**2
        )

        # Periodic Kernels
        diff = np.abs(np.subtract.outer(t1, t2))
        K_weekly = self.weekly_signal_variance * np.exp(
            -2 * (np.sin(np.pi * diff / 7) ** 2) / self.weekly_length_scale**2
        )
        K_monthly = self.monthly_signal_variance * np.exp(
            -2 * (np.sin(np.pi * diff / 30) ** 2) / self.monthly_length_scale**2
        )
        K_annual = self.annual_signal_variance * np.exp(
            -2 * (np.sin(np.pi * diff / 365) ** 2) / self.annual_length_scale**2
        )

        return K_trend + K_weekly + K_monthly + K_annual

    def fit(
        self, times_measured: NDArray, weights_measured: NDArray
    ) -> "GaussianProcess":
        """
        Fit the Gaussian Process by computing and storing the inverse
        covariance matrix.
        """
        self.X_train = self._prepare_X(times_measured)
        self.y_train = np.atleast_1d(weights_measured)
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(self.X_train.shape[0])
        self.K_inv = np.linalg.inv(K)
        return self

    def predict(self, times: NDArray) -> NDArray:
        mean, _ = self.predict_with_uncertainty(times)
        return mean

    def predict_with_uncertainty(self, X_test, full_cov=False):
        """
        Predict the mean and variance at test points.

        If full_cov is False, compute only the diagonal of the predictive
        covariance efficiently.

        :param X_test: Test input data (1D or 2D array).
        :param full_cov: If True, compute the full covariance matrix.
        :return: tuple (predictive_mean, predictive_variance)
        """
        X_test = self._prepare_X(X_test)
        K_trans = self.kernel(self.X_train, X_test)  # shape: (n_train, n_test)
        predictive_mean = K_trans.T.dot(self.K_inv).dot(self.y_train)
        if full_cov:
            K_test = self.kernel(X_test, X_test)
            predictive_cov = K_test - K_trans.T.dot(self.K_inv).dot(K_trans)
            predictive_variance = np.clip(np.diag(predictive_cov), 0, np.inf)
        else:
            # Diagonal of k(x,x) equals sum of signal variances for these kernels.
            diag_K_test = (
                self.trend_signal_variance
                + self.weekly_signal_variance
                + self.monthly_signal_variance
                + self.annual_signal_variance
            )
            diag_K_test = np.full(X_test.shape[0], diag_K_test)
            variance_correction = np.sum((self.K_inv.dot(K_trans)) * K_trans, axis=0)
            predictive_variance = np.clip(diag_K_test - variance_correction, 0, np.inf)
        return predictive_mean, predictive_variance

    def log_marginal_likelihood(self, X_train, y_train):
        """
        Compute the log marginal likelihood of the current model hyperparameters.
        """
        X_train = self._prepare_X(X_train)
        y_train = np.atleast_1d(y_train)  # Ensure y_train is 1D
        K = self.kernel(X_train, X_train)
        K += self.noise_variance * np.eye(X_train.shape[0])
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite.")
        K_inv = np.linalg.inv(K)
        lml = (
            -0.5 * np.dot(y_train, np.dot(K_inv, y_train))
            - 0.5 * logdet
            - (X_train.shape[0] / 2) * np.log(2 * np.pi)
        )
        return lml

    def get_log_params(self):
        return {
            "trend_length_scale": np.log(self.trend_length_scale),
            "trend_signal_variance": np.log(self.trend_signal_variance),
            "weekly_length_scale": np.log(self.weekly_length_scale),
            "weekly_signal_variance": np.log(self.weekly_signal_variance),
            "monthly_length_scale": np.log(self.monthly_length_scale),
            "monthly_signal_variance": np.log(self.monthly_signal_variance),
            "annual_length_scale": np.log(self.annual_length_scale),
            "annual_signal_variance": np.log(self.annual_signal_variance),
            "noise_variance": np.log(self.noise_variance),
        }

    def set_params_from_log(self, log_params):
        self.trend_length_scale = np.exp(log_params["trend_length_scale"])
        self.trend_signal_variance = np.exp(log_params["trend_signal_variance"])
        self.weekly_length_scale = np.exp(log_params["weekly_length_scale"])
        self.weekly_signal_variance = np.exp(log_params["weekly_signal_variance"])
        self.monthly_length_scale = np.exp(log_params["monthly_length_scale"])
        self.monthly_signal_variance = np.exp(log_params["monthly_signal_variance"])
        self.annual_length_scale = np.exp(log_params["annual_length_scale"])
        self.annual_signal_variance = np.exp(log_params["annual_signal_variance"])
        self.noise_variance = np.exp(log_params["noise_variance"])

    def optimize_hyperparameters_log(
        self,
        X_train,
        y_train,
        n_iters=100,
        learning_rate=1e-2,
        epsilon=1e-5,
        n_restarts=5,
    ):
        """
        Optimize all hyperparameters in log-space using gradient ascent with
        random restarts. This version updates the GP object immediately with
        the changed parameter values.
        """
        X_train = self._prepare_X(X_train)
        y_train = np.atleast_1d(y_train)
        params_to_optimize = [
            "trend_length_scale",
            "trend_signal_variance",
            "weekly_length_scale",
            "weekly_signal_variance",
            "monthly_length_scale",
            "monthly_signal_variance",
            "annual_length_scale",
            "annual_signal_variance",
            "noise_variance",
        ]
        best_lml = -np.inf
        best_log_params = None

        pb = tqdm(range(n_restarts * n_iters), desc="Optimizing hyperparameters")
        for _restart in range(n_restarts):
            # Start with current log parameters and add a small random perturbation.
            log_params = self.get_log_params()
            for key in params_to_optimize:
                log_params[key] += np.random.randn() * 0.1

            # Set the current parameters into the object
            self.set_params_from_log(log_params)
            for _it in range(n_iters):
                grads = {}
                # Update object's parameters from the current log_params.
                self.set_params_from_log(log_params)
                try:
                    current_lml = self.log_marginal_likelihood(X_train, y_train)
                except ValueError:
                    current_lml = -np.inf
                for param in params_to_optimize:
                    original_val = log_params[param]
                    # Forward difference
                    log_params[param] = original_val + epsilon
                    self.set_params_from_log(log_params)
                    try:
                        lml_plus = self.log_marginal_likelihood(X_train, y_train)
                    except ValueError:
                        lml_plus = current_lml
                    # Backward difference
                    log_params[param] = original_val - epsilon
                    self.set_params_from_log(log_params)
                    try:
                        lml_minus = self.log_marginal_likelihood(X_train, y_train)
                    except ValueError:
                        lml_minus = current_lml
                    # Reset parameter value back to original
                    log_params[param] = original_val
                    grad = (lml_plus - lml_minus) / (2 * epsilon)
                    grads[param] = grad

                # Update log_params and immediately set the new params on the GP.
                for param in params_to_optimize:
                    log_params[param] += learning_rate * grads[param]
                self.set_params_from_log(log_params)

                try:
                    current_lml = self.log_marginal_likelihood(X_train, y_train)
                except ValueError:
                    current_lml = -np.inf
                pb.set_description_str(f"lml = {current_lml:.3f}")
                pb.update()

            try:
                self.set_params_from_log(log_params)
                final_lml = self.log_marginal_likelihood(X_train, y_train)
            except ValueError:
                final_lml = -np.inf

            if final_lml > best_lml:
                best_lml = final_lml
                best_log_params = log_params.copy()

        if best_log_params is not None:
            self.set_params_from_log(best_log_params)
        return best_lml
