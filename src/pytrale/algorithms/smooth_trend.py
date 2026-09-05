"""Smooth-trend state-space model solved by Kalman filtering and RTS smoothing.

This module implements the algorithm described in
`docs/CONTEXT_trale_interpolation.md`. See `SmoothTrend` for the model and
the references it is based on.
"""

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from pytrale.algorithms.base import Interpolator

# A body weight series whose observation noise is estimated as exactly zero
# would make the Kalman gain degenerate; this floor keeps `S` positive.
_MIN_NOISE_VARIANCE = 1e-12

# Equivalent kernel bandwidth [days] used when there are too few measurements
# to estimate one, chosen to match the app's `medium` smoothing level.
_DEFAULT_BANDWIDTH = 6.0


class SmoothTrendPosterior(NamedTuple):
    """Posterior of the smooth-trend model at a set of query times.

    Attributes:
        mean (NDArray): Posterior mean weight [kg].
        std (NDArray): Posterior standard deviation of the weight [kg].
        trend (NDArray): Posterior mean rate of change [kg/day].
        trend_std (NDArray): Posterior standard deviation of the rate
            of change [kg/day].
    """

    mean: NDArray
    std: NDArray
    trend: NDArray
    trend_std: NDArray


class _FilterOutput(NamedTuple):
    """Everything the backward pass and the likelihood need from the filter."""

    mu: NDArray
    nu: NDArray
    p11: NDArray
    p12: NDArray
    p22: NDArray
    pred11: NDArray
    pred12: NDArray
    pred22: NDArray
    dt: NDArray
    start: int
    sum_sq: float
    sum_log_s: float
    n_used: int


def _filter_forward(
    times: NDArray,
    values: NDArray,
    is_observed: NDArray,
    noise_variance: float,
    diffusion: float,
    trend_prior_variance: float,
    robust_c: float | None,
) -> _FilterOutput:
    """Run the Kalman filter of the smooth-trend model over sorted times.

    The recursion starts at the first observation, so points before it are
    left untouched here and are handled by `_smooth_backward`. Points
    without an observation only get a prediction step, which is how missing
    days (and query times) are handled natively — nothing is filled in.

    Args:
        times (NDArray): Sorted 1D array of times [days].
        values (NDArray): Observed weights [kg]; entries where
            `is_observed` is False are ignored.
        is_observed (NDArray): Boolean mask marking the entries of `values`
            that carry an actual measurement.
        noise_variance (float): Observation noise variance `R` [kg²].
        diffusion (float): Rate diffusion `q` [kg²/day³] scaling `Q(dt)`.
        trend_prior_variance (float): Prior variance of the initial rate,
            in units of `noise_variance` [day⁻²].
        robust_c (float | None): Reweighting threshold in standard
            deviations, or None for a plain Gaussian filter.

    Returns:
        _FilterOutput: Filtered and predicted moments at every index, plus
            the two accumulators of the prediction-error decomposition of
            the log-likelihood.
    """
    n = len(times)
    mu = np.zeros(n)
    nu = np.zeros(n)
    p11 = np.zeros(n)
    p12 = np.zeros(n)
    p22 = np.zeros(n)
    pred11 = np.zeros(n)
    pred12 = np.zeros(n)
    pred22 = np.zeros(n)
    dt = np.zeros(n)

    start = int(np.argmax(is_observed))

    # Exact posterior of the level under a flat prior given the first
    # observation, so that observation is not counted twice.
    cur_mu = float(values[start])
    cur_nu = 0.0
    cur_p11 = noise_variance
    cur_p12 = 0.0
    cur_p22 = trend_prior_variance * noise_variance

    mu[start] = cur_mu
    nu[start] = cur_nu
    p11[start] = cur_p11
    p12[start] = cur_p12
    p22[start] = cur_p22
    pred11[start] = cur_p11
    pred12[start] = cur_p12
    pred22[start] = cur_p22

    sum_sq = 0.0
    sum_log_s = 0.0
    n_used = 0

    for i in range(start + 1, n):
        step = float(times[i] - times[i - 1])
        dt[i] = step

        mu_pred = cur_mu + step * cur_nu
        nu_pred = cur_nu
        p11_pred = (
            cur_p11
            + 2.0 * step * cur_p12
            + step * step * cur_p22
            + diffusion * step**3 / 3.0
        )
        p12_pred = cur_p12 + step * cur_p22 + diffusion * step * step / 2.0
        p22_pred = cur_p22 + diffusion * step

        pred11[i] = p11_pred
        pred12[i] = p12_pred
        pred22[i] = p22_pred

        if is_observed[i]:
            delta = float(values[i]) - mu_pred
            s = p11_pred + noise_variance

            # The likelihood must stay that of the plain Gaussian model, so
            # it is accumulated before any reweighting widens `R`.
            sum_sq += delta * delta / s
            sum_log_s += math.log(s)
            n_used += 1

            if robust_c is not None:
                z = abs(delta) / math.sqrt(s)
                if z > robust_c:
                    s = p11_pred + noise_variance * (z / robust_c) ** 2

            k1 = p11_pred / s
            k2 = p12_pred / s

            cur_mu = mu_pred + k1 * delta
            cur_nu = nu_pred + k2 * delta
            cur_p11 = p11_pred * (1.0 - k1)
            cur_p12 = p12_pred * (1.0 - k1)
            cur_p22 = p22_pred - k2 * p12_pred
        else:
            cur_mu = mu_pred
            cur_nu = nu_pred
            cur_p11 = p11_pred
            cur_p12 = p12_pred
            cur_p22 = p22_pred

        mu[i] = cur_mu
        nu[i] = cur_nu
        p11[i] = cur_p11
        p12[i] = cur_p12
        p22[i] = cur_p22

    return _FilterOutput(
        mu=mu,
        nu=nu,
        p11=p11,
        p12=p12,
        p22=p22,
        pred11=pred11,
        pred12=pred12,
        pred22=pred22,
        dt=dt,
        start=start,
        sum_sq=sum_sq,
        sum_log_s=sum_log_s,
        n_used=n_used,
    )


def _smooth_backward(
    filtered: _FilterOutput, diffusion: float, times: NDArray
) -> SmoothTrendPosterior:
    """Run the Rauch-Tung-Striebel smoother over a filtered sequence.

    Args:
        filtered (_FilterOutput): Output of `_filter_forward`.
        diffusion (float): Rate diffusion `q` [kg²/day³], as used in the
            forward pass.
        times (NDArray): The same sorted times passed to `_filter_forward`.

    Returns:
        SmoothTrendPosterior: Smoothed moments at every index.
    """
    n = len(times)
    mu_s = filtered.mu.copy()
    nu_s = filtered.nu.copy()
    var_mu = filtered.p11.copy()
    cov = filtered.p12.copy()
    var_nu = filtered.p22.copy()

    for i in range(n - 2, filtered.start - 1, -1):
        step = float(filtered.dt[i + 1])
        a = filtered.pred11[i + 1]
        b = filtered.pred12[i + 1]
        d = filtered.pred22[i + 1]
        # det Q(dt) = q² dt⁴/12 > 0 for dt > 0, and P is positive definite
        # for dt = 0, so the predicted covariance is never singular.
        det = a * d - b * b

        m11 = filtered.p11[i] + step * filtered.p12[i]
        m12 = filtered.p12[i]
        m21 = filtered.p12[i] + step * filtered.p22[i]
        m22 = filtered.p22[i]

        g11 = (m11 * d - m12 * b) / det
        g12 = (m12 * a - m11 * b) / det
        g21 = (m21 * d - m22 * b) / det
        g22 = (m22 * a - m21 * b) / det

        res_mu = mu_s[i + 1] - (filtered.mu[i] + step * filtered.nu[i])
        res_nu = nu_s[i + 1] - filtered.nu[i]
        mu_s[i] = filtered.mu[i] + g11 * res_mu + g12 * res_nu
        nu_s[i] = filtered.nu[i] + g21 * res_mu + g22 * res_nu

        dp11 = var_mu[i + 1] - a
        dp12 = cov[i + 1] - b
        dp22 = var_nu[i + 1] - d
        c11 = g11 * dp11 + g12 * dp12
        c12 = g11 * dp12 + g12 * dp22
        c21 = g21 * dp11 + g22 * dp12
        c22 = g21 * dp12 + g22 * dp22

        var_mu[i] = filtered.p11[i] + c11 * g11 + c12 * g12
        cov[i] = filtered.p12[i] + c11 * g21 + c12 * g22
        var_nu[i] = filtered.p22[i] + c21 * g21 + c22 * g22

    _extend_before_first(
        mu_s, nu_s, var_mu, cov, var_nu, filtered.start, diffusion, times
    )

    return SmoothTrendPosterior(
        mean=mu_s,
        std=np.sqrt(np.maximum(var_mu, 0.0)),
        trend=nu_s,
        trend_std=np.sqrt(np.maximum(var_nu, 0.0)),
    )


def _extend_before_first(
    mu_s: NDArray,
    nu_s: NDArray,
    var_mu: NDArray,
    cov: NDArray,
    var_nu: NDArray,
    start: int,
    diffusion: float,
    times: NDArray,
) -> None:
    """Fill in the smoothed moments at times before the first observation.

    No data precedes `start`, so the state there only connects to the data
    through the state at `start`. Under a flat prior the result is the
    exact reverse transition, `x = F(dt)⁻¹ x_start` with covariance
    `F(dt)⁻¹ (P + Q(dt)) F(dt)⁻ᵀ`, which is why the band keeps widening
    backwards instead of collapsing onto an extrapolated line.

    Args:
        mu_s (NDArray): Smoothed levels, modified in place.
        nu_s (NDArray): Smoothed rates, modified in place.
        var_mu (NDArray): Smoothed level variances, modified in place.
        cov (NDArray): Smoothed level/rate covariances, modified in place.
        var_nu (NDArray): Smoothed rate variances, modified in place.
        start (int): Index of the first observation.
        diffusion (float): Rate diffusion `q` [kg²/day³].
        times (NDArray): Sorted times [days].

    Returns:
        None
    """
    for i in range(start - 1, -1, -1):
        step = float(times[start] - times[i])
        a = var_mu[start] + diffusion * step**3 / 3.0
        b = cov[start] + diffusion * step * step / 2.0
        d = var_nu[start] + diffusion * step

        mu_s[i] = mu_s[start] - step * nu_s[start]
        nu_s[i] = nu_s[start]
        var_mu[i] = a - 2.0 * step * b + step * step * d
        cov[i] = b - step * d
        var_nu[i] = d


def _concentrated_objective(
    log10_lam: float,
    times: NDArray,
    weights: NDArray,
    trend_prior_variance: float,
) -> float:
    """Evaluate the concentrated negative log-likelihood at one `lam`.

    The observation scale is concentrated out: writing `R = sigma²` and
    `Q = sigma² lam Q0(dt)`, every covariance scales with `sigma²`, so the
    Kalman gains — and hence the state estimates — do not depend on it.
    Running the filter at `sigma² = 1` therefore yields the profile
    likelihood `m log(sum(delta²/S)) + sum(log S)` up to an additive
    constant, leaving `lam` as the only free parameter.

    Args:
        log10_lam (float): Base-10 logarithm of `lam` [day⁻³].
        times (NDArray): Sorted measurement times [days].
        weights (NDArray): Measured weights [kg].
        trend_prior_variance (float): Prior variance of the initial rate,
            in units of the observation noise variance [day⁻²].

    Returns:
        float: The concentrated negative log-likelihood, up to a constant.
    """
    lam = 10.0**log10_lam
    filtered = _filter_forward(
        times,
        weights,
        np.ones(len(times), dtype=bool),
        noise_variance=1.0,
        diffusion=lam,
        trend_prior_variance=trend_prior_variance,
        robust_c=None,
    )
    # Near-noiseless data drives the residual sum to zero; the same floor
    # the noise estimate uses keeps the logarithm finite.
    residual = max(filtered.sum_sq, filtered.n_used * _MIN_NOISE_VARIANCE)
    return filtered.n_used * math.log(residual) + filtered.sum_log_s


class SmoothTrend(Interpolator):
    """Gaussian-process smoother for sparse, noisy weight measurements.

    The latent weight follows a once-integrated Wiener process — Harvey's
    local linear trend with the level disturbance set to zero, also known
    as the smooth trend model (Harvey 1989, §5.4):

        d(level) = rate dt,   d(rate) = sqrt(q) dW,   y = level + noise

    Dropping the level disturbance buys three things: one hyperparameter
    fewer (it is only weakly identified against the rate disturbance), a
    continuously differentiable curve, and the exact equivalence between
    the smoothed posterior and the cubic smoothing spline (Wahba 1978;
    Wecker & Ansley 1983), which is what makes the smoothness parameter
    interpretable as a kernel bandwidth.

    Inference is a Kalman filter followed by an RTS smoother, which is the
    exact Gaussian-process posterior in O(N) rather than O(N³) (Hartikainen
    & Sarkka 2010). Missing days are not filled in: a day without a
    measurement simply gets no update step. Query times are handled the
    same way, so `predict` returns the exact posterior at arbitrary times
    rather than a linear interpolation of a precomputed grid.

    Hyperparameters are estimated by maximum likelihood from the
    prediction-error decomposition. The observation scale is available in
    closed form once concentrated out, so the fit reduces to a
    one-dimensional search for `lam` over a fixed logarithmic grid — no
    gradient-based optimizer is involved, which keeps the whole algorithm
    portable to environments without one.

    Robustness comes from reweighting the innovations: an observation
    further than `robust_c` standard deviations from its own prediction
    has its noise variance inflated by `(z / robust_c)**2`. That is one
    IRLS step of a Student-t observation likelihood with
    `robust_c**2 - 1` degrees of freedom (Lange, Little & Taylor 1989;
    Nickisch, Solin & Grigorievskiy 2018), so the influence of a gross
    error redescends towards zero rather than merely being bounded as it
    would be under Huber weighting. Reweighting is applied only when
    producing the final curve; `lam` and the noise scale are estimated
    with the plain Gaussian filter, so the likelihood being maximized
    remains the model's own.

    Attributes:
        lam_ (float | None): Fitted smoothness `lam = q / sigma_eps²`
            [day⁻³], or None if there were too few measurements to run the
            model.
        sigma_eps_ (float | None): Fitted observation noise standard
            deviation [kg], or None as above.
        bandwidth_ (float | None): Equivalent kernel bandwidth
            `lam_ ** -0.25` [days] (Silverman 1984), or None as above.
    """

    def __init__(
        self,
        lam: float | None = None,
        sigma_eps: float | None = None,
        robust_c: float | None = 2.5,
        lam_grid_size: int = 40,
        lam_log10_range: tuple[float, float] = (-8.0, -2.0),
        trend_prior_variance: float = 0.035,
        min_measurements_for_ml: int = 10,
    ) -> None:
        """Initialize the smoother.

        Args:
            lam (float | None): Smoothness `q / sigma_eps²` [day⁻³]. None
                estimates it by maximum likelihood. It relates to the
                equivalent kernel bandwidth `h` [days] as `lam = h ** -4`.
            sigma_eps (float | None): Observation noise standard deviation
                [kg]. None estimates it in closed form.
            robust_c (float | None): Innovation reweighting threshold in
                standard deviations, equivalent to a Student-t observation
                likelihood with `robust_c ** 2 - 1` degrees of freedom
                (about 5.3 at the default). None gives a plain Gaussian
                filter.
            lam_grid_size (int): Number of points in the logarithmic `lam`
                grid searched during fitting.
            lam_log10_range (tuple[float, float]): Base-10 logarithm of the
                smallest and largest `lam` considered. The default spans
                equivalent bandwidths from about 100 down to 3 days.
            trend_prior_variance (float): Prior variance of the initial
                rate, in units of the observation noise variance [day⁻²].
                Expressed relative to the noise so that concentrating the
                scale out of the likelihood stays exact; the default is
                about (0.15 kg/day)² at a typical `sigma_eps` of 0.8 kg.
            min_measurements_for_ml (int): Below this many measurements the
                default bandwidth is used instead of estimating `lam`.

        Returns:
            None
        """
        self.lam = lam
        self.sigma_eps = sigma_eps
        self.robust_c = robust_c
        self.lam_grid_size = lam_grid_size
        self.lam_log10_range = lam_log10_range
        self.trend_prior_variance = trend_prior_variance
        self.min_measurements_for_ml = min_measurements_for_ml

        self.lam_: float | None = None
        self.sigma_eps_: float | None = None
        self.bandwidth_: float | None = None

        self._times: NDArray | None = None
        self._weights: NDArray | None = None
        self._origin: float = 0.0

    def fit(self, times_measured: NDArray, weights_measured: NDArray) -> "SmoothTrend":
        """Fit the model to sparse measurements.

        Repeated measurements at the same time need no special treatment:
        they enter as consecutive update steps with `dt = 0`, which is the
        same posterior as averaging them with `R / k` while also giving the
        correct joint likelihood and letting the reweighting discount a
        single bad reading rather than a contaminated average.

        Args:
            times_measured (NDArray): 1D array of measurement times [days].
            weights_measured (NDArray): 1D array of measured weights [kg],
                same shape as `times_measured`.

        Returns:
            SmoothTrend: self, so calls can be chained with `predict`.

        Raises:
            ValueError: If the two arrays differ in shape, or if there is
                not at least one measurement.
        """
        times = np.asarray(times_measured, dtype=float).ravel()
        weights = np.asarray(weights_measured, dtype=float).ravel()
        if times.shape != weights.shape:
            raise ValueError(
                "times_measured and weights_measured must have the same shape."
            )
        if not times.size:
            raise ValueError("SmoothTrend requires at least one measurement.")

        order = np.argsort(times, kind="stable")
        # Weight exports carry times as days since the epoch (~2e4); working
        # relative to the first measurement keeps the differences exact.
        self._origin = float(times[order[0]])
        self._times = times[order] - self._origin
        self._weights = weights[order]

        self.lam_ = None
        self.sigma_eps_ = None
        self.bandwidth_ = None
        if times.size < 3:
            return self

        if self.lam is not None:
            self.lam_ = float(self.lam)
        elif times.size < self.min_measurements_for_ml:
            self.lam_ = _DEFAULT_BANDWIDTH**-4
        else:
            self.lam_ = self._estimate_lam()

        if self.sigma_eps is not None:
            self.sigma_eps_ = float(self.sigma_eps)
        else:
            self.sigma_eps_ = self._estimate_sigma_eps(self.lam_)
        self.bandwidth_ = self.lam_**-0.25

        return self

    def smooth(self, times: NDArray) -> SmoothTrendPosterior:
        """Compute the full posterior at the given query times.

        `predict` and its companions are thin wrappers around this method;
        call it directly to get all four quantities from a single pass.

        Args:
            times (NDArray): 1D array of query times [days].

        Returns:
            SmoothTrendPosterior: Mean, standard deviation, trend and trend
                standard deviation, each of the same shape as `times`. The
                standard deviations are NaN when fewer than three
                measurements were available, since the model is not fit in
                that case.

        Raises:
            RuntimeError: If called before `fit`.
        """
        if self._times is None or self._weights is None:
            raise RuntimeError("SmoothTrend must be fit before predict is called.")

        query = np.asarray(times, dtype=float).ravel() - self._origin
        if self.lam_ is None or self.sigma_eps_ is None:
            return self._degenerate_posterior(query)

        n_measured = len(self._times)
        all_times = np.concatenate([self._times, query])
        all_values = np.concatenate([self._weights, np.zeros(len(query))])
        is_observed = np.concatenate(
            [np.ones(n_measured, dtype=bool), np.zeros(len(query), dtype=bool)]
        )
        order = np.argsort(all_times, kind="stable")

        noise_variance = self.sigma_eps_**2
        posterior = _smooth_backward(
            _filter_forward(
                all_times[order],
                all_values[order],
                is_observed[order],
                noise_variance=noise_variance,
                diffusion=noise_variance * self.lam_,
                trend_prior_variance=self.trend_prior_variance,
                robust_c=self.robust_c,
            ),
            diffusion=noise_variance * self.lam_,
            times=all_times[order],
        )

        positions = np.empty(len(all_times), dtype=int)
        positions[order] = np.arange(len(all_times))
        idx = positions[n_measured:]
        return SmoothTrendPosterior(
            mean=posterior.mean[idx],
            std=posterior.std[idx],
            trend=posterior.trend[idx],
            trend_std=posterior.trend_std[idx],
        )

    def predict(self, times: NDArray) -> NDArray:
        """Estimate the weight at the given query times.

        Args:
            times (NDArray): 1D array of query times [days].

        Returns:
            NDArray: 1D array of estimated weights [kg], same shape as
                `times`.
        """
        return self.smooth(times).mean

    def predict_std(self, times: NDArray) -> NDArray:
        """Estimate the posterior uncertainty of the weight.

        Args:
            times (NDArray): 1D array of query times [days].

        Returns:
            NDArray: Posterior standard deviation of the weight [kg], same
                shape as `times`.
        """
        return self.smooth(times).std

    def predict_trend(self, times: NDArray) -> NDArray:
        """Estimate the rate of weight change at the given query times.

        Args:
            times (NDArray): 1D array of query times [days].

        Returns:
            NDArray: Estimated rate of change [kg/day], same shape as
                `times`.
        """
        return self.smooth(times).trend

    def predict_trend_std(self, times: NDArray) -> NDArray:
        """Estimate the posterior uncertainty of the rate of change.

        Args:
            times (NDArray): 1D array of query times [days].

        Returns:
            NDArray: Posterior standard deviation of the rate of change
                [kg/day], same shape as `times`.
        """
        return self.smooth(times).trend_std

    def _estimate_lam(self) -> float:
        """Estimate `lam` by maximum likelihood over a logarithmic grid.

        A fixed grid plus one parabolic refinement is used rather than an
        iterative optimizer: the objective is smooth and unimodal in
        practice, the result is deterministic, and there are no tolerances
        to tune.

        Returns:
            float: The maximizing `lam` [day⁻³].
        """
        low, high = self.lam_log10_range
        grid = np.linspace(low, high, self.lam_grid_size)
        objective = np.array(
            [
                _concentrated_objective(
                    x, self._times, self._weights, self.trend_prior_variance
                )
                for x in grid
            ]
        )

        best = int(np.argmin(objective))
        log10_lam = float(grid[best])
        if 0 < best < len(grid) - 1:
            left = objective[best - 1]
            right = objective[best + 1]
            curvature = left - 2.0 * objective[best] + right
            if curvature > 0.0:
                step = grid[1] - grid[0]
                shift = 0.5 * step * (left - right) / curvature
                log10_lam += float(np.clip(shift, -step, step))
        return 10.0**log10_lam

    def _estimate_sigma_eps(self, lam: float) -> float:
        """Estimate the observation noise scale in closed form.

        Args:
            lam (float): Smoothness `q / sigma_eps²` [day⁻³].

        Returns:
            float: Estimated observation noise standard deviation [kg].
        """
        filtered = _filter_forward(
            self._times,
            self._weights,
            np.ones(len(self._times), dtype=bool),
            noise_variance=1.0,
            diffusion=lam,
            trend_prior_variance=self.trend_prior_variance,
            robust_c=None,
        )
        variance = filtered.sum_sq / filtered.n_used
        return math.sqrt(max(variance, _MIN_NOISE_VARIANCE))

    def _degenerate_posterior(self, query: NDArray) -> SmoothTrendPosterior:
        """Return the fallback posterior for fewer than three measurements.

        Two measurements cannot separate a trend from noise and one cannot
        even define a trend, so the model is not run at all: the curve is
        the line through the points (or the constant), and the uncertainty
        is reported as NaN rather than as a number the data cannot support.

        Args:
            query (NDArray): Query times [days], relative to the fit origin.

        Returns:
            SmoothTrendPosterior: Mean and trend, with NaN uncertainties.
        """
        undefined = np.full(len(query), np.nan)
        if len(self._times) == 1:
            return SmoothTrendPosterior(
                mean=np.full(len(query), self._weights[0]),
                std=undefined,
                trend=np.zeros(len(query)),
                trend_std=undefined,
            )

        span = self._times[1] - self._times[0]
        # Two readings on the same day carry no rate information at all.
        slope = 0.0 if span == 0 else (self._weights[1] - self._weights[0]) / span
        return SmoothTrendPosterior(
            mean=self._weights[0] + slope * (query - self._times[0]),
            std=undefined,
            trend=np.full(len(query), slope),
            trend_std=undefined,
        )
