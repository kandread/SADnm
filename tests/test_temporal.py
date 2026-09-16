"""Tests for the GP temporal smoother (learned hydrograph prior)."""
import numpy as np
import pytest
from scipy.optimize import minimize

from sadnm.temporal import (ou_kernel, matern32_kernel, matern52_kernel,
                                 KERNELS, gp_posterior, gp_marginal_nll)


@pytest.mark.parametrize('kernel', KERNELS.values())
def test_kernel_properties(kernel):
    t = np.linspace(0, 100, 10)
    K = kernel(t, t, sigma=1.0, tau=20.0)
    assert np.allclose(np.diag(K), 1.0) # variance = sigma^2 (k(0)=sigma^2)
    assert np.all(K >= 0) and np.all(K <= 1.0001) # decays with lag
    assert np.min(np.linalg.eigvalsh(K)) > -1e-8 # valid covariance (PSD)


def test_smoother_kernels_couple_farther():
    """Smoother Matern kernels retain more correlation at moderate lag than OU."""
    t = np.array([0.0, 8.0]) # 8-day lag, tau=10
    c_ou = float(ou_kernel(t, t, 1.0, 10.0)[0, 1])
    c_32 = float(matern32_kernel(t, t, 1.0, 10.0)[0, 1])
    c_52 = float(matern52_kernel(t, t, 1.0, 10.0)[0, 1])
    assert c_ou < c_32 < c_52


def test_posterior_interpolates_obs_when_low_noise():
    t = np.array([0.0, 10.0, 30.0, 60.0])
    u = np.array([0.5, -0.3, 0.2, -0.1])
    noise = np.full(4, 1e-3)
    mean, var = gp_posterior(t, u, noise, t, sigma=1.0, tau=20.0)
    assert np.allclose(mean, u, atol=1e-2)
    assert np.all(var >= 0)


def test_posterior_denoises():
    """Posterior RMSE to a smooth truth is below the noisy-obs RMSE."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 200, 40))
    truth = 0.6 * np.sin(t / 40.0)
    obs = truth + rng.normal(0, 0.3, 40)
    mean, _ = gp_posterior(t, obs, np.full(40, 0.3), t, sigma=0.6, tau=40.0)
    assert np.sqrt(np.mean((mean - truth) ** 2)) < np.sqrt(np.mean((obs - truth) ** 2))


@pytest.mark.parametrize('kernel', KERNELS.values())
def test_marginal_nll_finite_and_smooth(kernel):
    """NLL and its finite-difference gradient are finite across the prior range."""
    t = np.array([0.0, 3.0, 9.0, 20.0]); u = np.array([0.4, -0.2, 0.1, -0.3])
    noise = np.full(4, 0.2)

    def f(p):
        return gp_marginal_nll(t, u, noise, np.exp(p[0]), np.exp(p[1]), kernel=kernel)

    p = np.array([np.log(0.5), np.log(15.0)])
    h = 1e-5
    g = [(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(2)]
    assert np.all(np.isfinite(g)) and np.isfinite(f(p))


def test_learned_kernel_recovers_timescale():
    """Fitting the marginal likelihood (Nelder-Mead) recovers the generating tau."""
    rng = np.random.default_rng(2)
    tau_true = 25.0
    t = np.arange(0, 400.0)
    K = ou_kernel(t, t, 0.5, tau_true) + 1e-4 * np.eye(len(t))
    u = np.linalg.cholesky(K) @ rng.normal(size=len(t))
    noise = np.full(len(t), 0.05)

    def loss(p):
        return gp_marginal_nll(t, u, noise, np.exp(p[0]), np.exp(p[1]))

    res = minimize(loss, np.array([np.log(0.5), np.log(10.0)]), method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    tau_fit = float(np.exp(res.x[1]))
    assert 0.5 * tau_true < tau_fit < 2.0 * tau_true, f"tau {tau_fit} vs {tau_true}"
