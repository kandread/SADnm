"""
SADnm Stage 2, temporal assimilation: the learned hydrograph prior as a Gaussian
process over log-discharge anomalies, fused with the noisy per-overpass physics
estimates that the uniform-flow inversion produces independently per overpass.

Idea: the inversion solves each overpass's Q independently, discarding temporal
structure. Discharge is temporally correlated (recessions, event persistence). We
model the log-Q deviation from the monthly-prior level as a GP with a learned
covariance kernel (the hydrograph prior). At inference the GP posterior given the
physics estimates (with their noise) denoises the hydrograph and yields calibrated
uncertainty strictly per reach (one reach's time series).

The state is `u(t) = log Q(t) - mu(t)` where mu is the (monthly) prior level, so
the GP models the anomaly and the level still rides the prior.

The only operation is a small (n_overpass x n_overpass) Cholesky
solve per reach. No autodiff / GPU needed — the kernel hyper-parameters are a
3-vector calibrated once offline (see sadnm.core.fit_temporal).
"""

import numpy as np
import scipy.linalg as sla

_JITTER = 1e-6


def ou_kernel(t1, t2, sigma, tau):
    """Ornstein-Uhlenbeck (Matern-1/2) covariance: sigma^2 * exp(-|dt|/tau).
    Rough sample paths (Markov): an overpass couples mainly to its immediate
    neighbour. Captures persistence with timescale tau (days). t in days."""
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    dt = np.abs(t1[:, None] - t2[None, :])
    return sigma ** 2 * np.exp(-dt / tau)


def matern32_kernel(t1, t2, sigma, tau):
    """Matern-3/2: sigma^2 (1 + s) exp(-s), s = sqrt(3)|dt|/tau. Once-differentiable
    sample paths, borrows strength across several neighbours, matching the smooth
    recession structure of hydrographs better than OU."""
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    s = np.sqrt(3.0) * np.abs(t1[:, None] - t2[None, :]) / tau
    return sigma ** 2 * (1.0 + s) * np.exp(-s)


def matern52_kernel(t1, t2, sigma, tau):
    """Matern-5/2: sigma^2 (1 + s + s^2/3) exp(-s), s = sqrt(5)|dt|/tau. Twice-
    differentiable, smoothest of the three; strongest coherent multi-neighbour
    coupling on densely-sampled reaches."""
    t1 = np.asarray(t1, float); t2 = np.asarray(t2, float)
    s = np.sqrt(5.0) * np.abs(t1[:, None] - t2[None, :]) / tau
    return sigma ** 2 * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


# all kernels have k(0) = sigma^2, so the posterior prior-variance term is unchanged
KERNELS = {'ou': ou_kernel, 'matern32': matern32_kernel, 'matern52': matern52_kernel}


def _chol(t_obs, noise, sigma, tau, kernel=ou_kernel):
    noise = np.asarray(noise, float)
    K = kernel(t_obs, t_obs, sigma, tau) + np.diag(noise ** 2 + _JITTER)
    return sla.cho_factor(K, lower=True)


def gp_posterior(t_obs, u_obs, noise, t_query, sigma, tau, kernel=ou_kernel):
    """
    GP posterior over the anomaly u at t_query given noisy obs (t_obs, u_obs, noise).
    Returns (mean, var) at t_query. Zero-mean prior (u is the prior-anomaly).
    `kernel` selects the temporal covariance (ou / matern32 / matern52).
    """
    u_obs = np.asarray(u_obs, float)
    cf = _chol(t_obs, noise, sigma, tau, kernel)
    alpha = sla.cho_solve(cf, u_obs)
    Ksx = kernel(t_query, t_obs, sigma, tau)
    mean = Ksx @ alpha
    v = sla.cho_solve(cf, Ksx.T) # (n_obs, n_query)
    var = sigma ** 2 - np.sum(Ksx * v.T, axis=1) # diag(Kss) - Ksx Kxx^-1 Ksx^T
    return mean, np.maximum(var, 0.0)


def gp_marginal_nll(t_obs, u_obs, noise, sigma, tau, kernel=ou_kernel):
    """Negative log marginal likelihood of the obs anomalies under the GP prior.
    Used to learn the kernel (sigma, tau) on gauge hydrographs."""
    u_obs = np.asarray(u_obs, float)
    cf = _chol(t_obs, noise, sigma, tau, kernel)
    L = cf[0]
    alpha = sla.cho_solve(cf, u_obs)
    n = u_obs.shape[0]
    return float(0.5 * u_obs @ alpha
                 + np.sum(np.log(np.abs(np.diag(L))))
                 + 0.5 * n * np.log(2 * np.pi))
