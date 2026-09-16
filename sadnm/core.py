"""
SADnm: next generation of the SWOT Assimilated Discharge algorithm.

The deployable discharge algorithm, composed of physically-interpretable stages
that each operate strictly per reach (no cross-reach inference at deploy time
except the optional Stage 3):

  Stage 1  uniform-flow inversion (sadnm.uniform_flow.invert_reach)
           Dingman power-law cross-section & Manning, anchored to the monthly
           prior level; one independent Q per overpass from that overpass's WSE
           node profile. cal_resid is the at-inference quality/selection signal.

  Stage 2  temporal assimilation (this module & sadnm.temporal)
           A learned GP hydrograph prior over the log-Q anomaly (deviation from
           the monthly prior) denoises the per-overpass Stage-1 estimates and
           yields predictive uncertainty. The GP hyper-parameters (sigma_proc,
           tau, sigma_obs) are calibrated ONCE on the training gauges and frozen
           (see sadnm.config); deployment does NOT refit.

  Stage 3  mass conservation (optional; sadnm.spatial)
           Spatial coupling across SWORD-connected reaches. Needs cross-reach
           orchestration (an extra Confluence module), so it is an optional
           enhancement, not part of the per-reach deployable core.

Pure numpy/scipy. The numerical core is a per-reach uniform-flow solve
plus a small Cholesky GP solve.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from sadnm.temporal import gp_posterior, ou_kernel
from sadnm.uniform_flow import invert_reach, InversionConfig, ReachResult, months_of


@dataclass(frozen=True)
class TemporalParams:
    """Calibrated-once GP hydrograph-prior hyper-parameters (Stage 2)."""
    sigma_proc: float        # process std of the log-Q anomaly
    tau: float               # temporal correlation length (days)
    sigma_obs: float         # Stage-1 (physics) observation noise in log space


@dataclass(frozen=True)
class SADnmResult:
    ok: bool
    reason: str = ""
    q: np.ndarray | None = None              # Stage-2 smoothed discharge at overpasses
    q_phys: np.ndarray | None = None         # Stage-1 raw per-overpass discharge
    log_sigma: np.ndarray | None = None      # predictive log-Q std (1-sigma)
    overpass_idx: np.ndarray | None = None
    cal_resid: float = np.nan                # Stage-1 quality / selection signal
    r_shape: float = np.nan                  # Dingman shape r
    d0: float = np.nan                       # Stage-1 baseflow depth (m)
    C: float = np.nan                        # Stage-1 level coefficient (log)
    used_spline: bool = False


def smooth_reach(days, q_phys, mu, params: TemporalParams, kernel=ou_kernel):
    """Stage 2 on one reach. `mu` is the per-overpass monthly-prior log level.
    Returns (q_smoothed, log_sigma) at the same overpass times.
    The GP models u = log(q) - mu (the anomaly); the level rides the prior."""
    d = np.asarray(days, float)
    q_phys = np.asarray(q_phys, float); mu = np.asarray(mu, float)
    u_obs = np.log(np.maximum(q_phys, 1e-6)) - mu
    noise = np.full(u_obs.shape[0], params.sigma_obs)
    mean, var = gp_posterior(d, u_obs, noise, d, params.sigma_proc, params.tau, kernel=kernel)
    log_sigma = np.sqrt(var + params.sigma_obs ** 2)
    return np.exp(mean + mu), log_sigma


def fit_temporal(reaches, kernel=ou_kernel, init=(0.5, 25.0, 0.3)) -> TemporalParams:
    """Offline calibration of Stage-2 hyper-parameters on a set of gauged reaches.
    `reaches`: iterable of dicts with keys days, q_phys, gauge, mu (1-D arrays).
    Objective: MSE of the GP-smoothed anomaly vs the gauge anomaly.

    Run once to produce the frozen config (sadnm.config); deployment
    loads those constants and never calls this. Gradient-free (Nelder-Mead) on the
    3 log-parameters, which is robust and dependency-light for a one-time fit."""
    rj = [(np.asarray(e['days'], float),
           np.log(np.maximum(np.asarray(e['q_phys'], float), 1e-6)) - np.asarray(e['mu'], float),
           np.log(np.maximum(np.asarray(e['gauge'], float), 1e-6)) - np.asarray(e['mu'], float))
          for e in reaches]

    def loss(p):
        sigma, tau, sobs = np.exp(p)
        errs = []
        for d, uo, ut in rj:
            noise = np.full(uo.shape[0], sobs)
            mean, _ = gp_posterior(d, uo, noise, d, sigma, tau, kernel=kernel)
            errs.append(np.mean((mean - ut) ** 2))
        return float(np.mean(errs))

    p0 = np.log(np.asarray(init, float))
    res = minimize(loss, p0, method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-5, maxiter=2000))
    sigma, tau, sobs = np.exp(res.x)
    return TemporalParams(float(sigma), float(tau), float(sobs))


def run_reach(wse_norm, width_norm, node_mask, overpass_mask, node_id, overpass_time_s,
              monthly_q, norm_stats, params: TemporalParams,
              eval_overpass_idx=None, cfg: InversionConfig = InversionConfig(),
              kernel=ou_kernel) -> SADnmResult:
    """Full per-reach SADnm: Stage 1 (uniform-flow inversion) then Stage 2
    (temporal assimilation). The monthly-prior level `mu` is derived internally
    from `monthly_q` at the returned overpasses' months (the caller cannot know
    which overpasses Stage 1 returns). Returns the smoothed Q + uncertainty."""
    overpass_time_s = np.asarray(overpass_time_s)
    monthly_q = np.asarray(monthly_q)
    res: ReachResult = invert_reach(wse_norm, width_norm, node_mask, overpass_mask,
                                    node_id, overpass_time_s, monthly_q, norm_stats,
                                    eval_overpass_idx=eval_overpass_idx, cfg=cfg)
    if not res.ok:
        return SADnmResult(ok=False, reason=res.reason)
    ts = overpass_time_s[res.overpass_idx]
    mu = np.log(np.maximum(monthly_q[months_of(ts)], 1e-6))     # prior level per returned overpass
    q_s, log_sig = smooth_reach(ts / 86400.0, res.q, mu, params, kernel=kernel)
    return SADnmResult(ok=True, q=q_s, q_phys=res.q, log_sigma=log_sig,
                       overpass_idx=res.overpass_idx, cal_resid=res.cal_resid,
                       r_shape=res.r_shape, d0=res.d0, C=res.C, used_spline=res.used_spline)
