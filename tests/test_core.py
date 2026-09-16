"""Tests for the SADnm Stage 2 fit & smooth on synthetic data."""

import numpy as np

from sadnm.core import TemporalParams, smooth_reach, fit_temporal
from sadnm.config import save_config, load_config, DEFAULT_TEMPORAL
from sadnm.uniform_flow import InversionConfig


def test_config_round_trip(tmp_path):
    p = tmp_path / 'cfg.json'
    tp = TemporalParams(sigma_proc=0.41, tau=6.0, sigma_obs=0.31)
    save_config(tp, kernel='ou', inversion=InversionConfig(rmse_max=0.8), path=p)
    t2, kern, inv = load_config(p)
    assert t2 == tp and kern == 'ou' and inv.rmse_max == 0.8


def test_config_default_fallback(tmp_path):
    tp, kern, inv = load_config(tmp_path / 'missing.json') # no file -> defaults
    assert tp == DEFAULT_TEMPORAL and isinstance(inv, InversionConfig)


def _synthetic_reach(rng, n=40, noise=0.3):
    days = np.sort(rng.uniform(0, 300, n))
    mu = np.full(n, np.log(100.0)) # flat monthly prior level
    truth_anom = 0.5 * np.sin(days / 40.0) # smooth hydrograph anomaly
    gauge = np.exp(mu + truth_anom)
    q_phys = np.exp(mu + truth_anom + rng.normal(0, noise, n)) # noisy physics
    return dict(days=days, q_phys=q_phys, gauge=gauge, mu=mu, truth=np.exp(mu + truth_anom))


def test_smooth_reach_denoises():
    rng = np.random.default_rng(0)
    e = _synthetic_reach(rng)
    params = TemporalParams(sigma_proc=0.5, tau=40.0, sigma_obs=0.3)
    q_s, log_sig = smooth_reach(e['days'], e['q_phys'], e['mu'], params)
    rmse_raw = np.sqrt(np.mean((np.log(e['q_phys']) - np.log(e['truth'])) ** 2))
    rmse_smo = np.sqrt(np.mean((np.log(q_s) - np.log(e['truth'])) ** 2))
    assert rmse_smo < rmse_raw # smoothing reduces error
    assert np.all(log_sig > 0) and q_s.shape == e['days'].shape


def test_fit_temporal_recovers_reasonable_params():
    rng = np.random.default_rng(1)
    reaches = [_synthetic_reach(rng) for _ in range(20)]
    params = fit_temporal(reaches)
    assert 0.05 < params.sigma_proc < 3.0
    assert 1.0 < params.tau < 500.0
    assert 0.1 < params.sigma_obs < 0.6 # should recover the ~0.3 injected noise
    # fitted params should denoise held-out reaches ON AVERAGE (a single draw can regress)
    raws, smos = [], []
    for _ in range(8):
        e = _synthetic_reach(rng)
        q_s, _ = smooth_reach(e['days'], e['q_phys'], e['mu'], params)
        raws.append(np.sqrt(np.mean((np.log(e['q_phys']) - np.log(e['truth'])) ** 2)))
        smos.append(np.sqrt(np.mean((np.log(q_s) - np.log(e['truth'])) ** 2)))
    assert np.mean(smos) < np.mean(raws)
