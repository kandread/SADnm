"""Tests for the per-reach uniform flow inversion orchestration module."""
import datetime
from pathlib import Path

import numpy as np
import pytest
import zarr

from sadnm.uniform_flow import (
    InversionConfig, invert_reach, robust_stage, estimate_r, months_of, A_FLOOR,
)

_EPOCH = datetime.datetime(2000, 1, 1, 0, 0, 0)
STORE = Path('data/store.zarr')


# Unit pieces

def test_robust_stage_recovers_level_through_outliers():
    cfg = InversionConfig()
    x = np.arange(20, dtype=float)
    y = 100.0 - 0.01 * x # gentle downstream tilt, level ~100 at mid
    y[3] += 5.0; y[15] -= 6.0 # two outlier nodes
    st, rmse, n = robust_stage(x, y, x_ref=9.5, cfg=cfg)
    assert abs(st - (100.0 - 0.01 * 9.5)) < 0.05
    assert n >= 18 # outliers rejected


def test_estimate_r_recovers_shape():
    cfg = InversionConfig()
    r_true = 0.45
    dH = np.linspace(0.5, 8.0, 100)
    logW = (1.0 / r_true) * np.log(dH + 1.0) # W ~ depth^(1/r), d0=1
    a, r, inv_r = estimate_r(dH, logW, d0=1.0, cfg=cfg)
    assert abs(r - r_true) < 0.05
    assert abs(a - (A_FLOOR + 1.0 / r_true)) < 0.1


# Synthetic end-to-end recovery

def _synthetic_reach(r_true=0.5, n_nodes=20, seed=0):
    """Build a synthetic reach whose stage/width follow a Dingman channel driven
    by a known discharge series; return the inversion inputs + the true Q."""
    rng = np.random.default_rng(seed)
    D_b, W_b, n_man, Z0 = 10.0, 300.0, 0.03, 100.0
    a = A_FLOOR + 1.0 / r_true
    S = 1e-4
    K = W_b * (r_true / (r_true + 1)) ** (5 / 3) / (n_man * D_b ** (1 / r_true))
    logC = np.log(K) + 0.5 * np.log(S) # log Q = a*log(depth) + logC

    # 36 monthly overpasses (3 yrs) with a seasonal + event discharge
    nt = 36
    base = 1e9
    times = base + np.arange(nt) * 30 * 86400.0
    months = months_of(times)
    season = 1.0 + 0.8 * np.sin(2 * np.pi * months / 12.0)
    Q_true = 200.0 * season * np.exp(0.3 * rng.standard_normal(nt)) # m^3/s
    depth = np.exp((np.log(Q_true) - logC) / a) # invert flow law
    H_reach = Z0 + depth # reach stage

    # Node profiles: tilted plane per overpass + WSE noise; widths via Dingman + noise
    x = np.arange(n_nodes)
    tilt = 1e-4 * 200.0 # ~m per node
    wse = np.zeros((nt, n_nodes)); width = np.zeros((nt, n_nodes))
    for t in range(nt):
        prof = H_reach[t] - tilt * (x - (n_nodes - 1) / 2)
        wse[t] = prof + 0.08 * rng.standard_normal(n_nodes) # ~8 cm WSE noise
        d_node = np.maximum(prof - Z0, 0.1)
        W = W_b * (d_node / D_b) ** (1 / r_true)
        width[t] = W * (1 + 0.20 * rng.standard_normal(n_nodes)) # 20% width noise

    node_mask = np.ones((nt, n_nodes), bool)
    overpass_mask = np.ones(nt, bool)
    node_id = np.arange(1, n_nodes + 1)
    # normalise (the module de-normalises with these stats)
    wse_mean, wse_std = wse.mean(), wse.std()
    w_mean, w_std = width.mean(), width.std()
    wse_n = (wse - wse_mean) / wse_std
    width_n = (width - w_mean) / w_std
    norm_stats = dict(wse_mean=wse_mean, wse_std=wse_std, width_mean=w_mean, width_std=w_std)

    monthly_q = np.array([Q_true[months == m].mean() if np.any(months == m) else 0.0
                          for m in range(12)])
    return dict(wse_norm=wse_n, width_norm=width_n, node_mask=node_mask,
                overpass_mask=overpass_mask, node_id=node_id, overpass_time_s=times,
                monthly_q=monthly_q, norm_stats=norm_stats), Q_true, r_true


def test_synthetic_recovers_r_and_dynamics():
    inputs, Q_true, r_true = _synthetic_reach(r_true=0.5, seed=1)
    res = invert_reach(**inputs)
    assert res.ok, res.reason
    # shape r recovered to within the width-noise tolerance
    assert abs(res.r_shape - r_true) < 0.2, f"r {res.r_shape} vs {r_true}"
    # inverted discharge correlates strongly with truth
    q_true_eval = Q_true[res.overpass_idx]
    r = np.corrcoef(np.log(res.q), np.log(q_true_eval))[0, 1]
    assert r > 0.85, f"correlation {r}"


def test_synthetic_convex_channel():
    inputs, Q_true, r_true = _synthetic_reach(r_true=0.35, seed=2)
    res = invert_reach(**inputs)
    assert res.ok
    assert res.r_shape < 1.0 # recovers convex channel


def test_spline_path_recovers_dynamics():
    """The spline + spatial-Z0 path also recovers shape and dynamics."""
    inputs, Q_true, r_true = _synthetic_reach(r_true=0.5, seed=3)
    cfg = InversionConfig(use_spline=True)
    res = invert_reach(**inputs, cfg=cfg)
    assert res.ok, res.reason
    assert abs(res.r_shape - r_true) < 0.25
    r = np.corrcoef(np.log(res.q), np.log(Q_true[res.overpass_idx]))[0, 1]
    assert r > 0.85, f"correlation {r}"


# Parity against the prototype on real reaches

@pytest.mark.skipif(not STORE.exists(), reason="zarr store not present")
def test_parity_on_real_reaches():
    store = zarr.open(str(STORE), mode='r')
    val_g = np.array(store['gauge']['val_reach_idx'])
    ov = np.array(store['gauge']['val_pairs']['overpass_idx'])
    q = np.array(store['gauge']['val_pairs']['q'])
    pm = np.array(store['gauge']['val_pairs']['mask'])

    # bucket lookup
    obs = store['obs']
    pos = {}
    for bkey in obs.keys():
        for li, gp in enumerate(np.array(obs[bkey]['reach_idx'])):
            pos[int(gp)] = (bkey, int(li))

    r_shapes, corrs, n_ok = [], [], 0
    for i in range(min(200, len(val_g))):
        gpos = int(val_g[i])
        if gpos not in pos:
            continue
        bkey, li = pos[gpos]
        bg = obs[bkey]
        ns = dict(
            wse_mean=float(store['norm']['wse_mean'][gpos]), wse_std=float(store['norm']['wse_std'][gpos]),
            width_mean=float(store['norm']['width_mean'][gpos]), width_std=float(store['norm']['width_std'][gpos]))
        pmask = pm[i].astype(bool)
        oidx = ov[i][pmask]; gq = np.maximum(q[i][pmask], 1e-6)
        if len(oidx) < 5:
            continue
        res = invert_reach(
            np.clip(np.array(bg['wse_norm'][li]), -10, 10),
            np.clip(np.array(bg['width_norm'][li]), -10, 10),
            np.array(bg['node_mask'][li]), np.array(bg['overpass_mask'][li]).astype(bool),
            np.array(bg['node_id'][li]), np.array(bg['overpass_time_s'][li]),
            np.array(store['prior']['monthly_q'][gpos]), ns,
            eval_overpass_idx=oidx,
        )
        if not res.ok:
            continue
        n_ok += 1
        r_shapes.append(res.r_shape)
        # align inverted Q to the gauge pairs actually evaluated
        keep = np.isin(oidx, res.overpass_idx)
        if keep.sum() >= 5:
            corrs.append(np.corrcoef(np.log(res.q), np.log(gq[keep]))[0, 1])

    assert n_ok >= 50, f"only {n_ok} reaches inverted"
    # parity with prototype: convex median shape (~0.35) and high median correlation (~0.89)
    assert 0.3 <= np.median(r_shapes) <= 1.0, f"median r {np.median(r_shapes)}"
    assert np.median(corrs) > 0.75, f"median corr {np.median(corrs)}"
