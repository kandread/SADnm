"""
SADnm Stage 3, mass conservation as spatial assimilation on the SWORD river graph.

It is expected that connected reaches share their log-Q ANOMALY (deviation from the monthly prior) at
relatively high correlation, decaying smoothly with network distance which is signal the prior cannot
see. This module fuses each reach's noisy per-reach estimate with its neighbours' via a graph-Laplacian
Gaussian Markov random field (the spatial analog of the Stage-2 temporal GP).

For a single time slice of the anomaly field x (one value per reach):

    posterior precision  P = M + lam * L + eps * I
    posterior mean       x_hat = P^{-1} (M y)
    posterior variance   diag(P^{-1})

where
    y      per-reach observed anomaly (Stage-1/2 estimate); arbitrary where missing
    M      diag(mask / obs_noise^2) — observation precision; 0 for uninvertible reaches
    L      weighted graph Laplacian D - W (neighbours pulled together)
    lam    coupling strength (calibrated on training gauges)
    eps    anchor to the prior mean (anomaly 0) for isolated / far reaches

Observed reaches are pulled toward their estimate; unobserved (uninvertible)
reaches are interpolated from neighbours (footprint extension); isolated reaches
fall back to the prior with wide uncertainty. Everything is a differentiable
linear solve. The level still rides the monthly prior: Q = prior * exp(x_hat).
"""

import collections
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import netCDF4 as nc

# graph construction (SWORD topology)

def build_adjacency(sword_dir: str | Path):
    """Undirected reach adjacency from SWORD rch_id_up / rch_id_dn (all continents)."""
    adj = collections.defaultdict(set)
    for f in sorted(Path(sword_dir).glob('*_sword_v17.nc')):
        d = nc.Dataset(f); r = d.groups['reaches']
        rid = np.asarray(r['reach_id'][:]).astype(np.int64)
        up = np.asarray(r['rch_id_up'][:]).astype(np.int64) # (4, n)
        dn = np.asarray(r['rch_id_dn'][:]).astype(np.int64)
        for i in range(len(rid)):
            a = int(rid[i])
            for j in range(up.shape[0]):
                for nb in (int(up[j, i]), int(dn[j, i])):
                    if nb > 0:
                        adj[a].add(nb); adj[nb].add(a)
        d.close()
    return adj


def hop_neighbors(adj, src, K):
    """BFS hop distance from src out to K hops: {reach_id: hops}."""
    seen = {src: 0}; frontier = [src]
    for depth in range(1, K + 1):
        nf = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen[v] = depth; nf.append(v)
        frontier = nf
    return seen


def decay_weight(hops, rho=0.978):
    """Edge weight from the gate's measured anomaly-correlation decay (~rho per hop)."""
    return rho ** np.asarray(hops, float)


def weighted_laplacian(W):
    """Graph Laplacian L = D - W for a dense (N,N) symmetric weight matrix."""
    W = np.asarray(W, float)
    return np.diag(np.sum(W, axis=1)) - W


# GMRF solve (per timestep)

def graph_gmrf(y, mask, noise, W, lam, eps):
    """
    Posterior of the anomaly field on a (sub)graph for one time slice.

    y      (N,)   observed anomaly per node (value ignored where mask==0)
    mask   (N,)   1 if node has an observation, else 0
    noise  (N,)   observation std per node (in log-anomaly space)
    W      (N,N)  symmetric non-negative edge weights (0 diagonal)
    lam    scalar spatial coupling strength
    eps    scalar prior-anchor (pulls toward anomaly 0)

    Returns (mean, var) over all N nodes. Unobserved nodes are filled from
    neighbours; isolated nodes return ~0 (prior) with variance ~1/eps.
    """
    y = np.asarray(y, float); mask = np.asarray(mask, float); noise = np.asarray(noise, float)
    N = y.shape[0]
    m = mask / (noise ** 2) # per-node obs precision
    P = np.diag(m + eps) + lam * weighted_laplacian(W)
    cf = sla.cho_factor(P, lower=True)
    mean = sla.cho_solve(cf, m * y)
    var = np.diag(sla.cho_solve(cf, np.eye(N)))
    return mean, var


def build_weight_matrix(reach_ids, adj, K=6, rho=0.978):
    """Dense (N,N) weight matrix among `reach_ids`, edges = connected within K hops,
    weight = decay_weight(hops). Coupling through intermediate ungauged reaches is
    captured by the hop distance. Returns (W, index_of) for the node ordering."""
    idx = {int(r): i for i, r in enumerate(reach_ids)}
    N = len(reach_ids)
    W = np.zeros((N, N))
    rid_set = set(idx)
    for r in reach_ids:
        seen = hop_neighbors(adj, int(r), K)
        for nb, h in seen.items():
            if h > 0 and nb in rid_set:
                w = float(decay_weight(h, rho))
                W[idx[int(r)], idx[nb]] = w; W[idx[nb], idx[int(r)]] = w
    return W, idx
