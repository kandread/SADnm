"""Tests for the Stage-3 graph GMRF (spatial assimilation)."""
import numpy as np

from sadnm.spatial import graph_gmrf, weighted_laplacian, build_weight_matrix


def _two_node_W(w=1.0):
    return np.array([[0.0, w], [w, 0.0]])


def test_unobserved_node_filled_from_neighbor():
    """Node 1 unobserved -> pulled toward observed node 0's value via the edge."""
    mean, var = graph_gmrf(np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                           np.array([0.1, 0.1]), _two_node_W(1.0), lam=5.0, eps=1e-3)
    assert mean[1] > 0.5 # interpolated toward the neighbour's 1.0
    assert var[1] > var[0] # less certain than the observed node


def test_isolated_unobserved_falls_back_to_prior():
    """No edges, no observation -> posterior ~0 (prior mean) with variance ~1/eps."""
    mean, var = graph_gmrf(np.array([3.0]), np.array([0.0]), np.array([0.1]),
                           np.zeros((1, 1)), lam=1.0, eps=0.5)
    assert abs(float(mean[0])) < 1e-6
    assert abs(float(var[0]) - 1.0 / 0.5) < 1e-6


def test_strong_observation_recovers_value():
    """Well-observed, weak coupling/anchor -> posterior ~ observation."""
    y = np.array([2.0, -1.0])
    mean, _ = graph_gmrf(y, np.array([1.0, 1.0]), np.array([0.05, 0.05]),
                         _two_node_W(1.0), lam=1e-3, eps=1e-4)
    assert np.allclose(mean, y, atol=1e-2)


def test_coupling_shrinks_neighbors_together():
    """Two observed nodes with strong coupling -> posteriors move toward each other."""
    y = np.array([1.0, -1.0]); mask = np.array([1.0, 1.0]); noise = np.array([0.3, 0.3])
    m0, _ = graph_gmrf(y, mask, noise, _two_node_W(1.0), lam=0.0, eps=1e-4)
    m1, _ = graph_gmrf(y, mask, noise, _two_node_W(1.0), lam=10.0, eps=1e-4)
    assert abs(m1[0] - m1[1]) < abs(m0[0] - m0[1]) # closer together under coupling


def test_coupling_strength_monotone():
    """Stronger coupling pulls an unobserved node further toward its neighbour."""
    y = np.array([1.0, 0.0]); mask = np.array([1.0, 0.0]); noise = np.array([0.1, 0.1])
    weak, _ = graph_gmrf(y, mask, noise, _two_node_W(1.0), lam=0.5, eps=1e-3)
    strong, _ = graph_gmrf(y, mask, noise, _two_node_W(1.0), lam=20.0, eps=1e-3)
    assert strong[1] > weak[1]
    assert np.all(np.isfinite(weak)) and np.all(np.isfinite(strong))


def test_laplacian_psd():
    W = np.array([[0., 1., 2.], [1., 0., 0.5], [2., 0.5, 0.]])
    L = weighted_laplacian(W)
    assert np.min(np.linalg.eigvalsh(L)) > -1e-8 # PSD
    assert np.allclose(L @ np.ones(3), 0.0, atol=1e-8) # constant in null space


def test_build_weight_matrix_small_chain():
    adj = {1: {2}, 2: {1, 3}, 3: {2}}
    W, idx = build_weight_matrix([1, 2, 3], adj, K=6, rho=0.9)
    assert W[idx[1], idx[2]] > W[idx[1], idx[3]] > 0 # 1-2 (1 hop) > 1-3 (2 hops)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
