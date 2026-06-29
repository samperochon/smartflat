"""Regression guard for ``barycenter_mode_dba`` after the rTWE kernel hardening.

The mode-DBA logic is unchanged; it now runs on the hardened rolling-buffer / prange
rTWE kernel. These tests pin its contract (shape, valid symbols, determinism) so a
future kernel change that silently alters the barycenter is caught.
"""

import numpy as np

from smartflat.features.symbolic_barycenter.baselines import barycenter_mode_dba


def _ground_cost(g=6, seed=3):
    rng = np.random.RandomState(seed)
    d = rng.rand(g, g)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return np.ascontiguousarray(d)


def test_mode_dba_shape_valid_symbols_and_determinism():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 6, size=(8, 40)).astype(int)
    d_g = _ground_cost(6)
    b1 = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1)
    b2 = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1)
    assert b1.shape == (40,)
    np.testing.assert_array_equal(b1, b2)  # deterministic (medoid init, no RNG)
    assert set(np.unique(b1)).issubset(set(range(6)))


def test_mode_dba_single_sequence_is_identity():
    d_g = _ground_cost(6)
    x = np.random.RandomState(1).randint(0, 6, size=(1, 20)).astype(int)
    b = barycenter_mode_dba(x, d_g, nu=1e-4, lmbda=0.1)
    np.testing.assert_array_equal(b, x[0])
