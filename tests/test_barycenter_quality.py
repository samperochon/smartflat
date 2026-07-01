"""Tests for the FGW barycenter and the representation-quality harness (Kickoff F).

Mirrors ``tests/test_barycenter_dba.py``: pins the shape/valid-symbol/determinism
contract of :func:`barycenter_fgw`, the frequency<->structure knob, and the
method-agnostic scoring of :func:`score_barycenter_quality` over the ``{build, distance,
kind}`` registry. Small ``n_nodes`` / ``max_iter`` keep FGW fast under
``NUMBA_THREADING_LAYER=workqueue``.
"""

import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.baselines import (
    barycenter_fgw,
    default_baseline_methods,
    extra_experiment_methods,
    fgw_methods,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import (
    build_fgw_registry,
    quality_table,
    score_barycenter_quality,
)


def _ground_cost(g=6, seed=3):
    rng = np.random.RandomState(seed)
    d = rng.rand(g, g)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return np.ascontiguousarray(d)


def _cohort(g=6, l=32, seed=0):
    """Two groups with distinct frequency profiles."""
    rng = np.random.RandomState(seed)
    xa = rng.choice(g, size=(5, l), p=[.4, .2, .15, .1, .1, .05])
    xb = rng.choice(g, size=(6, l), p=[.1, .1, .15, .2, .2, .25])
    X = np.vstack([xa, xb]).astype(int)
    labels = np.array(['A'] * 5 + ['B'] * 6, dtype=object)
    return X, labels


# --------------------------------------------------------------------------- FGW builder

def test_fgw_shape_valid_symbols_and_determinism():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 6, size=(7, 32)).astype(int)
    d_g = _ground_cost(6)
    b1 = barycenter_fgw(x, d_g, alpha=0.5, n_nodes=16, feature='mds',
                        max_iter=15, random_state=0)
    b2 = barycenter_fgw(x, d_g, alpha=0.5, n_nodes=16, feature='mds',
                        max_iter=15, random_state=0)
    assert b1.shape == (16,)
    np.testing.assert_array_equal(b1, b2)                 # deterministic given the seed
    assert set(np.unique(b1)).issubset(set(range(6)))


def test_fgw_both_encodings_valid():
    rng = np.random.RandomState(1)
    x = rng.randint(0, 6, size=(6, 40)).astype(int)
    d_g = _ground_cost(6)
    for feat in ('mds', 'onehot'):
        b = barycenter_fgw(x, d_g, alpha=0.5, n_nodes=20, feature=feat,
                           max_iter=15, random_state=0)
        assert b.shape == (20,)
        assert set(np.unique(b)).issubset(set(range(6)))


def test_fgw_alpha_knob_is_live():
    """alpha=0 (feature/frequency) and alpha=1 (GW/structure) give different barycenters."""
    rng = np.random.RandomState(2)
    x = rng.randint(0, 6, size=(6, 40)).astype(int)
    d_g = _ground_cost(6)
    b0 = barycenter_fgw(x, d_g, alpha=0.0, n_nodes=20, feature='onehot',
                        max_iter=20, random_state=0)
    b1 = barycenter_fgw(x, d_g, alpha=1.0, n_nodes=20, feature='onehot',
                        max_iter=20, random_state=0)
    assert not np.array_equal(b0, b1)


def test_fgw_ragged_input_is_resampled():
    rng = np.random.RandomState(3)
    x = [rng.randint(0, 6, size=rng.randint(20, 40)) for _ in range(4)]
    d_g = _ground_cost(6)
    b = barycenter_fgw(x, d_g, alpha=0.5, n_nodes=16, feature='mds',
                       max_iter=15, random_state=0)
    assert b.shape == (16,)
    assert set(np.unique(b)).issubset(set(range(6)))


def test_fgw_bad_feature_raises():
    d_g = _ground_cost(6)
    x = np.random.RandomState(0).randint(0, 6, size=(4, 16))
    with pytest.raises(ValueError):
        barycenter_fgw(x, d_g, feature='bogus', n_nodes=16, max_iter=5)


# --------------------------------------------------------------------------- quality harness

def _small_registry(d_g, g, l):
    methods = {
        'majority_voting': default_baseline_methods(d_g, nu=1e-4, lmbda=0.1)['majority_voting'],
        'wasserstein': default_baseline_methods(d_g, nu=1e-4, lmbda=0.1)['wasserstein'],
        'k_medoid': default_baseline_methods(d_g, nu=1e-4, lmbda=0.1)['k_medoid'],
        'transition': extra_experiment_methods(d_g, g, nu=1e-4, lmbda=0.1)['transition'],
    }
    methods.update(fgw_methods(d_g, n_nodes=l, alpha=0.5, max_iter=12))
    return methods


def test_score_columns_and_determinism():
    from smartflat.engine.distances._rtwe import rtwe_pairwise_distance
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    d_pw = rtwe_pairwise_distance(X.astype(np.float64), nu=1e-4, lmbda=0.1,
                                  precomputed_distances=d_g)
    methods = _small_registry(d_g, g, l)
    df = score_barycenter_quality(X, labels, methods, d_g, D_pairwise=d_pw,
                                  n_inits=2, random_state=42)
    assert list(df.columns) == ['method', 'group', 'metric', 'init', 'value']
    df2 = score_barycenter_quality(X, labels, methods, d_g, D_pairwise=d_pw,
                                   n_inits=2, random_state=42)
    assert df.equals(df2)                                  # deterministic
    assert set(df['method']) == set(methods)
    assert set(df['group']) == {'A', 'B'}


def test_histogram_output_nan_pattern():
    """Wasserstein histogram: no sequence -> rTWE/segment metrics NaN, frequency defined."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = {'wasserstein':
               default_baseline_methods(d_g, nu=1e-4, lmbda=0.1)['wasserstein']}
    q = quality_table(score_barycenter_quality(X, labels, methods, d_g, n_inits=1))
    assert np.isnan(q.loc['wasserstein', 'inertia_rtwe'])
    assert np.isnan(q.loc['wasserstein', 'n_segments'])
    assert np.isfinite(q.loc['wasserstein', 'freq_fidelity'])
    assert np.isfinite(q.loc['wasserstein', 'inertia_native'])


def test_transition_output_structure_only():
    """Transition-matrix barycenter: structure defined (it IS the group mean), freq NaN."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = {'transition': extra_experiment_methods(d_g, g)['transition']}
    q = quality_table(score_barycenter_quality(X, labels, methods, d_g, n_inits=1))
    assert np.isnan(q.loc['transition', 'freq_fidelity'])
    # the barycenter is the group-mean transition matrix -> ~0 distance to itself
    assert q.loc['transition', 'struct_preservation'] < 1e-6


def test_stability_zero_for_deterministic_builder():
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = {'majority_voting':
               default_baseline_methods(d_g)['majority_voting']}
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=3)
    stab = df[df['metric'] == 'stability_inertia_rtwe']['value']
    assert np.allclose(stab.values, 0.0)


def test_medoid_requires_d_pairwise():
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = {'k_medoid': default_baseline_methods(d_g, nu=1e-4, lmbda=0.1)['k_medoid']}
    with pytest.raises(ValueError):
        score_barycenter_quality(X, labels, methods, d_g, n_inits=1)


def test_alpha_sweep_registry_runs():
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    sweep = build_fgw_registry(d_g, alphas=[1e-3, 1.0], n_nodes=l,
                               encodings=('onehot',), max_iter=12)
    q = quality_table(score_barycenter_quality(X, labels, sweep, d_g, n_inits=1))
    assert set(q.index) == {'fgw_onehot_a0.001', 'fgw_onehot_a1'}
    assert np.isfinite(q['inertia_rtwe']).all()
