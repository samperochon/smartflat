"""Tests for the discreteness-fairness levers (Kickoff K).

Pins the two ADDITIVE levers that make Family-B averagers treat the discrete symbol
space consistently, reported ablation-style (existing keys/defaults unchanged):

- **Lever 2 -- ground-cost-consistent decode knob** on
  :func:`project_real_to_symbolic` (``decode={'euclidean','dg'}``): ``'dg'`` snaps a real
  profile ``m`` to ``argmin_c m[c]`` (the vocabulary-restricted 1-medoid under ``D_G``, the
  geometry the harness scores in), vs the current Euclidean nearest-``D_G``-row. The two can
  differ; both are lossless round-trips for pure symbols. Threaded through
  ``barycenter_{dba_dtw,softdtw,ssg,fgw}`` and exposed as the ``*_dg`` registry variants.
- **Lever 3 -- per-iteration re-discretised categorical variants** (``dba_dtw_cat`` /
  ``ssg_cat`` / ``soft_dtw_bary_cat``): keep the reference a valid symbol string every
  iteration (like mode-DBA, but mean-then-snap). The invariant is guaranteed by the atomic
  :func:`_snap_and_reembed` helper (tested directly) + a horizon sweep + an embed-input spy.

Small sizes keep them fast under ``NUMBA_THREADING_LAYER=workqueue`` (set in conftest and
here). ``tslearn`` / ``ot`` are optional (``importorskip`` per test that needs them).
"""

import os

import numpy as np
import pytest

os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')  # fork-safe rTWE

from smartflat.features.symbolic_barycenter import baselines as B
from smartflat.features.symbolic_barycenter.baselines import (
    _snap_and_reembed,
    barycenter_dba_dtw,
    discreteness_lever_methods,
    embed_symbolic_to_real,
    project_real_to_symbolic,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import (
    quality_table,
    score_barycenter_quality,
)

EXPECTED_KEYS = {
    'dba_dtw_dg', 'soft_dtw_bary_dg', 'ssg_dg', 'fgw_onehot_dg',   # Lever 2
    'dba_dtw_cat', 'ssg_cat', 'soft_dtw_bary_cat',                 # Lever 3
}


from _bary_helpers import _cohort, _ground_cost


# --------------------------------------------------------------------------- Lever 2: decode knob

def test_dg_decode_shapes():
    """decode='dg' collapses the last (symbol) axis for both (T, G) and (N, T, G)."""
    d_g = _ground_cost(6)
    rng = np.random.RandomState(0)
    m2 = rng.rand(10, 6)                                   # (T, G)
    m3 = rng.rand(4, 10, 6)                                # (N, T, G)
    assert project_real_to_symbolic(m2, d_g, decode='dg').shape == (10,)
    assert project_real_to_symbolic(m3, d_g, decode='dg').shape == (4, 10)


def test_dg_decode_is_deterministic():
    d_g = _ground_cost(6)
    m = np.random.RandomState(1).rand(10, 6)
    a = project_real_to_symbolic(m, d_g, decode='dg')
    b = project_real_to_symbolic(m, d_g, decode='dg')
    np.testing.assert_array_equal(a, b)


def test_dg_and_euclidean_decode_can_differ():
    """Worked example: for the DBA mean of aligned bag {0, 0, 2}, the ground-cost medoid
    (dg) is symbol 1 while the Euclidean nearest-row decode picks symbol 0."""
    d_g = np.array([[0.0, 1.0, 4.0],
                    [1.0, 0.0, 1.0],
                    [4.0, 1.0, 0.0]])
    m = (d_g[0] + d_g[0] + d_g[2]) / 3.0                   # = [4/3, 1, 8/3]
    m = m[None, :]                                          # (T=1, G=3)
    np.testing.assert_array_equal(project_real_to_symbolic(m, d_g, decode='dg'), [1])
    np.testing.assert_array_equal(project_real_to_symbolic(m, d_g, decode='euclidean'), [0])


def test_round_trip_lossless_both_decodes():
    """Embedding a pure symbol string then decoding recovers it under BOTH decodes
    (D_G diagonal is 0 and the off-diagonal is strictly positive)."""
    d_g = _ground_cost(6)
    X = np.random.RandomState(2).randint(0, 6, size=(4, 20)).astype(np.int64)
    emb = embed_symbolic_to_real(X, d_g)
    np.testing.assert_array_equal(project_real_to_symbolic(emb, d_g, decode='euclidean'), X)
    np.testing.assert_array_equal(project_real_to_symbolic(emb, d_g, decode='dg'), X)


def test_default_decode_is_euclidean_and_unchanged():
    """A 2-positional-arg call (no decode) equals the explicit Euclidean decode -- the
    additive param does not change existing callers."""
    d_g = _ground_cost(6)
    m = np.random.RandomState(3).rand(10, 6)
    np.testing.assert_array_equal(
        project_real_to_symbolic(m, d_g),
        project_real_to_symbolic(m, d_g, decode='euclidean'),
    )


def test_bad_decode_raises():
    d_g = _ground_cost(6)
    m = np.random.RandomState(4).rand(10, 6)
    with pytest.raises(ValueError):
        project_real_to_symbolic(m, d_g, decode='nope')


# --------------------------------------------------------------------------- Lever 3: atomic helper

def test_snap_and_reembed_invariant():
    """The atomic categorical step: snapped is a valid integer symbol string and the
    re-embedding is exactly its D_G rows (so a loop that sets barycenter=reembed stays
    categorical by construction)."""
    d_g = _ground_cost(6)
    bary_real = np.random.RandomState(5).rand(15, 6)
    for decode in ('euclidean', 'dg'):
        snapped, reembed = _snap_and_reembed(bary_real, d_g, decode)
        assert np.issubdtype(snapped.dtype, np.integer)
        assert set(np.unique(snapped)).issubset(set(range(6)))
        assert snapped.shape == (15,)
        np.testing.assert_array_equal(reembed, d_g[snapped])


# --------------------------------------------------------------------------- Lever 3: dba_dtw_cat

@pytest.mark.parametrize('max_iter', [1, 2, 3])
def test_dba_dtw_cat_valid_symbols_every_horizon(max_iter):
    """At every iteration budget the categorical DBA returns a valid symbol string."""
    d_g = _ground_cost(6)
    X = np.random.RandomState(6).randint(0, 6, size=(5, 24)).astype(np.int64)
    b = barycenter_dba_dtw(X, d_g, max_iter=max_iter, random_state=0,
                           discretise_each_iter=True, decode='dg')
    assert b.ndim == 1 and b.shape == (24,)
    assert np.issubdtype(b.dtype, np.integer)
    assert set(np.unique(b)).issubset(set(range(6)))


def test_dba_dtw_cat_reference_is_integer_every_iteration(monkeypatch):
    """Spy on embed_symbolic_to_real: the categorical loop only ever re-embeds integer
    symbol strings, proving the reference is discrete on every iteration."""
    d_g = _ground_cost(6)
    X = np.random.RandomState(7).randint(0, 6, size=(5, 24)).astype(np.int64)
    orig = embed_symbolic_to_real
    calls = {'n': 0}

    def spy(x_symbolic, dg):
        arr = np.asarray(x_symbolic)
        assert np.issubdtype(arr.dtype, np.integer), f"non-integer re-embed input: {arr.dtype}"
        calls['n'] += 1
        return orig(x_symbolic, dg)

    monkeypatch.setattr(B, 'embed_symbolic_to_real', spy)
    b = barycenter_dba_dtw(X, d_g, max_iter=3, random_state=0,
                           discretise_each_iter=True, decode='dg')
    assert calls['n'] >= 2                                  # initial embed + >=1 per-iter re-embed
    assert set(np.unique(b)).issubset(set(range(6)))


def test_dba_dtw_default_unchanged():
    """discretise_each_iter=False + no decode arg == the explicit continuous/Euclidean call."""
    d_g = _ground_cost(6)
    X = np.random.RandomState(8).randint(0, 6, size=(5, 24)).astype(np.int64)
    a = barycenter_dba_dtw(X, d_g, max_iter=5, random_state=0)
    b = barycenter_dba_dtw(X, d_g, max_iter=5, random_state=0,
                           discretise_each_iter=False, decode='euclidean')
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- Lever 3: tslearn variants

def test_ssg_cat_valid_and_reproducible():
    pytest.importorskip('tslearn')
    d_g = _ground_cost(6)
    X = np.random.RandomState(9).randint(0, 6, size=(6, 32)).astype(np.int64)
    b1 = B.barycenter_ssg_cat(X, d_g, n_rounds=3, inner_max_iter=4, random_state=0)
    b2 = B.barycenter_ssg_cat(X, d_g, n_rounds=3, inner_max_iter=4, random_state=0)
    assert set(np.unique(b1)).issubset(set(range(6)))
    np.testing.assert_array_equal(b1, b2)                   # fixed seed -> reproducible


def test_softdtw_cat_valid_and_deterministic():
    pytest.importorskip('tslearn')
    d_g = _ground_cost(6)
    X = np.random.RandomState(10).randint(0, 6, size=(6, 32)).astype(np.int64)
    b1 = B.barycenter_softdtw_cat(X, d_g, n_rounds=3, inner_max_iter=4)
    b2 = B.barycenter_softdtw_cat(X, d_g, n_rounds=3, inner_max_iter=4)
    assert set(np.unique(b1)).issubset(set(range(6)))
    np.testing.assert_array_equal(b1, b2)                   # deterministic (L-BFGS-B)


# --------------------------------------------------------------------------- Lever 2: FGW one-hot dg

def test_fgw_onehot_dg_valid_symbols():
    pytest.importorskip('ot')
    d_g = _ground_cost(6)
    X = np.random.RandomState(11).randint(0, 6, size=(6, 32)).astype(np.int64)
    b = B.barycenter_fgw(X, d_g, feature='onehot', n_nodes=32, max_iter=20,
                         random_state=0, decode='dg')
    assert b.shape == (32,)
    assert set(np.unique(b)).issubset(set(range(6)))


def test_fgw_mds_dg_raises():
    pytest.importorskip('ot')
    d_g = _ground_cost(6)
    X = np.random.RandomState(12).randint(0, 6, size=(6, 32)).astype(np.int64)
    with pytest.raises(ValueError):
        B.barycenter_fgw(X, d_g, feature='mds', n_nodes=32, decode='dg')


# --------------------------------------------------------------------------- registry + harness

def test_lever_methods_keys():
    d_g = _ground_cost(6)
    assert set(discreteness_lever_methods(d_g)) == EXPECTED_KEYS


def test_lever_methods_registered_and_scored():
    """Every lever variant scores through the unchanged harness with finite metrics."""
    pytest.importorskip('tslearn')
    pytest.importorskip('ot')
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = discreteness_lever_methods(
        d_g, dba_max_iters=5, sdtw_max_iter=5, ssg_max_iter=5,
        fgw_n_nodes=l, cat_rounds=3, cat_inner_iter=4)
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=2, random_state=42)
    assert list(df.columns) == ['method', 'group', 'metric', 'init', 'value']
    assert set(df['method']) == EXPECTED_KEYS
    inertia = df[(df['metric'] == 'inertia_rtwe') & (df['init'].notna())]['value']
    assert len(inertia) > 0
    assert np.isfinite(inertia.to_numpy()).all()


def test_lever_quality_table_pivot():
    pytest.importorskip('tslearn')
    pytest.importorskip('ot')
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = discreteness_lever_methods(
        d_g, dba_max_iters=5, sdtw_max_iter=5, ssg_max_iter=5,
        fgw_n_nodes=l, cat_rounds=3, cat_inner_iter=4)
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=1, random_state=42)
    q = quality_table(df)
    assert EXPECTED_KEYS.issubset(set(q.index))
    assert 'inertia_rtwe' in q.columns
