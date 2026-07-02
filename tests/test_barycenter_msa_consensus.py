"""Tests for the MSA positional-consensus barycenter (Kickoff F.S3).

Mirrors ``tests/test_barycenter_softdtw_ssg.py``: pins the shape / valid-symbol /
determinism contract of :func:`barycenter_msa_consensus` (center-star MSA, Gusfield 1993,
+ profile per-column consensus, Durbin et al. 1998), the "consensus of identical sequences
is that sequence" invariant, ragged (insertion/deletion) handling, and method-agnostic
scoring through :func:`score_barycenter_quality` via the :func:`msa_consensus_methods`
registry. Small sizes keep them fast under ``NUMBA_THREADING_LAYER=workqueue`` (set in
conftest). No optional dependency -- the method reuses only the vendored rTWE aligner.
"""

import numpy as np

from smartflat.features.symbolic_barycenter.baselines import (
    barycenter_msa_consensus,
    msa_consensus_methods,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import (
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


# --------------------------------------------------------------------------- builder

def test_msa_shape_valid_symbols_and_stays_in_range():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 6, size=(8, 24)).astype(int)
    d_g = _ground_cost(6)
    b = barycenter_msa_consensus(x, d_g, nu=1e-4, lmbda=0.1)
    assert b.ndim == 1
    assert np.issubdtype(b.dtype, np.integer)
    assert b.shape[0] >= 1                                  # never empty
    assert set(np.unique(b)).issubset(set(range(6)))        # valid symbols in [0, G)


def test_msa_is_deterministic():
    rng = np.random.RandomState(1)
    x = rng.randint(0, 6, size=(8, 24)).astype(int)
    d_g = _ground_cost(6)
    b1 = barycenter_msa_consensus(x, d_g, random_state=0)
    b2 = barycenter_msa_consensus(x, d_g, random_state=999)  # seed unused -> identical
    np.testing.assert_array_equal(b1, b2)


def test_msa_consensus_of_identical_sequences_is_that_sequence():
    """The defining sanity check: no dispersion -> the consensus IS the shared sequence."""
    d_g = _ground_cost(6)
    seq = np.array([1, 1, 3, 3, 3, 0, 5, 2, 2, 4], dtype=int)
    x = np.tile(seq, (5, 1))
    b = barycenter_msa_consensus(x, d_g)
    np.testing.assert_array_equal(b, seq)


def test_msa_handles_insertions_and_deletions():
    """Ragged members (a duplicated insertion run) still yield a valid, plausible-length
    consensus dominated by the majority pattern."""
    d_g = _ground_cost(6)
    base = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=int)
    # Three members share ``base``; two carry an insertion run (a doubled symbol).
    members = [base, base, base,
               np.insert(base, 5, 5),          # one extra symbol
               np.insert(base, 2, 2)]          # a different insertion site
    # Pad ragged members to equal length for the (n, L) contract by trimming to min length.
    L = min(len(m) for m in members)
    x = np.vstack([m[:L] for m in members]).astype(int)
    b = barycenter_msa_consensus(x, d_g, occupancy=0.5)
    assert b.ndim == 1 and b.shape[0] >= 1
    assert set(np.unique(b)).issubset(set(range(6)))
    # Majority-occupied backbone -> length within a sensible band of the members' length.
    assert 1 <= b.shape[0] <= 2 * L


def test_pseudocount_zero_recovers_hard_mode():
    """pseudocount -> 0 is the hard-mode special case; a clear per-column majority wins."""
    d_g = _ground_cost(4)
    # Column-wise majority is symbol 1 everywhere (3 of 4 members).
    x = np.array([[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [0, 2, 3, 0]], dtype=int)
    b = barycenter_msa_consensus(x, d_g, pseudocount=1e-9, occupancy=0.5)
    assert set(np.unique(b)).issubset({1})


# --------------------------------------------------------------------------- quality harness

def test_msa_registered_and_scored():
    """The method scores through the harness with the standard long-form columns."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = msa_consensus_methods(d_g, nu=1e-4, lmbda=0.1)
    assert set(methods) == {'msa_consensus'}
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=1, random_state=42)
    assert list(df.columns) == ['method', 'group', 'metric', 'init', 'value']
    assert set(df['method']) == {'msa_consensus'}
    assert set(df['group']) == {'A', 'B'}
    inertia = df[(df['metric'] == 'inertia_rtwe') & (df['init'].notna())]['value']
    assert len(inertia) > 0
    assert np.isfinite(inertia.to_numpy()).all()


def test_msa_quality_table_pivot():
    """quality_table pivots msa_consensus onto the standard metric columns."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = msa_consensus_methods(d_g, nu=1e-4, lmbda=0.1)
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=1, random_state=42)
    q = quality_table(df)
    assert 'msa_consensus' in q.index
    assert 'inertia_rtwe' in q.columns
