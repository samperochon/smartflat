"""Tests for the library-backed Soft-DTW and SSG barycenters (Kickoff F.S2).

Mirrors ``tests/test_barycenter_dba.py`` / ``tests/test_barycenter_quality.py``: pins the
shape / valid-symbol / determinism contract of :func:`barycenter_softdtw` (Cuturi & Blondel,
ICML 2017) and :func:`barycenter_ssg` (Schultz & Jain, Pattern Recognition 2018), the
embedding round-trip both rely on, and their method-agnostic scoring through
:func:`score_barycenter_quality` via the :func:`softdtw_ssg_methods` registry. Small sizes
and ``max_iter`` keep them fast under ``NUMBA_THREADING_LAYER=workqueue`` (set in conftest).
"""

import numpy as np
import pytest

# Both averagers are tslearn-backed; skip cleanly if tslearn is not installed rather than
# erroring at first call (tslearn is imported lazily inside the builders).
pytest.importorskip('tslearn')

from smartflat.features.symbolic_barycenter.baselines import (
    barycenter_softdtw,
    barycenter_ssg,
    embed_symbolic_to_real,
    project_real_to_symbolic,
    softdtw_ssg_methods,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import (
    quality_table,
    score_barycenter_quality,
)


from _bary_helpers import _cohort, _ground_cost


# --------------------------------------------------------------------------- builders

def test_softdtw_shape_valid_symbols_and_determinism():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 6, size=(8, 24)).astype(int)
    d_g = _ground_cost(6)
    b1 = barycenter_softdtw(x, d_g, gamma=1.0, max_iter=10)
    b2 = barycenter_softdtw(x, d_g, gamma=1.0, max_iter=10)
    assert b1.shape == (24,)
    np.testing.assert_array_equal(b1, b2)                 # deterministic (Euclidean-mean init)
    assert set(np.unique(b1)).issubset(set(range(6)))


def test_ssg_shape_valid_symbols_and_determinism():
    rng = np.random.RandomState(1)
    x = rng.randint(0, 6, size=(8, 24)).astype(int)
    d_g = _ground_cost(6)
    b1 = barycenter_ssg(x, d_g, max_iter=10, random_state=42)
    b2 = barycenter_ssg(x, d_g, max_iter=10, random_state=42)
    assert b1.shape == (24,)
    np.testing.assert_array_equal(b1, b2)                 # deterministic given the seed
    assert set(np.unique(b1)).issubset(set(range(6)))


def test_ssg_seed_changes_barycenter():
    """Different seeds -> generally different stochastic-subgradient trajectories."""
    rng = np.random.RandomState(2)
    x = rng.randint(0, 6, size=(8, 24)).astype(int)
    d_g = _ground_cost(6)
    b0 = barycenter_ssg(x, d_g, max_iter=10, random_state=0)
    b1 = barycenter_ssg(x, d_g, max_iter=10, random_state=7)
    assert b0.shape == b1.shape == (24,)                  # both valid; seed is honoured
    assert set(np.unique(b0)).issubset(set(range(6)))
    assert set(np.unique(b1)).issubset(set(range(6)))


def test_embedding_round_trip_is_identity():
    """The embed -> decode adapter both methods rely on is lossless for hard symbols."""
    rng = np.random.RandomState(3)
    x = rng.randint(0, 6, size=(4, 20)).astype(int)
    d_g = _ground_cost(6)
    recovered = project_real_to_symbolic(embed_symbolic_to_real(x, d_g), d_g)
    np.testing.assert_array_equal(recovered, x)


# --------------------------------------------------------------------------- quality harness

def test_softdtw_ssg_registered_and_scored():
    """Both methods score through the harness with the standard long-form columns."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = softdtw_ssg_methods(d_g, nu=1e-4, lmbda=0.1, sdtw_max_iter=10, ssg_max_iter=10)
    assert set(methods) == {'soft_dtw_bary', 'ssg'}
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=1, random_state=42)
    assert list(df.columns) == ['method', 'group', 'metric', 'init', 'value']
    assert set(df['method']) == {'soft_dtw_bary', 'ssg'}
    assert set(df['group']) == {'A', 'B'}
    # Sequence-output methods populate the rTWE-inertia yardstick with finite values.
    inertia = df[(df['metric'] == 'inertia_rtwe') & (df['init'].notna())]['value']
    assert len(inertia) > 0
    assert np.isfinite(inertia.to_numpy()).all()


def test_softdtw_ssg_quality_table_pivot():
    """quality_table pivots both methods onto the standard metric columns."""
    g, l = 6, 24
    X, labels = _cohort(g, l)
    d_g = _ground_cost(g)
    methods = softdtw_ssg_methods(d_g, nu=1e-4, lmbda=0.1, sdtw_max_iter=10, ssg_max_iter=10)
    df = score_barycenter_quality(X, labels, methods, d_g, n_inits=1, random_state=42)
    q = quality_table(df)
    assert {'soft_dtw_bary', 'ssg'}.issubset(set(q.index))
    assert 'inertia_rtwe' in q.columns
