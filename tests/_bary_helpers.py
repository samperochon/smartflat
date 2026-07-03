"""Shared synthetic builders for the barycenter / distance test suite.

Extracted (Kickoff L, internal-consistency fix) from the byte-identical copies that had
been pasted into ``test_barycenter_{quality,softdtw_ssg,msa_consensus,discreteness_levers,
dba}.py`` and ``test_distances_{rtwe,eshape_dtw}.py``. These are arg-taking *helper
functions* (not pytest fixtures) so call sites stay ``_ground_cost(6)`` / ``_cohort(g, l)``
/ ``_seq(rng, 15)`` verbatim -- behaviour is byte-for-byte unchanged, only the definition
is now single-sourced. ``tests/`` is on ``sys.path`` under pytest's default (prepend) import
mode, so ``from _bary_helpers import ...`` resolves for every test module.
"""

import numpy as np


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


def _seq(rng, length, n_symbols=5):
    """Create a (1, length) int64 symbolic sequence."""
    return rng.randint(0, n_symbols, size=(1, length)).astype(np.int64)
