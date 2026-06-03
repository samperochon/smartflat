"""Unit tests for Edit-Shape DTW distance (PAPER_BRIDGE item A2).

Covers: symmetry, identity, cost matrix shape, step_sequ parameter effect,
pairwise distance properties, and alignment path validity.

Note: Edit-Shape DTW uses rTWE as inner cost, so these tests also
implicitly verify the rTWE integration path.
"""

import numpy as np
import pytest

from smartflat.engine.distances import (
    eshape_dtw_alignment_path,
    eshape_dtw_cost_matrix,
    eshape_dtw_distance,
    eshape_dtw_pairwise_distance,
)


def _seq(rng, length, n_symbols=5):
    """Create a (1, length) int64 symbolic sequence."""
    return rng.randint(0, n_symbols, size=(1, length)).astype(np.int64)


# ---------------------------------------------------------------------------
# eshape_dtw_distance
# ---------------------------------------------------------------------------


class TestEshapeDtwDistance:
    def test_symmetry(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        d_xy = eshape_dtw_distance(x, y, precomputed_distances=D5)
        d_yx = eshape_dtw_distance(y, x, precomputed_distances=D5)
        assert d_xy == pytest.approx(d_yx)

    def test_identity(self, D5):
        x = _seq(np.random.RandomState(7), 8)
        assert eshape_dtw_distance(x, x, precomputed_distances=D5) == pytest.approx(
            0.0
        )

    def test_non_negative(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        assert eshape_dtw_distance(x, y, precomputed_distances=D5) >= 0.0

    def test_different_params(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        d1 = eshape_dtw_distance(x, y, nu=0.001, lmbda=1.0, precomputed_distances=D5)
        d2 = eshape_dtw_distance(x, y, nu=0.1, lmbda=0.01, precomputed_distances=D5)
        assert d1 != pytest.approx(d2, abs=1e-8)


# ---------------------------------------------------------------------------
# eshape_dtw_cost_matrix
# ---------------------------------------------------------------------------


class TestEshapeDtwCostMatrix:
    def test_shape_step_1(self, D5):
        """Cost matrix shape = ((N-1)//step, (M-1)//step) for step_sequ=1."""
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        cm = eshape_dtw_cost_matrix(x, y, precomputed_distances=D5, step_sequ=1)
        assert cm.shape == (7, 5)

    def test_shape_step_2(self, D5):
        """step_sequ=2 yields a smaller cost matrix."""
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        cm = eshape_dtw_cost_matrix(x, y, precomputed_distances=D5, step_sequ=2)
        assert cm.shape == (3, 2)

    def test_last_cell_matches_distance(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        cm = eshape_dtw_cost_matrix(x, y, precomputed_distances=D5)
        d = eshape_dtw_distance(x, y, precomputed_distances=D5)
        assert cm[-1, -1] == pytest.approx(d)

    def test_step_sequ_changes_distance(self, D5):
        """Different step_sequ values generally yield different distances."""
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 10), _seq(rng, 8)
        d1 = eshape_dtw_distance(x, y, precomputed_distances=D5, step_sequ=1)
        d2 = eshape_dtw_distance(x, y, precomputed_distances=D5, step_sequ=2)
        # Not guaranteed to differ, but with random sequences it almost certainly will
        assert d1 != pytest.approx(d2, abs=1e-8)


# ---------------------------------------------------------------------------
# eshape_dtw_pairwise_distance
# ---------------------------------------------------------------------------


class TestEshapeDtwPairwise:
    def test_diagonal_zero(self, D5):
        rng = np.random.RandomState(1)
        X = np.stack([_seq(rng, 8) for _ in range(3)])  # (3, 1, 8)
        D = eshape_dtw_pairwise_distance(X, precomputed_distances=D5)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_symmetric(self, D5):
        rng = np.random.RandomState(2)
        X = np.stack([_seq(rng, 8) for _ in range(3)])
        D = eshape_dtw_pairwise_distance(X, precomputed_distances=D5)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_cross_mode_shape(self, D5):
        rng = np.random.RandomState(3)
        X = np.stack([_seq(rng, 8) for _ in range(3)])
        Y = np.stack([_seq(rng, 8) for _ in range(2)])
        D = eshape_dtw_pairwise_distance(X, Y, precomputed_distances=D5)
        assert D.shape == (3, 2)

    def test_matches_pointwise(self, D5):
        """Pairwise distances match individual eshape_dtw_distance calls."""
        rng = np.random.RandomState(4)
        X = np.stack([_seq(rng, 8) for _ in range(3)])
        D = eshape_dtw_pairwise_distance(X, precomputed_distances=D5)
        for i in range(3):
            for j in range(i + 1, 3):
                d_ij = eshape_dtw_distance(X[i], X[j], precomputed_distances=D5)
                assert D[i, j] == pytest.approx(d_ij)


# ---------------------------------------------------------------------------
# eshape_dtw_alignment_path
# ---------------------------------------------------------------------------


class TestEshapeDtwAlignmentPath:
    def test_path_monotonicity(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        path, _ = eshape_dtw_alignment_path(x, y, precomputed_distances=D5)
        for k in range(1, len(path)):
            assert path[k][0] >= path[k - 1][0]
            assert path[k][1] >= path[k - 1][1]

    def test_path_endpoints(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 8), _seq(rng, 6)
        path, _ = eshape_dtw_alignment_path(x, y, precomputed_distances=D5)
        # Cost matrix shape is ((N-1)//step - 1, (M-1)//step - 1) after padding removal
        cm = eshape_dtw_cost_matrix(x, y, precomputed_distances=D5)
        assert path[0] == (0, 0)
        assert path[-1] == (cm.shape[0] - 1, cm.shape[1] - 1)
