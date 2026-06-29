"""Unit tests for rTWE distance (PAPER_BRIDGE item A1).

Covers: symmetry, identity, triangle inequality, cost matrix shape,
alignment path validity, pairwise distance properties, 1D/2D equivalence,
edge cases, and Numba JIT compilation smoke tests.
"""

import numpy as np
import pytest

from smartflat.engine.distances import (
    rtwe_alignment_path,
    rtwe_alignment_path_with_costs,
    rtwe_cost_matrix,
    rtwe_distance,
    rtwe_pairwise_distance,
)


def _seq(rng, length, n_symbols=5):
    """Create a (1, length) int64 symbolic sequence."""
    return rng.randint(0, n_symbols, size=(1, length)).astype(np.int64)


# ---------------------------------------------------------------------------
# rtwe_distance
# ---------------------------------------------------------------------------


class TestRtweDistance:
    def test_symmetry(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        assert rtwe_distance(x, y, precomputed_distances=D5) == pytest.approx(
            rtwe_distance(y, x, precomputed_distances=D5)
        )

    def test_identity(self, D5):
        x = _seq(np.random.RandomState(7), 15)
        assert rtwe_distance(x, x, precomputed_distances=D5) == pytest.approx(0.0)

    def test_non_negative(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        assert rtwe_distance(x, y, precomputed_distances=D5) >= 0.0

    def test_triangle_inequality(self, D5):
        rng = np.random.RandomState(13)
        a, b, c = _seq(rng, 10), _seq(rng, 10), _seq(rng, 10)
        d_ab = rtwe_distance(a, b, precomputed_distances=D5)
        d_bc = rtwe_distance(b, c, precomputed_distances=D5)
        d_ac = rtwe_distance(a, c, precomputed_distances=D5)
        assert d_ac <= d_ab + d_bc + 1e-10

    def test_1d_2d_equivalence(self, D5):
        rng = np.random.RandomState(42)
        x_1d = rng.randint(0, 5, size=10).astype(np.int64)
        y_1d = rng.randint(0, 5, size=8).astype(np.int64)
        d_1d = rtwe_distance(x_1d, y_1d, precomputed_distances=D5)
        d_2d = rtwe_distance(
            x_1d.reshape(1, -1), y_1d.reshape(1, -1), precomputed_distances=D5
        )
        assert d_1d == pytest.approx(d_2d)

    def test_different_params_give_different_distances(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        d1 = rtwe_distance(x, y, nu=0.001, lmbda=1.0, precomputed_distances=D5)
        d2 = rtwe_distance(x, y, nu=0.1, lmbda=0.01, precomputed_distances=D5)
        assert d1 != pytest.approx(d2, abs=1e-8)

    def test_jit_second_call(self, D5):
        x, y = _seq(np.random.RandomState(0), 5), _seq(np.random.RandomState(1), 5)
        d1 = rtwe_distance(x, y, precomputed_distances=D5)
        d2 = rtwe_distance(x, y, precomputed_distances=D5)
        assert d1 == pytest.approx(d2)


# ---------------------------------------------------------------------------
# rtwe_cost_matrix
# ---------------------------------------------------------------------------


class TestRtweCostMatrix:
    def test_shape_unequal_length(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        cm = rtwe_cost_matrix(x, y, precomputed_distances=D5)
        assert cm.shape == (15, 12)

    def test_shape_equal_length(self, D5):
        rng = np.random.RandomState(0)
        x, y = _seq(rng, 10), _seq(rng, 10)
        cm = rtwe_cost_matrix(x, y, precomputed_distances=D5)
        assert cm.shape == (10, 10)

    def test_last_cell_matches_distance(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        cm = rtwe_cost_matrix(x, y, precomputed_distances=D5)
        d = rtwe_distance(x, y, precomputed_distances=D5)
        assert cm[-1, -1] == pytest.approx(d)


# ---------------------------------------------------------------------------
# rtwe_pairwise_distance
# ---------------------------------------------------------------------------


class TestRtwePairwise:
    def test_diagonal_zero(self, D5):
        rng = np.random.RandomState(1)
        X = np.stack([_seq(rng, 10) for _ in range(3)])  # (3, 1, 10)
        D = rtwe_pairwise_distance(X, precomputed_distances=D5)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_symmetric(self, D5):
        rng = np.random.RandomState(2)
        X = np.stack([_seq(rng, 10) for _ in range(3)])
        D = rtwe_pairwise_distance(X, precomputed_distances=D5)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_cross_mode_shape(self, D5):
        rng = np.random.RandomState(3)
        X = np.stack([_seq(rng, 10) for _ in range(3)])
        Y = np.stack([_seq(rng, 10) for _ in range(2)])
        D = rtwe_pairwise_distance(X, Y, precomputed_distances=D5)
        assert D.shape == (3, 2)

    def test_matches_pointwise(self, D5):
        """Pairwise distances match individual rtwe_distance calls."""
        rng = np.random.RandomState(4)
        X = np.stack([_seq(rng, 10) for _ in range(3)])
        D = rtwe_pairwise_distance(X, precomputed_distances=D5)
        for i in range(3):
            for j in range(i + 1, 3):
                d_ij = rtwe_distance(X[i], X[j], precomputed_distances=D5)
                assert D[i, j] == pytest.approx(d_ij)


# ---------------------------------------------------------------------------
# rtwe_alignment_path
# ---------------------------------------------------------------------------


class TestRtweAlignmentPath:
    def test_path_monotonicity(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        path, _ = rtwe_alignment_path(x, y, precomputed_distances=D5)
        for k in range(1, len(path)):
            assert path[k][0] >= path[k - 1][0]
            assert path[k][1] >= path[k - 1][1]

    def test_path_endpoints(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        path, _ = rtwe_alignment_path(x, y, precomputed_distances=D5)
        assert path[0] == (0, 0)
        assert path[-1] == (x.shape[-1] - 1, y.shape[-1] - 1)

    def test_distance_matches_rtwe_distance(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        _, dist = rtwe_alignment_path(x, y, precomputed_distances=D5)
        d = rtwe_distance(x, y, precomputed_distances=D5)
        assert dist == pytest.approx(d)


# ---------------------------------------------------------------------------
# rtwe_alignment_path_with_costs
# ---------------------------------------------------------------------------


class TestRtweAlignmentPathWithCosts:
    def test_distance_matches_rtwe_distance(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        dist, _, _, _ = rtwe_alignment_path_with_costs(
            x, y, precomputed_distances=D5
        )
        d = rtwe_distance(x, y, precomputed_distances=D5)
        assert dist == pytest.approx(d)

    def test_path_costs_length_matches_path(self, D5):
        rng = np.random.RandomState(7)
        x, y = _seq(rng, 15), _seq(rng, 12)
        _, _, path, path_costs = rtwe_alignment_path_with_costs(
            x, y, precomputed_distances=D5
        )
        assert len(path_costs) == len(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRtweEdgeCases:
    def test_length_1(self, D5):
        x = np.array([[3]], dtype=np.int64)
        y = np.array([[1]], dtype=np.int64)
        d = rtwe_distance(x, y, precomputed_distances=D5)
        assert d >= 0.0
        assert np.isfinite(d)

    def test_equal_sequences_distance_zero(self, D5):
        x = np.array([[1, 2, 3, 4, 0]], dtype=np.int64)
        d = rtwe_distance(x, x, precomputed_distances=D5)
        assert d == pytest.approx(0.0)

    def test_single_symbol_repeated(self, D5):
        """Sequence of identical symbols — self-distance should be zero."""
        x = np.array([[2, 2, 2, 2, 2]], dtype=np.int64)
        d = rtwe_distance(x, x, precomputed_distances=D5)
        assert d == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hardened kernel: rolling-buffer distance + prange pairwise
# ---------------------------------------------------------------------------


class TestRtweRollingEquivalence:
    """The memory-light rolling-buffer ``rtwe_distance`` must equal the full cost
    matrix's bottom-right cell (the alignment-path oracle) across L incl. embedding
    scale, and the parallel (prange) pairwise must match a serial reference exactly."""

    NU, LMBDA = 1e-4, 0.1

    @pytest.mark.parametrize("length", [1, 2, 5, 10, 64, 257, 1024, 2048])
    def test_rolling_matches_full_matrix_equal_length(self, D28, length):
        rng = np.random.RandomState(length)
        x = rng.randint(0, 28, size=length).astype(np.int64)
        y = rng.randint(0, 28, size=length).astype(np.int64)
        d_roll = rtwe_distance(x, y, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28)
        d_full = rtwe_cost_matrix(
            x, y, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
        )[-1, -1]
        assert d_roll == pytest.approx(d_full, rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("la,lb", [(10, 7), (257, 311), (1024, 777)])
    def test_rolling_matches_full_matrix_unequal_length(self, D28, la, lb):
        rng = np.random.RandomState(la + lb)
        x = rng.randint(0, 28, size=la).astype(np.int64)
        y = rng.randint(0, 28, size=lb).astype(np.int64)
        d_roll = rtwe_distance(x, y, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28)
        d_full = rtwe_cost_matrix(
            x, y, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
        )[-1, -1]
        assert d_roll == pytest.approx(d_full, rel=1e-9, abs=1e-9)

    def test_parallel_pairwise_matches_serial(self, D28):
        """prange self-pairwise == serial loop of rtwe_distance (no race, deterministic)."""
        rng = np.random.RandomState(0)
        n, length = 12, 256
        X = np.vstack(
            [rng.randint(0, 28, size=length) for _ in range(n)]
        ).astype(np.int64)
        d_par = rtwe_pairwise_distance(
            X, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
        )
        d_ser = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d_ser[i, j] = rtwe_distance(
                    X[i], X[j], nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
                )
                d_ser[j, i] = d_ser[i, j]
        np.testing.assert_allclose(d_par, d_ser, rtol=0, atol=0)

    def test_parallel_cross_matches_serial(self, D28):
        """prange cross-pairwise (X vs Y) == serial reference, exactly."""
        rng = np.random.RandomState(1)
        X = np.vstack([rng.randint(0, 28, size=200) for _ in range(6)]).astype(np.int64)
        Y = np.vstack([rng.randint(0, 28, size=200) for _ in range(4)]).astype(np.int64)
        d_par = rtwe_pairwise_distance(
            X, Y, nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
        )
        d_ser = np.zeros((6, 4))
        for i in range(6):
            for j in range(4):
                d_ser[i, j] = rtwe_distance(
                    X[i], Y[j], nu=self.NU, lmbda=self.LMBDA, precomputed_distances=D28
                )
        np.testing.assert_allclose(d_par, d_ser, rtol=0, atol=0)
