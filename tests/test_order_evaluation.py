"""Unit tests for the frequency-preserving order-shuffle null (Kickoff E).

Covers the shuffle invariants (the load-bearing "frequency is held fixed" property),
the dwell-invariant run-transition feature, and the ``order_information`` evaluator --
including a planted-order **positive control** (delta_auc CI > 0) and a frequency-only
**negative control** (delta_auc CI brackets 0). Small ``n_shuffles``/``n_repeats`` for
speed.
"""

import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.baselines import (
    histogram_features,
    _transition_matrix,
)
from smartflat.features.symbolic_barycenter.order_evaluation import (
    _rle,
    token_shuffle,
    runlength_shuffle,
    order_shuffle_null,
    run_transition_features,
    order_information,
)


@pytest.fixture
def X_sym():
    """4 symbolic sequences, length 20, alphabet 0-4 (with runs, for RLE tests)."""
    rng = np.random.RandomState(42)
    base = rng.randint(0, 5, size=(4, 20)).astype(np.int64)
    # inject runs so RLE is non-trivial
    base[:, 5:9] = base[:, 5:6]
    return [row for row in base]


# ---------------------------------------------------------------------------
# RLE
# ---------------------------------------------------------------------------

class TestRLE:
    def test_roundtrip(self):
        seq = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 1])
        symbols, lengths = _rle(seq)
        assert np.array_equal(symbols, [0, 1, 2, 3, 1])
        assert np.array_equal(lengths, [3, 1, 2, 4, 1])
        assert np.array_equal(np.repeat(symbols, lengths), seq)

    def test_empty(self):
        symbols, lengths = _rle(np.array([], dtype=int))
        assert symbols.size == 0 and lengths.size == 0


# ---------------------------------------------------------------------------
# Shuffle invariants -- frequency is held fixed
# ---------------------------------------------------------------------------

class TestShuffles:
    @pytest.fixture
    def seq(self):
        return np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 1])

    def test_token_preserves_multiset(self, seq):
        rng = np.random.default_rng(0)
        sh = token_shuffle(seq, rng)
        assert sh.shape == seq.shape
        assert np.array_equal(np.bincount(sh, minlength=5),
                              np.bincount(seq, minlength=5))

    def test_runlength_preserves_multiset(self, seq):
        rng = np.random.default_rng(0)
        sh = runlength_shuffle(seq, rng)
        assert sh.shape == seq.shape
        assert np.array_equal(np.bincount(sh, minlength=5),
                              np.bincount(seq, minlength=5))

    def test_runlength_preserves_run_multiset(self):
        # every run has a DISTINCT symbol, so no permutation can merge runs ->
        # the (symbol, run-length) multiset is exactly invariant.
        seq = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4])
        s0, l0 = _rle(seq)
        ref = sorted(zip(s0.tolist(), l0.tolist()))
        rng = np.random.default_rng(0)
        for _ in range(25):
            s, l = _rle(runlength_shuffle(seq, rng))
            assert sorted(zip(s.tolist(), l.tolist())) == ref

    def test_token_deterministic_under_seed(self, seq):
        a = token_shuffle(seq, np.random.default_rng(7))
        b = token_shuffle(seq, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_runlength_deterministic_under_seed(self, seq):
        a = runlength_shuffle(seq, np.random.default_rng(7))
        b = runlength_shuffle(seq, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_empty_sequence(self):
        rng = np.random.default_rng(0)
        for fn in (token_shuffle, runlength_shuffle):
            out = fn(np.array([], dtype=int), rng)
            assert out.size == 0

    @pytest.mark.parametrize('kind', ['token', 'runlength'])
    def test_order_shuffle_null_keeps_per_row_histogram(self, X_sym, kind):
        # The load-bearing invariant: frequency is held fixed for EVERY sequence.
        G = 5
        H0 = histogram_features(X_sym, G)
        rng = np.random.default_rng(3)
        Xs = order_shuffle_null(X_sym, kind=kind, rng=rng)
        assert len(Xs) == len(X_sym)
        assert np.allclose(histogram_features(Xs, G), H0)


# ---------------------------------------------------------------------------
# Run-transition features (dwell-invariant)
# ---------------------------------------------------------------------------

class TestRunTransitionFeatures:
    def test_shape(self, X_sym):
        G = 5
        T = run_transition_features(X_sym, G)
        assert T.shape == (len(X_sym), G * G)

    def test_matches_collapsed_transition_matrix(self):
        seq = np.array([0, 0, 1, 1, 1, 2, 0])
        G = 3
        collapsed = _rle(seq)[0]            # [0, 1, 2, 0]
        expected = _transition_matrix(collapsed, G).reshape(-1)
        assert np.allclose(run_transition_features([seq], G)[0], expected)

    def test_single_run_is_zero(self):
        # one run -> no transitions -> zero feature row
        T = run_transition_features([np.array([2, 2, 2, 2])], 4)
        assert T.shape == (1, 16) and T.sum() == 0.0

    def test_dwell_invariant(self):
        # same run grammar, different dwell times -> identical run-transition feature
        a = np.array([0, 1, 1, 2])
        b = np.array([0, 0, 0, 1, 2, 2, 2, 2])
        G = 3
        assert np.allclose(run_transition_features([a], G),
                           run_transition_features([b], G))


# ---------------------------------------------------------------------------
# order_information -- columns, bounds, and the two scientific controls
# ---------------------------------------------------------------------------

EXPECTED_COLS = {
    'comparison', 'feature', 'shuffle', 'classifier', 'n_shuffles',
    'auc_intact', 'auc_null_mean', 'delta_auc', 'ci_low', 'ci_high',
    'p_perm', 'order_helps',
}


def _two_group_labels(n_a, n_b):
    return np.array(['HEALTHY'] * n_a + ['RIL'] * n_b, dtype=object)


class TestOrderInformation:
    def test_columns_and_bounds(self):
        rng = np.random.RandomState(0)
        X = [rng.randint(0, 4, size=30) for _ in range(40)]
        labels = _two_group_labels(20, 20)
        res = order_information(X, labels, G=4, feature='transition',
                                shuffle='token', n_repeats=2, n_folds=5,
                                n_shuffles=20, random_state=1)
        assert set(res.columns) == EXPECTED_COLS
        assert {'HEALTHY_vs_RIL', 'CONTROL_vs_PATIENT'} <= set(res['comparison'])
        for c in ['auc_intact', 'auc_null_mean']:
            assert ((res[c] >= 0) & (res[c] <= 1)).all()
        assert ((res['ci_low'] <= res['delta_auc'] + 1e-9)
                & (res['delta_auc'] - 1e-9 <= res['ci_high'])).all()
        assert (res['order_helps'] == (res['ci_low'] > 0)).all()

    def test_positive_control_planted_order(self):
        # identical 50/50 symbol frequency, different transition grammar:
        # group A alternates, group B is blocked. Order is the ONLY signal.
        rng = np.random.default_rng(0)
        L, n = 40, 22

        def alt(ph):
            return np.array([(i + ph) % 2 for i in range(L)])

        def block(rot):
            return np.roll(np.array([0] * (L // 2) + [1] * (L // 2)), rot)

        X = ([alt(int(rng.integers(2))) for _ in range(n)]
             + [block(int(rng.integers(L))) for _ in range(n)])
        labels = _two_group_labels(n, n)
        res = order_information(X, labels, G=4, feature='transition',
                                shuffle='token', n_repeats=2, n_folds=5,
                                n_shuffles=60, random_state=1)
        row = res[res.comparison == 'HEALTHY_vs_RIL'].iloc[0]
        # the planted order is recovered: high intact AUC, near-chance shuffled,
        # delta CI strictly above 0.
        assert row['auc_intact'] > 0.9
        assert row['ci_low'] > 0
        assert bool(row['order_helps']) is True

    def test_negative_control_frequency_only(self):
        # iid sequences, groups differ ONLY in marginal frequency, order random.
        # Even though raw AUC is high (frequency leaks), shuffling is
        # distribution-preserving so delta_auc ~ 0 and the CI brackets 0.
        rng = np.random.default_rng(0)
        L, n = 40, 22
        X = ([(rng.random(L) > 0.7).astype(int) for _ in range(n)]
             + [(rng.random(L) > 0.3).astype(int) for _ in range(n)])
        labels = _two_group_labels(n, n)
        res = order_information(X, labels, G=4, feature='transition',
                                shuffle='token', n_repeats=2, n_folds=5,
                                n_shuffles=60, random_state=1)
        row = res[res.comparison == 'HEALTHY_vs_RIL'].iloc[0]
        assert row['ci_low'] <= 0 <= row['ci_high']     # CI brackets 0
        assert bool(row['order_helps']) is False

    def test_runlength_shuffle_runs(self):
        # the run-length null path executes and returns valid bounded output
        rng = np.random.RandomState(2)
        X = [np.repeat(rng.randint(0, 4, size=15), rng.randint(1, 4, size=15))
             for _ in range(30)]
        labels = _two_group_labels(15, 15)
        res = order_information(X, labels, G=4, feature='run_transition',
                                shuffle='runlength', n_repeats=2, n_folds=5,
                                n_shuffles=20, random_state=1)
        assert set(res.columns) == EXPECTED_COLS
        assert ((res['auc_intact'] >= 0) & (res['auc_intact'] <= 1)).all()
