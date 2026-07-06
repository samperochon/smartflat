"""Tests for the synthetic generalization datasets (PAPER_TODO §2.3, Phase 1).

Covers the two invariants that make them meaningful: the order-discriminative set has an
*exactly matched* per-sequence multiset (so any group signal is order, not frequency) and
its planted order is detected (ΔAUC ``ci_low > 0`` — the converse of SDS2's §15 0/15);
the D1 set recovers a known centre exactly at zero noise and degrades with noise.
"""
import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.generalization.synthetic import (
    make_order_discriminative_dataset, make_exact_recovery_dataset, uniform_ground_cost)
from smartflat.features.symbolic_barycenter.order_evaluation import order_information


def test_order_discriminative_multiset_is_identical_across_all_sequences():
    for grammar, expected in (('order', ['ascending', 'descending']),
                              ('dwell', ['blocked', 'interleaved'])):
        for jitter in (0.0, 0.5, 1.0):
            X, y, G, D_G = make_order_discriminative_dataset(
                n_per_group=15, L=48, G=4, grammar=grammar, jitter=jitter, seed=0)
            counts = np.array([np.bincount(s, minlength=G) for s in X])
            assert (counts == counts[0]).all()       # every sequence: exact same symbol counts
            assert counts[0].tolist() == [12, 12, 12, 12]
            assert sorted(set(y)) == expected
            assert D_G.shape == (G, G) and np.array_equal(np.diag(D_G), np.zeros(G))


def test_order_dwell_grammar_matches_run_length_dwell():
    # 'order' matches dwell exactly (same run-length multiset); 'dwell' does not.
    def run_lengths(s):
        keep = np.concatenate(([True], s[1:] != s[:-1]))
        idx = np.flatnonzero(np.append(keep, True))
        return sorted(np.diff(idx).tolist())
    Xo, _, _, _ = make_order_discriminative_dataset(grammar='order', jitter=0.0, seed=0)
    assert all(run_lengths(s) == run_lengths(Xo[0]) for s in Xo)   # dwell identical everywhere


def test_order_discriminative_validity_proof_planted_order_is_detected():
    # jitter=0: order is the only signal -> ci_low > 0 (the SDS2-negative counterpoint).
    X, y, G, _ = make_order_discriminative_dataset(n_per_group=20, L=24, G=4, jitter=0.0, seed=0)
    row = order_information(X, y, G, feature='transition', shuffle='token',
                            n_repeats=2, n_folds=5, n_shuffles=30, random_state=1).iloc[0]
    assert row['comparison'] == 'ascending_vs_descending'
    assert row['auc_intact'] > 0.9
    assert row['ci_low'] > 0
    assert bool(row['order_helps']) is True


def test_order_discriminative_jitter_validation():
    with pytest.raises(ValueError):
        make_order_discriminative_dataset(L=48, G=5)     # 48 not divisible by 5
    with pytest.raises(ValueError):
        make_order_discriminative_dataset(jitter=1.5)


def test_exact_recovery_zero_noise_all_equal_center():
    regimes, center, G, D_G = make_exact_recovery_dataset(G=6, L=48, n_per=20,
                                                          noise_levels=(0.0, 0.15, 0.3), seed=0)
    X0, y0 = regimes[0.0]
    assert all(np.array_equal(x, center) for x in X0)     # exact-recovery target at p=0
    assert len(set(y0)) == 1 and y0[0] == 'regime_0.0'
    # per-position majority recovers the centre exactly at p=0 (all members identical)
    stacked = np.array(X0)
    mode = np.array([np.bincount(stacked[:, t]).argmax() for t in range(stacked.shape[1])])
    assert np.array_equal(mode, center)


def test_exact_recovery_noise_increases_mismatch_monotonically():
    regimes, center, G, _ = make_exact_recovery_dataset(G=6, L=60, n_per=40,
                                                        noise_levels=(0.0, 0.1, 0.25, 0.4), seed=0)
    frac = [np.mean([(x != center).mean() for x in regimes[p][0]]) for p in (0.0, 0.1, 0.25, 0.4)]
    assert frac[0] == 0.0
    assert frac[1] < frac[2] < frac[3]                    # more noise -> more mismatch
    assert abs(frac[3] - 0.4) < 0.1                       # mismatch tracks the requested rate
