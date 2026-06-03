"""Unit tests for barycenter baselines (PAPER_BRIDGE items A5-A7, B1-B3).

Tests verify output shape, value validity, and basic properties
for all six baseline methods plus embedding utilities.
"""

import numpy as np
import pandas as pd
import pytest

from smartflat.features.symbolic_barycenter.baselines import (
    barycenter_dba_dtw,
    barycenter_edit_median,
    barycenter_k_medoid,
    barycenter_majority_voting,
    barycenter_soft_dtw,
    barycenter_wasserstein,
    baseline_significance_tests,
    default_baseline_methods,
    dist_dtw,
    dist_edit,
    dist_hamming,
    dist_rtwe,
    dist_soft_dtw,
    dist_wasserstein_hist,
    embed_symbolic_to_real,
    evaluate_baselines,
    make_patient_control_labels,
    ordinal_cost_matrix,
    project_real_to_symbolic,
)


@pytest.fixture
def D5():
    """5x5 symmetric distance matrix with zero diagonal."""
    rng = np.random.RandomState(42)
    D = rng.rand(5, 5)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


@pytest.fixture
def X_sym():
    """Small set of symbolic sequences (4 sequences, length 20, alphabet 0-4)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 5, size=(4, 20)).astype(np.int64)


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_embed_shape(self, X_sym, D5):
        X_emb = embed_symbolic_to_real(X_sym, D5)
        assert X_emb.shape == (4, 20, 5)

    def test_project_roundtrip(self, D5):
        """Projecting D_G rows back should recover original symbols."""
        symbols = np.array([0, 1, 2, 3, 4])
        embedded = D5[symbols]  # (5, 5) — each row is D_G[s,:]
        recovered = project_real_to_symbolic(embedded, D5)
        np.testing.assert_array_equal(recovered, symbols)

    def test_project_batch(self, X_sym, D5):
        X_emb = embed_symbolic_to_real(X_sym, D5)
        X_proj = project_real_to_symbolic(X_emb, D5)
        np.testing.assert_array_equal(X_proj, X_sym)


# ---------------------------------------------------------------------------
# Tier A baselines
# ---------------------------------------------------------------------------


class TestDBADTW:
    def test_output_shape(self, X_sym, D5):
        bary = barycenter_dba_dtw(X_sym, D5, max_iters=3, random_state=42)
        assert bary.shape == (20,)

    def test_symbols_in_vocabulary(self, X_sym, D5):
        bary = barycenter_dba_dtw(X_sym, D5, max_iters=3, random_state=42)
        assert all(0 <= s < D5.shape[0] for s in bary)

    def test_no_nan(self, X_sym, D5):
        bary = barycenter_dba_dtw(X_sym, D5, max_iters=3, random_state=42)
        assert not np.any(np.isnan(bary))


class TestSoftDTW:
    def test_output_shape(self, X_sym, D5):
        bary = barycenter_soft_dtw(X_sym, D5, gamma=1.0, max_iter=3, random_state=42)
        assert bary.shape == (20,)

    def test_symbols_in_vocabulary(self, X_sym, D5):
        bary = barycenter_soft_dtw(X_sym, D5, gamma=1.0, max_iter=3, random_state=42)
        assert all(0 <= s < D5.shape[0] for s in bary)

    def test_no_nan(self, X_sym, D5):
        bary = barycenter_soft_dtw(X_sym, D5, gamma=1.0, max_iter=3, random_state=42)
        assert not np.any(np.isnan(bary))


class TestEditMedian:
    def test_output_valid_symbols(self, X_sym):
        bary = barycenter_edit_median(X_sym, n_alphabet=5, max_iter=3)
        assert all(0 <= s < 5 for s in bary)

    def test_output_not_empty(self, X_sym):
        bary = barycenter_edit_median(X_sym, n_alphabet=5, max_iter=3)
        assert len(bary) > 0

    def test_output_length_reasonable(self, X_sym):
        """Median should be roughly same length as inputs (can vary due to edits)."""
        bary = barycenter_edit_median(X_sym, n_alphabet=5, max_iter=3)
        assert 5 <= len(bary) <= 40  # generous bounds


# ---------------------------------------------------------------------------
# Tier B baselines
# ---------------------------------------------------------------------------


class TestWasserstein:
    def test_sums_to_one(self, X_sym, D5):
        bary = barycenter_wasserstein(X_sym, D5, reg=0.01)
        assert bary.sum() == pytest.approx(1.0, abs=1e-6)

    def test_shape(self, X_sym, D5):
        bary = barycenter_wasserstein(X_sym, D5)
        assert bary.shape == (D5.shape[0],)

    def test_non_negative(self, X_sym, D5):
        bary = barycenter_wasserstein(X_sym, D5)
        assert np.all(bary >= -1e-10)


class TestKMedoid:
    def test_returns_valid_index(self):
        D = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=float)
        idx = barycenter_k_medoid(D)
        assert 0 <= idx < 3

    def test_correct_medoid(self):
        """Sequence 1 has smallest sum of distances -> should be medoid."""
        D = np.array([[0, 1, 10], [1, 0, 2], [10, 2, 0]], dtype=float)
        assert barycenter_k_medoid(D) == 1


class TestMajorityVoting:
    def test_output_shape(self, X_sym):
        bary = barycenter_majority_voting(X_sym)
        assert bary.shape == (20,)

    def test_symbols_in_vocabulary(self, X_sym):
        bary = barycenter_majority_voting(X_sym)
        assert all(0 <= s < 5 for s in bary)

    def test_unanimous_columns(self):
        """When all sequences agree, mode should match."""
        X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.int64)
        bary = barycenter_majority_voting(X)
        np.testing.assert_array_equal(bary, [1, 2, 3])


# ---------------------------------------------------------------------------
# TWE ablation: ordinal cost matrix
# ---------------------------------------------------------------------------


class TestOrdinalCost:
    def test_shape_and_symmetry(self):
        D = ordinal_cost_matrix(5)
        assert D.shape == (5, 5)
        np.testing.assert_array_equal(D, D.T)

    def test_zero_diagonal(self):
        D = ordinal_cost_matrix(6)
        assert np.all(np.diag(D) == 0)

    def test_values_are_absolute_index_gap(self):
        D = ordinal_cost_matrix(4)
        expected = np.array([[0, 1, 2, 3],
                             [1, 0, 1, 2],
                             [2, 1, 0, 1],
                             [3, 2, 1, 0]], dtype=float)
        np.testing.assert_array_equal(D, expected)


# ---------------------------------------------------------------------------
# Native classification distances
# ---------------------------------------------------------------------------


class TestNativeDistances:
    def test_dtw_identity_and_symmetry(self, X_sym, D5):
        a, b = X_sym[0], X_sym[1]
        assert dist_dtw(a, a, D5) == pytest.approx(0.0, abs=1e-9)
        assert dist_dtw(a, b, D5) == pytest.approx(dist_dtw(b, a, D5))
        assert dist_dtw(a, b, D5) >= 0

    def test_soft_dtw_finite_and_symmetric(self, X_sym, D5):
        a, b = X_sym[0], X_sym[1]
        d_ab = dist_soft_dtw(a, b, D5, gamma=1.0)
        d_ba = dist_soft_dtw(b, a, D5, gamma=1.0)
        assert np.isfinite(d_ab)
        assert d_ab == pytest.approx(d_ba, abs=1e-6)

    def test_edit_identity_and_symmetry(self, X_sym):
        a, b = X_sym[0], X_sym[1]
        assert dist_edit(a, a) == 0
        assert dist_edit(a, b) >= 0
        assert dist_edit(a, b) == dist_edit(b, a)

    def test_hamming_range(self, X_sym):
        a, b = X_sym[0], X_sym[1]
        assert dist_hamming(a, a) == 0
        assert 0.0 <= dist_hamming(a, b) <= 1.0

    def test_wasserstein_hist_self_is_zero(self, X_sym, D5):
        seq = X_sym[0]
        G = D5.shape[0]
        M = D5 / D5.max()
        h = np.bincount(seq.astype(int), minlength=G).astype(float)
        h /= h.sum()
        assert dist_wasserstein_hist(seq, h, M) == pytest.approx(0.0, abs=1e-9)

    def test_rtwe_identity_and_symmetry(self, X_sym, D5):
        a, b = X_sym[0], X_sym[1]
        assert dist_rtwe(a, a, D5) == pytest.approx(0.0, abs=1e-6)
        assert dist_rtwe(a, b, D5) == pytest.approx(dist_rtwe(b, a, D5), abs=1e-6)
        assert dist_rtwe(a, b, D5) >= -1e-9

    def test_rtwe_ordinal_cost_runs(self, X_sym):
        """TWE ablation distance: rTWE with an ordinal cost is finite and >= 0."""
        a, b = X_sym[0], X_sym[1]
        D_ord = ordinal_cost_matrix(5)
        d = dist_rtwe(a, b, D_ord)
        assert np.isfinite(d) and d >= -1e-9


# ---------------------------------------------------------------------------
# Method registry + native-distance evaluation harness
# ---------------------------------------------------------------------------


@pytest.fixture
def labeled_data():
    """24 length-20 sequences over alphabet 0-4, 3 balanced groups, + D_pairwise."""
    rng = np.random.RandomState(0)
    n_per = 8
    groups = ['HEALTHY', 'TBI', 'RIL']
    X = rng.randint(0, 5, size=(n_per * 3, 20)).astype(np.int64)
    labels = np.array([g for g in groups for _ in range(n_per)], dtype=object)
    Dpw = rng.rand(n_per * 3, n_per * 3)
    Dpw = (Dpw + Dpw.T) / 2
    np.fill_diagonal(Dpw, 0.0)
    return X, labels, Dpw


class TestDefaultMethods:
    def test_six_baselines_present(self, D5):
        methods = default_baseline_methods(D5)
        assert set(methods) == {
            'dba_dtw', 'soft_dtw', 'edit_median',
            'wasserstein', 'k_medoid', 'majority_voting',
        }

    def test_specs_well_formed(self, D5):
        methods = default_baseline_methods(D5)
        for spec in methods.values():
            assert callable(spec['distance'])
            assert spec.get('kind') == 'medoid' or callable(spec['build'])


class TestEvaluateBaselines:
    def _cheap_methods(self):
        # Fast registry (no DBA/soft-DTW/numba) to exercise harness mechanics.
        return {
            'majority_voting': {
                'build': lambda X, seed: barycenter_majority_voting(X),
                'distance': dist_hamming,
            },
            'medoid': {'kind': 'medoid', 'distance': dist_hamming},
        }

    def test_schema_and_auc_range(self, labeled_data):
        X, labels, Dpw = labeled_data
        df = evaluate_baselines(
            X, labels, self._cheap_methods(), D_pairwise=Dpw,
            n_splits=2, n_inits=1, random_state=0,
        )
        assert list(df.columns) == ['method', 'split', 'init', 'comparison', 'auc']
        assert len(df) > 0
        assert df['auc'].between(0.0, 1.0).all()
        assert set(df['method'].unique()) == {'majority_voting', 'medoid'}
        assert set(df['comparison'].unique()) == {
            'HEALTHY_vs_RIL', 'HEALTHY_vs_TBI', 'RIL_vs_TBI',
        }

    def test_medoid_requires_d_pairwise(self, labeled_data):
        X, labels, _ = labeled_data
        with pytest.raises(ValueError):
            evaluate_baselines(
                X, labels, {'medoid': {'kind': 'medoid', 'distance': dist_hamming}},
                D_pairwise=None, n_splits=2, n_inits=1,
            )


class TestPooledLabels:
    def test_relabel(self):
        labels = np.array(['HEALTHY', 'TBI', 'RIL', 'HEALTHY'], dtype=object)
        pooled = make_patient_control_labels(labels)
        np.testing.assert_array_equal(
            pooled,
            np.array(['CONTROL', 'PATIENT', 'PATIENT', 'CONTROL'], dtype=object),
        )

    def test_pooled_comparison_in_evaluation(self, labeled_data):
        X, labels, _ = labeled_data
        pooled = make_patient_control_labels(labels)
        methods = {
            'maj': {
                'build': lambda X, seed: barycenter_majority_voting(X),
                'distance': dist_hamming,
            },
        }
        df = evaluate_baselines(
            X, pooled, methods, n_splits=2, n_inits=1, random_state=0,
        )
        assert set(df['comparison'].unique()) == {'CONTROL_vs_PATIENT'}


# ---------------------------------------------------------------------------
# Paired Wilcoxon + Benjamini-Hochberg significance
# ---------------------------------------------------------------------------


class TestSignificance:
    def _df(self):
        rng = np.random.RandomState(1)
        rows = []
        for split in range(8):
            base = 0.80 + rng.uniform(-0.02, 0.02)
            rows.append({'method': 'tw_twe', 'split': split, 'init': 0,
                         'comparison': 'A_vs_B', 'auc': base})
            rows.append({'method': 'worse', 'split': split, 'init': 0,
                         'comparison': 'A_vs_B', 'auc': base - 0.2})
            rows.append({'method': 'tie', 'split': split, 'init': 0,
                         'comparison': 'A_vs_B', 'auc': base})
        return pd.DataFrame(rows)

    def test_columns(self):
        out = baseline_significance_tests(self._df(), reference='tw_twe')
        for col in ['method', 'comparison', 'mean_auc', 'ref_mean_auc',
                    'delta', 'p_value', 'p_value_bh', 'significant']:
            assert col in out.columns

    def test_worse_method_flagged(self):
        out = baseline_significance_tests(self._df(), reference='tw_twe')
        row = out.set_index('method').loc['worse']
        assert row['delta'] > 0
        assert bool(row['significant'])

    def test_tied_method_not_flagged(self):
        out = baseline_significance_tests(self._df(), reference='tw_twe')
        row = out.set_index('method').loc['tie']
        assert not bool(row['significant'])
