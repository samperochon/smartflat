"""Unit tests for :mod:`smartflat.features.symbolic_barycenter.structure_metrics`.

Hand-checked values on tiny sequences (a perfectly periodic one, a maximally perseverative
one, a maximally diverse one), edge/NaN-safety, vocabulary-agnosticism (relabel invariance),
an LZ76 reference value, and smoke tests for the aggregator, group-stats, and the
leakage-guarded incremental harness.
"""

import numpy as np
import pandas as pd
import pytest

from smartflat.features.symbolic_barycenter import structure_metrics as sm


PERIODIC = np.array([0, 1, 0, 1, 0, 1])   # switch every step, background=0 half the time
PERSEV = np.array([2, 2, 2, 2])           # one sustained run, no background
UNIFORM = np.array([1, 2, 3, 4, 5, 6])    # all distinct, no repeats
EMPTY = np.array([], dtype=int)
SINGLE = np.array([7])
G = 8


class TestPeriodic:
    def test_perseveration_and_fragmentation(self):
        assert sm.immediate_repeat_rate(PERIODIC) == 0.0
        assert sm.switch_rate(PERIODIC) == 1.0
        assert sm.n_runs(PERIODIC) == 6.0
        assert sm.fragmentation_index(PERIODIC) == 3.0   # 6 runs / 2 distinct

    def test_run_length(self):
        assert sm.run_length_mean(PERIODIC) == 1.0
        assert sm.run_length_cv(PERIODIC) == 0.0
        assert sm.run_length_max(PERIODIC) == 1.0

    def test_dwell(self):
        assert sm.dwell_mean_over_states(PERIODIC, G, 0) == 1.0
        assert sm.dwell_cv_over_states(PERIODIC, G, 0) == 0.0

    def test_background_and_drift(self):
        assert sm.background_fraction(PERIODIC, 0) == 0.5
        assert sm.halves_tv_distance(PERIODIC, G, 0) == pytest.approx(1 / 3)
        assert sm.halves_js_divergence(PERIODIC, G, 0) == pytest.approx(0.081704, abs=1e-5)

    def test_grammar(self):
        # deterministic alternation -> zero transition entropy
        assert sm.transition_entropy(PERIODIC, G) == 0.0
        assert sm.bigram_coverage(PERIODIC, G) == pytest.approx(2 / (G * G))
        # LZ76 parse of [0,1,0,1,0,1] -> 3 phrases (hand-traced)
        assert sm._lz76_phrase_count(sm._rle(PERIODIC)[0]) == 3
        assert sm.lz76_complexity(PERIODIC) == pytest.approx(3 * np.log(6) / np.log(2) / 6)


class TestPerseverative:
    def test_perseveration(self):
        assert sm.immediate_repeat_rate(PERSEV) == 1.0
        assert sm.switch_rate(PERSEV) == 0.0
        assert sm.n_runs(PERSEV) == 1.0
        assert sm.fragmentation_index(PERSEV) == 1.0

    def test_run_length_and_dwell(self):
        assert sm.run_length_mean(PERSEV) == 4.0
        assert sm.run_length_max(PERSEV) == 4.0
        assert sm.run_length_cv(PERSEV) == 0.0
        assert sm.dwell_mean_over_states(PERSEV, G, 0) == 4.0

    def test_background_and_grammar_undefined(self):
        assert sm.background_fraction(PERSEV, 0) == 0.0
        # a single collapsed symbol has no transitions / bigrams
        assert np.isnan(sm.transition_entropy(PERSEV, G))
        assert np.isnan(sm.bigram_coverage(PERSEV, G))
        assert sm.lz76_complexity(PERSEV) == 0.0
        assert sm.halves_tv_distance(PERSEV, G, 0) == 0.0
        assert sm.halves_js_divergence(PERSEV, G, 0) == 0.0


class TestUniform:
    def test_max_diversity(self):
        assert sm.fragmentation_index(UNIFORM) == 1.0   # 6 runs / 6 distinct
        assert sm.switch_rate(UNIFORM) == 1.0
        assert sm.transition_entropy(UNIFORM, G) == 0.0  # each source seen once, deterministic
        assert sm._lz76_phrase_count(sm._rle(UNIFORM)[0]) == 6
        assert sm.lz76_complexity(UNIFORM) == pytest.approx(1.0)
        # disjoint halves -> maximal drift
        assert sm.halves_tv_distance(UNIFORM, G, 0) == 1.0
        assert sm.halves_js_divergence(UNIFORM, G, 0) == pytest.approx(1.0)


class TestEdgeAndNaN:
    def test_empty(self):
        for f in ('n_runs', 'switch_rate', 'run_length_mean', 'background_fraction',
                  'lz76_complexity'):
            assert np.isnan(getattr(sm, f)(EMPTY))
        assert np.isnan(sm.transition_entropy(EMPTY, G))

    def test_single(self):
        assert np.isnan(sm.switch_rate(SINGLE))
        assert np.isnan(sm.immediate_repeat_rate(SINGLE))
        assert sm.n_runs(SINGLE) == 1.0
        assert sm.background_fraction(SINGLE, 0) == 0.0
        assert np.isnan(sm.transition_entropy(SINGLE, G))

    def test_switch_is_one_minus_repeat(self):
        for seq in (PERIODIC, PERSEV, UNIFORM, np.array([0, 0, 1, 2, 2, 2, 1, 0])):
            assert sm.switch_rate(seq) == pytest.approx(1.0 - sm.immediate_repeat_rate(seq))


class TestVocabularyAgnostic:
    """Pure-structure metrics are invariant to a background-preserving relabeling."""

    STRUCT = ('switch_rate', 'n_runs', 'run_length_mean', 'run_length_cv', 'run_length_max',
              'fragmentation_index', 'immediate_repeat_rate')

    def _relabel(self, seq):
        # bijective relabel of non-background codes (0 stays background)
        return np.where(seq == 1, 5, np.where(seq == 2, 9, seq))

    def test_relabel_invariance_pure_structure(self):
        for seq in (PERIODIC, PERSEV, np.array([0, 1, 1, 2, 0, 2, 2])):
            r = self._relabel(seq)
            for f in self.STRUCT:
                a, b = getattr(sm, f)(seq), getattr(sm, f)(r)
                assert (np.isnan(a) and np.isnan(b)) or a == b
            gg = 12
            assert sm.transition_entropy(seq, gg) == pytest.approx(
                sm.transition_entropy(r, gg), nan_ok=True) or np.isnan(sm.transition_entropy(seq, gg))
            assert sm.lz76_complexity(seq) == pytest.approx(sm.lz76_complexity(r), nan_ok=True) \
                or np.isnan(sm.lz76_complexity(seq))

    def test_background_metrics_not_relabel_invariant(self):
        # background_fraction depends on WHICH code is background -- not a pure-structure metric
        seq = np.array([0, 1, 1, 1])
        assert sm.background_fraction(seq, 0) == 0.25
        assert sm.background_fraction(seq, 1) == 0.75


class TestLZ76Reference:
    def test_reference_phrase_count(self):
        # pinned Kaspar-Schuster-style binary reference (validates the parser, not just self-consistency)
        ref = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
                        0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
        assert sm._lz76_phrase_count(ref) == 10

    def test_degenerate_counts(self):
        assert sm._lz76_phrase_count(np.array([], int)) == 0
        assert sm._lz76_phrase_count(np.array([4])) == 1
        assert sm._lz76_phrase_count(np.array([4, 4, 4])) == 2  # '4' | '44'


class TestAggregator:
    def _df(self):
        return pd.DataFrame({
            'int_cat_segm_embedding_labels': [PERIODIC, PERSEV, UNIFORM, SINGLE],
            'pathologie': ['HEALTHY', 'TBI', 'RIL', 'HEALTHY'],
        })

    def test_shape_and_schema(self):
        out = sm.compute_structure_metrics(self._df(), labels_col='pathologie')
        assert len(out) == 4
        expected = ['length'] + list(sm.METRIC_NAMES) + ['background_fraction', 'pathologie']
        assert list(out.columns) == expected
        # numeric columns are float; index preserved
        for c in sm.METRIC_NAMES:
            assert np.issubdtype(out[c].dtype, np.floating)

    def test_nan_safe_on_degenerate_rows(self):
        df = pd.DataFrame({'int_cat_segm_embedding_labels': [EMPTY, SINGLE]})
        out = sm.compute_structure_metrics(df, G=8)
        assert out.shape[0] == 2
        assert np.isnan(out.loc[0, 'switch_rate'])       # empty
        assert np.isnan(out.loc[1, 'transition_entropy'])  # single symbol

    def test_G_inference(self):
        out = sm.compute_structure_metrics(self._df())  # G inferred = 8 (max 7 + 1)
        assert out['bigram_coverage'].iloc[0] == pytest.approx(2 / (8 * 8))

    def test_structure_features_drops_redundant(self):
        feats, names = sm.structure_features([PERIODIC, PERSEV, UNIFORM], G=8)
        assert 'background_fraction' not in names
        assert 'immediate_repeat_rate' not in names
        assert 'switch_rate' in names
        assert feats.shape == (3, len(names))
        assert not np.isnan(feats).any()   # NaNs imputed to column mean


def _synthetic_cohort(n_per=14, seed=0):
    """Two structurally-distinct groups: HEALTHY = blocky (low switch), TBI = fragmented."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per):
        # blocky: long runs -> low switch_rate, low fragmentation
        seq = np.repeat(rng.integers(1, 6, size=8), rng.integers(6, 12, size=8))
        X.append(np.asarray(seq)); y.append('HEALTHY')
    for _ in range(n_per):
        # fragmented: many short runs -> high switch_rate
        seq = rng.integers(1, 6, size=rng.integers(60, 90))
        X.append(np.asarray(seq)); y.append('TBI')
    return X, np.array(y, dtype=object)


class TestGroupStats:
    def test_shifted_metric_flagged_null_metric_not(self):
        rng = np.random.default_rng(1)
        n = 30
        labels = np.array(['HEALTHY'] * n + ['RIL'] * n, dtype=object)
        shifted = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n)])  # clear separation
        null = rng.normal(0, 1, 2 * n)
        mdf = pd.DataFrame({'shifted': shifted, 'null': null, 'length': rng.normal(500, 10, 2 * n)})
        res = sm.structure_group_stats(mdf, labels, n_boot=500, random_state=0)
        hr = res[(res['metric'] == 'shifted') & (res['comparison'] == 'HEALTHY_vs_RIL')].iloc[0]
        assert hr['cliffs_delta'] > 0.5            # RIL > HEALTHY
        assert hr['cliffs_ci_low'] > 0             # CI excludes 0
        assert bool(hr['significant'])             # survives BH
        nr = res[(res['metric'] == 'null') & (res['comparison'] == 'HEALTHY_vs_RIL')].iloc[0]
        assert not bool(nr['significant'])
        # BH column present and jointly computed
        assert 'p_value_bh' in res.columns

    def test_reports_length_correlation(self):
        X, labels = _synthetic_cohort()
        mdf = sm.compute_structure_metrics(
            pd.DataFrame({'int_cat_segm_embedding_labels': X}), G=8)
        res = sm.structure_group_stats(mdf, labels, n_boot=200, random_state=0)
        assert 'spearman_length' in res.columns


class TestIncrementalStructure:
    def test_smoke_schema_and_paired_folds(self):
        X, labels = _synthetic_cohort(n_per=16)
        folds, summary = sm.evaluate_incremental_structure(
            X, labels, G=8, classifiers=('logreg',),
            n_repeats=2, n_folds=3, random_state=0, n_boot=200)
        assert set(summary.columns) == {
            'comparison', 'classifier', 'mean_auc_hist', 'mean_auc_struct',
            'mean_auc_both', 'mean_delta', 'delta_ci_low', 'delta_ci_high',
            'wilcoxon_p', 'n_folds'}
        for col in ('mean_auc_hist', 'mean_auc_struct', 'mean_auc_both'):
            assert summary[col].between(0, 1).all()
        # struct/hist/both scored on identical (repeat,fold) folds
        counts = folds.groupby(['comparison', 'classifier', 'feature_set']).size()
        assert counts.nunique() == 1
        assert (summary['delta_ci_low'] <= summary['mean_delta']).all()
        assert (summary['mean_delta'] <= summary['delta_ci_high']).all()


class TestLengthControlled:
    def test_smoke_schema_and_cis(self):
        X, labels = _synthetic_cohort(n_per=16)
        summary = sm.evaluate_structure_length_controlled(
            X, labels, G=8, classifiers=('logreg',),
            n_repeats=2, n_folds=3, random_state=0, n_boot=200)
        assert set(summary.columns) == {
            'comparison', 'classifier', 'auc_hist', 'auc_hist_len', 'auc_hist_struct',
            'auc_hist_len_struct', 'delta_struct', 'delta_struct_ci_low', 'delta_struct_ci_high',
            'delta_len', 'delta_len_ci_low', 'delta_len_ci_high', 'delta_struct_given_len',
            'delta_struct_given_len_ci_low', 'delta_struct_given_len_ci_high', 'n_folds'}
        for col in ('auc_hist', 'auc_hist_len', 'auc_hist_struct', 'auc_hist_len_struct'):
            assert summary[col].between(0, 1).all()
        # each delta point estimate is bracketed by its bootstrap CI
        for d in ('delta_struct', 'delta_len', 'delta_struct_given_len'):
            assert (summary[d + '_ci_low'] <= summary[d]).all()
            assert (summary[d] <= summary[d + '_ci_high']).all()
