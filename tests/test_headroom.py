"""Tests for the frequency-headroom screen (PAPER_TODO §2, Phase 2).

Three groups of invariants:

- **Alignment.** ``headroom_table`` must produce exactly the rows ``order_information``
  produces, by the same recipe on the same folds — otherwise ``hist_auc`` and
  ``auc_intact`` are not comparable and the screen cannot gate the null.
- **Shuffle-invariance.** Both the screen *and* the restriction must be untouched by an
  order shuffle. This is what makes gating the null on them free of selection bias, so it
  is pinned here as an executable test rather than left as an argument in a docstring.
- **Construction.** The restriction must preserve relative order, produce a contiguous
  alphabet, and never let a real action inherit the background symbol.

Small ``n_shuffles``/``n_repeats`` throughout for speed (the module docstring's own idiom).
"""
import numpy as np
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold

from smartflat.features.symbolic_barycenter.evaluation import (
    _make_clf, _nested_cv_auc, _pairwise_subsets, histogram_features)
from smartflat.features.symbolic_barycenter.generalization.headroom import (
    class_vocabularies, headroom_table, pooled_markov_surrogate,
    restrict_to_shared_vocabulary)
from smartflat.features.symbolic_barycenter.generalization.synthetic import (
    make_order_discriminative_dataset)
from smartflat.features.symbolic_barycenter.order_evaluation import (
    order_information, order_shuffle_null)

G_TOY = 7   # 0 = background, 1..4 real actions, 5/6 class-exclusive markers


def _disjoint_vocab(n=8, L=12, seed=0):
    """Two classes with non-overlapping vocabularies -> frequency saturates (the 06m regime)."""
    rng = np.random.default_rng(seed)
    X = ([rng.choice([1, 2], size=L) for _ in range(n)]
         + [rng.choice([3, 4], size=L) for _ in range(n)])
    return X, np.array(['A'] * n + ['B'] * n, dtype=object), G_TOY


def _shared_vocab_with_markers(n=8, seed=0):
    """Shared actions {1,2} plus one class-exclusive marker each (5 for A, 6 for B).

    Every sequence is ``[0, 1, marker, 1, 2]``: deleting the marker abuts two runs of ``1``,
    so this fixture exercises the ``collapse_runs`` fork.
    """
    X = ([np.array([0, 1, 5, 1, 2]) for _ in range(n)]
         + [np.array([0, 2, 6, 2, 1]) for _ in range(n)])
    return X, np.array(['A'] * n + ['B'] * n, dtype=object), G_TOY


def test_rows_align_with_order_information():
    # class C is too small to stratify-split (3 < n_folds=5) -> both of its pairs must be
    # skipped by BOTH functions, identically.
    rng = np.random.default_rng(0)
    X = ([rng.choice([1, 2], size=12) for _ in range(8)]
         + [rng.choice([3, 4], size=12) for _ in range(8)]
         + [rng.choice([1, 3], size=12) for _ in range(3)])
    y = np.array(['A'] * 8 + ['B'] * 8 + ['C'] * 3, dtype=object)
    h = headroom_table(X, y, G_TOY, n_repeats=2, n_folds=5)
    o = order_information(X, y, G_TOY, n_repeats=2, n_folds=5, n_shuffles=2, random_state=42)
    assert list(h['comparison']) == list(o['comparison']) == ['A_vs_B']


def test_hist_auc_matches_manual_recipe_exactly():
    X, y, G = _disjoint_vocab()
    row = headroom_table(X, y, G, n_repeats=2, n_folds=5).iloc[0]
    comp, mask, yb = next(iter(_pairwise_subsets(np.asarray(y, dtype=object))))
    Xc = [np.asarray(X[i]) for i in np.flatnonzero(mask)]
    splits = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=2, random_state=42).split(np.zeros(len(Xc)), yb))
    pipe, grid = _make_clf('logreg')
    manual = float(np.mean(
        _nested_cv_auc(histogram_features(Xc, G), yb, splits, pipe, grid)))
    assert row['comparison'] == comp
    assert row['hist_auc'] == manual          # exact: same features, same folds, same grid


@pytest.mark.parametrize('kind', ['token', 'runlength'])
def test_headroom_is_shuffle_invariant(kind):
    # The screen must be blind to order -- this is precisely why gating the order-null on
    # it introduces no selection bias.
    X, y, G = _shared_vocab_with_markers()
    D_G = np.ones((G, G)) - np.eye(G)
    a = headroom_table(X, y, G, D_G, n_repeats=2, n_folds=5)
    b = headroom_table(order_shuffle_null(X, kind=kind, rng=np.random.default_rng(0)),
                       y, G, D_G, n_repeats=2, n_folds=5)
    for col in ('hist_auc', 'w_hist_dist', 'vocab_jaccard', 'n_shared',
                'n_marker_0', 'n_marker_1', 'headroom_band'):
        assert list(a[col]) == list(b[col]), col


@pytest.mark.parametrize('kind', ['token', 'runlength'])
def test_restriction_is_shuffle_invariant(kind):
    X, y, _ = _shared_vocab_with_markers()
    shuf = order_shuffle_null(X, kind=kind, rng=np.random.default_rng(0))
    _, _, _, _, i0 = restrict_to_shared_vocabulary(X, y, ('A', 'B'))
    _, _, _, _, i1 = restrict_to_shared_vocabulary(shuf, y, ('A', 'B'))
    assert np.array_equal(i0['shared'], i1['shared'])


def test_saturated_disjoint_vocab_hist_auc_is_one():
    X, y, G = _disjoint_vocab()
    D_G = np.ones((G, G)) - np.eye(G)
    row = headroom_table(X, y, G, D_G, n_repeats=2, n_folds=5).iloc[0]
    assert row['hist_auc'] == 1.0                  # the 06m ceiling: zero headroom
    assert row['headroom_band'] == 'saturated'
    assert row['vocab_jaccard'] == 0.0 and row['n_shared'] == 0
    assert row['w_hist_dist'] > 0                  # disjoint histograms are far apart


def test_identical_multiset_is_floor():
    # every sequence shares one exact multiset -> the histogram is a constant feature.
    X, y, G, D_G = make_order_discriminative_dataset(
        n_per_group=8, L=24, G=4, jitter=0.0, seed=0)
    row = headroom_table(X, y, G, D_G, n_repeats=2, n_folds=5).iloc[0]
    assert abs(row['hist_auc'] - 0.5) < 1e-6
    assert row['headroom_band'] == 'floor'         # a null here would be vacuous, not evidence


def test_headroom_handles_missing_D_G():
    X, y, G = _disjoint_vocab()
    assert np.isnan(headroom_table(X, y, G, None, n_repeats=2, n_folds=5)['w_hist_dist']).all()


def test_class_vocabularies_excludes_background_and_honours_min_frac():
    X, y, _ = _shared_vocab_with_markers(n=4)
    X = list(X) + [np.array([0, 1, 2, 3])]         # one stray '3' in a 5th A sequence
    y = np.append(np.asarray(y, dtype=object), 'A')
    assert 0 not in class_vocabularies(X, y)['A']  # background never in a vocabulary
    assert 3 in class_vocabularies(X, y)['A']
    assert 3 not in class_vocabularies(X, y, min_class_frac=0.5)['A']   # 1/5 < 0.5


def test_restrict_preserves_relative_order():
    X, y, _ = _shared_vocab_with_markers(n=3)
    X_r, _, _, _, info = restrict_to_shared_vocabulary(
        X, y, ('A', 'B'), collapse_runs=False, min_len=2)
    inv = {v: k for k, v in info['remap'].items()}
    # decoded back to original ids, each output must be a subsequence of its input
    for out, src in zip(X_r, X):
        it = iter(list(src))
        assert all(any(int(s) == int(t) for t in it) for s in (inv[v] for v in out))


def test_restrict_compact_alphabet_contiguous():
    X, y, _ = _shared_vocab_with_markers()
    X_r, _, G_c, _, info = restrict_to_shared_vocabulary(X, y, ('A', 'B'))
    assert np.array_equal(info['shared'], np.array([1, 2]))          # markers 5/6 dropped
    assert sorted(info['remap'].values()) == list(range(G_c))        # no gaps
    assert set(np.concatenate(X_r).tolist()) <= set(range(G_c))


def test_restrict_background_pinned_to_zero():
    # With background kept, it MUST remap to 0 -- compute_distance_matrix pins row/col 0 to
    # the row/col max, so a real action landing on 0 would silently inherit that geometry.
    X, y, G = _shared_vocab_with_markers()
    D_G = np.arange(G * G, dtype=float).reshape(G, G)
    _, _, G_c, D_c, info = restrict_to_shared_vocabulary(
        X, y, ('A', 'B'), D_G=D_G, keep_background=True)
    assert info['remap'][0] == 0
    assert np.array_equal(info['shared'], np.array([0, 1, 2]))
    # D_G must be SLICED from the original, never rebuilt on the remapped alphabet
    assert D_c.shape == (G_c, G_c)
    assert np.array_equal(D_c, D_G[np.ix_(info['shared'], info['shared'])])


def test_restrict_collapse_runs_flag():
    # '[0,1,5,1,2]' -> dropping marker 5 abuts two runs of '1'.
    X, y, _ = _shared_vocab_with_markers(n=3)
    keep, _, _, _, _ = restrict_to_shared_vocabulary(
        X, y, ('A', 'B'), collapse_runs=False, min_len=2)
    coll, _, _, _, _ = restrict_to_shared_vocabulary(
        X, y, ('A', 'B'), collapse_runs=True, min_len=2)
    assert keep[0].tolist() == [0, 0, 1]     # 1,1,2 remapped -> repeat retained
    assert coll[0].tolist() == [0, 1]        # ... and collapsed away
    assert len(keep[0]) > len(coll[0])


def test_restrict_rejects_pair_without_shared_vocabulary():
    X, y, _ = _disjoint_vocab()
    with pytest.raises(ValueError, match='shared symbol'):
        restrict_to_shared_vocabulary(X, y, ('A', 'B'))


def test_surrogate_preserves_multiset_exactly():
    X, y, G = _shared_vocab_with_markers()
    S = pooled_markov_surrogate(X, G, random_state=0)
    assert len(S) == len(X)
    for s, x in zip(S, X):
        assert np.array_equal(np.bincount(s, minlength=G),
                              np.bincount(np.asarray(x, dtype=int), minlength=G))


def test_surrogate_destroys_order_label_association():
    # Order resampled from a class-POOLED model -> the null must not reject on it.
    X, y, G, _ = make_order_discriminative_dataset(
        n_per_group=12, L=16, G=4, jitter=0.0, seed=0)
    S = pooled_markov_surrogate(X, G, random_state=0)
    row = order_information(S, y, G, feature='transition', shuffle='token',
                            n_repeats=2, n_folds=5, n_shuffles=30, random_state=1).iloc[0]
    assert bool(row['order_helps']) is False
