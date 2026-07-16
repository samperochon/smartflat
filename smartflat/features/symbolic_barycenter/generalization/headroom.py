"""Frequency-control for the ΔAUC order-shuffle null: *measure* headroom, *construct* it.

The §15 null (:func:`..order_evaluation.order_information`) reports ``delta_auc =
AUC_intact - mean(AUC_shuffled)``. The shuffle preserves each sequence's symbol multiset,
so the null asks "does order add anything **beyond frequency**?". That question is only
answerable when frequency leaves room for an answer.

**It often does not.** Notebook 06m ran the null on Breakfast's top-6 activities and got
``order_helps`` 0/60 -- but with ``auc_intact = 1.0`` on *all 60 cells*. The activities have
near-disjoint action vocabularies, so a bare unigram histogram already separates them
perfectly. With ``AUC_intact`` pinned at the ceiling, ``delta_auc`` is 0 by arithmetic and
the null **cannot fire even if order carried signal**. The 0/60 is vacuous, not evidence.
This module exists so that failure mode is *detected before* the null is run, not
rationalised afterwards.

:func:`headroom_table` measures, per class pair, the out-of-fold **unigram-histogram AUC**
-- the exact frequency channel the null holds fixed -- and assigns a ``headroom_band``:

- ``saturated`` (``hist_auc >= sat_hi``) -- frequency already separates the classes; the
  null has no room. The 06m failure.
- ``floor`` (``hist_auc <= floor_lo``) -- frequency separates nothing. Order *could* carry
  everything, but a null here is **ambiguous**: "order doesn't help" is indistinguishable
  from "these classes are exchangeable". The 06m failure mirrored, and the reason a
  one-sided ``1 - hist_auc`` score is the wrong instrument.
- ``sweet`` -- classes are distinguishable *and* frequency does not saturate. Only here is
  a ΔAUC null informative in both directions.

:func:`restrict_to_shared_vocabulary` *constructs* a sweet-band task from a saturated one
by dropping each pair's class-exclusive "marker" symbols, leaving only the actions both
classes perform. It asks the narrower, honest question: *given only the shared actions,
does their **order** distinguish the classes?*

**Gating the null on ``hist_auc`` is legitimate, and so is the restriction.** Both are
deterministic functions of shuffle-*invariant* quantities -- a sequence's histogram *is*
its multiset, and its vocabulary is that multiset's support; labels and splits (built from
``y`` and ``n`` alone) are likewise untouched by the shuffle. So both are *conditioning
variables*, not statistics: the conditional shuffle distribution is unchanged and no
selection bias enters. ``tests/test_headroom.py`` pins this as two executable invariants.

:func:`pooled_markov_surrogate` is the calibration decoy. The shuffle null tests *"order is
uniform given the multiset"*, which is strictly stronger than the question we care about,
*"order is independent of the label given the multiset"*. Real sequences violate the former
massively while possibly satisfying the latter -- and then shuffling **blurs** the frequency
signal (a sharp, structured transition matrix degrades into a noisy one) and ``delta_auc``
goes positive with zero order-label association. The surrogate holds each multiset exactly
while resampling order from a **class-pooled** model, so order carries provably no label
information by construction: running the *unmodified* null on it measures the probe's
empirical type-I rate at this dataset's own operating point.

PAPER_TODO §2. Additive: nothing here mutates the frozen §15/§17-20 harnesses.
"""
import numpy as np
import pandas as pd

from ..barycenter_quality import _w_hist
from ..evaluation import (
    _make_clf,
    _nested_cv_auc,
    _pairwise_subsets,
    histogram_features,
)


def _splits_like_order_information(n, y, n_folds, n_repeats, random_state):
    """The outer split list, built **byte-identically** to :func:`order_information`.

    Sharing the construction (same splitter, same ``np.zeros(n)`` dummy X, same seed) is
    what makes ``hist_auc`` and ``auc_intact`` a legitimate *paired* comparison on
    identical folds rather than two unrelated numbers.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(np.zeros(n), y))


def _band(auc, sat_hi, floor_lo):
    if auc >= sat_hi:
        return 'saturated'
    if auc <= floor_lo:
        return 'floor'
    return 'sweet'


def class_vocabularies(X, labels, *, background=0, min_class_frac=0.0):
    """``{class_label: sorted int array of symbols}`` -- the actions each class performs.

    ``background`` is excluded (it is the harness's reserved symbol 0, present everywhere
    and never discriminative). ``min_class_frac`` requires a symbol to appear in at least
    that fraction of the class's sequences before it counts as part of the vocabulary --
    a guard against a single stray annotation defining (and thereby un-dropping) a marker.
    """
    labels = np.asarray(labels, dtype=object)
    out = {}
    for lab in sorted(set(labels), key=str):
        seqs = [np.asarray(X[i], dtype=int) for i in np.flatnonzero(labels == lab)]
        n_seen = {}
        for s in seqs:
            for sym in np.unique(s):
                n_seen[int(sym)] = n_seen.get(int(sym), 0) + 1
        thresh = min_class_frac * len(seqs)
        out[lab] = np.array(
            sorted(s for s, c in n_seen.items() if s != background and c >= thresh),
            dtype=int)
    return out


def headroom_table(X, labels, G, D_G=None, *, classifier='logreg', n_repeats=3, n_folds=5,
                   random_state=42, background=0, sat_hi=0.95, floor_lo=0.55,
                   name='dataset'):
    """Per-class-pair frequency headroom: can the ΔAUC order-null fire at all?

    Rows are produced by the same :func:`..evaluation._pairwise_subsets` the null uses and
    are subject to the **same skip rule**, so they align one-to-one with
    :func:`..order_evaluation.order_information`'s rows on the same inputs.

    Parameters
    ----------
    X : ragged list of int sequences (or a 2-D ``(n, L)`` array).
    labels : array-like of class labels (a SDS2 vocabulary yields the three frozen
        comparisons; any other yields all unordered class pairs).
    G : int -- alphabet size (incl. background).
    D_G : ``(G, G)`` ground cost, optional. Supplies ``w_hist_dist``; ``None`` -> NaN.
    sat_hi, floor_lo : the two-sided band edges (see module docstring).

    Returns
    -------
    pd.DataFrame -- one row per comparison, columns: ``comparison, n, n_class0, n_class1,
    hist_auc, w_hist_dist, vocab_jaccard, n_shared, n_marker_0, n_marker_1, G,
    headroom_band, dataset``. ``hist_auc`` is the out-of-fold AUC of an L1-normalized
    unigram histogram -- the frequency channel the shuffle holds fixed, i.e. the exact
    reference ``auc_intact`` must beat for an order claim to mean anything.
    """
    labels = np.asarray(labels, dtype=object)
    pipe, grid = _make_clf(classifier)
    rows = []
    for comp, mask, y in _pairwise_subsets(labels):
        # identical skip rule to order_information -- otherwise the rows silently misalign
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_folds:
            continue
        idx = np.flatnonzero(mask)
        Xc = [np.asarray(X[i]) for i in idx]
        splits = _splits_like_order_information(
            len(Xc), y, n_folds, n_repeats, random_state)
        hist_auc = float(np.mean(
            _nested_cv_auc(histogram_features(Xc, G), y, splits, pipe, grid)))

        # vocabulary overlap + Wasserstein distance between the two class-mean histograms
        v = class_vocabularies(Xc, y, background=background)
        v0, v1 = set(v[0].tolist()), set(v[1].tolist())
        union = v0 | v1
        F = histogram_features(Xc, G)
        w = (_w_hist(F[y == 0].mean(0), F[y == 1].mean(0), D_G)
             if D_G is not None else np.nan)

        rows.append({
            'comparison': comp, 'n': len(Xc),
            'n_class0': int((y == 0).sum()), 'n_class1': int((y == 1).sum()),
            'hist_auc': hist_auc, 'w_hist_dist': w,
            'vocab_jaccard': len(v0 & v1) / len(union) if union else np.nan,
            'n_shared': len(v0 & v1),
            'n_marker_0': len(v0 - v1), 'n_marker_1': len(v1 - v0),
            'G': int(G), 'headroom_band': _band(hist_auc, sat_hi, floor_lo),
            'dataset': name,
        })
    return pd.DataFrame(rows)


def restrict_to_shared_vocabulary(X, labels, pair, *, D_G=None, background=0,
                                  keep_background=False, collapse_runs=True, min_len=2,
                                  min_class_frac=0.0):
    """Construct a frequency-controlled sub-task: keep only symbols **both** classes use.

    Drops each class's exclusive *marker* symbols -- the ones that let a bare histogram
    saturate -- and remaps the survivors to a compact ``0..G_c-1`` alphabet. The retained
    symbols keep their **relative order**, so the output is a genuine subsequence of the
    real annotation and any order conclusion is about real order.

    The honest reading of the resulting task is narrow: *given only the actions the two
    classes have in common, does the order of those actions distinguish them?* The
    restriction is label-derived, so the resulting ``hist_auc`` is the frequency AUC **of
    the sub-task**, not of the original task. (It is label-derived in the *anti*-leakage
    direction -- it removes perfectly class-diagnostic features rather than keeping them --
    and it is shuffle-invariant, so the null stays conditionally valid; see the module
    docstring.)

    Parameters
    ----------
    pair : ``(label_a, label_b)`` -- the two classes to restrict and retain.
    D_G : ``(G, G)`` ground cost, optional -> returned **sliced** to the kept symbols.
    keep_background : keep the background symbol in the shared set (default ``False``:
        background is present in every class and carries no contrast).
    collapse_runs : after deletion, two runs of the same symbol can become adjacent
        (``a b a`` -> ``a a`` once ``b`` is dropped). ``True`` re-collapses them, so the
        output stays a run-symbol sequence and the multiset changes accordingly; ``False``
        leaves the repeat, which preserves the multiset but creates a run of length 2 --
        which in turn makes ``runlength_shuffle`` differ from ``token_shuffle`` and
        ``run_transition_features`` differ from ``transition_features``. This flag is a
        real fork in what the 2x2 order grid measures, not a formatting detail.
    min_len : drop restricted sequences shorter than this (they carry no transition).

    Returns
    -------
    ``(X_r, labels_r, G_compact, D_G_compact, info)`` -- ``info`` carries ``shared``
    (original ids, sorted), ``markers`` (per class), ``remap``, ``n_dropped`` (per class),
    ``lengths``, and the flags.

    Notes
    -----
    ``D_G_compact`` is a **slice** ``D_G[np.ix_(shared, shared)]`` and must never be
    rebuilt via :func:`.action_segmentation.build_action_seg_ground_cost` on the remapped
    sequences: :func:`..vocab.compute_distance_matrix`'s ``max_rows_cols_pre`` pins row/col
    **0** to the row/col max because symbol 0 is the harness's reserved background. With
    ``keep_background=False`` the compact alphabet has no background, so rebuilding would
    silently give the lowest-numbered *real* action background's geometry.
    """
    labels = np.asarray(labels, dtype=object)
    a, b = pair
    vocs = class_vocabularies(X, labels, background=background,
                              min_class_frac=min_class_frac)
    for lab in (a, b):
        if lab not in vocs:
            raise KeyError(f"class {lab!r} not in labels; known: {sorted(vocs, key=str)}")
    va, vb = set(vocs[a].tolist()), set(vocs[b].tolist())
    shared = va & vb
    if keep_background:
        shared = shared | {int(background)}
    if len(shared) < 2:
        raise ValueError(
            f"{a!r} vs {b!r}: only {len(shared)} shared symbol(s) -- no order to test")
    shared = np.array(sorted(shared), dtype=int)
    remap = {int(s): i for i, s in enumerate(shared)}

    X_r, labels_r, n_dropped = [], [], {a: 0, b: 0}
    for i in np.flatnonzero(np.isin(labels, [a, b])):
        seq = np.array([remap[int(s)] for s in np.asarray(X[i], dtype=int)
                        if int(s) in remap], dtype=int)
        if collapse_runs and seq.size:
            seq = seq[np.concatenate(([True], seq[1:] != seq[:-1]))]
        if len(seq) < min_len:
            n_dropped[labels[i]] += 1
            continue
        X_r.append(seq)
        labels_r.append(labels[i])

    G_c = len(shared)
    assert sorted(remap.values()) == list(range(G_c)), 'compact alphabet must be contiguous'
    if keep_background:
        assert remap[int(background)] == 0, 'background must stay pinned to symbol 0'
    D_c = np.asarray(D_G)[np.ix_(shared, shared)] if D_G is not None else None
    info = {
        'shared': shared, 'remap': remap,
        'markers': {a: np.array(sorted(va - set(shared.tolist())), dtype=int),
                    b: np.array(sorted(vb - set(shared.tolist())), dtype=int)},
        'n_dropped': n_dropped, 'lengths': np.array([len(s) for s in X_r], dtype=int),
        'keep_background': keep_background, 'collapse_runs': collapse_runs,
    }
    return X_r, np.array(labels_r, dtype=object), G_c, D_c, info


def pooled_markov_surrogate(X, G, *, random_state=0):
    """Calibration decoy: keep every multiset exactly, resample order from a **pooled** model.

    Pass the *pooled two-class* ``X`` of one comparison. A first-order transition model is
    fit on all of it (Laplace-smoothed), then each sequence is re-arranged by drawing,
    without replacement, from **its own** symbol multiset with the next symbol weighted by
    ``P(next | current)`` and by what remains. So:

    - every sequence's histogram is **bit-identical** to the original -> the
      frequency-label relationship (and hence ``hist_auc``) is fully preserved;
    - order is drawn from a model fit on **both** classes -> order is independent of the
      label *by construction*.

    Running the unmodified :func:`..order_evaluation.order_information` on the result must
    therefore fail to reject; the rate at which it *does* reject is the probe's empirical
    type-I rate at this dataset's operating point. This matters because the shuffle null is
    anti-conservative when frequency signal is present (module docstring), and the artifact
    is regime-dependent -- so it has to be calibrated against the real data's own
    structure rather than an arbitrary synthetic.
    """
    rng = np.random.default_rng(random_state)
    T = np.ones((G, G), dtype=np.float64)          # Laplace prior: never a zero row
    for s in X:
        s = np.asarray(s, dtype=int)
        for u, v in zip(s[:-1], s[1:]):
            T[u, v] += 1.0

    out = []
    for s in X:
        remaining = np.bincount(np.asarray(s, dtype=int), minlength=G).astype(float)
        avail = np.flatnonzero(remaining)
        cur = int(rng.choice(avail, p=remaining[avail] / remaining[avail].sum()))
        remaining[cur] -= 1
        seq = [cur]
        while remaining.sum() > 0:
            avail = np.flatnonzero(remaining)
            w = T[cur, avail] * remaining[avail]
            cur = int(rng.choice(avail, p=w / w.sum()))
            remaining[cur] -= 1
            seq.append(cur)
        out.append(np.array(seq, dtype=int))
    return out
