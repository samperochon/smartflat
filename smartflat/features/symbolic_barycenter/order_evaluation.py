"""Frequency-preserving order-shuffle null: how much group signal lives in *order*.

The barycenter paper's premise is that symbol *order* carries group-discriminative
structure beyond symbol *frequency*. The prior probe (``baselines.evaluate_incremental_
ordering``) added a high-dimensional bigram block on top of the unigram histogram and
read the incremental held-out AUC -- a probe that overfits (``G*G`` features vs n~60-120)
and conflates "does ordering help" with "does this representation help".

This module is the higher-leverage, self-calibrating probe: a **frequency-preserving
order-shuffle null**. The best order-aware classifier is scored on the *intact*
sequences (``AUC_intact``) and on many *within-sequence shuffles* that preserve each
sequence's symbol multiset exactly (``AUC_shuffled``). The reported

    delta_auc = AUC_intact - mean(AUC_shuffled)

isolates the contribution of *order alone*, because the shuffle holds frequency fixed.
Crucially the **same pipeline** is applied to intact and shuffled data, so classifier
optimism/overfitting bias is shared and cancels in ``delta_auc`` -- the null is its own
control. A 95% CI on ``delta_auc`` is read off the shuffle distribution.

Two shuffle nulls, both per-sequence (so the unigram histogram is exactly invariant):

- ``token_shuffle``     -- uniform within-sequence permutation. Preserves the symbol
  multiset; destroys order **and** run-lengths/dwell. ``delta_auc`` against it measures
  *all* structure beyond bare frequency (transitions + dwell).
- ``runlength_shuffle`` -- run-length-encode, permute the *order of runs*, concatenate.
  Preserves the multiset **and** the per-symbol run-length (dwell-time) distribution;
  destroys only the sequencing of runs. ``delta_auc`` against it measures *pure
  transition/sequencing* signal, holding frequency and dwell fixed.

The two coincide at segment-level (runs are length-1) and diverge at embedding-level
(long dwell runs).

**This is a measurement, not a goal.** ``delta_auc`` CI bracketing 0 is the expected,
paper-hardening result (order carries no group signal beyond frequency). A CI strictly
above 0 is a real order effect worth a pre-registered follow-up.

Reuses the leakage-guarded split semantics and nested-CV harness from
:mod:`smartflat.features.symbolic_barycenter.baselines`.
"""

import numpy as np
import pandas as pd

from smartflat.features.symbolic_barycenter.baselines import (
    _make_clf,
    _nested_cv_auc,
    _pairwise_subsets,
    _transition_matrix,
    transition_features,
)


def _rle(seq):
    """Run-length-encode a 1-D integer sequence.

    Returns ``(symbols, lengths)`` -- the value of each maximal run of identical
    consecutive symbols and its length. ``symbols`` is the "collapsed" sequence (the
    action grammar); ``lengths`` are the dwell times. Empty input -> two empty arrays.
    """
    seq = np.asarray(seq)
    if seq.size == 0:
        return seq.astype(int), np.array([], dtype=int)
    change = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [seq.size]))
    return seq[starts].astype(int), (ends - starts).astype(int)


def token_shuffle(seq, rng):
    """Uniform within-sequence permutation (preserves the symbol multiset exactly)."""
    return rng.permutation(np.asarray(seq))


def runlength_shuffle(seq, rng):
    """Permute the *order of runs*, preserving the (symbol, run-length) multiset.

    Run-length-encodes ``seq``, shuffles the run order, and re-concatenates. The
    unigram histogram **and** the per-symbol run-length (dwell-time) distribution are
    invariant; only the sequencing of runs is destroyed.
    """
    seq = np.asarray(seq)
    symbols, lengths = _rle(seq)
    if symbols.size == 0:
        return seq.copy()
    order = rng.permutation(symbols.size)
    return np.repeat(symbols[order], lengths[order])


_SHUFFLES = {'token': token_shuffle, 'runlength': runlength_shuffle}


def order_shuffle_null(X, kind='token', rng=None):
    """Apply a per-sequence shuffle to every sequence; each row's histogram is invariant.

    Parameters
    ----------
    X : sequence of 1-D int arrays (ragged list) or a 2-D ``(n, L)`` array.
    kind : {'token', 'runlength'}.
    rng : np.random.Generator, optional (a fresh default generator if None).

    Returns
    -------
    list of 1-D int arrays -- the shuffled sequences (same per-row symbol multiset).
    """
    if rng is None:
        rng = np.random.default_rng()
    fn = _SHUFFLES[kind]
    return [fn(np.asarray(seq), rng) for seq in X]


def run_transition_features(X, G):
    """Per-sequence flattened bigram transitions between consecutive *distinct runs*.

    Unlike :func:`baselines.transition_features` (frame-level bigrams, dominated by the
    self-transition diagonal at embedding length), this collapses each sequence to its
    run-symbol sequence (the action grammar) first, so the feature is **dwell-invariant**
    -- it depends only on *which action follows which*, not how long each lasts. This is
    the cleanest pure-sequencing probe: under ``runlength_shuffle`` it changes only via
    run re-ordering.

    Returns
    -------
    np.ndarray of shape (n_sequences, G*G) -- zero row only for single-run sequences.
    """
    feats = np.zeros((len(X), G * G), dtype=np.float64)
    for i, seq in enumerate(X):
        symbols, _ = _rle(np.asarray(seq))
        feats[i] = _transition_matrix(symbols, G).reshape(-1)
    return feats


_FEATURES = {'transition': transition_features, 'run_transition': run_transition_features}


def order_information(
    X, labels, G,
    feature='transition', shuffle='token', classifier='logreg',
    n_repeats=3, n_folds=5, n_shuffles=200, random_state=42,
):
    """Frequency-preserving order-shuffle null: CI'd ``delta_auc(order)`` per comparison.

    For each of the three paper comparisons (HEALTHY_vs_RIL, RIL_vs_TBI,
    CONTROL_vs_PATIENT -- via the reused, leakage-guarded :func:`baselines._pairwise_
    subsets`), scores an order-aware classifier on the intact sequences and on
    ``n_shuffles`` within-sequence shuffles (same fixed ``RepeatedStratifiedKFold`` split
    list, nested inner ``GridSearchCV``), then reports ``delta_auc = AUC_intact -
    mean(AUC_shuffled)`` with a 95% CI from the shuffle distribution.

    Parameters
    ----------
    X : ragged list of int sequences (or 2-D ``(n, L)`` array). Full-length is fine --
        the order-aware features are per-sequence and cheap.
    labels : array-like of group labels (HEALTHY/RIL/TBI).
    G : int -- alphabet size (incl. background).
    feature : {'transition', 'run_transition'}.
        ``transition``     -- frame-level bigram transitions (dwell-dominated at
                              embedding length); pair with ``shuffle='runlength'`` to
                              read pure sequencing and ``'token'`` for transitions+dwell.
        ``run_transition`` -- dwell-invariant run-level transitions (clean sequencing).
    shuffle : {'token', 'runlength'} -- the frequency-preserving null (see module docstring).
    classifier : {'logreg', 'rf'} -- via :func:`baselines._make_clf`.
    n_repeats, n_folds : outer RepeatedStratifiedKFold geometry (split list fixed across
        intact and all shuffles, so the comparison is paired on identical folds).
    n_shuffles : number of shuffle replicates forming the null distribution.
    random_state : seeds the splitter and the (single) shuffle generator.

    Returns
    -------
    pd.DataFrame -- one row per comparison, columns:
        comparison, feature, shuffle, classifier, n_shuffles,
        auc_intact, auc_null_mean, delta_auc, ci_low, ci_high, p_perm, order_helps.
        ``ci_low``/``ci_high`` are the 95% CI of ``delta_auc``; ``p_perm`` is the
        one-sided permutation p-value (order helps); ``order_helps == (ci_low > 0)``.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    feature_fn = _FEATURES[feature]
    pipe, grid = _make_clf(classifier)
    rng = np.random.default_rng(random_state)

    rows = []
    for comp, mask, y in _pairwise_subsets(labels):
        # skip comparisons where a group is absent or too small to stratify-split
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_folds:
            continue
        idx = np.flatnonzero(mask)
        Xc = [np.asarray(X[i]) for i in idx]

        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        splits = list(cv.split(np.zeros(len(Xc)), y))

        auc_intact = float(np.mean(
            _nested_cv_auc(feature_fn(Xc, G), y, splits, pipe, grid)))

        null = np.empty(n_shuffles, dtype=np.float64)
        for r in range(n_shuffles):
            X_shuf = order_shuffle_null(Xc, kind=shuffle, rng=rng)
            null[r] = np.mean(_nested_cv_auc(feature_fn(X_shuf, G), y, splits, pipe, grid))

        delta = auc_intact - float(null.mean())
        ci_low = auc_intact - float(np.percentile(null, 97.5))
        ci_high = auc_intact - float(np.percentile(null, 2.5))
        p_perm = (1.0 + int(np.sum(null >= auc_intact))) / (1.0 + n_shuffles)
        rows.append({
            'comparison': comp, 'feature': feature, 'shuffle': shuffle,
            'classifier': classifier, 'n_shuffles': n_shuffles,
            'auc_intact': auc_intact, 'auc_null_mean': float(null.mean()),
            'delta_auc': delta, 'ci_low': ci_low, 'ci_high': ci_high,
            'p_perm': float(p_perm), 'order_helps': bool(ci_low > 0),
        })
    return pd.DataFrame(rows)
