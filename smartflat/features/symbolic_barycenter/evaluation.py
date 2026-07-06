"""Evaluation harness for the barycenter baselines — patient/control labelling,
nested-CV AUC helpers, discrimination/ordering evaluators, significance tests,
and bootstrap CIs.

Split out of ``baselines.py`` (arc-audit Phase 2). Uses ``_transition_matrix``
from :mod:`.distances` and ``barycenter_k_medoid`` from :mod:`.builders`.
Re-exported unchanged via the ``baselines`` shim.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


from .distances import (
    _transition_matrix,
)
from .builders import (
    barycenter_k_medoid,
)


def make_patient_control_labels(labels, patient=('TBI', 'RIL'), control=('HEALTHY',)):
    """Collapse group labels into pooled ``'PATIENT'`` vs ``'CONTROL'``.

    Run :func:`evaluate_baselines` a second time with these labels to obtain the
    pooled ``'CONTROL_vs_PATIENT'`` comparison, then concatenate it with the
    pairwise run to fill the paper's "Patient vs Control" table column.
    """
    patient, control = set(patient), set(control)
    out = np.empty(len(labels), dtype=object)
    for i, lab in enumerate(labels):
        if lab in patient:
            out[i] = 'PATIENT'
        elif lab in control:
            out[i] = 'CONTROL'
        else:
            out[i] = lab
    return out


def evaluate_baselines(
    X_symbolic, labels, methods, D_pairwise=None,
    n_splits=10, n_inits=3, random_state=42,
):
    """Run the 50/50 split evaluation protocol with native-distance scoring.

    Each method builds its own group barycenters on the training split and
    classifies test sequences by nearest group barycenter under that method's
    OWN native distance (TW-TWE for the proposed method and the ablation, DTW
    for DBA-DTW, soft-DTW for Soft-DTW, edit distance for the median string,
    Wasserstein for the histogram barycenter, lock-step Hamming for majority
    voting). This makes the cross-method AUC comparison apples-to-apples: every
    method is scored as it would actually be used.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    labels : np.ndarray of shape (n_sequences,)
        Group labels (e.g. 'HEALTHY', 'TBI', 'RIL', or pooled 'CONTROL'/'PATIENT').
    methods : dict
        Mapping ``method_name -> spec``. Each ``spec`` is a dict with:
          - ``'distance'`` : callable(test_seq_symbolic, barycenter) -> float (required).
          - ``'build'``    : callable(X_group_symbolic, seed) -> barycenter
            (required unless ``kind == 'medoid'``). The barycenter may be a
            symbolic sequence or a histogram, as long as ``'distance'`` consumes it.
          - ``'kind'``     : optional; ``'medoid'`` selects the within-group medoid
            from ``D_pairwise`` instead of calling ``'build'``.
        See :func:`default_baseline_methods` for the six standard baselines.
    D_pairwise : np.ndarray of shape (n_sequences, n_sequences), optional
        Precomputed pairwise distance matrix; required if any method has
        ``kind == 'medoid'``.
    n_splits : int
        Number of random 50/50 stratified splits.
    n_inits : int
        Number of random initializations per split.
    random_state : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        Results with columns: method, split, init, comparison, auc.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=object)
    unique_groups = sorted(np.unique(labels))
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=0.5, random_state=random_state,
    )

    records = []
    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_symbolic, labels)):
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test leakage detected"

        for init_idx in range(n_inits):
            seed = random_state + split_idx * 100 + init_idx

            for method_name, spec in methods.items():
                distance_fn = spec['distance']
                is_medoid = spec.get('kind') == 'medoid'

                # Build one barycenter per group on the training split
                group_barycenters = {}
                for grp in unique_groups:
                    grp_mask = labels[train_idx] == grp
                    if is_medoid:
                        if D_pairwise is None:
                            raise ValueError(
                                f"method '{method_name}' has kind='medoid' but "
                                "D_pairwise was not provided"
                            )
                        grp_global = train_idx[grp_mask]
                        D_grp = D_pairwise[np.ix_(grp_global, grp_global)]
                        group_barycenters[grp] = X_symbolic[grp_global[barycenter_k_medoid(D_grp)]]
                    else:
                        group_barycenters[grp] = spec['build'](
                            X_symbolic[train_idx][grp_mask], seed,
                        )

                # Classify each test sequence by nearest group barycenter
                # under the method's native distance
                test_dists = np.zeros((len(test_idx), len(unique_groups)))
                for gi, grp in enumerate(unique_groups):
                    bary = group_barycenters[grp]
                    for ti, tidx in enumerate(test_idx):
                        test_dists[ti, gi] = distance_fn(X_symbolic[tidx], bary)

                # Pairwise AUC-ROC (higher score -> second group of the pair)
                test_labels = labels[test_idx]
                for g1_idx, g1 in enumerate(unique_groups):
                    for g2_idx, g2 in enumerate(unique_groups):
                        if g1_idx >= g2_idx:
                            continue
                        mask = np.isin(test_labels, [g1, g2])
                        if mask.sum() < 4:
                            continue
                        y_true = (test_labels[mask] == g2).astype(int)
                        y_score = test_dists[mask, g1_idx] - test_dists[mask, g2_idx]
                        try:
                            auc = roc_auc_score(y_true, y_score)
                        except ValueError:
                            auc = 0.5

                        records.append({
                            'method': method_name,
                            'split': split_idx,
                            'init': init_idx,
                            'comparison': f'{g1}_vs_{g2}',
                            'auc': auc,
                        })

    return pd.DataFrame(records)


def baseline_significance_tests(df_all, reference='tw_twe', alpha=0.05):
    """Paired significance tests of each method vs a reference, with FDR control.

    For each comparison, collapses initializations to one AUC per split, then runs
    a paired Wilcoxon signed-rank test of the reference method against each other
    method across splits. p-values are corrected across all (method x comparison)
    tests with Benjamini-Hochberg (``statsmodels`` ``fdr_bh``), matching the
    multiple-testing convention used elsewhere in the paper.

    Parameters
    ----------
    df_all : pd.DataFrame
        Long-format output of :func:`evaluate_baselines` (columns
        method, split, init, comparison, auc), including the ``reference`` method.
    reference : str
        Method name treated as the proposed method (default ``'tw_twe'``).
    alpha : float
        Family-wise FDR level.

    Returns
    -------
    pd.DataFrame
        One row per (method != reference, comparison) with columns:
        method, comparison, mean_auc, ref_mean_auc, delta (ref - method),
        statistic, p_value, p_value_bh, significant.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    # One AUC per (method, comparison, split): average over initializations
    per_split = (
        df_all.groupby(['method', 'comparison', 'split'])['auc'].mean().reset_index()
    )

    rows = []
    for comp in sorted(per_split['comparison'].unique()):
        ref = (
            per_split[(per_split.method == reference) & (per_split.comparison == comp)]
            .set_index('split')['auc']
        )
        if ref.empty:
            continue
        for method in sorted(per_split['method'].unique()):
            if method == reference:
                continue
            cur = (
                per_split[(per_split.method == method) & (per_split.comparison == comp)]
                .set_index('split')['auc']
            )
            joined = pd.concat([ref, cur], axis=1, join='inner').dropna()
            if len(joined) < 1:
                continue
            ref_v = joined.iloc[:, 0].to_numpy()
            cur_v = joined.iloc[:, 1].to_numpy()
            if np.allclose(ref_v, cur_v):
                statistic, p_value = np.nan, 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(ref_v, cur_v)
                except ValueError:
                    statistic, p_value = np.nan, 1.0
            rows.append({
                'method': method,
                'comparison': comp,
                'mean_auc': float(cur_v.mean()),
                'ref_mean_auc': float(ref_v.mean()),
                'delta': float(ref_v.mean() - cur_v.mean()),
                'statistic': statistic,
                'p_value': float(p_value),
            })

    out = pd.DataFrame(rows)
    if len(out):
        reject, p_bh, _, _ = multipletests(
            out['p_value'].to_numpy(), alpha=alpha, method='fdr_bh',
        )
        out['p_value_bh'] = p_bh
        out['significant'] = reject
    return out


def histogram_features(X_symbolic, G):
    """Per-sequence L1-normalized unigram symbol-frequency histograms.

    The pure frequency summary -- discards all temporal ordering. Same count logic
    as :func:`barycenter_wasserstein`.

    Returns
    -------
    np.ndarray of shape (n_sequences, G)
    """
    feats = np.zeros((len(X_symbolic), G), dtype=np.float64)
    for i, seq in enumerate(X_symbolic):
        h = np.bincount(np.asarray(seq).astype(int), minlength=G).astype(float)
        s = h.sum()
        feats[i] = h / s if s > 0 else h
    return feats


def transition_features(X_symbolic, G):
    """Per-sequence flattened row-normalized bigram transition matrices.

    Each sequence's ``(G, G)`` transition matrix (:func:`_transition_matrix`) is
    flattened to a length-``G*G`` vector -- the ordering structure ("which action
    follows which") that the unigram histogram discards.

    Returns
    -------
    np.ndarray of shape (n_sequences, G*G)
    """
    feats = np.zeros((len(X_symbolic), G * G), dtype=np.float64)
    for i, seq in enumerate(X_symbolic):
        feats[i] = _transition_matrix(seq, G).reshape(-1)
    return feats


_SDS2_LABELS = frozenset({'HEALTHY', 'RIL', 'TBI'})


def _pairwise_subsets(labels):
    """Yield ``(comparison_name, mask, y_binary)`` group-discrimination comparisons.

    ``mask`` selects the two groups from ``labels``; ``y_binary`` is 1 for the second
    group of the pair. Two regimes, auto-selected from the label vocabulary:

    - **SDS2 cohort** (labels ⊆ {HEALTHY, RIL, TBI}): byte-identical to the three
      frozen paper comparisons — ``HEALTHY_vs_RIL``, ``RIL_vs_TBI``, and the pooled
      ``CONTROL_vs_PATIENT``. The committed §15/§16 numbers are unchanged.
    - **Any other vocabulary** (generalization datasets, PAPER_TODO §2): all unordered
      class pairs, so the same order/structure harnesses run unchanged on new datasets.
      To bound the comparisons on a many-class dataset, filter ``labels``/``X`` to the
      classes of interest before calling the harness.
    """
    labels = np.asarray(labels, dtype=object)
    uniq = {lab for lab in labels
            if lab is not None and not (isinstance(lab, float) and np.isnan(lab))}
    if uniq <= _SDS2_LABELS:
        pooled = make_patient_control_labels(labels)
        specs = [
            ('HEALTHY_vs_RIL', ('HEALTHY', 'RIL'), labels),
            ('RIL_vs_TBI', ('RIL', 'TBI'), labels),
            ('CONTROL_vs_PATIENT', ('CONTROL', 'PATIENT'), pooled),
        ]
        for name, (g1, g2), lab in specs:
            mask = np.isin(lab, [g1, g2])
            y = (lab[mask] == g2).astype(int)
            yield name, mask, y
    else:
        classes = sorted(uniq, key=str)
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                g1, g2 = classes[i], classes[j]
                mask = np.isin(labels, [g1, g2])
                y = (labels[mask] == g2).astype(int)
                yield f'{g1}_vs_{g2}', mask, y


def _make_clf(name):
    """Return ``(sklearn Pipeline, param_grid)`` for ``name`` in {'logreg', 'rf'}.

    Shared nested-CV estimator factory (extracted verbatim from
    :func:`evaluate_incremental_ordering`'s former local ``make_estimator``), so the
    incremental-ordering harness, the order-shuffle-null evaluator
    (``order_evaluation.order_information``), and any structure-feature evaluator all
    use one definition.

    - ``logreg``: ``StandardScaler`` + ``LogisticRegression(penalty='l2',
      solver='liblinear', max_iter=1000)``; grid ``{'clf__C': [0.01, 0.1, 1.0, 10.0]}``.
    - ``rf``: ``RandomForestClassifier(random_state=0)``; grid
      ``{'clf__n_estimators': [200], 'clf__max_depth': [3, 5, None]}``.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    if name == 'logreg':
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2', solver='liblinear', max_iter=1000)),
        ])
        grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
    elif name == 'rf':
        pipe = Pipeline([('clf', RandomForestClassifier(random_state=0))])
        grid = {'clf__n_estimators': [200], 'clf__max_depth': [3, 5, None]}
    else:
        raise ValueError(f"unknown classifier {name!r}")
    return pipe, grid


def _nested_cv_auc(F, y, splits, pipe, grid):
    """Per-fold held-out AUC via inner ``GridSearchCV(scoring='roc_auc')``.

    The shared inner loop behind every nested-CV ordering/structure evaluator.

    Parameters
    ----------
    F : np.ndarray of shape (n, d) -- feature matrix (already group-masked).
    y : np.ndarray of shape (n,) -- binary int labels.
    splits : list of (train_idx, test_idx) -- pre-enumerated so paired feature sets
        are evaluated on identical folds.
    pipe, grid : from :func:`_make_clf`.

    Returns
    -------
    list of float -- one AUC per split (in ``splits`` order); ``0.5`` on a degenerate
    single-class test fold. Inner-CV fold count is
    ``max(2, int(min(5, min class count in y[train])))``. No global state;
    deterministic given the inputs.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score

    aucs = []
    for tr, te in splits:
        inner = max(2, int(min(5, np.min(np.bincount(y[tr])))))
        gs = GridSearchCV(pipe, grid, scoring='roc_auc', cv=inner)
        gs.fit(F[tr], y[tr])
        score = gs.predict_proba(F[te])[:, 1]
        try:
            auc = roc_auc_score(y[te], score)
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return aucs


def evaluate_incremental_ordering(
    X_symbolic, labels, G, classifiers=('logreg', 'rf'),
    n_repeats=10, n_folds=5, random_state=42, n_boot=10000,
):
    """Incremental AUC of bigram ordering over the unigram histogram (Track A.1).

    Trains a plain classifier (nested CV) on two feature sets -- ``hist`` (unigram
    histogram only) and ``both`` (histogram + flattened bigram transitions) -- and
    reports the *incremental* held-out AUC of adding ordering, per pairwise
    comparison. This isolates "does temporal ordering carry group signal beyond
    frequency?" from the barycenter machinery.

    Protocol: outer ``RepeatedStratifiedKFold`` (paired across feature sets via a
    single pre-enumerated split list), inner ``GridSearchCV`` (roc_auc) for the
    classifier's regularization; test AUC per outer fold; paired percentile
    bootstrap 95% CI + Wilcoxon on the ``(both - hist)`` delta over folds.

    Caveat: the bigram block has ``G*G`` features (e.g. 784 at G=28) vs n as low as
    ~60; StandardScaler + inner-CV L2 (logreg) / RF mitigate but high-dim
    overfitting is real -- read the delta CI, not the raw ``both`` AUC.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
    labels : array-like of group labels (HEALTHY/RIL/TBI).
    G : int -- alphabet size (number of symbols incl. background).
    classifiers : tuple of {'logreg', 'rf'}.
    n_repeats, n_folds : outer RepeatedStratifiedKFold geometry.
    random_state : seed for the splitter and the bootstrap.
    n_boot : bootstrap resamples for the delta CI.

    Returns
    -------
    folds : pd.DataFrame
        columns: comparison, classifier, feature_set, repeat, fold, auc
    summary : pd.DataFrame
        one row per (comparison, classifier): mean_auc_hist, mean_auc_both,
        mean_delta, delta_ci_low, delta_ci_high, wilcoxon_p, n_folds
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from scipy.stats import wilcoxon

    X_symbolic = np.asarray(X_symbolic)
    H = histogram_features(X_symbolic, G)
    T = transition_features(X_symbolic, G)
    feature_sets = {'hist': H, 'both': np.hstack([H, T])}

    fold_rows = []
    for comp, mask, y in _pairwise_subsets(labels):
        # skip comparisons where a group is absent or too small to stratify-split
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_folds:
            continue
        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state,
        )
        # Enumerate splits ONCE so 'hist' and 'both' are evaluated on identical folds.
        splits = list(cv.split(np.zeros(mask.sum()), y))
        for clf_name in classifiers:
            pipe, grid = _make_clf(clf_name)
            for fs_name, F_full in feature_sets.items():
                aucs = _nested_cv_auc(F_full[mask], y, splits, pipe, grid)
                for k, auc in enumerate(aucs):
                    fold_rows.append({
                        'comparison': comp, 'classifier': clf_name,
                        'feature_set': fs_name,
                        'repeat': k // n_folds, 'fold': k % n_folds, 'auc': auc,
                    })

    folds = pd.DataFrame(fold_rows)

    rng = np.random.default_rng(random_state)
    sum_rows = []
    for (comp, clf_name), g in folds.groupby(['comparison', 'classifier']):
        wide = g.pivot_table(
            index=['repeat', 'fold'], columns='feature_set', values='auc',
        ).dropna()
        delta = (wide['both'] - wide['hist']).to_numpy()
        idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
        boot = delta[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if np.allclose(delta, 0):
            wp = 1.0
        else:
            try:
                _, wp = wilcoxon(wide['both'].to_numpy(), wide['hist'].to_numpy())
            except ValueError:
                wp = 1.0
        sum_rows.append({
            'comparison': comp, 'classifier': clf_name,
            'mean_auc_hist': float(wide['hist'].mean()),
            'mean_auc_both': float(wide['both'].mean()),
            'mean_delta': float(delta.mean()),
            'delta_ci_low': float(lo), 'delta_ci_high': float(hi),
            'wilcoxon_p': float(wp), 'n_folds': int(len(delta)),
        })
    summary = pd.DataFrame(sum_rows)
    return folds, summary


def _per_split_auc(df, method, comparison):
    """Per-split AUC (inits averaged) for one method+comparison, indexed by split."""
    sub = df[(df['method'] == method) & (df['comparison'] == comparison)]
    return sub.groupby('split')['auc'].mean()


def bootstrap_auc_ci(df, method, comparison, n_boot=10000, ci=0.95, random_state=0):
    """Percentile bootstrap CI for a method's mean per-split AUC (Track A.2).

    Resamples the per-split AUC values (inits averaged) from an
    :func:`evaluate_baselines` result. Returns ``(mean, low, high)``, or
    ``(nan, nan, nan)`` if the method/comparison is absent.
    """
    v = _per_split_auc(df, method, comparison).to_numpy()
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = v[idx].mean(axis=1)
    p = (1 - ci) / 2 * 100
    lo, hi = np.percentile(boot, [p, 100 - p])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_delta_ci(df, method_a, method_b, comparison, n_boot=10000, ci=0.95,
                       random_state=0):
    """Paired bootstrap CI for the ``method_a - method_b`` per-split AUC delta.

    Splits are matched (inits averaged), then the split-level paired deltas are
    resampled. Returns ``(mean_delta, low, high)``. The pre-registered TBI-vs-RIL
    test is "positive" iff ``low > 0`` (the ordering method's advantage over the
    histogram excludes 0).
    """
    a = _per_split_auc(df, method_a, comparison)
    b = _per_split_auc(df, method_b, comparison)
    joined = pd.concat([a, b], axis=1, join='inner').dropna()
    if len(joined) == 0:
        return float('nan'), float('nan'), float('nan')
    delta = (joined.iloc[:, 0] - joined.iloc[:, 1]).to_numpy()
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(axis=1)
    p = (1 - ci) / 2 * 100
    lo, hi = np.percentile(boot, [p, 100 - p])
    return float(delta.mean()), float(lo), float(hi)
