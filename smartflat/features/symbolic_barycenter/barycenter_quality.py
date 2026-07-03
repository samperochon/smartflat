"""Barycenter *representation-quality* harness (Kickoff F of the E->G->F arc).

The group-discrimination question is settled (RESULTS_HANDOFF §12-§16: symbol frequency
dominates; E's §15 order-shuffle-null finds 0/15 order signals). This module scores the
**other** axis -- how faithfully a barycenter *represents* the group it averages -- so that
every averaging method in the ``{build, distance, kind}`` registry (see
:func:`smartflat.features.symbolic_barycenter.baselines.evaluate_baselines`) is judged
identically, without reference to classification.

Metrics, per (method, group):

- ``inertia_rtwe``      : mean rTWE distance of members to the barycenter (the common
  yardstick across methods; NaN when the barycenter is not a symbol sequence).
- ``inertia_native``    : mean of the method's OWN distance of members to the barycenter
  (how the method is actually used).
- ``freq_fidelity``     : Wasserstein-on-D_G distance between the barycenter's symbol
  histogram and the pooled-group histogram (lower = better; defined for all output kinds
  that carry a symbol distribution).
- ``struct_preservation``: Frobenius distance between the barycenter's bigram transition
  matrix and the group-mean transition matrix (temporal structure the histogram discards).
- ``entropy_bits``      : Shannon entropy of the barycenter's symbol histogram -- the
  §12.3 mode-collapse lens (group sequences ~3.8 bits; mode-DBA collapses to 2.8-3.2).
- ``n_distinct``        : number of distinct symbols used by the barycenter.
- ``n_segments``        : run-length segment count of the barycenter (compression).
- ``stability_inertia_rtwe`` / ``stability_histogram`` : spread across random inits
  (0 for deterministic builders); emitted once per (method, group) with ``init = NaN``.

Output kinds handled (auto-detected): symbol sequence (most methods), symbol histogram
(``wasserstein``), medoid index (``k_medoid`` -> its member sequence), and (G, G)
transition matrix (``transition`` -> structure metric only).
"""

import numpy as np
import pandas as pd

from smartflat.engine.distances._rtwe import rtwe_distance
from smartflat.features.symbolic_barycenter.baselines import (
    _transition_matrix,
    barycenter_transition_matrix,
)

_METRICS = [
    'inertia_rtwe', 'inertia_native', 'freq_fidelity', 'struct_preservation',
    'entropy_bits', 'n_distinct', 'n_segments',
]

# Canonical discreteness-family taxonomy for every registry key (see
# ``BARYCENTER_METHOD_DISCRETENESS.md`` / RESULTS_HANDOFF §18.4). Single-sourced here so
# notebooks/reports import it instead of re-copying the literal (it had been pasted into
# 06i/06j/06k). Family A = native-categorical (never leave symbol space); Family B =
# continuous relaxation + decode. The two ``wasserstein``/``transition`` labels carry the
# object-kind nuance those methods have (a distribution / a matrix, no per-position symbol).
FAMILY = {
    # Family A -- native-categorical
    'tw_twe_mode': 'A', 'shape_dba': 'A', 'majority_voting': 'A', 'edit_median': 'A',
    'k_medoid': 'A', 'msa_consensus': 'A',
    'wasserstein': 'A (freq only)', 'transition': 'A (matrix)',
    # Family B -- continuous relaxation + decode
    'dba_dtw': 'B', 'soft_dtw': 'B', 'soft_dtw_bary': 'B', 'ssg': 'B',
    'fgw_mds': 'B', 'fgw_onehot': 'B',
    # Family B lever variants (Kickoff K) -- ground-cost decode / per-iteration categorical
    'dba_dtw_dg': 'B (D_G decode)', 'soft_dtw_bary_dg': 'B (D_G decode)',
    'ssg_dg': 'B (D_G decode)', 'fgw_onehot_dg': 'B (D_G decode)',
    'dba_dtw_cat': 'B->cat (per-iter)', 'ssg_cat': 'B->cat (per-iter)',
    'soft_dtw_bary_cat': 'B->cat (per-iter)',
}


def family_of(method):
    """Return the discreteness-family label ('A'/'B' variants) for a registry key.

    Exact match in :data:`FAMILY` first; otherwise infer from the key suffix so the FGW
    alpha-sweep keys (``fgw_{enc}_a{alpha}`` from :func:`build_fgw_registry`) and any
    ``*_dg`` / ``*_cat`` lever key still resolve. Returns ``'?'`` for an unknown method.
    """
    if method in FAMILY:
        return FAMILY[method]
    if method.endswith('_cat'):
        return 'B->cat (per-iter)'
    if method.endswith('_dg'):
        return 'B (D_G decode)'
    if method.startswith('fgw_'):
        return 'B'
    return '?'


def _hist_from_seq(seq, G):
    """Normalized symbol histogram (length G, sums to 1) of an integer sequence."""
    h = np.bincount(np.asarray(seq).astype(int), minlength=G).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else h


def _entropy_bits(hist):
    """Shannon entropy (bits) of a probability vector."""
    p = np.asarray(hist, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _n_segments(seq):
    """Number of maximal constant runs (run-length segments) in a symbol sequence."""
    s = np.asarray(seq).astype(int)
    return int(1 + np.count_nonzero(np.diff(s))) if s.size else 0


def _w_hist(p, q, M):
    """Wasserstein distance between two symbol histograms under ground cost M."""
    import ot
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(ot.emd2(p, q, np.ascontiguousarray(M, dtype=np.float64)))


def _resolve_barycenter(bary_raw, is_medoid, X_group, G):
    """Coerce a raw barycenter to ``(seq, hist, kind)``.

    ``seq``  : integer symbol sequence, or None (histogram / matrix outputs).
    ``hist`` : normalized symbol histogram, or None (matrix output).
    ``kind`` : 'sequence' | 'histogram' | 'matrix'.
    """
    if is_medoid:
        seq = np.asarray(X_group[int(bary_raw)]).astype(int)
        return seq, _hist_from_seq(seq, G), 'sequence'
    b = np.asarray(bary_raw)
    if b.ndim == 2 and b.shape[0] == b.shape[1] == G:
        return None, None, 'matrix'                        # transition-matrix barycenter
    if (b.ndim == 1 and b.shape[0] == G
            and np.issubdtype(b.dtype, np.floating) and abs(b.sum() - 1.0) < 1e-6):
        return None, b.astype(np.float64), 'histogram'     # Wasserstein histogram barycenter
    seq = b.astype(int)
    return seq, _hist_from_seq(seq, G), 'sequence'


def _metrics_one(spec, bary_native, seq, hist, kind, X_group, D_G, G, M,
                 group_hist, group_T, rtwe_nu, rtwe_lmbda):
    """Compute the metric dict for one barycenter against its group.

    ``bary_native`` is the object the method's own distance consumes -- the medoid's
    member sequence for ``kind='medoid'``, else the raw build output (sequence /
    histogram / transition matrix).
    """
    distance_fn = spec['distance']
    out = {m: np.nan for m in _METRICS}

    # native inertia: the method's own distance against its own barycenter object.
    out['inertia_native'] = float(np.mean([distance_fn(m, bary_native) for m in X_group]))

    # rTWE inertia: common yardstick, only for symbol-sequence outputs.
    if seq is not None:
        bary_f = seq.astype(np.float64)
        Dc = np.ascontiguousarray(D_G, dtype=np.float64)
        out['inertia_rtwe'] = float(np.mean([
            rtwe_distance(np.asarray(m, dtype=np.float64), bary_f,
                          nu=rtwe_nu, lmbda=rtwe_lmbda, precomputed_distances=Dc)
            for m in X_group
        ]))

    # frequency fidelity: for any output carrying a symbol distribution.
    if hist is not None:
        out['freq_fidelity'] = _w_hist(hist, group_hist, M)
        out['entropy_bits'] = _entropy_bits(hist)
        out['n_distinct'] = float(np.count_nonzero(hist))

    # temporal-structure preservation.
    if kind == 'matrix':
        out['struct_preservation'] = float(np.linalg.norm(np.asarray(bary_native) - group_T))
    elif seq is not None:
        out['struct_preservation'] = float(
            np.linalg.norm(_transition_matrix(seq, G) - group_T))

    # compression / run structure.
    if seq is not None:
        out['n_segments'] = float(_n_segments(seq))

    return out


def score_barycenter_quality(X_symbolic, labels, methods, D_G, D_pairwise=None,
                             n_inits=3, random_state=42, rtwe_nu=1e-4, rtwe_lmbda=0.1):
    """Score every method's per-group barycenter on representation-quality metrics.

    Parameters
    ----------
    X_symbolic : (n_sequences, L) int array
        Equal-length integer symbolic sequences (e.g. ``load_g28_cohort(upsample_to=128)``).
    labels : (n_sequences,) array
        Group labels (e.g. 'HEALTHY' / 'TBI' / 'RIL').
    methods : dict
        ``method_name -> spec`` registry with ``'build'`` / ``'distance'`` / optional
        ``'kind'='medoid'`` -- exactly the contract of
        :func:`smartflat.features.symbolic_barycenter.baselines.evaluate_baselines`.
    D_G : (G, G) ndarray
        rTWE / Wasserstein ground cost.
    D_pairwise : (n_sequences, n_sequences) ndarray, optional
        Pairwise distances; required if any method has ``kind='medoid'``.
    n_inits : int
        Random initialisations per (method, group) -- drives the stability metrics.
    random_state : int
        Base seed.

    Returns
    -------
    pd.DataFrame
        Long-form with columns ``method, group, metric, init, value``. Base metrics carry
        an integer ``init``; the two ``stability_*`` rows carry ``init = NaN``.
    """
    X = np.asarray(X_symbolic)
    labels = np.asarray(labels, dtype=object)
    G = np.asarray(D_G).shape[0]
    M = np.asarray(D_G, dtype=np.float64)
    M = M / M.max() if M.max() > 0 else M
    groups = sorted(np.unique(labels))

    records = []
    for grp in groups:
        idx = np.where(labels == grp)[0]
        X_group = [np.asarray(X[i]).astype(int) for i in idx]
        group_hist = _hist_from_seq(np.concatenate(X_group), G)
        group_T = barycenter_transition_matrix(X_group, G)

        for method_name, spec in methods.items():
            is_medoid = spec.get('kind') == 'medoid'
            if is_medoid and D_pairwise is None:
                raise ValueError(
                    f"method '{method_name}' has kind='medoid' but D_pairwise was not provided")

            per_init = []
            for init_idx in range(n_inits):
                seed = random_state + init_idx
                if is_medoid:
                    D_grp = D_pairwise[np.ix_(idx, idx)]
                    bary_raw = spec['build'](D_grp, seed) if 'build' in spec \
                        else int(np.argmin(D_grp.sum(axis=1)))
                else:
                    bary_raw = spec['build'](np.asarray(X_group), seed)

                seq, hist, kind = _resolve_barycenter(bary_raw, is_medoid, X_group, G)
                bary_native = seq if is_medoid else bary_raw
                m = _metrics_one(spec, bary_native, seq, hist, kind, X_group, D_G, G, M,
                                 group_hist, group_T, rtwe_nu, rtwe_lmbda)
                per_init.append((m, hist))
                for metric, value in m.items():
                    records.append(dict(method=method_name, group=grp, metric=metric,
                                        init=init_idx, value=value))

            # stability across inits (0 for deterministic builders).
            inertias = np.array([mi['inertia_rtwe'] for mi, _ in per_init], dtype=np.float64)
            records.append(dict(method=method_name, group=grp,
                                metric='stability_inertia_rtwe', init=np.nan,
                                value=float(np.nanstd(inertias)) if np.isfinite(inertias).any()
                                else np.nan))
            hists = [h for _, h in per_init if h is not None]
            hist_std = float(np.mean(np.std(np.vstack(hists), axis=0))) if len(hists) > 1 \
                else (0.0 if hists else np.nan)
            records.append(dict(method=method_name, group=grp,
                                metric='stability_histogram', init=np.nan, value=hist_std))

    return pd.DataFrame.from_records(
        records, columns=['method', 'group', 'metric', 'init', 'value'])


def quality_table(df, agg='mean'):
    """Pivot :func:`score_barycenter_quality` output to a methods x metric table.

    Averages base metrics over inits and groups; keeps the stability rows (already
    init-aggregated) as their own columns. Returns a ``method``-indexed DataFrame.
    """
    base = df[df['init'].notna()]
    stab = df[df['init'].isna()]
    t = (base.groupby(['method', 'metric'])['value'].agg(agg).unstack('metric'))
    if len(stab):
        s = (stab.groupby(['method', 'metric'])['value'].agg(agg).unstack('metric'))
        t = t.join(s)
    order = [m for m in _METRICS + ['stability_inertia_rtwe', 'stability_histogram']
             if m in t.columns]
    return t[order]


def build_fgw_registry(D_G, alphas, n_nodes=128, encodings=('mds', 'onehot'),
                       nu=1e-4, lmbda=0.1, max_iter=100):
    """Registry of FGW methods across an ``alpha`` sweep and encodings (no cherry-picking).

    Produces entries named ``fgw_{encoding}_a{alpha}`` so a single
    :func:`score_barycenter_quality` call yields the full frequency<->structure
    decomposition. Each is scored by the native rTWE distance.
    """
    from smartflat.features.symbolic_barycenter.baselines import barycenter_fgw, dist_rtwe

    reg = {}
    for enc in encodings:
        for a in alphas:
            name = f"fgw_{enc}_a{a:g}"
            reg[name] = {
                'build': (lambda X, seed, a=a, enc=enc: barycenter_fgw(
                    X, D_G, alpha=a, n_nodes=n_nodes, feature=enc,
                    max_iter=max_iter, random_state=seed)),
                'distance': (lambda seq, bary: dist_rtwe(
                    seq, bary, D_G, nu=nu, lmbda=lmbda)),
            }
    return reg
