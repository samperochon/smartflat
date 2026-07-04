"""Behavioral-structure metrics over symbolic action sequences.

A reusable, vocabulary-agnostic library of **per-sequence** statistics that quantify
*local execution structure* -- fragmentation, perseveration, complexity, and drift --
of a symbolic action sequence. It is the counterpart to the *frequency* summary
(``baselines.histogram_features``) and the *global-order* summary
(``order_evaluation.run_transition_features``): those two are already shown, on the SDS2
G=28 cohort, to carry the group signal / no incremental signal respectively
(``RESULTS_HANDOFF_barycenters.md`` §12.7/§13.4/§15). Dysexecutive syndrome is
phenomenologically about *local disorganisation and perseveration* -- how fragmented or
perseverative an execution is, which neither the frequency histogram nor a global-order
test captures. Two sequences with identical symbol frequencies can differ sharply here.
This module tests that hypothesis; the metrics may or may not separate groups, and the
group-analysis helpers below report effect sizes + bootstrap CIs + BH correction either way.

Scientific framing (honest): a null result is a valid, reportable outcome.

Design
------
- Each metric is one small, documented function over **one 1D integer sequence** plus, where
  needed, the alphabet size ``G`` and the ``background`` code. All return a Python ``float``
  (``nan`` where undefined). Uniform edge rules: empty -> ``nan``; length-1 -> ``nan`` for any
  metric that needs an adjacent pair, defined value otherwise.
- ``compute_structure_metrics`` aggregates every metric over a DataFrame of sequences into a
  flat one-row-per-sequence scalar table; ``structure_features`` returns the de-collinearised
  matrix consumed by the leakage-guarded predictive harness.
- Run-length encoding reuses Kickoff-E's ``order_evaluation._rle`` (import, don't re-copy);
  the transition matrix reuses ``baselines._transition_matrix``; the nested-CV inner loop
  reuses ``baselines._make_clf`` / ``baselines._nested_cv_auc``.

Representation
--------------
Metrics whose meaning depends on *duration* (perseveration, run-length, dwell, fragmentation,
background fraction, drift) run on the native (embedding-level) sequence. Metrics about the
*action grammar* (transition entropy, n-gram coverage, LZ complexity) run on the RLE-collapsed
(segment-level) sequence by default and expose ``on={'embedding','segment'}``. Embedding-level
run/dwell metrics confound duration with organisation; segment-level metrics strip duration.

References
----------
- Shannon (1948); Cover & Thomas, *Elements of Information Theory* (2006), §4 -- transition entropy.
- Sandson & Albert, *Neuropsychologia* (1984) -- perseveration as immediate repetition.
- Monsell, *Trends Cogn. Sci.* (2003); Stuss & Alexander, *Phil. Trans. R. Soc.* (2007) -- switching/fragmentation.
- Yu, *Artif. Intell.* (2010) -- semi-Markov dwell/sojourn times.
- Manning & Schütze, *Foundations of Statistical NLP* (1999), §6 -- n-gram diversity.
- Lempel & Ziv, *IEEE Trans. IT* (1976); Kaspar & Schuster, *Phys. Rev. A* (1987) -- LZ76 complexity.
- Lin, *IEEE Trans. IT* (1991) -- Jensen-Shannon divergence.
"""

import numpy as np
import pandas as pd

from smartflat.features.symbolic_barycenter.baselines import (
    _transition_matrix, _pairwise_subsets, _make_clf, _nested_cv_auc,
    histogram_features,
)
from smartflat.features.symbolic_barycenter.order_evaluation import _rle


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _as_seq(seq):
    """Coerce to a 1D int ndarray."""
    return np.asarray(seq).astype(int).ravel()


def _hist(seq, G):
    """L1-normalized unigram histogram over ``G`` bins (empty -> zeros)."""
    s = _as_seq(seq)
    h = np.bincount(s, minlength=G).astype(float)
    tot = h.sum()
    return h / tot if tot > 0 else h


def _n_distinct(seq):
    s = _as_seq(seq)
    return int(np.unique(s).size) if s.size else 0


# ---------------------------------------------------------------------------
# Perseveration / run-length / fragmentation (duration-sensitive: embedding-level)
# ---------------------------------------------------------------------------

def immediate_repeat_rate(seq):
    """Fraction of adjacent pairs that are equal (immediate perseveration).

    ``mean(seq[1:] == seq[:-1])``. On the smoothed embedding sequence this is
    dominated by within-run dwell; it is exactly ``1 - switch_rate``. ``nan`` for
    ``len < 2``. Ref: Sandson & Albert (1984).
    """
    s = _as_seq(seq)
    if s.size < 2:
        return float('nan')
    return float(np.mean(s[1:] == s[:-1]))


def n_runs(seq):
    """Number of maximal constant runs (segments) in the sequence.

    ``len == 0`` -> ``nan``; a single-symbol sequence -> ``1.0``.
    """
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    values, _ = _rle(s)
    return float(values.size)


def switch_rate(seq):
    """Fraction of adjacent pairs that are a state switch = ``(n_runs - 1)/(L - 1)``.

    Length-normalized fragmentation; ``== 1 - immediate_repeat_rate``. ``nan`` for
    ``len < 2``. Ref: Monsell (2003).
    """
    s = _as_seq(seq)
    if s.size < 2:
        return float('nan')
    return float((n_runs(s) - 1.0) / (s.size - 1.0))


def run_length_mean(seq):
    """Mean run length (mean action duration). ``L / n_runs``; length-dependent.

    ``len == 0`` -> ``nan``. Ref: bout-length ethology.
    """
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    _, lengths = _rle(s)
    return float(lengths.mean())


def run_length_cv(seq):
    """Coefficient of variation of run lengths (length-robust dispersion).

    ``std(ddof=0)/mean``. ``len == 0`` -> ``nan``; a single run -> ``0.0``.
    """
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    _, lengths = _rle(s)
    m = lengths.mean()
    if m == 0:
        return float('nan')
    return float(lengths.std(ddof=0) / m)


def run_length_max(seq):
    """Longest run length (max sustained dwell). ``len == 0`` -> ``nan``."""
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    _, lengths = _rle(s)
    return float(lengths.max())


def fragmentation_index(seq):
    """Re-entry rate = ``n_runs / n_distinct_states``.

    How many runs per distinct action -- a periodic revisiting sequence scores high, a
    blocked sequence scores 1. ``len == 0`` -> ``nan``. Ref: Stuss & Alexander (2007).
    """
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    nd = _n_distinct(s)
    if nd == 0:
        return float('nan')
    return float(n_runs(s) / nd)


def _state_dwells(seq, G, background):
    """Return dict ``state -> list of run lengths`` for occurring non-background states."""
    s = _as_seq(seq)
    values, lengths = _rle(s)
    out = {}
    for v, ln in zip(values.tolist(), lengths.tolist()):
        if v == background:
            continue
        out.setdefault(v, []).append(ln)
    return out


def dwell_mean_over_states(seq, G=None, background=0):
    """Mean over occurring non-background states of that state's mean run length.

    Collapses the per-state dwell-time distribution to a single vocabulary-agnostic
    scalar (never emits ``G`` columns). ``nan`` if no non-background run exists.
    Ref: Yu (2010).
    """
    dwells = _state_dwells(seq, G, background)
    if not dwells:
        return float('nan')
    per_state_mean = [np.mean(v) for v in dwells.values()]
    return float(np.mean(per_state_mean))


def dwell_cv_over_states(seq, G=None, background=0):
    """Mean over occurring non-background states of the within-state run-length CV.

    A state seen in a single run contributes CV 0. ``nan`` if no non-background run.
    Ref: Yu (2010).
    """
    dwells = _state_dwells(seq, G, background)
    if not dwells:
        return float('nan')
    per_state_cv = []
    for v in dwells.values():
        arr = np.asarray(v, dtype=float)
        m = arr.mean()
        per_state_cv.append(arr.std(ddof=0) / m if m > 0 else 0.0)
    return float(np.mean(per_state_cv))


# ---------------------------------------------------------------------------
# Organisation / grammar (duration-agnostic: segment-level by default)
# ---------------------------------------------------------------------------

def transition_entropy(seq, G, on='segment'):
    """Occupancy-weighted mean Shannon entropy of the outgoing-transition distribution.

    For each state ``s`` with row-normalized outgoing distribution ``P[s, :]``
    (``baselines._transition_matrix``), the row entropy ``H_s = -Σ_j P[s,j] log2 P[s,j]``
    (with ``0·log0 = 0``); the sequence value is ``Σ_s w_s H_s`` where ``w_s`` is the
    fraction of transitions originating in ``s``. A deterministic grammar -> 0; a maximally
    unpredictable one -> ``log2 G``. Units: bits. ``on='segment'`` collapses runs first
    (grammar-level organisation, not dwell). ``nan`` if fewer than 2 transitions exist.
    Ref: Shannon (1948); Cover & Thomas (2006).
    """
    s = _as_seq(seq)
    if on == 'segment':
        s, _ = _rle(s)
    if s.size < 2:
        return float('nan')
    P = _transition_matrix(s, G)
    src = np.bincount(s[:-1], minlength=G).astype(float)
    w = src / src.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        row_ent = -np.nansum(np.where(P > 0, P * np.log2(P), 0.0), axis=1)
    return float(np.sum(w * row_ent))


def _distinct_ngrams(seq, k):
    """Number of distinct observed ordered ``k``-grams (as tuples)."""
    s = _as_seq(seq)
    if s.size < k:
        return None
    grams = {tuple(s[t:t + k].tolist()) for t in range(s.size - k + 1)}
    return len(grams)


def bigram_coverage(seq, G, on='segment'):
    """Distinct observed bigrams / ``G**2`` -- how much of the transition space is used.

    ``on='segment'`` (default) counts distinct action *transitions*, not within-run repeats.
    ``nan`` if the (collapsed) sequence has ``len < 2``. Ref: Manning & Schütze (1999).
    """
    s = _as_seq(seq)
    if on == 'segment':
        s, _ = _rle(s)
    d = _distinct_ngrams(s, 2)
    if d is None:
        return float('nan')
    return float(d / (G * G))


def trigram_coverage(seq, G, on='segment'):
    """Distinct observed trigrams / ``G**3``. ``nan`` if the sequence has ``len < 3``.

    Ref: Manning & Schütze (1999).
    """
    s = _as_seq(seq)
    if on == 'segment':
        s, _ = _rle(s)
    d = _distinct_ngrams(s, 3)
    if d is None:
        return float('nan')
    return float(d / (G ** 3))


def _lz76_phrase_count(seq):
    """Number of distinct LZ76 phrases ``c(n)`` of a 1D int sequence.

    Exact Lempel-Ziv (1976) production complexity via the Kaspar & Schuster (1987)
    pointer scan. ``len == 0`` -> 0; ``len == 1`` -> 1. This is the load-bearing integer
    the tests pin.
    """
    u = _as_seq(seq)
    n = u.size
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, l, k, k_max, c = 0, 1, 1, 1, 1
    while True:
        if u[i + k - 1] == u[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def lz76_complexity(seq, G=None, on='segment'):
    """Length-normalized Lempel-Ziv (LZ76) complexity of the action grammar.

    Parses the RLE-collapsed sequence (``on='segment'``, default) into ``c`` distinct LZ76
    phrases and normalizes ``c_norm = c · log_{G_eff}(n) / n`` -- so a random string over the
    effective alphabet tends to 1. ``G_eff = max(2, n_distinct)`` (a low-alphabet string is
    not penalised against the full ``G``); pass ``G`` only to override the base. ``nan`` for
    empty input; ``0.0`` for a single-symbol (collapsed length 1) sequence.
    Ref: Lempel & Ziv (1976); Kaspar & Schuster (1987).
    """
    s = _as_seq(seq)
    if on == 'segment':
        s, _ = _rle(s)
    n = s.size
    if n == 0:
        return float('nan')
    if n == 1:
        return 0.0
    c = _lz76_phrase_count(s)
    g_eff = max(2, _n_distinct(s))
    base = float(g_eff if G is None else G)
    return float(c * (np.log(n) / np.log(base)) / n)


# ---------------------------------------------------------------------------
# Background fraction & stationarity/drift (embedding-level)
# ---------------------------------------------------------------------------

def background_fraction(seq, background=0):
    """Fraction of tokens equal to the background code.

    NOTE: this is exactly unigram-histogram bin ``background`` -- a pure frequency
    restatement, not structure. Reported for description, excluded from the
    "structure beyond frequency" predictive vector. ``len == 0`` -> ``nan``.
    """
    s = _as_seq(seq)
    if s.size == 0:
        return float('nan')
    return float(np.mean(s == background))


def halves_tv_distance(seq, G, background=0):
    """Total-variation distance between first- and second-half unigram histograms.

    ``0.5 · Σ|p - q|`` where ``p, q`` are the L1-normalized histograms of the two halves
    (split at ``L // 2``). A stationary sequence -> 0. ``nan`` for ``len < 2``.
    """
    s = _as_seq(seq)
    if s.size < 2:
        return float('nan')
    mid = s.size // 2
    p, q = _hist(s[:mid], G), _hist(s[mid:], G)
    return float(0.5 * np.abs(p - q).sum())


def halves_js_divergence(seq, G, background=0):
    """Jensen-Shannon divergence (bits) between first- and second-half histograms.

    ``JS = 0.5·KL(p‖m) + 0.5·KL(q‖m)`` with ``m = (p + q)/2`` -- a finite, symmetric drift
    measure (avoids the ∞ of raw symmetric-KL when a symbol is absent in one half). Range
    ``[0, 1]`` bits. ``nan`` for ``len < 2``. Ref: Lin (1991).
    """
    s = _as_seq(seq)
    if s.size < 2:
        return float('nan')
    mid = s.size // 2
    p, q = _hist(s[:mid], G), _hist(s[mid:], G)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


# ---------------------------------------------------------------------------
# Per-sequence metric registry + aggregators
# ---------------------------------------------------------------------------

# name -> callable(seq, G, background). Order defines the output-column order.
_METRICS = {
    'immediate_repeat_rate': lambda s, G, bg: immediate_repeat_rate(s),
    'switch_rate':           lambda s, G, bg: switch_rate(s),
    'n_runs':                lambda s, G, bg: n_runs(s),
    'run_length_mean':       lambda s, G, bg: run_length_mean(s),
    'run_length_cv':         lambda s, G, bg: run_length_cv(s),
    'run_length_max':        lambda s, G, bg: run_length_max(s),
    'dwell_mean_over_states': lambda s, G, bg: dwell_mean_over_states(s, G, bg),
    'dwell_cv_over_states':  lambda s, G, bg: dwell_cv_over_states(s, G, bg),
    'fragmentation_index':   lambda s, G, bg: fragmentation_index(s),
    'transition_entropy':    lambda s, G, bg: transition_entropy(s, G),
    'bigram_coverage':       lambda s, G, bg: bigram_coverage(s, G),
    'trigram_coverage':      lambda s, G, bg: trigram_coverage(s, G),
    'lz76_complexity':       lambda s, G, bg: lz76_complexity(s),
    'halves_tv_distance':    lambda s, G, bg: halves_tv_distance(s, G, bg),
    'halves_js_divergence':  lambda s, G, bg: halves_js_divergence(s, G, bg),
}

# Columns that restate frequency or are exactly collinear with a kept column; excluded
# from the predictive "structure beyond frequency" vector so the incremental test is honest.
_REDUNDANT = ('background_fraction', 'immediate_repeat_rate')

METRIC_NAMES = tuple(_METRICS.keys())


def _infer_G(sequences):
    mx = -1
    for s in sequences:
        s = _as_seq(s)
        if s.size:
            mx = max(mx, int(s.max()))
    return mx + 1 if mx >= 0 else 1


def compute_structure_metrics(df, seq_col='int_cat_segm_embedding_labels', labels_col=None,
                              G=None, background=0, include_background_fraction=True,
                              include_length=True):
    """Compute every structure metric over a DataFrame of symbolic sequences.

    One row per input sequence (index preserved), one float column per metric -- a flat
    scalar table suitable for group statistics. Vocabulary-agnostic and NaN-safe: any metric
    that is undefined for a sequence yields ``nan`` (never raises).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``seq_col`` (each cell a 1D array-like of integer codes; ragged lengths
        are fine).
    seq_col : str
        Column holding the symbolic sequences.
    labels_col : str, optional
        If given, its values are copied through to the output (for downstream ``groupby``);
        the metric columns stay purely numeric.
    G : int, optional
        Alphabet size. If ``None``, inferred as ``max(code) + 1`` over all sequences.
    background : int
        Background code (default 0).
    include_background_fraction : bool
        Add the ``background_fraction`` column (a frequency restatement; on by default for
        description).
    include_length : bool
        Add the ``length`` column (sequence length) -- the covariate used to length-control
        the group analysis.

    Returns
    -------
    pd.DataFrame
        Indexed like ``df``; columns ``[length?] + METRIC_NAMES + [background_fraction?] + [labels_col?]``.
    """
    sequences = list(df[seq_col].values)
    if G is None:
        G = _infer_G(sequences)

    rows = []
    for seq in sequences:
        s = _as_seq(seq)
        row = {}
        if include_length:
            row['length'] = float(s.size)
        for name, fn in _METRICS.items():
            try:
                row[name] = float(fn(s, G, background))
            except Exception:
                row[name] = float('nan')
        if include_background_fraction:
            try:
                row['background_fraction'] = float(background_fraction(s, background))
            except Exception:
                row['background_fraction'] = float('nan')
        rows.append(row)

    out = pd.DataFrame(rows, index=df.index)
    if labels_col is not None and labels_col in df.columns:
        out[labels_col] = df[labels_col].values
    return out


def structure_features(X_symbolic, G, background=0, drop_redundant=True):
    """Per-sequence structure metrics as a feature matrix (mirror of ``histogram_features``).

    Returns an ``(n_sequences, k)`` float array for the predictive harness. With
    ``drop_redundant=True`` (default) the frequency-restatement / exactly-collinear columns
    (``background_fraction``, ``immediate_repeat_rate``) are dropped, so an incremental test of
    "structure beyond frequency" is not inflated by re-encoding the histogram. NaNs (undefined
    metrics on degenerate sequences) are imputed to the column mean so the matrix is
    classifier-ready; on the real cohort (all long sequences) no imputation occurs.

    Returns
    -------
    (feats, names) : (np.ndarray, list[str])
    """
    names = [n for n in METRIC_NAMES if not (drop_redundant and n in _REDUNDANT)]
    feats = np.full((len(X_symbolic), len(names)), np.nan, dtype=np.float64)
    for i, seq in enumerate(X_symbolic):
        s = _as_seq(seq)
        for j, name in enumerate(names):
            try:
                feats[i, j] = float(_METRICS[name](s, G, background))
            except Exception:
                feats[i, j] = np.nan
    # column-mean impute (nan-safe); all-nan column -> 0.
    col_mean = np.nanmean(feats, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(feats))
    feats[inds] = np.take(col_mean, inds[1])
    return feats, names


# ---------------------------------------------------------------------------
# Honest group analysis
# ---------------------------------------------------------------------------

def _cliffs_delta(x, y):
    """Cliff's delta effect size via the Mann-Whitney U identity ``2U/(n1 n2) - 1``.

    ``x`` is group 1, ``y`` is group 2; positive means ``y`` tends to exceed ``x``.
    Returns ``(delta, p_value)`` (two-sided Mann-Whitney U p). ``(nan, nan)`` if a group is
    empty or all values are tied/NaN.
    """
    from scipy.stats import mannwhitneyu
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float('nan'), float('nan')
    try:
        u, p = mannwhitneyu(y, x, alternative='two-sided')  # U for y over x
    except ValueError:
        return 0.0, 1.0
    delta = 2.0 * u / (x.size * y.size) - 1.0
    return float(delta), float(p)


def _cliffs_delta_fast(x, y):
    """Cliff's delta only (no p-value) via ``mean(sign(y_j - x_i))`` broadcasting.

    Used in the bootstrap inner loop where the scipy Mann-Whitney call is too slow.
    Assumes ``x, y`` are already finite 1D arrays.
    """
    if x.size == 0 or y.size == 0:
        return float('nan')
    return float(np.sign(y[:, None] - x[None, :]).mean())


def _hedges_g(x, y):
    """Hedges' g (bias-corrected standardized mean difference), ``y`` minus ``x``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return float('nan')
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    if sp2 <= 0:
        return 0.0
    d = (y.mean() - x.mean()) / np.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)  # small-sample correction
    return float(j * d)


def structure_group_stats(metrics_df, labels, metric_cols=None, length_col='length',
                          n_boot=10000, alpha=0.05, random_state=0):
    """Descriptive per-metric group comparison with effect sizes, bootstrap CIs, and BH.

    For every ``(metric × comparison)`` over the three paper comparisons
    (``_pairwise_subsets``: HEALTHY_vs_RIL, RIL_vs_TBI, CONTROL_vs_PATIENT) reports the group
    means, **Cliff's delta** (+ percentile bootstrap 95% CI), Hedges' g, the Mann-Whitney p,
    and a **Benjamini-Hochberg** ``p_value_bh`` computed jointly over the entire
    ``metric × comparison`` family (pre-committed; not per-comparison). Also reports each
    metric's Spearman correlation with sequence length, so a group effect that is really a
    length effect is visible (length is a known covariate: several metrics scale with L).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`compute_structure_metrics` (must contain the metric columns and,
        for the length-confound column, ``length_col``).
    labels : array-like of group labels (HEALTHY/RIL/TBI), aligned to ``metrics_df`` rows.
    metric_cols : list[str], optional
        Which columns to test. Defaults to all numeric columns except ``length`` and
        ``pathologie``.
    length_col : str
        Column used for the Spearman length-confound diagnostic (skipped if absent).
    n_boot, alpha, random_state : bootstrap resamples, BH alpha, seed.

    Returns
    -------
    pd.DataFrame
        One row per ``(metric, comparison)``: ``metric, comparison, n1, n2, mean_g1, mean_g2,
        cliffs_delta, cliffs_ci_low, cliffs_ci_high, hedges_g, spearman_length, p_value,
        p_value_bh, significant``.
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import spearmanr

    labels = np.asarray(labels, dtype=object)
    if metric_cols is None:
        drop = {length_col, 'pathologie'}
        metric_cols = [c for c in metrics_df.columns
                       if c not in drop and np.issubdtype(metrics_df[c].dtype, np.number)]

    rng = np.random.default_rng(random_state)
    has_len = length_col in metrics_df.columns
    rows = []
    for metric in metric_cols:
        vals = metrics_df[metric].to_numpy(dtype=float)
        spear = float('nan')
        if has_len:
            lv = metrics_df[length_col].to_numpy(dtype=float)
            ok = ~np.isnan(vals) & ~np.isnan(lv)
            if ok.sum() > 2 and np.unique(vals[ok]).size > 1:
                spear = float(spearmanr(vals[ok], lv[ok]).correlation)
        for comp, mask, y in _pairwise_subsets(labels):
            v = vals[mask]
            g1, g2 = v[y == 0], v[y == 1]
            delta, p = _cliffs_delta(g1, g2)
            g = _hedges_g(g1, g2)
            # paired-free bootstrap CI on Cliff's delta (resample within each group)
            a = g1[~np.isnan(g1)]
            b = g2[~np.isnan(g2)]
            if a.size and b.size:
                boot = np.empty(n_boot)
                for k in range(n_boot):
                    ba = a[rng.integers(0, a.size, a.size)]
                    bb = b[rng.integers(0, b.size, b.size)]
                    boot[k] = _cliffs_delta_fast(ba, bb)
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo = hi = float('nan')
            rows.append({
                'metric': metric, 'comparison': comp,
                'n1': int(a.size), 'n2': int(b.size),
                'mean_g1': float(np.nanmean(g1)) if a.size else float('nan'),
                'mean_g2': float(np.nanmean(g2)) if b.size else float('nan'),
                'cliffs_delta': delta, 'cliffs_ci_low': float(lo), 'cliffs_ci_high': float(hi),
                'hedges_g': g, 'spearman_length': spear, 'p_value': p,
            })

    out = pd.DataFrame(rows)
    valid = out['p_value'].notna()
    out['p_value_bh'] = np.nan
    out['significant'] = False
    if valid.any():
        reject, p_bh, _, _ = multipletests(
            out.loc[valid, 'p_value'].to_numpy(), alpha=alpha, method='fdr_bh')
        out.loc[valid, 'p_value_bh'] = p_bh
        out.loc[valid, 'significant'] = reject
    return out


def evaluate_incremental_structure(
    X_symbolic, labels, G, background=0, classifiers=('logreg', 'rf'),
    n_repeats=10, n_folds=5, random_state=42, n_boot=10000,
):
    """Leakage-guarded incremental AUC of structure features over the frequency histogram.

    Mirrors :func:`baselines.evaluate_incremental_ordering` (same nested-CV protocol, same
    shared inner loop ``baselines._nested_cv_auc``), but with three feature sets:

    - ``hist``: the unigram symbol-frequency histogram (``histogram_features``);
    - ``struct``: the de-collinearised structure metrics (``structure_features``);
    - ``both``: ``[hist ⊕ struct]``.

    The confirmatory quantity is the paired ``both - hist`` delta (does local structure add
    held-out signal *beyond* frequency?), reported with a percentile bootstrap 95% CI +
    Wilcoxon; ``mean_auc_struct`` (with its own read against 0.5) says whether structure alone
    discriminates. Outer ``RepeatedStratifiedKFold`` split list is enumerated once so all three
    feature sets are scored on identical folds.

    .. note::
        Caveat -- the ``both - hist`` CI resamples the per-``(repeat, fold)`` paired deltas
        as if i.i.d., but repeated-CV folds share subjects across repeats, so the effective
        sample size is smaller than ``n_repeats * n_folds`` and the CI **width is mildly
        anti-conservative** (optimistically narrow). The point delta and its **sign are
        unaffected** (they are well inside the interval); the confirmatory reads used in the
        arc all remain valid. Escalating to a subject-level / block bootstrap is deferred
        unless a paper claim needs the tighter guarantee.

    Parameters
    ----------
    X_symbolic : list of int arrays (ragged ok) or (n, T) array.
    labels : array-like of group labels (HEALTHY/RIL/TBI).
    G : int -- alphabet size (incl. background).
    background : int -- background code.
    classifiers : tuple of {'logreg', 'rf'}.
    n_repeats, n_folds, random_state, n_boot : as in ``evaluate_incremental_ordering``.

    Returns
    -------
    folds : pd.DataFrame
        columns: comparison, classifier, feature_set, repeat, fold, auc
    summary : pd.DataFrame
        one row per (comparison, classifier): mean_auc_hist, mean_auc_struct, mean_auc_both,
        mean_delta (both-hist), delta_ci_low, delta_ci_high, wilcoxon_p, n_folds
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from scipy.stats import wilcoxon

    seqs = list(X_symbolic)
    H = histogram_features(seqs, G)
    S, _ = structure_features(seqs, G, background=background, drop_redundant=True)
    feature_sets = {'hist': H, 'struct': S, 'both': np.hstack([H, S])}

    fold_rows = []
    for comp, mask, y in _pairwise_subsets(labels):
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_folds:
            continue
        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
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
    for (comp, clf_name), grp in folds.groupby(['comparison', 'classifier']):
        wide = grp.pivot_table(
            index=['repeat', 'fold'], columns='feature_set', values='auc').dropna()
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
            'mean_auc_struct': float(wide['struct'].mean()),
            'mean_auc_both': float(wide['both'].mean()),
            'mean_delta': float(delta.mean()),
            'delta_ci_low': float(lo), 'delta_ci_high': float(hi),
            'wilcoxon_p': float(wp), 'n_folds': int(len(delta)),
        })
    summary = pd.DataFrame(sum_rows)
    return folds, summary


def evaluate_structure_length_controlled(
    X_symbolic, labels, G, background=0, classifiers=('logreg', 'rf'),
    n_repeats=10, n_folds=5, random_state=42, n_boot=10000,
):
    """Does local structure add signal *beyond frequency AND sequence length*? (confound control).

    Several structure metrics scale with sequence length, and on SDS2 length is itself
    group-discriminative (RIL administrations are longer). So the raw
    :func:`evaluate_incremental_structure` gain of ``[hist⊕struct]`` over ``hist`` can partly be a
    *duration* effect. This adds a standardized ``length`` column to the frequency baseline and
    reports the **incremental AUC of struct over ``[hist⊕length]``** -- the honest "structure beyond
    both frequency and duration" quantity. Same leakage-guarded nested-CV harness
    (``_make_clf``/``_nested_cv_auc``) on four paired feature sets: ``hist``, ``hist+len``,
    ``hist+struct``, ``hist+len+struct``.

    Use this, not :func:`evaluate_incremental_structure` alone, when reporting a structure claim on a
    cohort whose sequence lengths differ by group: a positive ``delta_struct_given_len`` CI (excludes
    0) is the length-robust result.

    .. note::
        Two caveats, both inert here. (1) The ``length`` column is standardised with a
        **global** mean/std computed over all administrations (test folds included). This is
        not a leakage concern: RF is scale-invariant and the logreg pipe re-standardises
        inside each training fold, so the global scaling only sets the units of a column the
        classifiers ignore or re-scale. (2) The incremental-delta CIs use the same
        per-``(repeat, fold)`` i.i.d. bootstrap as :func:`evaluate_incremental_structure`, so
        their **width is mildly anti-conservative** (repeated-CV folds share subjects); the
        point deltas and their signs are unaffected.

    Returns
    -------
    summary : pd.DataFrame
        one row per (comparison, classifier): ``auc_hist, auc_hist_len, auc_hist_struct,
        auc_hist_len_struct``; the three incremental deltas each with a percentile-bootstrap 95% CI
        -- ``delta_struct`` (struct beyond frequency), ``delta_len`` (length beyond frequency), and
        ``delta_struct_given_len`` (struct beyond frequency+length, the decisive one) with
        ``*_ci_low``/``*_ci_high``; ``n_folds``.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    seqs = list(X_symbolic)
    H = histogram_features(seqs, G)
    S, _ = structure_features(seqs, G, background=background, drop_redundant=True)
    lengths = np.array([_as_seq(s).size for s in seqs], dtype=float)
    sd = lengths.std()
    L = ((lengths - lengths.mean()) / (sd if sd > 0 else 1.0)).reshape(-1, 1)
    feature_sets = {
        'hist': H, 'hist_len': np.hstack([H, L]),
        'hist_struct': np.hstack([H, S]), 'hist_len_struct': np.hstack([H, L, S]),
    }

    fold_rows = []
    for comp, mask, y in _pairwise_subsets(labels):
        if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < n_folds:
            continue
        cv = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        splits = list(cv.split(np.zeros(mask.sum()), y))
        for clf_name in classifiers:
            pipe, grid = _make_clf(clf_name)
            for fs_name, F_full in feature_sets.items():
                aucs = _nested_cv_auc(F_full[mask], y, splits, pipe, grid)
                for k, auc in enumerate(aucs):
                    fold_rows.append({
                        'comparison': comp, 'classifier': clf_name, 'feature_set': fs_name,
                        'k': k, 'auc': auc})
    folds = pd.DataFrame(fold_rows)

    rng = np.random.default_rng(random_state)

    def _delta_ci(wide, a, b):
        d = (wide[a] - wide[b]).to_numpy()
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    sum_rows = []
    for (comp, clf_name), grp in folds.groupby(['comparison', 'classifier']):
        wide = grp.pivot_table(index='k', columns='feature_set', values='auc').dropna()
        ds, ds_lo, ds_hi = _delta_ci(wide, 'hist_struct', 'hist')
        dl, dl_lo, dl_hi = _delta_ci(wide, 'hist_len', 'hist')
        dsl, dsl_lo, dsl_hi = _delta_ci(wide, 'hist_len_struct', 'hist_len')
        sum_rows.append({
            'comparison': comp, 'classifier': clf_name,
            'auc_hist': float(wide['hist'].mean()),
            'auc_hist_len': float(wide['hist_len'].mean()),
            'auc_hist_struct': float(wide['hist_struct'].mean()),
            'auc_hist_len_struct': float(wide['hist_len_struct'].mean()),
            'delta_struct': ds, 'delta_struct_ci_low': ds_lo, 'delta_struct_ci_high': ds_hi,
            'delta_len': dl, 'delta_len_ci_low': dl_lo, 'delta_len_ci_high': dl_hi,
            'delta_struct_given_len': dsl, 'delta_struct_given_len_ci_low': dsl_lo,
            'delta_struct_given_len_ci_high': dsl_hi, 'n_folds': int(len(wide)),
        })
    return pd.DataFrame(sum_rows)
