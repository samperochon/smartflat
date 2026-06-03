"""Baseline barycenter methods for comparison with TW-TWE + DBA (PAPER_BRIDGE items A5-A7, B1-B3).

Provides six alternative barycenter computation methods with a uniform
interface, plus embedding utilities and an evaluation framework.

Design decision: symbolic sequences are embedded into real-valued space
using prototype-distance rows (D_G[s, :]) to preserve Wasserstein structure.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def embed_symbolic_to_real(X_symbolic, D_G):
    """Map symbolic integer sequences to real-valued time series via D_G rows.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (symbol indices into D_G).
    D_G : np.ndarray of shape (G, G)
        Symmetric prototype distance matrix.

    Returns
    -------
    np.ndarray of shape (n_sequences, n_timepoints, G)
        Real-valued embedded sequences where each symbol is replaced by its
        D_G row vector.
    """
    return D_G[X_symbolic.astype(int)]


def project_real_to_symbolic(X_real, D_G):
    """Project real-valued embedded sequences back to symbolic via nearest prototype.

    Parameters
    ----------
    X_real : np.ndarray of shape (n_sequences, n_timepoints, G) or (n_timepoints, G)
        Real-valued embedded sequences.
    D_G : np.ndarray of shape (G, G)
        Symmetric prototype distance matrix.

    Returns
    -------
    np.ndarray
        Integer-valued symbolic sequences (same shape as input minus last dim).
    """
    if X_real.ndim == 2:
        # Single sequence: (T, G)
        dists = np.linalg.norm(X_real[:, None, :] - D_G[None, :, :], axis=2)
        return np.argmin(dists, axis=1)
    # Batch: (N, T, G)
    dists = np.linalg.norm(X_real[:, :, None, :] - D_G[None, None, :, :], axis=3)
    return np.argmin(dists, axis=2)


def _symbols_to_str(seq):
    """Encode an integer symbol sequence as a string (offset to printable ASCII).

    Shared by the edit-distance median baseline and its native distance.
    """
    return ''.join(chr(int(s) + 33) for s in seq)


def ordinal_cost_matrix(G):
    """Plain ordinal ground cost ``D_ord[i, j] = |i - j|`` over the alphabet.

    Drop-in replacement for the Wasserstein-based prototype cost ``D_G`` used to
    build the **TWE ablation** baseline: it strips the temporal-occurrence
    (Wasserstein) structure and treats symbols as bare ordinal indices, isolating
    the contribution of the Wasserstein component of TW-TWE.

    Parameters
    ----------
    G : int
        Alphabet size.

    Returns
    -------
    np.ndarray of shape (G, G)
        Symmetric ordinal cost matrix with zero diagonal.
    """
    idx = np.arange(G)
    return np.abs(idx[:, None] - idx[None, :]).astype(float)


# ---------------------------------------------------------------------------
# Tier A baselines (A5, A6, A7)
# ---------------------------------------------------------------------------

def _dtw_cost_matrix(x, y):
    """Accumulated DTW cost matrix between two real-valued sequences.

    Parameters
    ----------
    x : np.ndarray of shape (T1, d)
    y : np.ndarray of shape (T2, d)

    Returns
    -------
    np.ndarray of shape (T1 + 1, T2 + 1)
        Accumulated cost matrix; ``D[T1, T2]`` is the DTW distance.
    """
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(x[i - 1] - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def _dtw_alignment(x, y):
    """Compute DTW alignment path between two real-valued sequences.

    Parameters
    ----------
    x : np.ndarray of shape (T1, d)
    y : np.ndarray of shape (T2, d)

    Returns
    -------
    list of (i, j) tuples
        Alignment path.
    """
    n, m = len(x), len(y)
    D = _dtw_cost_matrix(x, y)

    # Backtrack
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            idx = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if idx == 0:
                i, j = i - 1, j - 1
            elif idx == 1:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    return path[::-1]


def barycenter_dba_dtw(X_symbolic, D_G, max_iters=30, tol=1e-5, random_state=None):
    """Baseline A5: DBA with standard DTW on prototype-distance embeddings.

    Pure numpy/scipy implementation (Petitjean et al. 2011).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences.
    D_G : np.ndarray of shape (G, G)
        Prototype distance matrix.
    max_iters : int
        Maximum DBA iterations.
    tol : float
        Convergence tolerance.
    random_state : int or None
        Random seed for initialization.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)
    N, T, G = X_emb.shape

    rng = np.random.RandomState(random_state)
    barycenter = X_emb[rng.randint(N)].copy()  # (T, G)
    prev_cost = np.inf

    for _ in range(max_iters):
        # Accumulate aligned values per timestep
        assoc = [[] for _ in range(T)]
        total_cost = 0.0

        for i in range(N):
            path = _dtw_alignment(barycenter, X_emb[i])
            for bi, si in path:
                if 0 <= bi < T:
                    assoc[bi].append(X_emb[i, si])
            total_cost += np.linalg.norm(barycenter - X_emb[i])

        # Update barycenter: mean of aligned values at each timestep
        for t in range(T):
            if assoc[t]:
                barycenter[t] = np.mean(assoc[t], axis=0)

        if abs(prev_cost - total_cost) < tol:
            break
        prev_cost = total_cost

    return project_real_to_symbolic(barycenter, D_G)


def _soft_dtw_grad(barycenter, X_emb, gamma):
    """Compute Soft-DTW gradient w.r.t. barycenter (simplified).

    Uses a smoothed min over DTW alignment costs.
    """
    N, T, G = X_emb.shape
    grad = np.zeros_like(barycenter)

    for i in range(N):
        # Compute soft-DTW cost matrix
        n, m = len(barycenter), T
        R = np.full((n + 2, m + 2), np.inf)
        R[0, 0] = 0
        for ii in range(1, n + 1):
            for jj in range(1, m + 1):
                cost = np.sum((barycenter[ii - 1] - X_emb[i, jj - 1]) ** 2)
                softmin = -gamma * np.log(
                    np.exp(-R[ii - 1, jj - 1] / gamma)
                    + np.exp(-R[ii - 1, jj] / gamma)
                    + np.exp(-R[ii, jj - 1] / gamma)
                )
                R[ii, jj] = cost + softmin

        # Approximate gradient via DTW alignment
        path = _dtw_alignment(barycenter, X_emb[i])
        for bi, si in path:
            if 0 <= bi < len(barycenter):
                grad[bi] += 2 * (barycenter[bi] - X_emb[i, si])

    return grad / N


def _softmin3(a, b, c, gamma):
    """Numerically stable soft-minimum of three values (log-sum-exp form)."""
    m = min(a, b, c)
    if np.isinf(m):
        return m
    return m - gamma * np.log(
        np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    )


def _soft_dtw_cost(x, y, gamma):
    """Soft-DTW discrepancy between two real-valued sequences (Cuturi & Blondel 2017).

    Parameters
    ----------
    x : np.ndarray of shape (T1, d)
    y : np.ndarray of shape (T2, d)
    gamma : float
        Smoothing parameter.

    Returns
    -------
    float
        Soft-DTW value ``R[T1, T2]`` (a discrepancy, not necessarily non-negative).
    """
    n, m = len(x), len(y)
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sum((x[i - 1] - y[j - 1]) ** 2)
            R[i, j] = cost + _softmin3(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], gamma)
    return float(R[n, m])


def barycenter_soft_dtw(X_symbolic, D_G, gamma=1.0, max_iter=30, random_state=None):
    """Baseline A6: Soft-DTW barycenter on prototype-distance embeddings.

    Gradient-descent approximation (Cuturi & Blondel 2017).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences.
    D_G : np.ndarray of shape (G, G)
        Prototype distance matrix.
    gamma : float
        Soft-DTW smoothing parameter.
    max_iter : int
        Maximum iterations.
    random_state : int or None
        Random seed for initialization.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)

    rng = np.random.RandomState(random_state)
    barycenter = X_emb[rng.randint(X_emb.shape[0])].copy()

    lr = 0.05
    for it in range(max_iter):
        grad = _soft_dtw_grad(barycenter, X_emb, gamma)
        barycenter -= lr * grad
        lr *= 0.95  # decay

    return project_real_to_symbolic(barycenter, D_G)


def barycenter_edit_median(X_symbolic, n_alphabet, max_iter=20):
    """Baseline A7: Edit-distance median string via iterative local search.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences.
    n_alphabet : int
        Size of the symbol alphabet (symbols in [0, n_alphabet-1]).
    max_iter : int
        Maximum local search iterations.

    Returns
    -------
    np.ndarray of shape (variable,)
        Symbolic median sequence (integer-valued).
    """
    import Levenshtein

    def _from_str(s):
        return np.array([ord(c) - 33 for c in s], dtype=np.int64)

    strs = [_symbols_to_str(seq) for seq in X_symbolic]
    n = len(strs)

    # Step 1: Set median — sequence minimizing sum of edit distances
    dist_sums = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_sums[i] += Levenshtein.distance(strs[i], strs[j])
    median_idx = np.argmin(dist_sums)
    median = list(strs[median_idx])

    def _total_dist(candidate):
        s = ''.join(candidate)
        return sum(Levenshtein.distance(s, t) for t in strs)

    best_cost = _total_dist(median)
    alphabet = [chr(a + 33) for a in range(n_alphabet)]

    # Step 2: Iterative local search
    for _ in range(max_iter):
        improved = False
        for pos in range(len(median)):
            # Try substitution
            for sym in alphabet:
                if sym == median[pos]:
                    continue
                candidate = median.copy()
                candidate[pos] = sym
                cost = _total_dist(candidate)
                if cost < best_cost:
                    median = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break

            # Try deletion
            candidate = median[:pos] + median[pos + 1:]
            if len(candidate) > 0:
                cost = _total_dist(candidate)
                if cost < best_cost:
                    median = candidate
                    best_cost = cost
                    improved = True
                    break

        if not improved:
            break

    return _from_str(''.join(median))


# ---------------------------------------------------------------------------
# Tier B baselines (B1, B2, B3)
# ---------------------------------------------------------------------------

def barycenter_wasserstein(X_symbolic, D_G, reg=0.01):
    """Baseline B1: Wasserstein barycenter of symbol frequency histograms.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences.
    D_G : np.ndarray of shape (G, G)
        Prototype distance matrix (used as ground cost).
    reg : float
        Entropic regularization parameter.

    Returns
    -------
    np.ndarray of shape (G,)
        Barycenter histogram (sums to 1).
    """
    import ot

    G = D_G.shape[0]
    # Build duration-weighted histograms
    histograms = []
    for seq in X_symbolic:
        h = np.bincount(seq.astype(int), minlength=G).astype(float)
        h /= h.sum()
        histograms.append(h)

    A = np.column_stack(histograms)  # (G, n_sequences)
    weights = np.ones(len(histograms)) / len(histograms)
    M = D_G / D_G.max()  # Normalize cost matrix

    bary = ot.barycenter(A, M, reg, weights=weights)
    return bary


def barycenter_k_medoid(D_pairwise):
    """Baseline B2: k-Medoid — select existing sequence minimizing total distance.

    Parameters
    ----------
    D_pairwise : np.ndarray of shape (n_sequences, n_sequences)
        Pairwise distance matrix within the group.

    Returns
    -------
    int
        Index of the medoid sequence.
    """
    return int(np.argmin(D_pairwise.sum(axis=1)))


def barycenter_majority_voting(X_symbolic):
    """Baseline B3: Per-timestep majority voting (lock-step, no alignment).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (must be equal length).

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter via per-timestep mode.
    """
    result = np.zeros(X_symbolic.shape[1], dtype=np.int64)
    for t in range(X_symbolic.shape[1]):
        result[t] = int(stats.mode(X_symbolic[:, t], keepdims=False).mode)
    return result


# ---------------------------------------------------------------------------
# Native classification distances (one per method)
# ---------------------------------------------------------------------------

def dist_dtw(seq, bary, D_G):
    """DTW distance between two symbolic sequences in D_G-embedded space."""
    D = _dtw_cost_matrix(
        D_G[np.asarray(seq).astype(int)], D_G[np.asarray(bary).astype(int)],
    )
    return float(D[-1, -1])


def dist_soft_dtw(seq, bary, D_G, gamma=1.0):
    """Soft-DTW discrepancy between two symbolic sequences in D_G-embedded space."""
    return _soft_dtw_cost(
        D_G[np.asarray(seq).astype(int)], D_G[np.asarray(bary).astype(int)], gamma,
    )


def dist_edit(seq, bary):
    """Levenshtein edit distance between two symbolic sequences."""
    import Levenshtein
    return float(Levenshtein.distance(_symbols_to_str(seq), _symbols_to_str(bary)))


def dist_hamming(seq, bary):
    """Lock-step normalized Hamming distance (truncated to the shorter length)."""
    seq = np.asarray(seq)
    bary = np.asarray(bary)
    L = min(len(seq), len(bary))
    if L == 0:
        return 1.0
    return float(np.mean(seq[:L] != bary[:L]))


def dist_wasserstein_hist(seq, bary_hist, M):
    """Wasserstein distance between a sequence's symbol histogram and a barycenter histogram.

    Parameters
    ----------
    seq : np.ndarray
        Symbolic sequence (integer-valued).
    bary_hist : np.ndarray of shape (G,)
        Barycenter histogram (sums to 1).
    M : np.ndarray of shape (G, G)
        Normalized ground-cost matrix.
    """
    import ot
    G = len(bary_hist)
    h = np.bincount(np.asarray(seq).astype(int), minlength=G).astype(float)
    h /= h.sum()
    return float(ot.emd2(h, np.asarray(bary_hist, dtype=float), M))


def dist_rtwe(seq, bary, D_cost, nu=0.001, lmbda=1.0, window=None):
    """TW-TWE (registered Time Warp Edit) distance with a given ground cost.

    Used as the native distance for the proposed method (``D_cost = D_G``),
    the TWE ablation (``D_cost = ordinal_cost_matrix(G)``), and k-medoid.
    """
    from smartflat.engine.distances._rtwe import rtwe_distance
    return float(rtwe_distance(
        np.asarray(seq, dtype=np.float64),
        np.asarray(bary, dtype=np.float64),
        window=window, nu=nu, lmbda=lmbda,
        precomputed_distances=np.asarray(D_cost, dtype=np.float64),
    ))


def default_baseline_methods(D_G, gamma=1.0, nu=0.001, lmbda=1.0, window=None):
    """Build the six standard-baseline registry for :func:`evaluate_baselines`.

    Each baseline is paired with its NATIVE classification distance so the
    cross-method AUC comparison scores every method as it would actually be
    used: DBA-DTW->DTW, Soft-DTW->soft-DTW, edit-median->edit distance,
    Wasserstein->Wasserstein (histogram), k-medoid->TW-TWE, majority-voting->
    lock-step Hamming.

    Parameters
    ----------
    D_G : np.ndarray of shape (G, G)
        Prototype ground-cost matrix.
    gamma : float
        Soft-DTW smoothing parameter.
    nu, lmbda, window :
        TW-TWE hyperparameters for the k-medoid native distance; they must match
        the settings used to precompute the ``D_pairwise`` passed to
        :func:`evaluate_baselines`.

    Returns
    -------
    dict
        ``method_name -> spec`` registry (see :func:`evaluate_baselines`).
    """
    G = D_G.shape[0]
    M = D_G / D_G.max() if D_G.max() > 0 else D_G.copy()
    return {
        'dba_dtw': {
            'build': lambda X, seed: barycenter_dba_dtw(X, D_G, random_state=seed),
            'distance': lambda seq, bary: dist_dtw(seq, bary, D_G),
        },
        'soft_dtw': {
            'build': lambda X, seed: barycenter_soft_dtw(X, D_G, gamma=gamma, random_state=seed),
            'distance': lambda seq, bary: dist_soft_dtw(seq, bary, D_G, gamma=gamma),
        },
        'edit_median': {
            'build': lambda X, seed: barycenter_edit_median(X, n_alphabet=G),
            'distance': lambda seq, bary: dist_edit(seq, bary),
        },
        'wasserstein': {
            'build': lambda X, seed: barycenter_wasserstein(X, D_G),
            'distance': lambda seq, bary: dist_wasserstein_hist(seq, bary, M),
        },
        'k_medoid': {
            'kind': 'medoid',
            'distance': lambda seq, bary: dist_rtwe(seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
        },
        'majority_voting': {
            'build': lambda X, seed: barycenter_majority_voting(X),
            'distance': lambda seq, bary: dist_hamming(seq, bary),
        },
    }


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


# ---------------------------------------------------------------------------
# Evaluation framework
# ---------------------------------------------------------------------------

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
