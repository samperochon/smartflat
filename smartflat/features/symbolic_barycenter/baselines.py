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


def barycenter_mode_dba(X_symbolic, D_G, nu=0.001, lmbda=1.0, max_iter=10, random_state=None):
    """Mode-based DBA barycenter for categorical symbolic sequences.

    Mean-based DBA averages nominal prototype indices (e.g. symbols 2 and 70 -> 36),
    which is meaningless for categorical data and erases the symbol-frequency signal.
    Mode-based DBA instead warps every sequence to the current reference via the rTWE
    alignment path and takes the per-position MODE of the aligned symbols, iterating to
    convergence. The reference is initialised to the within-group rTWE medoid.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Ground-cost matrix used for the rTWE alignment.
    nu, lmbda : float
        rTWE stiffness / edit penalty.
    max_iter : int
        Maximum number of mode-DBA refinement iterations.
    random_state : int or None
        Unused; kept for a uniform ``build(X, seed)`` signature.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Mode-based symbolic barycenter.
    """
    from collections import Counter
    from smartflat.engine.distances._rtwe import (
        rtwe_alignment_path, rtwe_pairwise_distance,
    )
    X = np.asarray(X_symbolic).astype(int)
    if len(X) == 1:
        return X[0].copy()
    Dc = np.asarray(D_G, dtype=np.float64)
    Xa = X.astype(np.float64)[:, None, :]
    D = rtwe_pairwise_distance(Xa, nu=nu, lmbda=lmbda, precomputed_distances=Dc)
    ref = X[int(np.argmin(D.sum(axis=1)))].copy()
    for _ in range(max_iter):
        votes = [[] for _ in range(len(ref))]
        for s in X:
            path, _ = rtwe_alignment_path(
                s.astype(np.float64), ref.astype(np.float64), Dc, nu=nu, lmbda=lmbda,
            )
            for (i, j) in path:
                if 0 <= j < len(ref) and 0 <= i < len(s):
                    votes[j].append(int(s[i]))
        new_ref = np.array(
            [Counter(v).most_common(1)[0][0] if v else int(ref[j])
             for j, v in enumerate(votes)],
            dtype=np.int64,
        )
        if np.array_equal(new_ref, ref):
            break
        ref = new_ref
    return ref


def barycenter_mean_rtwe_dba(X_symbolic, D_G, nu=1e-4, lmbda=0.1, max_iter=50, tol=1e-7,
                             init='random', project='round', allow_background=False,
                             random_state=None):
    """Mean-based (Petitjean) DBA with rTWE alignment -- faithful thesis reconstruction.

    The paper's barycenter used a forked aeon ``elastic_barycenter_average(
    method='petitjean', distance='twe', precomputed_distances=D_G, ...)`` that is not
    available here (stock aeon lacks ``precomputed_distances``). This reconstructs it:
    each sequence is warped to the current reference via the rTWE alignment path
    (D_G inner cost), and every reference position is updated to the MEAN of the
    aligned symbol indices. Mean-averaging nominal category indices is categorically
    questionable (the motivation for :func:`barycenter_mode_dba`); this function
    exists to reproduce/diagnose the thesis behaviour and to ablate mean-vs-mode.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Ground cost used for the rTWE alignment.
    nu, lmbda : float
        rTWE stiffness / edit penalty (thesis: 1e-4 / 0.1).
    max_iter, tol : int, float
        Petitjean iteration budget / convergence tolerance.
    init : {'random', 'medoid'}
        'random' picks a seeded member (thesis); 'medoid' uses the rTWE medoid
        (deterministic, ignores ``random_state``).
    project : {'round', 'dg', 'none'}
        How to turn each position's real-valued mean into the returned barycenter:
        'round' = nearest integer index (default; usable by the rTWE p_match feature);
        'dg' = symbol minimizing summed D_G to the aligned symbols (Frechet mean in the
        D_G geometry, categorically principled); 'none' = keep the fractional mean
        (thesis-faithful; score with the stock-TWE Match feature, not the rTWE one).
    allow_background : bool
        If True, background (code 0) symbols are excluded from the position means.
    random_state : int or None
        Seed for 'random' init.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Mean-based barycenter (int if ``project`` in {'round', 'dg'}, float if 'none').
    """
    from smartflat.engine.distances._rtwe import (
        rtwe_alignment_path, rtwe_pairwise_distance,
    )
    X = np.asarray(X_symbolic).astype(int)
    Dc = np.asarray(D_G, dtype=np.float64)
    G = Dc.shape[0]
    if len(X) == 1:
        return X[0].astype(np.float64) if project == 'none' else X[0].copy()
    if init == 'medoid':
        Xa = X.astype(np.float64)[:, None, :]
        D = rtwe_pairwise_distance(Xa, nu=nu, lmbda=lmbda, precomputed_distances=Dc)
        ref = X[int(np.argmin(D.sum(axis=1)))].astype(np.float64).copy()
    else:
        rng = np.random.default_rng(random_state)
        ref = X[int(rng.integers(len(X)))].astype(np.float64).copy()
    L = len(ref)
    votes = [[] for _ in range(L)]
    for _ in range(max_iter):
        ref_align = np.clip(np.rint(ref), 0, G - 1)  # integer ref for D_G indexing
        sums = np.zeros(L)
        counts = np.zeros(L)
        votes = [[] for _ in range(L)]
        for s in X:
            path, _ = rtwe_alignment_path(
                s.astype(np.float64), ref_align, Dc, nu=nu, lmbda=lmbda,
            )
            for (i, j) in path:
                if 0 <= j < L and 0 <= i < len(s):
                    si = int(s[i])
                    if allow_background and si == 0:
                        continue
                    sums[j] += si
                    counts[j] += 1
                    votes[j].append(si)
        new_ref = ref.copy()
        nz = counts > 0
        new_ref[nz] = sums[nz] / counts[nz]
        if np.allclose(new_ref, ref, atol=tol):
            ref = new_ref
            break
        ref = new_ref
    if project == 'none':
        return ref
    if project == 'dg':
        out = np.empty(L, dtype=np.int64)
        for j in range(L):
            if votes[j]:
                a = np.asarray(votes[j])
                out[j] = int(np.argmin(Dc[:, a].sum(axis=1)))
            else:
                out[j] = int(np.clip(round(float(ref[j])), 0, G - 1))
        return out
    return np.clip(np.rint(ref).astype(np.int64), 0, G - 1)


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


def pmatch_to_barycenter(seq, bary, D_G, nu=0.001, lmbda=1.0, window=None):
    """Proportion of exactly-matching symbols along the rTWE alignment between a sequence
    and a barycenter. This is the paper's discriminative feature; higher means closer."""
    from smartflat.engine.distances._rtwe import rtwe_alignment_path
    path, _ = rtwe_alignment_path(
        np.asarray(seq, dtype=np.float64), np.asarray(bary, dtype=np.float64),
        np.asarray(D_G, dtype=np.float64), window=window, nu=nu, lmbda=lmbda,
    )
    s = np.asarray(seq).astype(int)
    b = np.asarray(bary).astype(int)
    nmatch = ntot = 0
    for k in range(1, len(path)):
        di = path[k][0] - path[k - 1][0]
        dj = path[k][1] - path[k - 1][1]
        if di == 1 and dj == 1:
            ntot += 1
            if s[path[k][0]] == b[path[k][1]]:
                nmatch += 1
    return nmatch / max(ntot, 1)


def dist_neg_pmatch(seq, bary, D_G, nu=0.001, lmbda=1.0, window=None):
    """Negative p_match, usable as a 'distance' in :func:`evaluate_baselines` (lower=closer)."""
    return -pmatch_to_barycenter(seq, bary, D_G, nu=nu, lmbda=lmbda, window=window)


def pmatch_to_barycenter_stock_twe(seq, bary, nu=1e-4, lmbda=0.1):
    """Thesis ``Match_normalized`` feature: fraction of diagonal 'Match' steps along the
    STOCK aeon TWE alignment (Euclidean inner cost, no D_G) where the (rounded) barycenter
    symbol equals the sequence symbol.

    Unlike :func:`pmatch_to_barycenter` (which uses the rTWE alignment and so needs an
    integer barycenter to index D_G), this uses plain TWE and tolerates a fractional
    barycenter -- it is the faithful scorer for ``barycenter_mean_rtwe_dba(project='none')``.
    """
    from aeon.distances import twe_alignment_path
    b = np.asarray(bary, dtype=np.float64).reshape(1, -1)
    s = np.asarray(seq, dtype=np.float64).reshape(1, -1)
    path, _ = twe_alignment_path(b, s, nu=nu, lmbda=lmbda)
    bi = np.rint(np.asarray(bary, dtype=np.float64)).astype(int)
    si = np.rint(np.asarray(seq, dtype=np.float64)).astype(int)
    nmatch = ntot = 0
    for k in range(1, len(path)):
        di = path[k][0] - path[k - 1][0]
        dj = path[k][1] - path[k - 1][1]
        if di == 1 and dj == 1:
            ntot += 1
            if bi[path[k][0]] == si[path[k][1]]:
                nmatch += 1
    return nmatch / max(ntot, 1)


def dist_neg_pmatch_stock(seq, bary, nu=1e-4, lmbda=0.1):
    """Negative stock-TWE Match_normalized, usable as a 'distance' (lower=closer)."""
    return -pmatch_to_barycenter_stock_twe(seq, bary, nu=nu, lmbda=lmbda)


# ---------------------------------------------------------------------------
# Experimental methods (beat-majority-voting) -- registered via
# extra_experiment_methods(), kept out of the frozen default_baseline_methods set.
# ---------------------------------------------------------------------------

def _transition_matrix(seq, G):
    """Row-normalized bigram transition matrix (G, G) of one symbolic sequence."""
    s = np.asarray(seq).astype(int)
    M = np.zeros((G, G), dtype=np.float64)
    if len(s) > 1:
        np.add.at(M, (s[:-1], s[1:]), 1.0)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def barycenter_transition_matrix(X_symbolic, G):
    """Group barycenter = mean of per-sequence row-normalized bigram transition matrices.

    Encodes which action tends to follow which -- temporal-ordering structure that the
    unigram symbol-frequency histogram discards. Each administration contributes equally
    (mean over per-sequence row-stochastic matrices); the result is re-row-normalized so
    every row that has any mass sums to 1. The 'barycenter' is a (G, G) stochastic matrix;
    pair it with :func:`dist_transition`.
    """
    T = np.mean([_transition_matrix(s, G) for s in X_symbolic], axis=0)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return T / rs


def dist_transition(seq, bary_T):
    """Frobenius distance between a sequence's transition matrix and a barycenter matrix."""
    G = np.asarray(bary_T).shape[0]
    return float(np.linalg.norm(_transition_matrix(seq, G) - np.asarray(bary_T)))


def dist_eshape_dtw(seq, bary, D_cost, nu=1e-4, lmbda=0.1, window=None, step_sequ=2):
    """Edit-Shape DTW outer-loop distance (rTWE inner cost) between a sequence and barycenter.

    Uses the temporal-shape structure of the outer DTW alignment over the rTWE inner cost.
    O(L^2) pure-Python rTWE calls -- budget tightly (small L, larger ``step_sequ``, fewer
    splits), like ``soft_dtw``.
    """
    from smartflat.engine.distances._eshape_dtw import eshape_dtw_distance
    return float(eshape_dtw_distance(
        np.asarray(seq, dtype=np.float64).reshape(1, -1),
        np.asarray(bary, dtype=np.float64).reshape(1, -1),
        window=window, nu=nu, lmbda=lmbda,
        precomputed_distances=np.asarray(D_cost, dtype=np.float64), step_sequ=step_sequ,
    ))


def barycenter_soft_mode_dba(X_symbolic, D_G, nu=1e-4, lmbda=0.1, beta=4.0, max_iter=10,
                             random_state=None):
    """Soft categorical DBA: per-position soft voting in the D_G geometry.

    Like :func:`barycenter_mode_dba`, but each aligned symbol ``s`` contributes a soft
    distribution ``softmax(-beta * D_G[:, s])`` over candidate symbols (rather than a
    single hard vote), accumulated per reference position; the position is set to the
    argmax. Symbols close under D_G reinforce one another, reducing the mode-collapse of
    the hard per-position majority while staying categorical. ``beta -> inf`` recovers
    the hard mode.
    """
    from smartflat.engine.distances._rtwe import (
        rtwe_alignment_path, rtwe_pairwise_distance,
    )
    X = np.asarray(X_symbolic).astype(int)
    Dc = np.asarray(D_G, dtype=np.float64)
    G = Dc.shape[0]
    if len(X) == 1:
        return X[0].copy()
    Z = np.exp(-beta * (Dc - Dc.min(axis=0, keepdims=True)))
    P = Z / Z.sum(axis=0, keepdims=True)  # P[c, s] = soft mass on candidate c from symbol s
    Xa = X.astype(np.float64)[:, None, :]
    D = rtwe_pairwise_distance(Xa, nu=nu, lmbda=lmbda, precomputed_distances=Dc)
    ref = X[int(np.argmin(D.sum(axis=1)))].copy()
    for _ in range(max_iter):
        acc = np.zeros((len(ref), G))
        for s in X:
            path, _ = rtwe_alignment_path(
                s.astype(np.float64), ref.astype(np.float64), Dc, nu=nu, lmbda=lmbda,
            )
            for (i, j) in path:
                if 0 <= j < len(ref) and 0 <= i < len(s):
                    acc[j] += P[:, int(s[i])]
        new_ref = np.array(
            [int(np.argmax(acc[j])) if acc[j].sum() > 0 else int(ref[j])
             for j in range(len(ref))],
            dtype=np.int64,
        )
        if np.array_equal(new_ref, ref):
            break
        ref = new_ref
    return ref


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


def extra_experiment_methods(D_G, G, nu=1e-4, lmbda=0.1, window=None, step_sequ=2):
    """Registry of the beat-majority-voting experimental methods.

    Kept separate from :func:`default_baseline_methods` so the six-baseline contract
    test (``test_six_baselines_present``) stays stable. Merge with ``|`` in the notebook.

    - ``'transition'`` : bigram transition-matrix barycenter + Frobenius distance --
      temporal-ordering structure the unigram histogram discards.
    - ``'eshape_dtw'`` : mode-DBA barycenter scored by the Edit-Shape DTW outer loop
      (rTWE inner cost). EXPENSIVE -- run at small L / reduced budget.
    - ``'shape_dba'``  : soft-mode (D_G-soft-vote) DBA barycenter + rTWE p_match feature.

    Returns
    -------
    dict
        ``method_name -> spec`` registry (see :func:`evaluate_baselines`).
    """
    return {
        'transition': {
            'build': lambda X, seed: barycenter_transition_matrix(X, G),
            'distance': lambda seq, bary: dist_transition(seq, bary),
        },
        'eshape_dtw': {
            'build': lambda X, seed: barycenter_mode_dba(X, D_G, nu=nu, lmbda=lmbda),
            'distance': lambda seq, bary: dist_eshape_dtw(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window, step_sequ=step_sequ),
        },
        'shape_dba': {
            'build': lambda X, seed: barycenter_soft_mode_dba(
                X, D_G, nu=nu, lmbda=lmbda, random_state=seed),
            'distance': lambda seq, bary: dist_neg_pmatch(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
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


# ---------------------------------------------------------------------------
# Track A: decisive ordering-vs-frequency tests
# (incremental AUC of bigram ordering over the unigram histogram; bootstrap CIs)
# ---------------------------------------------------------------------------

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


def _pairwise_subsets(labels):
    """Yield ``(comparison_name, mask, y_binary)`` for the three paper comparisons.

    ``mask`` selects the two groups from ``labels`` (HEALTHY/RIL/TBI or the pooled
    CONTROL/PATIENT); ``y_binary`` is 1 for the second group of the pair.
    """
    labels = np.asarray(labels, dtype=object)
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
