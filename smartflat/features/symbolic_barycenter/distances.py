"""rTWE operating-point constants, symbolic<->real embedding/decode, DTW &
soft-DTW primitives, and every ``dist_*`` / ``pmatch_*`` distance function.

Split out of ``baselines.py`` (arc-audit Phase 2); imported nothing internal —
the vendored ``_rtwe`` / ``_eshape_dtw`` engines stay lazily imported inside the
functions that use them. Re-exported unchanged via the ``baselines`` shim.
"""


import numpy as np


# ---------------------------------------------------------------------------
# Canonical rTWE operating point (smartflat layer)
# ---------------------------------------------------------------------------
# Every smartflat-layer barycenter/distance/registry default references these so
# "the rTWE cost" has ONE operating point. The vendored engine/distances/_rtwe.py
# kernel keeps its own aeon-style defaults; smartflat callers always pass these
# explicitly. Unified in Kickoff M (arc-audit Phase 1); see ARC_AUDIT.md Dimension 5.
RTWE_NU = 1e-4      # rTWE stiffness    (was drifting: 0.001 vs 1e-4)
RTWE_LMBDA = 0.1    # rTWE edit penalty (was drifting: 1.0   vs 0.1)


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


def project_real_to_symbolic(X_real, D_G, decode='euclidean'):
    """Project real-valued embedded sequences back to symbolic via nearest prototype.

    Parameters
    ----------
    X_real : np.ndarray of shape (n_sequences, n_timepoints, G) or (n_timepoints, G)
        Real-valued embedded sequences.
    D_G : np.ndarray of shape (G, G)
        Symmetric prototype distance matrix.
    decode : {'euclidean', 'dg'}
        Decode rule for snapping a real profile ``m`` (a point in the ``D_G``-row
        embedding) back to a hard symbol (Lever 2 -- discreteness fairness):

        - ``'euclidean'`` (default, unchanged): nearest ``D_G`` row,
          ``argmin_c ||m - D_G[c]||_2`` -- Euclidean in distance-profile space.
        - ``'dg'``: ground-cost-consistent decode ``argmin_c m[c]``. For a DBA
          arithmetic mean ``m = mean_i D_G[s_i]`` over the aligned bag ``{s_i}``,
          ``m[c] = mean_i D_G[c, s_i]`` is the mean rTWE ground cost from candidate
          ``c`` to the bag (``D_G`` symmetric), so ``argmin_c m[c]`` is the
          vocabulary-restricted 1-medoid under ``D_G`` -- the same geometry the
          harness scores in (rTWE's substitution cost is the direct lookup
          ``D_G[a, b]``). Exact for DBA means; a profile-argmin heuristic for the
          optimised soft-DTW/SSG centroids. Lossless round-trip for pure symbols
          (``D_G`` diagonal is 0), provided the off-diagonal is strictly positive.

    Returns
    -------
    np.ndarray
        Integer-valued symbolic sequences (same shape as input minus last dim).
    """
    if decode not in ('euclidean', 'dg'):
        raise ValueError(f"decode must be 'euclidean' or 'dg', got {decode!r}")
    if decode == 'dg':
        # Ground-cost 1-medoid: argmin_c m[c]. axis=-1 covers (T, G) -> (T,)
        # and (N, T, G) -> (N, T) uniformly.
        return np.argmin(X_real, axis=-1)
    if X_real.ndim == 2:
        # Single sequence: (T, G)
        dists = np.linalg.norm(X_real[:, None, :] - D_G[None, :, :], axis=2)
        return np.argmin(dists, axis=1)
    # Batch: (N, T, G)
    dists = np.linalg.norm(X_real[:, :, None, :] - D_G[None, None, :, :], axis=3)
    return np.argmin(dists, axis=2)


def _snap_and_reembed(bary_real, D_G, decode):
    """Snap a single real barycenter ``(T, G)`` to symbols and re-embed it.

    The atomic categorical step (Lever 3): decode the running real barycenter to a
    valid symbol string via :func:`project_real_to_symbolic`, then map it back to its
    ``D_G`` rows via :func:`embed_symbolic_to_real`. Returns ``(snapped, reembedded)``
    with ``snapped`` an integer ``(T,)`` sequence and ``reembedded`` its ``(T, G)``
    embedding, so callers can keep the reference categorical every iteration.
    """
    snapped = project_real_to_symbolic(np.asarray(bary_real), D_G, decode=decode)  # (T,)
    return snapped, embed_symbolic_to_real(snapped[None, :], D_G)[0]


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


def dist_rtwe(seq, bary, D_cost, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None):
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


def pmatch_to_barycenter(seq, bary, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None):
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


def dist_neg_pmatch(seq, bary, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None):
    """Negative p_match, usable as a 'distance' in :func:`evaluate_baselines` (lower=closer)."""
    return -pmatch_to_barycenter(seq, bary, D_G, nu=nu, lmbda=lmbda, window=window)


def pmatch_to_barycenter_stock_twe(seq, bary, nu=RTWE_NU, lmbda=RTWE_LMBDA):
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


def dist_neg_pmatch_stock(seq, bary, nu=RTWE_NU, lmbda=RTWE_LMBDA):
    """Negative stock-TWE Match_normalized, usable as a 'distance' (lower=closer)."""
    return -pmatch_to_barycenter_stock_twe(seq, bary, nu=nu, lmbda=lmbda)


def _transition_matrix(seq, G):
    """Row-normalized bigram transition matrix (G, G) of one symbolic sequence."""
    s = np.asarray(seq).astype(int)
    M = np.zeros((G, G), dtype=np.float64)
    if len(s) > 1:
        np.add.at(M, (s[:-1], s[1:]), 1.0)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def dist_transition(seq, bary_T):
    """Frobenius distance between a sequence's transition matrix and a barycenter matrix."""
    G = np.asarray(bary_T).shape[0]
    return float(np.linalg.norm(_transition_matrix(seq, G) - np.asarray(bary_T)))


def dist_eshape_dtw(seq, bary, D_cost, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None, step_sequ=2):
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


def _classical_mds(D_G, dim=None):
    """Classical (Torgerson) MDS embedding of a distance matrix.

    Returns coordinates ``E`` of shape ``(G, k)`` whose pairwise Euclidean distances best
    approximate ``D_G`` (exact when ``D_G`` is Euclidean). Deterministic (symmetric
    eigendecomposition), unlike SMACOF, so the FGW node features and the decode step are
    reproducible. Keeps the positive-eigenvalue axes (optionally capped to ``dim``).
    """
    D = np.asarray(D_G, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J.dot(D ** 2).dot(J)               # double-centred squared distances
    w, V = np.linalg.eigh(B)                       # ascending eigenvalues
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    keep = w > 1e-9
    if dim is not None:
        cap = np.zeros(n, dtype=bool)
        cap[:dim] = True
        keep &= cap
    return V[:, keep] * np.sqrt(w[keep])
