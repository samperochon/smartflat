"""The ``barycenter_*`` constructors — one per averaging method, uniform signature.

Split out of ``baselines.py`` (arc-audit Phase 2). Pulls embedding/decode and
cost primitives from :mod:`.distances`; the vendored rTWE engine stays lazily
imported inside the functions. Re-exported unchanged via the ``baselines`` shim.
"""


import numpy as np
from scipy import stats


from .distances import (
    RTWE_NU, RTWE_LMBDA, embed_symbolic_to_real, project_real_to_symbolic, _snap_and_reembed, _symbols_to_str, _dtw_alignment, _soft_dtw_grad, _transition_matrix, _classical_mds,
)


def barycenter_dba_dtw(X_symbolic, D_G, max_iter=30, tol=1e-5, random_state=None,
                       discretise_each_iter=False, decode='euclidean', max_iters=None):
    """Baseline A5: DBA with standard DTW on prototype-distance embeddings.

    Pure numpy/scipy implementation (Petitjean et al. 2011).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences.
    D_G : np.ndarray of shape (G, G)
        Prototype distance matrix.
    max_iter : int
        Maximum DBA iterations.
    max_iters : int or None
        Deprecated alias for ``max_iter`` (emits ``DeprecationWarning`` if set);
        kept for back-compat with pre-Kickoff-M callers.
    tol : float
        Convergence tolerance (continuous mode only).
    random_state : int or None
        Random seed for initialization.
    discretise_each_iter : bool
        Lever 3 (categorical variant): when True, snap the running barycenter back
        to symbols and re-embed **after each iteration**, so the reference is a valid
        symbol string throughout (like mode-DBA, but mean-then-snap rather than a hard
        vote). Convergence then uses the discrete fixed-point (unchanged snapped
        symbols) instead of the real-valued ``tol`` check, which can oscillate once
        quantised. Default False = the standard continuous DBA (snap once at the end).
    decode : {'euclidean', 'dg'}
        Decode rule handed to :func:`project_real_to_symbolic` (Lever 2). Default
        ``'euclidean'`` = unchanged.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    if max_iters is not None:  # deprecated alias for max_iter (Kickoff M rename)
        import warnings
        warnings.warn(
            "barycenter_dba_dtw(max_iters=...) is deprecated; use max_iter=...",
            DeprecationWarning, stacklevel=2,
        )
        max_iter = max_iters

    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)
    N, T, G = X_emb.shape

    rng = np.random.RandomState(random_state)
    barycenter = X_emb[rng.randint(N)].copy()  # (T, G)
    prev_cost = np.inf
    prev_snapped = None

    for _ in range(max_iter):
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

        if discretise_each_iter:
            # Keep the reference categorical: snap to symbols + re-embed each iter.
            snapped, barycenter = _snap_and_reembed(barycenter, D_G, decode)
            if prev_snapped is not None and np.array_equal(snapped, prev_snapped):
                break
            prev_snapped = snapped
        else:
            if abs(prev_cost - total_cost) < tol:
                break
            prev_cost = total_cost

    return project_real_to_symbolic(barycenter, D_G, decode=decode)


def barycenter_soft_dtw(X_symbolic, D_G, gamma=1.0, max_iter=30, random_state=None,
                        decode='euclidean'):
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
    decode : {'euclidean', 'dg'}
        Decode rule handed to :func:`project_real_to_symbolic` (Lever 2). Default
        ``'euclidean'`` = unchanged; parity with the tslearn sibling
        :func:`barycenter_softdtw`.

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

    return project_real_to_symbolic(barycenter, D_G, decode=decode)


def barycenter_softdtw(X_symbolic, D_G, gamma=1.0, max_iter=50, random_state=None,
                       decode='euclidean'):
    """Soft-DTW barycenter (Cuturi & Blondel, ICML 2017) via tslearn's L-BFGS-B solver.

    Library-backed counterpart to the hand-rolled :func:`barycenter_soft_dtw`: the
    symbolic sequences are embedded into real space by their ``D_G`` rows
    (:func:`embed_symbolic_to_real`), averaged with ``tslearn.barycenters.softdtw_barycenter``
    (minimises the Frobenius-regularised soft-DTW Frechet functional), and the continuous
    centroid is decoded back to hard symbols by nearest ``D_G`` row
    (:func:`project_real_to_symbolic`). ``gamma -> 0`` recovers DTW.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Prototype ground-cost matrix (the embedding).
    gamma : float
        Soft-DTW smoothing parameter (larger = smoother).
    max_iter : int
        Maximum L-BFGS-B iterations.
    random_state : int or None
        Accepted for the ``{build, distance}`` registry contract but unused: the
        L-BFGS-B barycenter is deterministic (Euclidean-mean initialisation).
    decode : {'euclidean', 'dg'}
        Decode rule handed to :func:`project_real_to_symbolic` (Lever 2). Default
        ``'euclidean'`` = unchanged.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    try:
        from tslearn.barycenters import softdtw_barycenter
    except ImportError as exc:  # pragma: no cover - exercised only without tslearn
        raise ImportError(
            "barycenter_softdtw requires tslearn. Install with: pip install tslearn"
        ) from exc
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)
    bary = softdtw_barycenter(X_emb, gamma=gamma, max_iter=max_iter)  # (T, G)
    return project_real_to_symbolic(np.asarray(bary), D_G, decode=decode)


def barycenter_ssg(X_symbolic, D_G, max_iter=30, random_state=None, decode='euclidean'):
    """Stochastic-subgradient DTW averaging (SSG; Schultz & Jain, Pattern Recognition 2018).

    Uses ``tslearn.barycenters.dtw_barycenter_averaging_subgradient`` on the ``D_G``
    embedding (:func:`embed_symbolic_to_real`), decoding the continuous centroid back to
    hard symbols by nearest ``D_G`` row (:func:`project_real_to_symbolic`). Unlike DBA's
    full batch mean-under-alignment update, SSG takes stochastic subgradient steps over the
    DTW Frechet functional; the update order is seeded by ``random_state`` (reproducible).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Prototype ground-cost matrix (the embedding).
    max_iter : int
        Maximum subgradient epochs.
    random_state : int or None
        Seed for the stochastic update order (determinism guarantee).
    decode : {'euclidean', 'dg'}
        Decode rule handed to :func:`project_real_to_symbolic` (Lever 2). Default
        ``'euclidean'`` = unchanged.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    try:
        from tslearn.barycenters import dtw_barycenter_averaging_subgradient
    except ImportError as exc:  # pragma: no cover - exercised only without tslearn
        raise ImportError(
            "barycenter_ssg requires tslearn. Install with: pip install tslearn"
        ) from exc
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)
    bary = dtw_barycenter_averaging_subgradient(
        X_emb, max_iter=max_iter, random_state=random_state,
    )  # (T, G)
    return project_real_to_symbolic(np.asarray(bary), D_G, decode=decode)


def _categorical_outer_loop(X_emb, D_G, run, n_rounds, decode):
    """Wrap a library averager in an outer loop that re-discretises between rounds.

    Lever 3 for the tslearn-backed averagers (which cannot snap mid-solve): call
    ``run(X_emb, ref)`` with a small inner ``max_iter``, snap the result to symbols +
    re-embed (:func:`_snap_and_reembed`), and feed that back as the init for the next
    round -- so the reference is a valid symbol string throughout. ``ref`` is ``None``
    on the first round (tslearn uses its default init). Early-exits at the discrete
    fixed-point (unchanged snapped symbols). Returns the final integer ``(T,)`` sequence.
    """
    ref = None
    prev = None
    snapped = None
    for _ in range(n_rounds):
        bary = run(X_emb, ref)
        snapped, ref = _snap_and_reembed(bary, D_G, decode)
        if prev is not None and np.array_equal(snapped, prev):
            break
        prev = snapped
    return snapped


def barycenter_ssg_cat(X_symbolic, D_G, n_rounds=6, inner_max_iter=8, decode='dg',
                       random_state=None):
    """Categorical SSG barycenter -- Lever 3 counterpart of :func:`barycenter_ssg`.

    Runs ``tslearn.barycenters.dtw_barycenter_averaging_subgradient`` inside
    :func:`_categorical_outer_loop`, re-discretising (snap + re-embed) after each round
    so the reference stays a valid symbol string throughout (the "stay categorical"
    property of mode-DBA, applied to SSG). ``decode='dg'`` makes each per-round snap the
    ground-cost 1-medoid (:func:`project_real_to_symbolic`). Reproducible under a fixed
    ``random_state`` (fixed init each round + fixed seed).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Prototype ground-cost matrix (the embedding).
    n_rounds : int
        Outer re-discretisation rounds.
    inner_max_iter : int
        Subgradient epochs per round (tslearn's ``max_iter``).
    decode : {'euclidean', 'dg'}
        Per-round decode rule; ``'dg'`` = ground-cost medoid (default).
    random_state : int or None
        Seed for the stochastic update order (determinism guarantee).

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    try:
        from tslearn.barycenters import dtw_barycenter_averaging_subgradient
    except ImportError as exc:  # pragma: no cover - exercised only without tslearn
        raise ImportError(
            "barycenter_ssg_cat requires tslearn. Install with: pip install tslearn"
        ) from exc
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)

    def run(x_emb, ref):
        return dtw_barycenter_averaging_subgradient(
            x_emb, init_barycenter=ref, max_iter=inner_max_iter, random_state=random_state)

    return _categorical_outer_loop(X_emb, D_G, run, n_rounds, decode)


def barycenter_softdtw_cat(X_symbolic, D_G, gamma=1.0, n_rounds=6, inner_max_iter=8,
                           decode='dg', random_state=None):
    """Categorical Soft-DTW barycenter -- Lever 3 counterpart of :func:`barycenter_softdtw`.

    Runs ``tslearn.barycenters.softdtw_barycenter`` inside :func:`_categorical_outer_loop`,
    re-discretising (snap + re-embed) after each round so the reference stays a valid symbol
    string throughout. ``decode='dg'`` makes each per-round snap the ground-cost 1-medoid.
    Deterministic (L-BFGS-B from the re-embedded init each round).

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length).
    D_G : np.ndarray of shape (G, G)
        Prototype ground-cost matrix (the embedding).
    gamma : float
        Soft-DTW smoothing parameter.
    n_rounds : int
        Outer re-discretisation rounds.
    inner_max_iter : int
        L-BFGS-B iterations per round (tslearn's ``max_iter``).
    decode : {'euclidean', 'dg'}
        Per-round decode rule; ``'dg'`` = ground-cost medoid (default).
    random_state : int or None
        Accepted for the ``{build, distance}`` registry contract but unused (deterministic).

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Symbolic barycenter sequence (integer-valued).
    """
    try:
        from tslearn.barycenters import softdtw_barycenter
    except ImportError as exc:  # pragma: no cover - exercised only without tslearn
        raise ImportError(
            "barycenter_softdtw_cat requires tslearn. Install with: pip install tslearn"
        ) from exc
    X_emb = embed_symbolic_to_real(X_symbolic, D_G)  # (N, T, G)

    def run(x_emb, ref):
        return softdtw_barycenter(x_emb, gamma=gamma, max_iter=inner_max_iter, init=ref)

    return _categorical_outer_loop(X_emb, D_G, run, n_rounds, decode)


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


def barycenter_mode_dba(X_symbolic, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, max_iter=10, random_state=None):
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


def barycenter_msa_consensus(X_symbolic, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None,
                             pseudocount=1.0, occupancy=0.5, random_state=None):
    """Center-star MSA + profile positional consensus (Family A, native-categorical).

    A statistically-principled positional consensus for symbolic sequences that stays
    entirely in symbol space -- no ``D_G`` embedding and no decode (contrast the Family-B
    ``dba_dtw``/``soft_dtw_bary``/``ssg``/``fgw_*``). The cooking task is a *recipe*, so a
    per-position probabilistic consensus is the natural Family-A generalisation of
    :func:`barycenter_majority_voting` (multiple alignment instead of lock-step) and of
    :func:`barycenter_mode_dba` (a proper multiple alignment with insertion columns and an
    occupancy rule, instead of an iterative single reference whose length is clamped to the
    reference).

    Algorithm -- **center-star MSA** (Gusfield 1993; 2-approximation under a metric ground cost)
    with a **profile per-column consensus** (Durbin, Eddy, Krogh & Mitchison 1998):

    1. Pick the star **center** = within-group rTWE medoid (argmin summed pairwise rTWE).
    2. Align every member pairwise to the center via :func:`rtwe_alignment_path` (the same
       vendored aligner used by ``mode_dba``); the center's positions are the alignment
       backbone, extra member symbols mapped to one center column are **insertions**.
    3. **Merge** the pairwise-to-center alignments into one MSA, padding insertions with
       gaps ("once a gap, always a gap"; Feng-Doolittle 1987 / ClustalW, Thompson 1994).
    4. **Profile consensus:** per MSA column, add ``pseudocount`` (Laplace) to the symbol
       counts and take the argmax; ``mode`` voting is the ``pseudocount -> 0`` special case.
    5. **Occupancy rule:** keep only columns whose non-gap occupancy >= ``occupancy`` (the
       50% match-state rule; Durbin et al. 1998, Ch. 5), yielding a gap-free symbol sequence.

    Parameters
    ----------
    X_symbolic : np.ndarray of shape (n_sequences, n_timepoints)
        Integer-valued symbolic sequences (equal length under the L=128 harness).
    D_G : np.ndarray of shape (G, G)
        Ground-cost matrix used inside the rTWE alignment / medoid selection.
    nu, lmbda : float
        rTWE stiffness / edit penalty.
    window : float or None
        Sakoe-Chiba band passed to the rTWE aligner.
    pseudocount : float
        Laplace pseudocount added to every symbol's per-column count (>0 -> profile,
        ->0 -> hard mode).
    occupancy : float
        Minimum non-gap fraction for a column to survive into the consensus.
    random_state : int or None
        Unused; the method is deterministic. Kept for a uniform ``build(X, seed)`` signature.

    Returns
    -------
    np.ndarray of shape (n_consensus_timepoints,)
        Gap-free symbolic barycenter; a genuine symbol sequence in ``[0, G)``.

    References
    ----------
    Gusfield (1993), Bull. Math. Biol. 55:141. Feng & Doolittle (1987), J. Mol. Evol.
    25:351. Thompson, Higgins & Gibson (1994), Nucleic Acids Res. 22:4673. Durbin, Eddy,
    Krogh & Mitchison (1998), Biological Sequence Analysis, Cambridge Univ. Press, Ch. 5.
    """
    from smartflat.engine.distances._rtwe import (
        rtwe_alignment_path, rtwe_pairwise_distance,
    )
    X = np.asarray(X_symbolic).astype(int)
    if len(X) == 1:
        return X[0].copy()
    n = len(X)
    G = int(np.asarray(D_G).shape[0])
    Dc = np.asarray(D_G, dtype=np.float64)

    # 1. star center = within-group rTWE medoid.
    Xa = X.astype(np.float64)[:, None, :]
    D = rtwe_pairwise_distance(Xa, nu=nu, lmbda=lmbda, window=window,
                               precomputed_distances=Dc)
    c = int(np.argmin(D.sum(axis=1)))
    center = X[c]
    Lc = len(center)

    # 2-3. align every member to the center; per center column, the ordered symbols each
    # member contributes (>1 => insertion run). GAP is the sentinel -1.
    GAP = -1
    mapping = [[[] for _ in range(Lc)] for _ in range(n)]  # mapping[k][i] -> [symbols]
    for k in range(n):
        member = X[k]
        Lm = len(member)
        path, _ = rtwe_alignment_path(
            center.astype(np.float64), member.astype(np.float64), Dc,
            window=window, nu=nu, lmbda=lmbda,
        )
        for (i, j) in path:               # i -> center col, j -> member pos
            if 0 <= i < Lc and 0 <= j < Lm:
                mapping[k][i].append(int(member[j]))

    # width of each center column = 1 match sub-column + its insertion sub-columns.
    widths = [max(1, max(len(mapping[k][i]) for k in range(n))) for i in range(Lc)]

    # 4-5. per (center col, sub-col): profile argmax over non-gap symbols + occupancy gate.
    consensus, occ = [], []
    for i in range(Lc):
        for sub in range(widths[i]):
            counts = np.full(G, float(pseudocount))
            n_nongap = 0
            for k in range(n):
                syms = mapping[k][i]
                sym = syms[sub] if sub < len(syms) else GAP
                if sym != GAP:
                    counts[sym] += 1.0
                    n_nongap += 1
            consensus.append(int(np.argmax(counts)))       # ties -> lowest symbol index
            occ.append(n_nongap / n)
    consensus = np.asarray(consensus, dtype=np.int64)
    occ = np.asarray(occ)

    keep = occ >= occupancy
    if not keep.any():                    # never emit an empty barycenter
        keep = np.zeros_like(occ, dtype=bool)
        keep[int(np.argmax(occ))] = True
    return consensus[keep].astype(np.int64)


def barycenter_mean_rtwe_dba(X_symbolic, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, max_iter=50, tol=1e-7,
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


def barycenter_soft_mode_dba(X_symbolic, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, beta=4.0, max_iter=10,
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


def barycenter_fgw(X_symbolic, D_G, alpha=0.5, n_nodes=128, feature='mds', mds_dim=None,
                   max_iter=100, random_state=None, decode='euclidean'):
    """Fused Gromov-Wasserstein (FGW) barycenter of symbolic sequences.

    FGW (Vayer, Chapel, Flamary, Tavenard, Courty, *ICML 2019* / *Algorithms 2020*; POT
    ``ot.gromov.fgw_barycenters``) averages structured objects by trading off a feature
    optimal-transport term and a Gromov-Wasserstein structure term. Each sequence is a
    graph: nodes = timesteps, node features = a per-symbol vector, structure ``C`` = the
    intra-sequence timestamp-distance matrix (normalized ``|i-j|``). In POT's convention
    ``alpha`` weights the **structure** (GW) term, so ``alpha -> 0`` is a feature-only
    (Wasserstein / symbol-frequency) average and ``alpha -> 1`` is structure-only (GW) --
    one knob for the frequency<->structure decomposition (``alpha=0`` recovers the
    histogram baseline as a nested special case; see :func:`barycenter_wasserstein`).

    Two node-feature encodings:

    - ``feature='mds'``    : classical-MDS embedding of ``D_G`` (so FGW's internal
      Euclidean feature cost approximates the Wasserstein ground cost ``D_G``);
    - ``feature='onehot'`` : one-hot symbols (feature cost is D_G-agnostic).

    Parameters
    ----------
    X_symbolic : (n_sequences, n_nodes) int array, or a (ragged) list of int arrays
        If already length ``n_nodes`` it is used as-is; otherwise every sequence is
        resampled to ``n_nodes`` via :func:`smartflat.utils.utils.upsample_sequence`.
    D_G : (G, G) ndarray
        Prototype ground-cost matrix.
    alpha : float in [0, 1]
        FGW structure<->feature trade-off (POT: weight on the GW structure term). POT
        documents the open interval ``0 < alpha < 1``; the endpoints run in POT 0.9.5 and
        are the mathematically nested cases. ``alpha=0`` is clamped to a tiny epsilon only
        if a future POT rejects the literal endpoint.
    n_nodes : int
        Number of barycenter nodes (= resampled length). O(n_nodes^2)/iteration -- gate.
    feature : {'mds', 'onehot'}
        Node-feature encoding (see above).
    mds_dim : int, optional
        Cap on MDS dimensionality (default: all positive-eigenvalue axes).
    max_iter : int
        POT FGW iterations.
    random_state : int, optional
        Seed for POT's random initialisation (init-stability across seeds).
    decode : {'euclidean', 'dg'}
        Decode rule for the continuous FGW centroids (Lever 2 -- discreteness fairness):

        - ``'euclidean'`` (default, unchanged): nearest prototype in the FGW **feature**
          space, ``argmin_c ||Xb - proto[c]||_2`` (for one-hot this is ``argmax_c Xb[c]``).
        - ``'dg'`` (``feature='onehot'`` only): ground-cost barycentric projection. The
          centroid row ``Xb[t]`` is read as a soft membership over symbols -- clipped to
          non-negative and L1-normalised to a distribution ``p`` -- and decoded to the
          symbol minimising the **expected rTWE ground cost**, ``argmin_c sum_k p[k] D_G[c, k]``
          (``= argmin_c (D_G @ p)[c]``, the same ``D_G`` substitution cost the harness scores
          in). The clip+renormalise map ``Xb -> p`` is the only researcher choice; degenerate
          all-non-positive rows fall back to the Euclidean decode. Raises for ``feature='mds'``
          (MDS feature cost already approximates ``D_G``, so the Euclidean decode is already
          distance-consistent -- a separate ``dg`` variant would be degenerate).

    Returns
    -------
    (n_nodes,) int array
        Symbolic barycenter (continuous FGW feature centroids decoded to a hard symbol).
    """
    import ot

    X = np.asarray(X_symbolic) if not isinstance(X_symbolic, list) else X_symbolic
    if not isinstance(X, list) and getattr(X, 'ndim', 0) == 2 and X.shape[1] == n_nodes:
        Xr = X.astype(int)
    else:
        from smartflat.utils.utils import upsample_sequence
        Xr = np.vstack([
            upsample_sequence(np.asarray(s).astype(int), n_nodes) for s in X_symbolic
        ]).astype(int)

    G = np.asarray(D_G).shape[0]
    if feature == 'mds':
        proto = _classical_mds(D_G, dim=mds_dim)          # (G, k)
    elif feature == 'onehot':
        proto = np.eye(G, dtype=np.float64)               # (G, G)
    else:
        raise ValueError(f"feature must be 'mds' or 'onehot', got {feature!r}")

    Ys = [proto[x] for x in Xr]                            # each (n_nodes, d)
    pos = np.arange(n_nodes, dtype=np.float64)[:, None]
    C = np.abs(pos - pos.T)
    cmax = C.max()
    if cmax > 0:
        C = C / cmax
    Cs = [C for _ in range(len(Xr))]

    a = max(float(alpha), 1e-9) if alpha <= 0 else float(alpha)
    out = ot.gromov.fgw_barycenters(
        N=n_nodes, Ys=Ys, Cs=Cs, alpha=a, max_iter=max_iter, random_state=random_state,
    )
    Xb = out[0]                                            # (n_nodes, d) continuous centroids
    if decode not in ('euclidean', 'dg'):
        raise ValueError(f"decode must be 'euclidean' or 'dg', got {decode!r}")
    if decode == 'dg':
        if feature != 'onehot':
            raise ValueError(
                "decode='dg' for FGW is defined only for feature='onehot' "
                "(MDS feature cost already approximates D_G)."
            )
        p = np.clip(Xb, 0.0, None)                         # Xb -> distribution over symbols
        s = p.sum(axis=1, keepdims=True)
        good = s[:, 0] > 0
        p = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
        out_sym = (p @ np.asarray(D_G)).argmin(axis=1).astype(np.int64)  # argmin_c E_k[D_G[c, k]]
        if not good.all():                                 # degenerate rows -> Euclidean fallback
            eu = np.linalg.norm(Xb[:, None, :] - proto[None, :, :], axis=2).argmin(axis=1)
            out_sym[~good] = eu[~good]
        return out_sym
    d = np.linalg.norm(Xb[:, None, :] - proto[None, :, :], axis=2)   # (n_nodes, G)
    return d.argmin(axis=1).astype(np.int64)
