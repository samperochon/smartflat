"""The six ``*_methods`` registry builders — each returns a dict of
``{name: {build, distance, kind}}`` specs wiring a barycenter constructor to a
scoring distance.

Split out of ``baselines.py`` (arc-audit Phase 2); wires constructors from
:mod:`.builders` to distances from :mod:`.distances`. Re-exported unchanged via
the ``baselines`` shim.
"""


from .distances import (
    RTWE_NU, RTWE_LMBDA, dist_dtw, dist_soft_dtw, dist_edit, dist_hamming, dist_wasserstein_hist, dist_rtwe, dist_neg_pmatch, dist_transition, dist_eshape_dtw,
)
from .builders import (
    barycenter_dba_dtw, barycenter_soft_dtw, barycenter_softdtw, barycenter_ssg, barycenter_ssg_cat, barycenter_softdtw_cat, barycenter_edit_median, barycenter_wasserstein, barycenter_majority_voting, barycenter_mode_dba, barycenter_msa_consensus, barycenter_transition_matrix, barycenter_soft_mode_dba, barycenter_fgw,
)


def default_baseline_methods(D_G, gamma=1.0, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None):
    """Build the six standard-baseline registry for :func:`evaluate_baselines`.

    Each baseline is paired with its NATIVE classification distance so the
    cross-method AUC comparison scores every method as it would actually be
    used: DBA-DTW->DTW, Soft-DTW->soft-DTW, edit-median->edit distance,
    Wasserstein->Wasserstein (histogram), k-medoid->TW-TWE, majority-voting->
    lock-step Hamming.

    .. note::
        ``dba_dtw`` is scored here by its native ``dist_dtw``. In
        :func:`discreteness_lever_methods` the sibling DBA keys (``dba_dtw_dg`` /
        ``dba_dtw_cat``) are scored by ``dist_rtwe`` instead. The shared
        ``inertia_rtwe`` yardstick (fixed ``RTWE_NU``/``RTWE_LMBDA``) is comparable
        across every registry, but ``inertia_native`` for the DBA base method is
        **not comparable** between these two registries (different distance). This
        divergence is intentional -- each registry scores its methods on the distance
        they would actually be used with -- and is documented rather than standardised.

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


def extra_experiment_methods(D_G, G, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None, step_sequ=2):
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


def fgw_methods(D_G, n_nodes=128, alpha=0.5, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None, mds_dim=None,
                max_iter=100):
    """Fused Gromov-Wasserstein barycenter registry (the representation-quality centrepiece).

    Two entries, both built by :func:`barycenter_fgw` and scored by the native rTWE
    distance (like ``k_medoid``): ``fgw_mds`` (classical-MDS node features, feature cost
    ~ D_G) and ``fgw_onehot`` (one-hot node features, D_G-agnostic). ``alpha`` is the
    frequency<->structure knob (POT weights the GW structure term). O(n_nodes^2)/iteration
    -- keep ``n_nodes`` gated (default 128). Merge with ``|`` alongside
    :func:`default_baseline_methods` / :func:`extra_experiment_methods` in the notebook.
    """
    return {
        'fgw_mds': {
            'build': lambda X, seed: barycenter_fgw(
                X, D_G, alpha=alpha, n_nodes=n_nodes, feature='mds', mds_dim=mds_dim,
                max_iter=max_iter, random_state=seed),
            'distance': lambda seq, bary: dist_rtwe(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
        },
        'fgw_onehot': {
            'build': lambda X, seed: barycenter_fgw(
                X, D_G, alpha=alpha, n_nodes=n_nodes, feature='onehot',
                max_iter=max_iter, random_state=seed),
            'distance': lambda seq, bary: dist_rtwe(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
        },
    }


def softdtw_ssg_methods(D_G, gamma=1.0, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None,
                        sdtw_max_iter=50, ssg_max_iter=30):
    """Library-backed Soft-DTW + SSG barycenter registry (mirrors :func:`fgw_methods`).

    Two ``{build, distance}`` entries scored by :func:`score_barycenter_quality` exactly
    like the other methods:

    - ``soft_dtw_bary`` -- :func:`barycenter_softdtw` (Cuturi & Blondel, ICML 2017),
      paired with its NATIVE soft-DTW distance (:func:`dist_soft_dtw`).
    - ``ssg`` -- :func:`barycenter_ssg` (Schultz & Jain, Pattern Recognition 2018), paired
      with the NATIVE rTWE distance (:func:`dist_rtwe`, like ``fgw_*``/``k_medoid``).

    Both embed via ``D_G`` and decode to hard symbols, so the harness auto-computes every
    axis. O(L^2)/iteration -- length-gated (cheap at L=128 with tslearn's compiled solver;
    the full-length L~5162 / full-cohort run is a scale job). Merge with ``|`` alongside
    :func:`default_baseline_methods` / :func:`fgw_methods` in the notebook.
    """
    return {
        'soft_dtw_bary': {
            'build': lambda X, seed: barycenter_softdtw(
                X, D_G, gamma=gamma, max_iter=sdtw_max_iter, random_state=seed),
            'distance': lambda seq, bary: dist_soft_dtw(seq, bary, D_G, gamma=gamma),
        },
        'ssg': {
            'build': lambda X, seed: barycenter_ssg(
                X, D_G, max_iter=ssg_max_iter, random_state=seed),
            'distance': lambda seq, bary: dist_rtwe(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
        },
    }


def msa_consensus_methods(D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None,
                          pseudocount=1.0, occupancy=0.5):
    """MSA positional-consensus barycenter registry (mirrors :func:`softdtw_ssg_methods`).

    A single ``{build, distance}`` entry scored by :func:`score_barycenter_quality` exactly
    like the other methods:

    - ``msa_consensus`` -- :func:`barycenter_msa_consensus` (center-star MSA, Gusfield 1993,
      + profile per-column consensus, Durbin et al. 1998), paired with the NATIVE rTWE
      distance (:func:`dist_rtwe`, like ``ssg``/``fgw_*``/``k_medoid``).

    This is a **Family A** native-categorical method: it stays in symbol space (no ``D_G``
    embedding, no decode), so its output is a genuine symbol sequence and the harness
    auto-computes every axis. O(n^2 * L^2) for the pairwise medoid selection -- length-gated
    (L=128 preview; the full-length L~5162 / full-cohort run is a scale job for pomme). Merge
    with ``|`` alongside :func:`default_baseline_methods` / :func:`fgw_methods` /
    :func:`softdtw_ssg_methods` in the notebook.
    """
    return {
        'msa_consensus': {
            'build': lambda X, seed: barycenter_msa_consensus(
                X, D_G, nu=nu, lmbda=lmbda, window=window,
                pseudocount=pseudocount, occupancy=occupancy, random_state=seed),
            'distance': lambda seq, bary: dist_rtwe(
                seq, bary, D_G, nu=nu, lmbda=lmbda, window=window),
        },
    }


def discreteness_lever_methods(D_G, gamma=1.0, nu=RTWE_NU, lmbda=RTWE_LMBDA, window=None,
                               dba_max_iters=30, sdtw_max_iter=50, ssg_max_iter=30,
                               fgw_alpha=0.5, fgw_n_nodes=128, fgw_max_iter=100,
                               cat_rounds=6, cat_inner_iter=8):
    """Discreteness-fairness lever variants registry (Kickoff K; mirrors :func:`softdtw_ssg_methods`).

    Seven ADDITIVE ``{build, distance}`` entries beside the existing Family-B methods;
    the existing keys (``dba_dtw``/``soft_dtw_bary``/``ssg``/``fgw_onehot``) are untouched,
    so results stay byte-for-byte reproducible. Reported ablation-style (both ways, no
    cherry-picked winner). Each native distance mirrors the base method (``dist_rtwe`` for
    DBA/SSG/FGW, ``dist_soft_dtw`` for the soft-DTW pair), so the harness scores every axis.

    .. note::
        The DBA keys here (``dba_dtw_dg``/``dba_dtw_cat``) use ``dist_rtwe`` as their
        native distance, whereas ``dba_dtw`` in :func:`default_baseline_methods` uses
        ``dist_dtw``. The shared ``inertia_rtwe`` yardstick is comparable across every
        registry, but ``inertia_native`` for the DBA base method is **not comparable**
        between the two registries (see the matching note there).

    **Lever 2 -- ground-cost-consistent decode** (``decode='dg'``: ``argmin_c m[c]``, the
    vocabulary-restricted 1-medoid under ``D_G``; snap once at the end). Same
    ``random_state``/``n_inits`` as the Euclidean sibling -> the two share the same continuous
    barycenter and differ *only* in the decode:

    - ``dba_dtw_dg``        -- :func:`barycenter_dba_dtw` with ``decode='dg'``.
    - ``soft_dtw_bary_dg``  -- :func:`barycenter_softdtw` with ``decode='dg'``.
    - ``ssg_dg``            -- :func:`barycenter_ssg` with ``decode='dg'``.
    - ``fgw_onehot_dg``     -- :func:`barycenter_fgw` (``feature='onehot'``) with ``decode='dg'``
      (barycentric ground-cost projection; the ``feature='mds'`` case is degenerate, so it is
      not exposed here).

    **Lever 3 -- per-iteration re-discretised categorical variants** (keep the reference a valid
    symbol string throughout, like mode-DBA but mean-then-snap; each per-iter snap is the ``'dg'``
    ground-cost medoid):

    - ``dba_dtw_cat``       -- :func:`barycenter_dba_dtw` with ``discretise_each_iter=True``.
    - ``ssg_cat``           -- :func:`barycenter_ssg_cat` (outer loop over tslearn SSG).
    - ``soft_dtw_bary_cat`` -- :func:`barycenter_softdtw_cat` (outer loop over tslearn Soft-DTW).

    Merge with ``|`` alongside :func:`default_baseline_methods` / :func:`fgw_methods` /
    :func:`softdtw_ssg_methods` in the notebook (length-gated; the full-length run is a pomme job).
    """
    def dist_rtwe_native(seq, bary):
        return dist_rtwe(seq, bary, D_G, nu=nu, lmbda=lmbda, window=window)

    def dist_sdtw_native(seq, bary):
        return dist_soft_dtw(seq, bary, D_G, gamma=gamma)

    return {
        # --- Lever 2: ground-cost (D_G) decode, snap once at the end ---
        'dba_dtw_dg': {
            'build': lambda X, seed: barycenter_dba_dtw(
                X, D_G, max_iter=dba_max_iters, random_state=seed, decode='dg'),
            'distance': dist_rtwe_native,
        },
        'soft_dtw_bary_dg': {
            'build': lambda X, seed: barycenter_softdtw(
                X, D_G, gamma=gamma, max_iter=sdtw_max_iter, random_state=seed, decode='dg'),
            'distance': dist_sdtw_native,
        },
        'ssg_dg': {
            'build': lambda X, seed: barycenter_ssg(
                X, D_G, max_iter=ssg_max_iter, random_state=seed, decode='dg'),
            'distance': dist_rtwe_native,
        },
        'fgw_onehot_dg': {
            'build': lambda X, seed: barycenter_fgw(
                X, D_G, alpha=fgw_alpha, n_nodes=fgw_n_nodes, feature='onehot',
                max_iter=fgw_max_iter, random_state=seed, decode='dg'),
            'distance': dist_rtwe_native,
        },
        # --- Lever 3: per-iteration re-discretised categorical variants ---
        'dba_dtw_cat': {
            'build': lambda X, seed: barycenter_dba_dtw(
                X, D_G, max_iter=dba_max_iters, random_state=seed,
                discretise_each_iter=True, decode='dg'),
            'distance': dist_rtwe_native,
        },
        'ssg_cat': {
            'build': lambda X, seed: barycenter_ssg_cat(
                X, D_G, n_rounds=cat_rounds, inner_max_iter=cat_inner_iter,
                decode='dg', random_state=seed),
            'distance': dist_rtwe_native,
        },
        'soft_dtw_bary_cat': {
            'build': lambda X, seed: barycenter_softdtw_cat(
                X, D_G, gamma=gamma, n_rounds=cat_rounds, inner_max_iter=cat_inner_iter,
                decode='dg', random_state=seed),
            'distance': dist_sdtw_native,
        },
    }
