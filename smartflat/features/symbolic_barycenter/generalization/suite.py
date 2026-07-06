"""Reusable driver that runs the verified barycenter probes on ANY dataset.

A generalization dataset only needs to produce ``(X_symbolic, labels, G, D_G)`` — the
same four objects the SDS2 harness consumes. :func:`run_generalization_suite` then runs:

  - the §15 **order-null** (`order_evaluation.order_information`): does symbol *order*
    discriminate the groups beyond frequency? (``order_helps`` / ``ci_low > 0``); and
  - the §17-20 **representation-quality** harness
    (`barycenter_quality.score_barycenter_quality`): methods × quality for the group
    barycenters, on the common rTWE yardstick.

Nothing here is dataset-specific; it is the counterpart to the SDS2 notebooks (06f/06g…)
for the generalization datasets (PAPER_TODO §2).
"""
import numpy as np
import pandas as pd

from ..order_evaluation import order_information
from ..barycenter_quality import score_barycenter_quality, quality_table
from ..registries import default_baseline_methods
from ..distances import dist_rtwe, RTWE_NU, RTWE_LMBDA

# Same feature × shuffle grid the SDS2 order-null (§15) sweeps.
_ORDER_GRID = (('transition', 'token'), ('transition', 'runlength'),
               ('run_transition', 'token'), ('run_transition', 'runlength'))


def _ensure_rectangular(X, upsample_to):
    """Return an ``(n, L)`` int array for lock-step scoring.

    If ``X`` is already a rectangular numeric 2-D array it is returned as ints;
    otherwise the ragged sequences are resampled to ``upsample_to`` via the same
    :func:`smartflat.utils.utils.upsample_sequence` used by ``vocab.load_g28_cohort``.
    """
    if isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype != object:
        return X.astype(int)
    from smartflat.utils.utils import upsample_sequence
    return np.vstack([upsample_sequence(np.asarray(s), upsample_to) for s in X]).astype(int)


def _pairwise_rtwe(Xr, D_G, nu=RTWE_NU, lmbda=RTWE_LMBDA):
    """``(n, n)`` symmetric rTWE distance matrix — only needed for medoid methods."""
    n = len(Xr)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_rtwe(Xr[i], Xr[j], D_G, nu=nu, lmbda=lmbda)
            D[i, j] = D[j, i] = d
    return D


def run_generalization_suite(X, labels, G, D_G, *, name='dataset',
                             run_order=True, run_quality=True,
                             order_grid=_ORDER_GRID, n_shuffles=200,
                             methods=None, n_inits=2, upsample_to=128,
                             random_state=42):
    """Run the order-null and quality probes on one dataset.

    Parameters
    ----------
    X : ragged ``list`` of int sequences, or an ``(n, L)`` int array.
        Order-null uses per-sequence features (ragged is fine); quality resamples to a
        rectangular ``(n, upsample_to)`` if ``X`` is ragged.
    labels : ``(n,)`` group labels (any vocabulary — the harness auto-generates the
        class comparisons; SDS2's {HEALTHY,RIL,TBI} keep the frozen 3 comparisons).
    G : int alphabet size (incl. background 0). ``D_G`` : ``(G, G)`` ground cost.
    methods : registry dict; defaults to ``default_baseline_methods(D_G)``.

    Returns
    -------
    dict with keys ``'order'`` (§15 ΔAUC table) and/or ``'quality'`` (methods × metric
    table), each a ``pd.DataFrame`` carrying a ``dataset`` column.
    """
    results = {}
    if run_order:
        rows = [order_information(X, labels, G, feature=f, shuffle=s,
                                  n_shuffles=n_shuffles, random_state=random_state)
                for (f, s) in order_grid]
        results['order'] = pd.concat(rows, ignore_index=True).assign(dataset=name)
    if run_quality:
        Xr = _ensure_rectangular(X, upsample_to)
        methods = methods if methods is not None else default_baseline_methods(D_G)
        needs_pw = any(spec.get('kind') == 'medoid' for spec in methods.values())
        D_pw = _pairwise_rtwe(Xr, D_G) if needs_pw else None
        res = score_barycenter_quality(Xr, labels, methods, D_G, D_pairwise=D_pw,
                                       n_inits=n_inits, random_state=random_state)
        results['quality'] = quality_table(res).assign(dataset=name)
    return results
