"""Segment-scale, duration-explicit (RLE) symbolic representation + barycenter.

Motivation (method redesign, 2026-08): the frame-level representation the
proposed TW-TWE + mode-DBA method runs on is ~25x redundant (immediate-repeat
rate 0.95-0.97, mean run length 20-34 frames on SDS2), so elastic alignment
mostly stretches runs of identical symbols against each other -- which makes
the distance approximately order-insensitive and lets symbol frequency dominate.
The order signal lives at the *segment* scale, and the run durations (dwell
times) -- a real clinical signal -- are destroyed by per-participant length
normalization and resampling.

This module keeps both: every sequence is run-length encoded into
``(symbol, duration)`` pairs (median L~181 on SDS2 instead of ~5162), aligned
with the duration-aware rTWE of :mod:`smartflat.engine.distances._rtwe_duration`
(pointwise cost ``D_G[a, b] + gamma_dur * |log d_a - log d_b|``), and averaged
with a DBA whose per-position update is the ``D_G``-Frechet symbol mode plus the
geometric-mean duration. ``gamma_dur = 0`` recovers plain rTWE on the RLE symbol
sequence. No resampling step exists in this pipeline: the elastic distance
consumes the ragged sequences directly.

Packed format: one RLE sequence = ``np.ndarray`` of shape ``(2, L)`` -- row 0
the run symbols, row 1 the run durations in frames. Collections are stored in
1-D object arrays (:func:`rle_object_array`) so :func:`~.evaluation.
evaluate_baselines` can fancy-index them like the rectangular frame-level input.
"""

import numpy as np

from .distances import RTWE_NU, RTWE_LMBDA
from .order_evaluation import _rle
from smartflat.engine.distances._rtwe_duration import (
    rtwe_dur_alignment_path,
    rtwe_dur_distance,
    rtwe_dur_pairwise_distance,
)

# Default weight of the log-duration term in the pointwise cost. A 2x duration
# mismatch then costs 0.2 * log 2 ~ 0.14 -- deliberately below the smallest
# off-diagonal symbol cost of the G=28 ground cost (~0.39), so symbol identity
# stays the primary signal. Sweep it in experiments; 0 disables durations.
GAMMA_DUR = 0.2


def rle_encode(seq):
    """Run-length encode a frame-level symbolic sequence into a packed (2, L) array.

    Row 0 = run symbols (the "action grammar"), row 1 = run durations in frames.
    Thin wrapper over :func:`~.order_evaluation._rle` (the canonical RLE helper).
    """
    symbols, lengths = _rle(np.asarray(seq).astype(int))
    return np.vstack([symbols, lengths]).astype(np.int64)


def rle_decode(packed):
    """Inverse of :func:`rle_encode`: expand a packed (2, L) array to frame level."""
    packed = np.asarray(packed)
    return np.repeat(packed[0].astype(int), packed[1].astype(int))


def rle_object_array(X):
    """Pack a collection of frame-level sequences into a 1-D object array of RLE arrays.

    The object array supports the fancy indexing (``X[train_idx]``, boolean masks)
    that :func:`~.evaluation.evaluate_baselines` applies to its input, while the
    per-sequence lengths stay ragged.
    """
    out = np.empty(len(X), dtype=object)
    for i, seq in enumerate(X):
        out[i] = rle_encode(seq)
    return out


def merge_adjacent_runs(packed):
    """Merge adjacent identical symbols of a packed RLE sequence, summing durations.

    The DBA update can emit the same symbol at consecutive positions; merging keeps
    the barycenter a genuine RLE sequence (no zero-length or split runs) and lets
    its length adapt downward -- the lightweight form of an adaptive-length
    barycenter.
    """
    packed = np.asarray(packed)
    sym, dur = packed[0], packed[1]
    if sym.size == 0:
        return packed.copy()
    change = np.flatnonzero(np.diff(sym)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [sym.size]))
    merged_dur = np.array([dur[s:e].sum() for s, e in zip(starts, ends)])
    return np.vstack([sym[starts], merged_dur]).astype(np.int64)


# ---------------------------------------------------------------------------
# Distances / features on packed RLE sequences
# ---------------------------------------------------------------------------

def dist_rle_twe(seq, bary, D_G, gamma_dur=GAMMA_DUR, nu=RTWE_NU, lmbda=RTWE_LMBDA):
    """Duration-aware rTWE distance between two packed RLE sequences."""
    seq = np.asarray(seq)
    bary = np.asarray(bary)
    return rtwe_dur_distance(
        seq[0], seq[1], bary[0], bary[1], D_G, gamma=gamma_dur, nu=nu, lmbda=lmbda,
    )


def pmatch_rle(seq, bary, D_G, gamma_dur=GAMMA_DUR, nu=RTWE_NU, lmbda=RTWE_LMBDA,
               soft=False, duration_weighted=True):
    """Proportion of matching symbols along the duration-aware rTWE alignment.

    Parameters
    ----------
    seq, bary : np.ndarray of shape (2, L)
        Packed RLE sequences.
    soft : bool
        If ``True``, a diagonal step contributes ``1 - D_G[a, b] / D_G.max()``
        (credit for close-but-distinct symbols) instead of the hard ``a == b``
        indicator.
    duration_weighted : bool
        If ``True`` (default), each diagonal step is weighted by the mean frame
        duration of the two aligned runs, so agreement on long runs counts for
        the share of task time it represents -- the RLE analog of the
        frame-level p_match. ``False`` counts each run once.
    """
    seq = np.asarray(seq)
    bary = np.asarray(bary)
    D = np.asarray(D_G, dtype=np.float64)
    dmax = D.max() if D.max() > 0 else 1.0
    path, _ = rtwe_dur_alignment_path(
        seq[0], seq[1], bary[0], bary[1], D, gamma=gamma_dur, nu=nu, lmbda=lmbda,
    )
    num = den = 0.0
    for k in range(1, len(path)):
        di = path[k][0] - path[k - 1][0]
        dj = path[k][1] - path[k - 1][1]
        if di == 1 and dj == 1:
            i, j = path[k]
            w = 0.5 * (seq[1, i] + bary[1, j]) if duration_weighted else 1.0
            a, b = int(seq[0, i]), int(bary[0, j])
            credit = (1.0 - D[a, b] / dmax) if soft else float(a == b)
            num += w * credit
            den += w
    return num / max(den, 1e-12)


def dist_rle_neg_pmatch(seq, bary, D_G, gamma_dur=GAMMA_DUR, nu=RTWE_NU,
                        lmbda=RTWE_LMBDA, soft=False):
    """Negative RLE p_match, usable as a 'distance' in ``evaluate_baselines``."""
    return -pmatch_rle(seq, bary, D_G, gamma_dur=gamma_dur, nu=nu, lmbda=lmbda, soft=soft)


# ---------------------------------------------------------------------------
# Barycenter
# ---------------------------------------------------------------------------

def barycenter_rle_dba(X_rle, D_G, gamma_dur=GAMMA_DUR, nu=RTWE_NU, lmbda=RTWE_LMBDA,
                       max_iter=10, update='dg_frechet', random_state=None,
                       merge_runs=True, return_costs=False):
    """DBA barycenter on packed RLE sequences under the duration-aware rTWE.

    Same iterative structure as :func:`~.builders.barycenter_mode_dba` (medoid
    init, align-all / update-positions, fixed point or ``max_iter``), with two
    representation-level differences:

    - the per-position **symbol** update is the ``D_G``-Frechet mode
      ``argmin_c sum_votes D_G[c, v]^2`` (``update='dg_frechet'``, consistent
      with the squared-distance Frechet objective) or the plain ``'mode'``;
    - each position also carries a **duration**, updated to the geometric mean
      of the aligned runs' durations (the Frechet mean under the
      ``|log d - log d'|`` cost the alignment itself uses).

    The best-cost iterate (including the final one) is returned, and adjacent
    identical symbols are merged (``merge_runs=True``) so the result is a
    genuine RLE sequence whose length can adapt downward.

    Parameters
    ----------
    X_rle : sequence of np.ndarray of shape (2, L_i)
        Packed RLE sequences (ragged lengths fine; object arrays fine).
    D_G : np.ndarray of shape (G, G)
        Symbolic ground cost.
    gamma_dur, nu, lmbda : float
        Duration-aware rTWE parameters (see :func:`dist_rle_twe`).
    max_iter : int
        Refinement iteration budget.
    update : {'dg_frechet', 'mode'}
        Per-position symbol update rule.
    random_state : int or None
        Unused (medoid init is deterministic); kept for the uniform
        ``build(X, seed)`` signature.
    merge_runs : bool
        Merge adjacent identical barycenter symbols (summing durations).
    return_costs : bool
        Also return the per-iterate total alignment cost trace.

    Returns
    -------
    np.ndarray of shape (2, L_bary)
        Packed RLE barycenter. With ``return_costs``, ``(barycenter, costs)``.
    """
    if update not in ('dg_frechet', 'mode'):
        raise ValueError(f"update must be 'dg_frechet' or 'mode', got {update!r}")
    X = [np.asarray(s) for s in X_rle]
    if len(X) == 1:
        out = merge_adjacent_runs(X[0]) if merge_runs else X[0].copy()
        return (out, []) if return_costs else out
    Dc = np.asarray(D_G, dtype=np.float64)
    G = Dc.shape[0]
    Dc2 = Dc ** 2

    Dp = rtwe_dur_pairwise_distance(X, Dc, gamma=gamma_dur, nu=nu, lmbda=lmbda)
    ref = X[int(np.argmin(Dp.sum(axis=1)))].copy()

    candidates, costs = [], []
    for _ in range(max_iter):
        L = ref.shape[1]
        sym_votes = np.zeros((G, L))
        logdur_sum = np.zeros(L)
        logdur_n = np.zeros(L)
        iter_cost = 0.0
        for s in X:
            path, d = rtwe_dur_alignment_path(
                s[0], s[1], ref[0], ref[1], Dc, gamma=gamma_dur, nu=nu, lmbda=lmbda,
            )
            iter_cost += float(d)
            for (i, j) in path:
                if 0 <= j < L and 0 <= i < s.shape[1]:
                    sym_votes[int(s[0, i]), j] += 1.0
                    logdur_sum[j] += np.log(max(float(s[1, i]), 1.0))
                    logdur_n[j] += 1.0
        candidates.append(ref)
        costs.append(iter_cost)

        new_sym = np.empty(L, dtype=np.int64)
        new_dur = np.empty(L, dtype=np.int64)
        for j in range(L):
            if sym_votes[:, j].sum() > 0:
                if update == 'dg_frechet':
                    new_sym[j] = int(np.argmin(Dc2 @ sym_votes[:, j]))
                else:
                    new_sym[j] = int(np.argmax(sym_votes[:, j]))
                new_dur[j] = max(1, int(round(np.exp(logdur_sum[j] / logdur_n[j]))))
            else:
                new_sym[j] = int(ref[0, j])
                new_dur[j] = int(ref[1, j])
        new_ref = np.vstack([new_sym, new_dur])
        if new_ref.shape == ref.shape and np.array_equal(new_ref, ref):
            break
        ref = new_ref
    else:
        # max_iter exhausted without a fixed point: cost the final iterate too.
        candidates.append(ref)
        costs.append(float(sum(
            rtwe_dur_distance(s[0], s[1], ref[0], ref[1], Dc,
                              gamma=gamma_dur, nu=nu, lmbda=lmbda)
            for s in X
        )))

    best = candidates[int(np.argmin(costs))]
    out = merge_adjacent_runs(best) if merge_runs else best.copy()
    return (out, costs) if return_costs else out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def rle_methods(D_G, gamma_dur=GAMMA_DUR, nu=RTWE_NU, lmbda=RTWE_LMBDA,
                update='dg_frechet', soft_pmatch=False):
    """Registry of RLE-representation methods for :func:`~.evaluation.evaluate_baselines`.

    Pass an object array of packed RLE sequences (:func:`rle_object_array`) as
    ``X_symbolic``. Entries:

    - ``rle_dba_pmatch`` -- duration-aware RLE-DBA barycenter scored by the
      (negated) RLE p_match feature: the segment-scale analog of the proposed
      ``tw_twe_pmatch``.
    - ``rle_dba_twe``    -- same barycenter scored by the duration-aware rTWE
      distance itself (analog of ``tw_twe``).
    - ``rle_medoid``     -- ``kind='medoid'`` selection on a precomputed
      duration-aware pairwise matrix (pass ``D_pairwise`` from
      :func:`~smartflat.engine.distances._rtwe_duration.rtwe_dur_pairwise_distance`),
      scored by the duration-aware rTWE.
    """
    def build(X, seed):
        return barycenter_rle_dba(
            X, D_G, gamma_dur=gamma_dur, nu=nu, lmbda=lmbda,
            update=update, random_state=seed,
        )

    return {
        'rle_dba_pmatch': {
            'build': build,
            'distance': lambda seq, bary: dist_rle_neg_pmatch(
                seq, bary, D_G, gamma_dur=gamma_dur, nu=nu, lmbda=lmbda,
                soft=soft_pmatch),
        },
        'rle_dba_twe': {
            'build': build,
            'distance': lambda seq, bary: dist_rle_twe(
                seq, bary, D_G, gamma_dur=gamma_dur, nu=nu, lmbda=lmbda),
        },
        'rle_medoid': {
            'kind': 'medoid',
            'distance': lambda seq, bary: dist_rle_twe(
                seq, bary, D_G, gamma_dur=gamma_dur, nu=nu, lmbda=lmbda),
        },
    }
