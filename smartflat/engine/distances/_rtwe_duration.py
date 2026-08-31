"""Duration-aware registered Time Warp Edit (rTWE) distance for RLE sequences.

Extends :mod:`._rtwe` to the segment-scale, duration-explicit representation:
each element of a sequence is a ``(symbol, duration)`` pair (one element per
maximal run of identical symbols -- the output of run-length encoding), and the
pointwise cost becomes

    d((a, d_a), (b, d_b)) = D[a, b] + gamma * |log d_a - log d_b|

i.e. the symbolic ground cost plus a log-duration mismatch term. ``gamma = 0``
recovers plain rTWE on the RLE symbol sequence exactly. Because
``|log d_a - log d_b|`` is a metric on durations and ``D`` is a metric on
symbols, the sum is a metric on ``(symbol, duration)`` pairs, so the TWE
metric property (Marteau 2009) is preserved.

Boundary handling: sequences are padded with a **neutral** sentinel element
(symbol ``-1``, guarded to zero cost against everything) rather than the
literal symbol ``0`` used by :mod:`._rtwe` -- in the G=28 vocabulary code 0 is
the background symbol, so a real-symbol pad would collide with it. For a
zero-diagonal ground cost the two conventions give identical distances (the
only reachable pad lookup is the pad-pad term of the first match, which is 0
either way); the sentinel makes this true for *any* ground cost.

No Sakoe-Chiba window support: the RLE representation is ~25x shorter than the
frame-level one (median L~181 vs ~5162 on SDS2), so the full O(nm) recursion is
already cheap.
"""

from typing import List, Tuple

import numpy as np
from numba import njit

from smartflat.engine.distances._alignment_paths import compute_min_return_path

_PAD = -1


@njit(cache=False, fastmath=True)
def _pair_cost(a, la, b, lb, D, gamma):
    """Pointwise cost between (symbol, log-duration) pairs; 0 against the pad."""
    if a == _PAD or b == _PAD:
        return 0.0
    return D[a, b] + gamma * abs(la - lb)


@njit(cache=False, fastmath=True)
def _pad_rle(sym, logdur):
    """Prepend the neutral sentinel element to a (symbols, log-durations) pair."""
    n = sym.shape[0]
    ps = np.empty(n + 1, dtype=np.int64)
    pl = np.empty(n + 1, dtype=np.float64)
    ps[0] = _PAD
    pl[0] = 0.0
    ps[1:] = sym
    pl[1:] = logdur
    return ps, pl


@njit(cache=False, fastmath=True)
def _rtwe_dur_cost_matrix(px, plx, py, ply, D, gamma, nu, lmbda):
    """Full accumulated cost matrix on padded inputs (mirrors ``_rtwe_cost_matrix``)."""
    x_size = px.shape[0]
    y_size = py.shape[0]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    del_add = nu + lmbda
    for i in range(1, x_size):
        for j in range(1, y_size):
            del_x = (
                cost_matrix[i - 1, j]
                + _pair_cost(px[i - 1], plx[i - 1], px[i], plx[i], D, gamma)
                + del_add
            )
            del_y = (
                cost_matrix[i, j - 1]
                + _pair_cost(py[j - 1], ply[j - 1], py[j], ply[j], D, gamma)
                + del_add
            )
            match = (
                cost_matrix[i - 1, j - 1]
                + _pair_cost(px[i], plx[i], py[j], ply[j], D, gamma)
                + _pair_cost(px[i - 1], plx[i - 1], py[j - 1], ply[j - 1], D, gamma)
                + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
            )
            cost_matrix[i, j] = min(del_x, del_y, match)
    return cost_matrix[1:, 1:]


def _as_logdur(durations):
    """Durations (in frames, >= 1) -> float64 log-durations."""
    d = np.asarray(durations, dtype=np.float64)
    return np.log(np.maximum(d, 1.0))


def _prep(symbols, durations):
    sym = np.asarray(symbols).astype(np.int64)
    if sym.size == 0:
        raise ValueError("empty RLE sequence")
    return _pad_rle(sym, _as_logdur(durations))


def rtwe_dur_distance(x_sym, x_dur, y_sym, y_dur, D, gamma=0.0, nu=0.001, lmbda=1.0):
    """Duration-aware rTWE distance between two RLE sequences.

    Parameters
    ----------
    x_sym, y_sym : array-like of int
        Run symbols (indices into ``D``).
    x_dur, y_dur : array-like
        Run durations in frames (>= 1); compared on the log scale.
    D : np.ndarray of shape (G, G)
        Symbolic ground-cost matrix (zero diagonal).
    gamma : float
        Weight of the ``|log d_a - log d_b|`` duration term. ``0`` recovers
        plain rTWE on the symbol sequence.
    nu, lmbda : float
        TWE stiffness / edit penalty (indices are now *run* indices, so ``nu``
        is not commensurate with its frame-level value).

    Returns
    -------
    float
    """
    px, plx = _prep(x_sym, x_dur)
    py, ply = _prep(y_sym, y_dur)
    cm = _rtwe_dur_cost_matrix(
        px, plx, py, ply, np.asarray(D, dtype=np.float64), gamma, nu, lmbda,
    )
    return float(cm[-1, -1])


def rtwe_dur_alignment_path(
    x_sym, x_dur, y_sym, y_dur, D, gamma=0.0, nu=0.001, lmbda=1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """Duration-aware rTWE alignment path (mirrors ``rtwe_alignment_path``).

    Returns
    -------
    path : list of (i, j)
        Alignment path over run indices (0-based into the unpadded sequences).
    distance : float
    """
    px, plx = _prep(x_sym, x_dur)
    py, ply = _prep(y_sym, y_dur)
    cm = _rtwe_dur_cost_matrix(
        px, plx, py, ply, np.asarray(D, dtype=np.float64), gamma, nu, lmbda,
    )
    return compute_min_return_path(cm), float(cm[-1, -1])


def rtwe_dur_pairwise_distance(seqs, D, gamma=0.0, nu=0.001, lmbda=1.0):
    """Pairwise duration-aware rTWE over a collection of packed RLE sequences.

    Parameters
    ----------
    seqs : sequence of np.ndarray of shape (2, L_i)
        Packed RLE sequences (row 0 = symbols, row 1 = durations); ragged
        lengths are fine.
    D : np.ndarray of shape (G, G)
        Symbolic ground cost.
    gamma, nu, lmbda : float
        See :func:`rtwe_dur_distance`.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric distance matrix.
    """
    Dc = np.asarray(D, dtype=np.float64)
    prepped = [_prep(s[0], s[1]) for s in seqs]
    n = len(prepped)
    out = np.zeros((n, n))
    for i in range(n):
        pi, li = prepped[i]
        for j in range(i + 1, n):
            pj, lj = prepped[j]
            cm = _rtwe_dur_cost_matrix(pi, li, pj, lj, Dc, gamma, nu, lmbda)
            out[i, j] = out[j, i] = cm[-1, -1]
    return out
