"""Registered Time Warp Edit (rTWE) distance between symbolic sequences.

Thesis contribution (Chapter 6, Section 6.2): extends the Time Warp Edit
distance (Marteau 2009) to operate on **symbolic sequences** by replacing
Euclidean pointwise distances with lookups in a precomputed prototype
distance matrix.  This enables TWE-based alignment and barycenter averaging
for discrete action-prototype sequences.

Key difference from standard TWE:
    Standard TWE: d(x_i, y_j) = ||x_i - y_j||_2   (Euclidean)
    rTWE:         d(x_i, y_j) = D[x_i, y_j]        (precomputed matrix)

References
----------
Marteau, P.-F. (2009). Time Warp Edit Distance with Stiffness Adjustment
for Time Series Matching. IEEE TPAMI, 31(2), 306-318.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

from smartflat.engine.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from smartflat.engine.distances._bounding_matrix import create_bounding_matrix
from smartflat.engine.distances._utils import _convert_to_list, _is_multivariate


@njit(cache=False, fastmath=True, parallel=False)
def rtwe_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: np.ndarray = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    r"""Compute the rTWE distance between two symbolic sequences.

    Parameters
    ----------
    x : np.ndarray
        First sequence, shape ``(n_timepoints,)`` or ``(n_channels, n_timepoints)``.
    y : np.ndarray
        Second sequence, shape ``(n_timepoints,)`` or ``(n_channels, n_timepoints)``.
    window : float, default=None
        Sakoe-Chiba band width. If None, no constraint.
    nu : float, default=0.001
        Stiffness penalty for warping off the diagonal.
    lmbda : float, default=1.0
        Penalty for insert/delete operations.
    precomputed_distances : np.ndarray
        Pairwise prototype distance matrix D[i, j].
    itakura_max_slope : float, default=None
        Maximum slope for Itakura parallelogram bounding.

    Returns
    -------
    float
        rTWE distance between x and y.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _rtwe_distance(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda,
            precomputed_distances,
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _rtwe_distance(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda,
            precomputed_distances,
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=False, fastmath=True, parallel=False)
def rtwe_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: np.ndarray = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the rTWE cost matrix between two symbolic sequences.

    Parameters
    ----------
    x, y : np.ndarray
        Symbolic sequences (1D or 2D).
    window : float, default=None
        Sakoe-Chiba band width.
    nu : float, default=0.001
        Stiffness penalty.
    lmbda : float, default=1.0
        Edit penalty.
    precomputed_distances : np.ndarray
        Pairwise prototype distance matrix.
    itakura_max_slope : float, default=None
        Itakura parallelogram slope.

    Returns
    -------
    np.ndarray of shape (n_timepoints_x, n_timepoints_y)
        rTWE cost matrix.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        bounding_matrix = create_bounding_matrix(
            _x.shape[1], _y.shape[1], window, itakura_max_slope
        )
        return _rtwe_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda,
            precomputed_distances,
        )
    if x.ndim == 2 and y.ndim == 2:
        bounding_matrix = create_bounding_matrix(
            x.shape[1], y.shape[1], window, itakura_max_slope
        )
        return _rtwe_cost_matrix(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda,
            precomputed_distances,
        )
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=False, fastmath=True, parallel=False)
def _rtwe_distance_rolling(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
    nu: float, lmbda: float, precomputed_distances: np.ndarray,
) -> float:
    """Memory-light rTWE distance via a 2-row rolling buffer.

    Numerically identical to ``_rtwe_cost_matrix(...)[ -1, -1]`` (the bottom-right
    corner of the full cost matrix, which is what ``_rtwe_distance`` returns), but uses
    O(L) memory instead of O(L^2): each recurrence cell reads only ``prev[j]``,
    ``cur[j-1]`` and ``prev[j-1]``, so two 1-D buffers suffice. The full-matrix
    ``_rtwe_cost_matrix`` is retained for the alignment-path functions, which need the
    whole matrix for backtracking. Out-of-band interior cells are set to 0.0 to match the
    full-matrix kernel (which leaves them at their ``np.zeros`` initial value).
    """
    x_size = x.shape[1]
    y_size = y.shape[1]
    prev = np.empty(y_size)
    cur = np.empty(y_size)
    prev[0] = 0.0
    for j in range(1, y_size):
        prev[j] = np.inf

    del_add = nu + lmbda
    for i in range(1, x_size):
        cur[0] = np.inf
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                del_x = prev[j] + precomputed_distances[int(x[0, i - 1]), int(x[0, i])] + del_add
                del_y = cur[j - 1] + precomputed_distances[int(y[0, j - 1]), int(y[0, j])] + del_add
                match = (
                    prev[j - 1]
                    + precomputed_distances[int(x[0, i]), int(y[0, j])]
                    + precomputed_distances[int(x[0, i - 1]), int(y[0, j - 1])]
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )
                cur[j] = min(del_x, del_y, match)
            else:
                cur[j] = 0.0
        prev, cur = cur, prev

    return prev[y_size - 1]


@njit(cache=False, fastmath=True, parallel=False)
def _rtwe_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
    nu: float, lmbda: float, precomputed_distances: np.ndarray,
) -> float:
    return _rtwe_distance_rolling(
        x, y, bounding_matrix, nu, lmbda, precomputed_distances,
    )


@njit(cache=False, fastmath=True, parallel=False)
def _rtwe_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
    nu: float, lmbda: float, precomputed_distances: np.ndarray,
) -> np.ndarray:
    """Core rTWE cost matrix computation using precomputed prototype distances."""
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size, y_size))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf

    del_add = nu + lmbda

    for i in range(1, x_size):
        for j in range(1, y_size):
            if bounding_matrix[i - 1, j - 1]:
                # Deletion in x: cost of transitioning within x
                del_x = precomputed_distances[int(x[0, i - 1]), int(x[0, i])]
                del_x = cost_matrix[i - 1, j] + del_x + del_add

                # Deletion in y: cost of transitioning within y
                del_y = precomputed_distances[int(y[0, j - 1]), int(y[0, j])]
                del_y = cost_matrix[i, j - 1] + del_y + del_add

                # Match: cost of aligning x[i] with y[j]
                match_same = precomputed_distances[int(x[0, i]), int(y[0, j])]
                match_previous = precomputed_distances[int(x[0, i - 1]), int(y[0, j - 1])]
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same
                    + match_previous
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


@njit(cache=False, fastmath=True, parallel=False)
def _pad_arrs(x: np.ndarray) -> np.ndarray:
    """Prepend a zero column to each channel (for boundary conditions)."""
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    zero_arr = np.array([0], dtype=x.dtype)
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


def rtwe_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: np.ndarray = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute the rTWE pairwise distance between a set of symbolic sequences.

    Parameters
    ----------
    X : np.ndarray or List[np.ndarray]
        Collection of sequences, shape ``(n_cases, n_channels, n_timepoints)``
        or ``(n_cases, n_timepoints)``.
    y : np.ndarray or List[np.ndarray] or None, default=None
        Second collection. If None, compute self-pairwise distances.
    window : float, default=None
        Sakoe-Chiba band width.
    nu : float, default=0.001
        Stiffness penalty.
    lmbda : float, default=1.0
        Edit penalty.
    precomputed_distances : np.ndarray
        Pairwise prototype distance matrix.
    itakura_max_slope : float, default=None
        Itakura parallelogram slope.

    Returns
    -------
    np.ndarray of shape (n_cases, n_cases) or (n_cases, m_cases)
        Pairwise rTWE distance matrix.
    """
    multivariate_conversion = _is_multivariate(X, y)
    _X, unequal_length = _convert_to_list(X, "X", multivariate_conversion)
    if y is None:
        return _rtwe_pairwise_distance(
            _X, window, nu, lmbda, precomputed_distances, itakura_max_slope,
            unequal_length,
        )
    _y, unequal_length = _convert_to_list(y, "y", multivariate_conversion)
    return _rtwe_from_multiple_to_multiple_distance(
        _X, _y, window, nu, lmbda, precomputed_distances, itakura_max_slope,
        unequal_length,
    )


@njit(cache=False, fastmath=True, parallel=True)
def _rtwe_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    precomputed_distances: np.ndarray,
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    padded_X = NumbaList()
    for i in range(n_cases):
        padded_X.append(_pad_arrs(X[i]))

    # Shared bounding matrix for the equal-length case (a valid placeholder otherwise,
    # always defined so the prange body never reads an undefined variable).
    bounding_matrix = create_bounding_matrix(
        X[0].shape[1], X[0].shape[1], window, itakura_max_slope
    )

    # Flat upper-triangle prange: each (i, j) is an independent distance written to a
    # disjoint cell, so there is no race. Built on the O(L)-memory rolling buffer, so each
    # thread holds only a 2-row buffer (not an O(L^2) cost matrix) — peak memory stays flat
    # at high L (parallelising the full-matrix kernel would hold n_threads * O(L^2)).
    for k in prange(n_cases * n_cases):
        i = k // n_cases
        j = k % n_cases
        if j > i:
            x1, x2 = padded_X[i], padded_X[j]
            if unequal_length:
                bm = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            else:
                bm = bounding_matrix
            distances[i, j] = _rtwe_distance_rolling(
                x1, x2, bm, nu, lmbda, precomputed_distances,
            )

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            distances[j, i] = distances[i, j]

    return distances


@njit(cache=False, fastmath=True, parallel=True)
def _rtwe_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    precomputed_distances: np.ndarray,
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))

    padded_x = NumbaList()
    for i in range(n_cases):
        padded_x.append(_pad_arrs(x[i]))

    padded_y = NumbaList()
    for i in range(m_cases):
        padded_y.append(_pad_arrs(y[i]))

    bounding_matrix = create_bounding_matrix(
        x[0].shape[1], y[0].shape[1], window, itakura_max_slope
    )

    for k in prange(n_cases * m_cases):
        i = k // m_cases
        j = k % m_cases
        x1, y1 = padded_x[i], padded_y[j]
        if unequal_length:
            bm = create_bounding_matrix(
                x1.shape[1], y1.shape[1], window, itakura_max_slope
            )
        else:
            bm = bounding_matrix
        distances[i, j] = _rtwe_distance_rolling(
            x1, y1, bm, nu, lmbda, precomputed_distances,
        )
    return distances


@njit(cache=False, fastmath=True, parallel=False)
def rtwe_alignment_path_with_costs(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: np.ndarray = None,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[float, np.ndarray, List[Tuple[int, int]], List[float]]:
    """Compute rTWE alignment path with intermediate costs.

    Returns
    -------
    distance : float
        rTWE distance.
    cost_matrix : np.ndarray
        Full cost matrix.
    path : List[Tuple[int, int]]
        Alignment path indices.
    path_costs : List[float]
        Cost at each alignment step.
    """
    bounding_matrix = create_bounding_matrix(
        x.shape[-1], y.shape[-1], window, itakura_max_slope
    )
    cost_matrix = rtwe_cost_matrix(
        x, y, window, nu, lmbda, precomputed_distances, itakura_max_slope,
    )
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)

    path = compute_min_return_path(cost_matrix)
    path_costs = [cost_matrix[i, j] for (i, j) in path]
    distance = cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1]

    return distance, cost_matrix, path, path_costs


@njit(cache=False, fastmath=True)
def rtwe_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    precomputed_distances: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the rTWE alignment path between two symbolic sequences.

    .. warning::
        Argument-order quirk. Here ``precomputed_distances`` is the **3rd positional,
        required** argument, whereas its siblings ``rtwe_distance`` /
        ``rtwe_cost_matrix`` / ``rtwe_alignment_path_with_costs`` /
        ``eshape_dtw_alignment_path`` all place it **last, with a ``None`` default**.
        Always pass it **by keyword** (``precomputed_distances=D_G``) so a copied
        sibling call site does not silently bind ``window``/``nu`` to the wrong slot.
        The signature is retained (this is a ``@njit`` kernel; reordering would be a
        breaking positional change) -- the fix is the keyword convention.

    Parameters
    ----------
    x, y : np.ndarray
        Symbolic sequences to align.
    precomputed_distances : np.ndarray
        Pairwise prototype distance matrix.
    window : float, default=None
        Sakoe-Chiba band width.
    nu : float, default=0.001
        Stiffness penalty.
    lmbda : float, default=1.0
        Edit penalty.
    itakura_max_slope : float, default=None
        Itakura parallelogram slope.

    Returns
    -------
    path : List[Tuple[int, int]]
        Alignment path indices.
    distance : float
        rTWE distance.
    """
    bounding_matrix = create_bounding_matrix(
        x.shape[-1], y.shape[-1], window, itakura_max_slope
    )
    cost_matrix = rtwe_cost_matrix(
        x, y, window, nu, lmbda, precomputed_distances, itakura_max_slope,
    )
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
