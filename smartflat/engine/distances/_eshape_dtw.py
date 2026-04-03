"""Edit-Shape DTW distance between symbolic sequences using rTWE inner cost.

Thesis contribution (Chapter 6, Section 6.2): an Edit-Shape DTW algorithm
where the inner distance between time-series columns is computed using
Registered TWE (rTWE) with precomputed prototype distances.

The outer loop iterates over sequence columns (shape descriptors),
computing insertion, deletion, and match costs via rTWE at each cell.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from numba.typed import List as NumbaList

from smartflat.engine.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from smartflat.engine.distances._bounding_matrix import create_bounding_matrix
from smartflat.engine.distances._rtwe import rtwe_distance
from smartflat.engine.distances._utils import _convert_to_list, _is_multivariate


def eshape_dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
    step_sequ: int = 1,
) -> float:
    """Compute the Edit-Shape DTW distance between two symbolic sequences.

    Parameters
    ----------
    x, y : np.ndarray
        Symbolic sequences, shape ``(n_channels, n_timepoints)``.
    window : float, default=None
        Sakoe-Chiba band width for the outer alignment.
    nu : float, default=0.001
        Stiffness penalty.
    lmbda : float, default=1.0
        Edit penalty.
    precomputed_distances : np.ndarray, default=None
        Pairwise prototype distance matrix for rTWE inner cost.
    itakura_max_slope : float, default=None
        Itakura parallelogram slope.
    step_sequ : int, default=1
        Step size for subsampling columns.

    Returns
    -------
    float
        Edit-Shape DTW distance.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        nx = (x.shape[1] - 1) // step_sequ
        ny = (y.shape[1] - 1) // step_sequ
        bounding_matrix = create_bounding_matrix(nx, ny, window, itakura_max_slope)
        return _eshape_dtw_distance(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda,
            precomputed_distances, step_sequ,
        )
    if x.ndim == 2 and y.ndim == 2:
        nx = (x.shape[1] - 1) // step_sequ
        ny = (y.shape[1] - 1) // step_sequ
        bounding_matrix = create_bounding_matrix(nx, ny, window, itakura_max_slope)
        return _eshape_dtw_distance(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda,
            precomputed_distances, step_sequ,
        )
    raise ValueError("x and y must be 1D or 2D")


def eshape_dtw_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
    step_sequ: int = 1,
) -> np.ndarray:
    """Compute the Edit-Shape DTW cost matrix between two symbolic sequences.

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
    precomputed_distances : np.ndarray, default=None
        Pairwise prototype distance matrix.
    itakura_max_slope : float, default=None
        Itakura parallelogram slope.
    step_sequ : int, default=1
        Column subsampling step.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """
    if x.ndim == 1 and y.ndim == 1:
        _x = x.reshape((1, x.shape[0]))
        _y = y.reshape((1, y.shape[0]))
        nx = (_x.shape[1] - 1) // step_sequ
        ny = (_y.shape[1] - 1) // step_sequ
        bounding_matrix = create_bounding_matrix(nx, ny, window, itakura_max_slope)
        return _eshape_dtw_cost_matrix(
            _pad_arrs(_x), _pad_arrs(_y), bounding_matrix, nu, lmbda,
            precomputed_distances, step_sequ,
        )
    if x.ndim == 2 and y.ndim == 2:
        nx = (x.shape[1] - 1) // step_sequ
        ny = (y.shape[1] - 1) // step_sequ
        bounding_matrix = create_bounding_matrix(nx, ny, window, itakura_max_slope)
        return _eshape_dtw_cost_matrix(
            _pad_arrs(x), _pad_arrs(y), bounding_matrix, nu, lmbda,
            precomputed_distances, step_sequ,
        )
    raise ValueError("x and y must be 1D or 2D")


def _eshape_dtw_distance(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
    nu: float, lmbda: float,
    precomputed_distances: Optional[np.ndarray] = None,
    step_sequ: int = 1,
) -> float:
    return _eshape_dtw_cost_matrix(
        x, y, bounding_matrix, nu, lmbda, precomputed_distances, step_sequ,
    )[(x.shape[1] - 1) // step_sequ - 2, (y.shape[1] - 1) // step_sequ - 2]


def _eshape_dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray,
    nu: float, lmbda: float,
    precomputed_distances: Optional[np.ndarray] = None,
    step_sequ: int = 1,
) -> np.ndarray:
    """Core Edit-Shape DTW cost matrix using rTWE as inner distance."""
    x_size = x.shape[1]
    y_size = y.shape[1]
    nx = len(range(1, x_size, step_sequ))
    ny = len(range(1, y_size, step_sequ))

    x_cols = [np.ascontiguousarray(x[:, i][None, :]) for i in range(1, x_size, step_sequ)]
    y_cols = [np.ascontiguousarray(y[:, i][None, :]) for i in range(1, y_size, step_sequ)]

    nu_rtwe = 0.001
    lmbda_rtwe = 0.01

    cost_matrix = np.zeros((nx, ny))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    del_add = nu + lmbda

    for i in range(1, nx):
        for j in range(1, ny):
            if bounding_matrix[i - 1, j - 1]:
                xip, xi = x_cols[i - 1], x_cols[i]
                yjp, yj = y_cols[j - 1], y_cols[j]

                # Deletion in x
                del_x_rtwe_dist = rtwe_distance(
                    xip, xi, nu=nu_rtwe, lmbda=lmbda_rtwe,
                    precomputed_distances=precomputed_distances,
                )
                del_x = cost_matrix[i - 1, j] + del_x_rtwe_dist + del_add

                # Deletion in y
                del_y_rtwe_dist = rtwe_distance(
                    yjp, yj, nu=nu_rtwe, lmbda=lmbda_rtwe,
                    precomputed_distances=precomputed_distances,
                )
                del_y = cost_matrix[i, j - 1] + del_y_rtwe_dist + del_add

                # Match
                match_same_rtwe_d = rtwe_distance(
                    xi, yj, nu=nu_rtwe, lmbda=lmbda_rtwe,
                    precomputed_distances=precomputed_distances,
                )
                match_prev_rtwe_d = rtwe_distance(
                    xip, yjp, nu=nu_rtwe, lmbda=lmbda_rtwe,
                    precomputed_distances=precomputed_distances,
                )
                match = (
                    cost_matrix[i - 1, j - 1]
                    + match_same_rtwe_d
                    + match_prev_rtwe_d
                    + nu * (abs(i - j) + abs((i - 1) - (j - 1)))
                )

                cost_matrix[i, j] = min(del_x, del_y, match)

    return cost_matrix[1:, 1:]


def _pad_arrs(x: np.ndarray) -> np.ndarray:
    """Prepend a zero column to each channel (for boundary conditions)."""
    padded_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    zero_arr = np.array([0], dtype=x.dtype)
    for i in range(x.shape[0]):
        padded_x[i, :] = np.concatenate((zero_arr, x[i, :]))
    return padded_x


def eshape_dtw_pairwise_distance(
    X: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> np.ndarray:
    """Compute pairwise Edit-Shape DTW distances.

    Parameters
    ----------
    X : np.ndarray or List[np.ndarray]
        Collection of sequences.
    y : np.ndarray or List[np.ndarray] or None
        Second collection. If None, compute self-pairwise.
    window, nu, lmbda, precomputed_distances, itakura_max_slope
        See ``eshape_dtw_distance``.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix.
    """
    multivariate_conversion = _is_multivariate(X, y)
    _X, unequal_length = _convert_to_list(X, "X", multivariate_conversion)
    if y is None:
        return _eshape_dtw_pairwise_distance(
            _X, window, nu, lmbda, precomputed_distances, itakura_max_slope,
            unequal_length,
        )
    _y, unequal_length = _convert_to_list(y, "y", multivariate_conversion)
    return _eshape_dtw_from_multiple_to_multiple_distance(
        _X, _y, window, nu, lmbda, precomputed_distances, itakura_max_slope,
        unequal_length,
    )


def _eshape_dtw_pairwise_distance(
    X: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    precomputed_distances: Optional[np.ndarray],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(X)
    distances = np.zeros((n_cases, n_cases))

    if not unequal_length:
        n_timepoints = X[0].shape[1]
        bounding_matrix = create_bounding_matrix(
            n_timepoints, n_timepoints, window, itakura_max_slope
        )

    padded_X = NumbaList()
    for i in range(n_cases):
        padded_X.append(_pad_arrs(X[i]))

    for i in range(n_cases):
        for j in range(i + 1, n_cases):
            x1, x2 = padded_X[i], padded_X[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], x2.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _eshape_dtw_distance(
                x1, x2, bounding_matrix, nu, lmbda, precomputed_distances,
            )
            distances[j, i] = distances[i, j]

    return distances


def _eshape_dtw_from_multiple_to_multiple_distance(
    x: NumbaList[np.ndarray],
    y: NumbaList[np.ndarray],
    window: Optional[float],
    nu: float,
    lmbda: float,
    precomputed_distances: Optional[np.ndarray],
    itakura_max_slope: Optional[float],
    unequal_length: bool,
) -> np.ndarray:
    n_cases = len(x)
    m_cases = len(y)
    distances = np.zeros((n_cases, m_cases))
    if not unequal_length:
        bounding_matrix = create_bounding_matrix(
            x[0].shape[1], y[0].shape[1], window, itakura_max_slope
        )

    padded_x = NumbaList()
    for i in range(n_cases):
        padded_x.append(_pad_arrs(x[i]))

    padded_y = NumbaList()
    for i in range(m_cases):
        padded_y.append(_pad_arrs(y[i]))

    for i in range(n_cases):
        for j in range(m_cases):
            x1, y1 = padded_x[i], padded_y[j]
            if unequal_length:
                bounding_matrix = create_bounding_matrix(
                    x1.shape[1], y1.shape[1], window, itakura_max_slope
                )
            distances[i, j] = _eshape_dtw_distance(
                x1, y1, bounding_matrix, nu, lmbda, precomputed_distances,
            )
    return distances


def eshape_dtw_alignment_path(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    nu: float = 0.001,
    lmbda: float = 1.0,
    precomputed_distances: Optional[np.ndarray] = None,
    itakura_max_slope: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """Compute the Edit-Shape DTW alignment path.

    Returns
    -------
    path : List[Tuple[int, int]]
        Alignment path indices.
    distance : float
        Edit-Shape DTW distance.
    """
    bounding_matrix = create_bounding_matrix(
        x.shape[-1], y.shape[-1], window, itakura_max_slope
    )
    cost_matrix = eshape_dtw_cost_matrix(
        x, y, window, nu, lmbda, precomputed_distances, itakura_max_slope,
    )
    cost_matrix = _add_inf_to_out_of_bounds_cost_matrix(cost_matrix, bounding_matrix)
    return (
        compute_min_return_path(cost_matrix),
        cost_matrix[x.shape[-1] - 1, y.shape[-1] - 1],
    )
