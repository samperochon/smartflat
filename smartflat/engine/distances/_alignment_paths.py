"""Alignment path utilities for elastic distance computation.

Vendored from aeon v1.0.0 (BSD-3-Clause license).
https://github.com/aeon-toolkit/aeon

Original authors: aeon developers.
"""

from typing import List, Tuple

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def compute_min_return_path(cost_matrix: np.ndarray) -> List[Tuple]:
    """Compute the minimum return path through a cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray, of shape (n_timepoints_x, n_timepoints_y)
        Cost matrix.

    Returns
    -------
    List[Tuple]
        List of indices that make up the minimum return path.
    """
    x_size, y_size = cost_matrix.shape
    i, j = x_size - 1, y_size - 1
    alignment = []

    while i > 0 or j > 0:
        alignment.append((i, j))

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_index = np.argmin(
                np.array(
                    [
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1],
                    ]
                )
            )
            if min_index == 0:
                i, j = i - 1, j - 1
            elif min_index == 1:
                i -= 1
            else:
                j -= 1

    alignment.append((0, 0))
    return alignment[::-1]


@njit(cache=True, fastmath=True)
def _add_inf_to_out_of_bounds_cost_matrix(
    cost_matrix: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    """Set out-of-bounds cost matrix entries to infinity.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix to modify in-place.
    bounding_matrix : np.ndarray
        Boolean matrix indicating valid (True) and invalid (False) entries.

    Returns
    -------
    np.ndarray
        Modified cost matrix.
    """
    x_size, y_size = cost_matrix.shape
    for i in range(x_size):
        for j in range(y_size):
            if not np.isfinite(bounding_matrix[i, j]):
                cost_matrix[i, j] = np.inf
    return cost_matrix
