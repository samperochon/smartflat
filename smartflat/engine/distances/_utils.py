"""Array conversion utilities for pairwise distance computation.

Vendored from aeon (BSD-3-Clause license).
https://github.com/aeon-toolkit/aeon

Original authors: aeon developers.
"""

from typing import List, Optional, Union

import numpy as np
from numba.typed import List as NumbaList


def _is_multivariate(
    x: Union[np.ndarray, List[np.ndarray]],
    y: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
) -> bool:
    """Determine if input time series collections are multivariate."""
    if y is None:
        if isinstance(x, np.ndarray):
            x_dims = x.ndim
            if x_dims == 3:
                return x.shape[1] != 1
            return False
        if isinstance(x, (List, NumbaList)):
            x_dims = x[0].ndim
            if x_dims == 2:
                return x[0].shape[0] != 1
            return False
    else:
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_dims = x.ndim
            y_dims = y.ndim
            if x_dims < y_dims:
                return _is_multivariate(y, x)
            if x_dims == 3 and y_dims == 3:
                return not (x.shape[1] == 1 and y.shape[1] == 1)
            if x_dims == 3 and y_dims == 2:
                return x.shape[1] != 1
            if x_dims == 3 and y_dims == 1:
                return x.shape[1] != 1
            return False
        if isinstance(x, (List, NumbaList)) and isinstance(y, (List, NumbaList)):
            x_dims = x[0].ndim
            y_dims = y[0].ndim
            if x_dims < y_dims:
                return _is_multivariate(y, x)
            if x_dims == 1 or y_dims == 1:
                return False
            if x_dims == 2 and y_dims == 2:
                return not (x[0].shape[0] == 1 or y[0].shape[0] == 1)

        list_x = None
        ndarray_y: Optional[np.ndarray] = None
        if isinstance(x, (List, NumbaList)):
            list_x = x
            ndarray_y = y
        elif isinstance(y, (List, NumbaList)):
            list_x = y
            ndarray_y = x

        if list_x is not None and ndarray_y is not None:
            list_y = []
            if ndarray_y.ndim == 3:
                for i in range(ndarray_y.shape[0]):
                    list_y.append(ndarray_y[i])
            else:
                list_y = [ndarray_y]
            return _is_multivariate(list_x, list_y)

    raise ValueError("The format of your input is not supported.")


def _convert_to_list(
    x: Union[np.ndarray, List[np.ndarray]],
    name: str = "X",
    multivariate_conversion: bool = False,
) -> NumbaList[np.ndarray]:
    """Convert input collections to a NumbaList of 2D arrays.

    Parameters
    ----------
    x : np.ndarray or List[np.ndarray]
        One or more time series.
    name : str
        Variable name for error messages.
    multivariate_conversion : bool
        If True, treat 2D input as a single multivariate series.

    Returns
    -------
    NumbaList[np.ndarray]
        List of 2D arrays (n_channels, n_timepoints).
    bool
        Whether the time series have unequal lengths.
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return NumbaList(x), False
        elif x.ndim == 2:
            if multivariate_conversion:
                return NumbaList(x.reshape(1, x.shape[0], x.shape[1])), False
            return NumbaList(x.reshape(x.shape[0], 1, x.shape[1])), False
        elif x.ndim == 1:
            return NumbaList(x.reshape(1, 1, x.shape[0])), False
        else:
            raise ValueError(f"{name} must be 1D, 2D or 3D")
    elif isinstance(x, (List, NumbaList)):
        x_new = NumbaList()
        expected_n_timepoints = x[0].shape[-1]
        unequal_timepoints = False
        for i in range(len(x)):
            curr_x = x[i]
            if curr_x.shape[-1] != expected_n_timepoints:
                unequal_timepoints = True
            if x[i].ndim == 2:
                x_new.append(curr_x)
            elif x[i].ndim == 1:
                x_new.append(curr_x.reshape((1, curr_x.shape[0])))
            else:
                raise ValueError(f"{name} must include only 1D or 2D arrays")
        return x_new, unequal_timepoints
    else:
        raise ValueError(f"{name} must be either np.ndarray or List[np.ndarray]")
