"""Custom elastic distance metrics for symbolic sequences (Ch. 6).

This module contains thesis contributions for aligning and comparing
symbolic sequences derived from the recursive prototyping pipeline:

- **rTWE** (Registered Time Warp Edit): TWE adapted for discrete symbolic
  sequences using precomputed prototype distance matrices.
- **Edit-Shape DTW**: Outer Edit-Shape DTW using rTWE as the inner cost.

Utility modules (``_alignment_paths``, ``_bounding_matrix``, ``_utils``)
are vendored from the `aeon <https://github.com/aeon-toolkit/aeon>`_
time-series toolkit (BSD-3-Clause license) to make these distances
self-contained.
"""

from smartflat.engine.distances._rtwe import (
    rtwe_alignment_path,
    rtwe_alignment_path_with_costs,
    rtwe_cost_matrix,
    rtwe_distance,
    rtwe_pairwise_distance,
)
from smartflat.engine.distances._eshape_dtw import (
    eshape_dtw_alignment_path,
    eshape_dtw_cost_matrix,
    eshape_dtw_distance,
    eshape_dtw_pairwise_distance,
)

__all__ = [
    "rtwe_distance",
    "rtwe_cost_matrix",
    "rtwe_alignment_path",
    "rtwe_alignment_path_with_costs",
    "rtwe_pairwise_distance",
    "eshape_dtw_distance",
    "eshape_dtw_cost_matrix",
    "eshape_dtw_alignment_path",
    "eshape_dtw_pairwise_distance",
]
