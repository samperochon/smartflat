"""Symbolic barycenter estimation and hierarchical community detection (Ch. 6).

Implements the analysis pipeline from Chapter 6 of the thesis
("Symbolic Representation Analysis and Barycenter Estimation for the
Assessment of dysexecutive syndromes"):

- Edit-Shape DTW with rTWE cost matrix for symbolic sequence alignment: ``main``
- Hierarchical community detection and Markov chain analysis: ``hierarchical_states_transitions``
- Alignment path, cost matrix, and distance distribution visualization: ``visualization``
- Baseline barycenter methods for paper comparison: ``baselines`` (a back-compat shim
  over ``distances`` / ``builders`` / ``registries`` / ``evaluation``)
- Representation-quality harness: ``barycenter_quality``
- Order- and structure-evaluation: ``order_evaluation`` / ``structure_metrics``
- Shared G=28 cohort + ground-cost loaders: ``vocab``

Curated re-exports (arc-audit Phase 3). The F-arc analysis API is reachable directly at
package level -- ``from smartflat.features.symbolic_barycenter import barycenter_fgw`` -- in
addition to the full submodule paths. The large baseline/method surface is re-exported at
runtime from ``baselines.__all__`` (single source of truth) so this list never drifts.

The runnable scripts ``main`` / ``hierarchical_states_transitions`` are intentionally not
eagerly imported here (kept out of package-import cost); import them by submodule path.
"""

from . import (  # noqa: F401 -- re-exported for convenience + documented package map
    baselines,
    distances,
    builders,
    registries,
    evaluation,
    barycenter_quality,
    order_evaluation,
    structure_metrics,
    vocab,
    visualization,
)
from .barycenter_quality import (  # noqa: F401
    FAMILY,
    family_of,
    score_barycenter_quality,
    quality_table,
    build_fgw_registry,
)
from .order_evaluation import (  # noqa: F401
    order_information,
    order_shuffle_null,
    run_transition_features,
    runlength_shuffle,
    token_shuffle,
)
from .structure_metrics import (  # noqa: F401
    evaluate_incremental_structure,
    evaluate_structure_length_controlled,
    structure_group_stats,
    structure_features,
    compute_structure_metrics,
)
from .vocab import (  # noqa: F401
    build_g28_ground_cost,
    load_g28_cohort,
    make_ground_cost,
    compute_distance_matrix,
    build_category_codes,
    add_category_columns,
    recompute_category_temporal_D_G,
    pyramid_labels_to_category,
)

# The baseline/method surface (embed/decode, dist_*, barycenter_*, the six *_methods
# registries, evaluators, bootstrap CIs, RTWE_NU/RTWE_LMBDA) lives in the back-compat
# shim's __all__. Re-export its PUBLIC (non-underscore) names at runtime so this package
# surface stays in lock-step with baselines.__all__ -- no hardcoded list to drift.
_baseline_public = [_n for _n in baselines.__all__ if not _n.startswith("_")]
globals().update({_n: getattr(baselines, _n) for _n in _baseline_public})

_submodules = [
    "baselines", "distances", "builders", "registries", "evaluation",
    "barycenter_quality", "order_evaluation", "structure_metrics",
    "vocab", "visualization",
]
_harness_public = [
    "FAMILY", "family_of", "score_barycenter_quality", "quality_table",
    "build_fgw_registry",
    "order_information", "order_shuffle_null", "run_transition_features",
    "runlength_shuffle", "token_shuffle",
    "evaluate_incremental_structure", "evaluate_structure_length_controlled",
    "structure_group_stats", "structure_features", "compute_structure_metrics",
    "build_g28_ground_cost", "load_g28_cohort", "make_ground_cost",
    "compute_distance_matrix", "build_category_codes", "add_category_columns",
    "recompute_category_temporal_D_G", "pyramid_labels_to_category",
]
__all__ = _submodules + _harness_public + _baseline_public
