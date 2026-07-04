"""Baseline barycenter methods for comparison with TW-TWE + DBA (PAPER_BRIDGE items A5-A7, B1-B3).

Provides six alternative barycenter computation methods with a uniform
interface, plus embedding utilities and an evaluation framework.

Design decision: symbolic sequences are embedded into real-valued space
using prototype-distance rows (D_G[s, :]) to preserve Wasserstein structure.

This module is now a thin back-compat re-export shim (arc-audit Phase 2). The
implementations live in cohesive submodules and are re-exported here unchanged so
every ``from smartflat.features.symbolic_barycenter.baselines import X`` keeps
resolving -- public names via ``import *`` and private-but-imported helpers explicitly.

- :mod:`.distances`   -- rTWE constants, embed/decode, DTW/soft-DTW primitives, ``dist_*``/``pmatch_*``
- :mod:`.builders`    -- the ``barycenter_*`` constructors
- :mod:`.registries`  -- the six ``*_methods`` registry builders
- :mod:`.evaluation`  -- CV helpers, evaluators, significance tests, bootstrap CIs
"""

from .distances import *      # noqa: F401,F403
from .builders import *       # noqa: F401,F403
from .registries import *     # noqa: F401,F403
from .evaluation import *     # noqa: F401,F403

# Private-but-referenced helpers (not carried by ``import *``) — imported by
# order_evaluation.py / structure_metrics.py / tests via this shim path.
from .distances import (      # noqa: F401
    _snap_and_reembed, _symbols_to_str, _transition_matrix, _classical_mds, _dtw_cost_matrix, _dtw_alignment, _soft_dtw_grad, _softmin3, _soft_dtw_cost,
)
from .builders import _categorical_outer_loop  # noqa: F401
from .evaluation import (     # noqa: F401
    _make_clf, _nested_cv_auc, _pairwise_subsets, _per_split_auc,
)

__all__ = [
    "RTWE_NU",
    "RTWE_LMBDA",
    "embed_symbolic_to_real",
    "project_real_to_symbolic",
    "_snap_and_reembed",
    "_symbols_to_str",
    "ordinal_cost_matrix",
    "_dtw_cost_matrix",
    "_dtw_alignment",
    "_soft_dtw_grad",
    "_softmin3",
    "_soft_dtw_cost",
    "dist_dtw",
    "dist_soft_dtw",
    "dist_edit",
    "dist_hamming",
    "dist_wasserstein_hist",
    "dist_rtwe",
    "pmatch_to_barycenter",
    "dist_neg_pmatch",
    "pmatch_to_barycenter_stock_twe",
    "dist_neg_pmatch_stock",
    "_transition_matrix",
    "dist_transition",
    "dist_eshape_dtw",
    "_classical_mds",
    "barycenter_dba_dtw",
    "barycenter_soft_dtw",
    "barycenter_softdtw",
    "barycenter_ssg",
    "_categorical_outer_loop",
    "barycenter_ssg_cat",
    "barycenter_softdtw_cat",
    "barycenter_edit_median",
    "barycenter_wasserstein",
    "barycenter_k_medoid",
    "barycenter_majority_voting",
    "barycenter_mode_dba",
    "barycenter_msa_consensus",
    "barycenter_mean_rtwe_dba",
    "barycenter_transition_matrix",
    "barycenter_soft_mode_dba",
    "barycenter_fgw",
    "default_baseline_methods",
    "extra_experiment_methods",
    "fgw_methods",
    "softdtw_ssg_methods",
    "msa_consensus_methods",
    "discreteness_lever_methods",
    "make_patient_control_labels",
    "evaluate_baselines",
    "baseline_significance_tests",
    "histogram_features",
    "transition_features",
    "_pairwise_subsets",
    "_make_clf",
    "_nested_cv_auc",
    "evaluate_incremental_ordering",
    "_per_split_auc",
    "bootstrap_auc_ci",
    "bootstrap_delta_ci",
]
