"""Public-surface guard for the ``baselines`` module split (arc-audit Phase 2).

``smartflat/features/symbolic_barycenter/baselines.py`` was split into cohesive
modules (``distances`` / ``builders`` / ``registries`` / ``evaluation``) with
``baselines.py`` kept as a thin back-compat re-export shim. This test freezes the
full importable surface so the split — and any future edit to the shim — cannot
silently drop a name that a caller (source module, test, or notebook) relies on.

``NAMES`` is the complete set of top-level ``def``/constant symbols that existed in
``baselines.py`` before the split (AST-enumerated: 59 functions + 2 constants), a
superset of every symbol actually imported from ``…baselines`` anywhere in the repo
(public and private-but-imported alike: ``_make_clf``, ``_nested_cv_auc``,
``_pairwise_subsets``, ``_transition_matrix``, ``_snap_and_reembed``). Each must
remain importable *from the shim's fully-qualified path* — that path is what every
consumer uses.
"""

import importlib

BASELINES_PATH = "smartflat.features.symbolic_barycenter.baselines"

# The complete pre-split public surface (61 names). Grouped by the module each name
# moved to, for readability; the test asserts all of them resolve from the shim.

# distances.py — constants + embed/decode + DTW/soft-DTW primitives + dist_*/pmatch_*
DISTANCES_NAMES = [
    "RTWE_NU", "RTWE_LMBDA",
    "embed_symbolic_to_real", "project_real_to_symbolic", "_snap_and_reembed",
    "_symbols_to_str", "_classical_mds", "ordinal_cost_matrix", "_transition_matrix",
    "_dtw_cost_matrix", "_dtw_alignment", "_soft_dtw_grad", "_softmin3", "_soft_dtw_cost",
    "dist_dtw", "dist_soft_dtw", "dist_edit", "dist_hamming", "dist_wasserstein_hist",
    "dist_rtwe", "pmatch_to_barycenter", "dist_neg_pmatch",
    "pmatch_to_barycenter_stock_twe", "dist_neg_pmatch_stock", "dist_transition",
    "dist_eshape_dtw",
]

# builders.py — the 17 barycenter_* constructors (+ the categorical outer-loop helper)
BUILDERS_NAMES = [
    "barycenter_dba_dtw", "barycenter_soft_dtw", "barycenter_softdtw", "barycenter_ssg",
    "_categorical_outer_loop", "barycenter_ssg_cat", "barycenter_softdtw_cat",
    "barycenter_edit_median", "barycenter_wasserstein", "barycenter_k_medoid",
    "barycenter_majority_voting", "barycenter_mode_dba", "barycenter_msa_consensus",
    "barycenter_mean_rtwe_dba", "barycenter_transition_matrix", "barycenter_soft_mode_dba",
    "barycenter_fgw",
]

# registries.py — the six *_methods registry builders
REGISTRIES_NAMES = [
    "default_baseline_methods", "extra_experiment_methods", "fgw_methods",
    "softdtw_ssg_methods", "msa_consensus_methods", "discreteness_lever_methods",
]

# evaluation.py — CV helpers + evaluators + feature/label utilities
EVALUATION_NAMES = [
    "make_patient_control_labels", "evaluate_baselines", "baseline_significance_tests",
    "histogram_features", "transition_features", "_pairwise_subsets", "_make_clf",
    "_nested_cv_auc", "evaluate_incremental_ordering", "_per_split_auc",
    "bootstrap_auc_ci", "bootstrap_delta_ci",
]

NAMES = DISTANCES_NAMES + BUILDERS_NAMES + REGISTRIES_NAMES + EVALUATION_NAMES


def test_shim_reexports_full_surface():
    """Every pre-split symbol is importable from the baselines shim path."""
    m = importlib.import_module(BASELINES_PATH)
    missing = [n for n in NAMES if not hasattr(m, n)]
    assert not missing, f"names not re-exported by the baselines shim: {missing}"


def test_surface_has_no_duplicates():
    """The frozen name list is internally consistent (no accidental dup)."""
    assert len(NAMES) == len(set(NAMES)) == 61
