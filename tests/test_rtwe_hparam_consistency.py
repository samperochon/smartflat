"""Kickoff-M guard: the rTWE ``nu``/``lmbda`` operating point is single + canonical.

The arc's rTWE cost default drifted into four operating points across ~10 sub-sessions
(``ARC_AUDIT.md`` Dimension 5). Phase 1 collapses the *smartflat layer* to one canonical
point ``RTWE_NU=1e-4, RTWE_LMBDA=0.1`` behind module constants. This test locks that in:

  (a) every smartflat-layer ``nu``/``lmbda`` default references the canonical constants
      (none is the old ``0.001`` / ``1.0`` drift);
  (b) the bare default now equals the explicit canonical point equals a pinned oracle
      (``tests/_oracles/rtwe_hparam_oracle.csv``, frozen once at the canonical point);
  (c) the ``barycenter_dba_dtw`` ``max_iters`` -> ``max_iter`` rename keeps a
      behaviour-identical back-compat alias.

Fast + dependency-light: the numeric cells use pure-numpy methods only (no ot/tslearn),
so this runs in the frozen suite without the optional deps.
"""
import csv
import inspect
import os

import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.baselines import (
    RTWE_NU, RTWE_LMBDA,
    barycenter_mode_dba, barycenter_msa_consensus, barycenter_mean_rtwe_dba,
    dist_rtwe, pmatch_to_barycenter, dist_neg_pmatch,
    pmatch_to_barycenter_stock_twe, dist_neg_pmatch_stock, dist_eshape_dtw,
    barycenter_soft_mode_dba, default_baseline_methods, extra_experiment_methods,
    fgw_methods, softdtw_ssg_methods, msa_consensus_methods, discreteness_lever_methods,
    barycenter_dba_dtw,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import (
    score_barycenter_quality, quality_table,
)
from _bary_helpers import _ground_cost, _cohort

_ORACLE_CSV = os.path.join(os.path.dirname(__file__), "_oracles", "rtwe_hparam_oracle.csv")

# Every smartflat-layer function/registry that should now default to the canonical point.
_SMARTFLAT_LAYER = [
    barycenter_mode_dba, barycenter_msa_consensus, barycenter_mean_rtwe_dba,
    dist_rtwe, pmatch_to_barycenter, dist_neg_pmatch,
    pmatch_to_barycenter_stock_twe, dist_neg_pmatch_stock, dist_eshape_dtw,
    barycenter_soft_mode_dba, default_baseline_methods, extra_experiment_methods,
    fgw_methods, softdtw_ssg_methods, msa_consensus_methods, discreteness_lever_methods,
]


def _derive_cells():
    """The pinned nu/lmbda-sensitive cells (single-sourced for oracle + test).

    All quantities use only pure-numpy registry methods so no optional dep is needed.
    """
    D = _ground_cost(6)
    X, labels = _cohort(6, 32)
    cells = {}
    # (b) direct rTWE distances at the bare default (== canonical after unification).
    cells["dist_rtwe_0_3"] = float(dist_rtwe(X[0], X[3], D))
    cells["dist_rtwe_1_7"] = float(dist_rtwe(X[1], X[7], D))
    # quality_table yardstick cells (inertia_rtwe uses the score's rtwe_nu/rtwe_lmbda).
    # symbol-sequence methods -> non-NaN rTWE inertia (the common yardstick, which is
    # itself pinned at RTWE_NU/RTWE_LMBDA); wasserstein is skipped (histogram -> NaN).
    methods = {
        k: default_baseline_methods(D, nu=RTWE_NU, lmbda=RTWE_LMBDA)[k]
        for k in ("majority_voting", "edit_median")
    }
    res = score_barycenter_quality(X, labels, methods, D, n_inits=1, random_state=0)
    q = quality_table(res)
    cells["majority_voting_inertia_rtwe"] = float(q.loc["majority_voting", "inertia_rtwe"])
    cells["edit_median_inertia_rtwe"] = float(q.loc["edit_median", "inertia_rtwe"])
    return cells


def _load_oracle():
    with open(_ORACLE_CSV, newline="") as fh:
        return {row["name"]: float(row["value"]) for row in csv.DictReader(fh)}


# ---- (a) single canonical operating point --------------------------------------

def test_canonical_constants_value():
    assert RTWE_NU == 1e-4
    assert RTWE_LMBDA == 0.1


@pytest.mark.parametrize("fn", _SMARTFLAT_LAYER, ids=lambda f: f.__name__)
def test_smartflat_layer_default_is_canonical(fn):
    """No smartflat-layer nu/lmbda default is the old 0.001 / 1.0 drift."""
    params = inspect.signature(fn).parameters
    assert params["nu"].default == RTWE_NU, f"{fn.__name__} nu default drifted"
    assert params["lmbda"].default == RTWE_LMBDA, f"{fn.__name__} lmbda default drifted"
    # the specific drifted values must be gone
    assert params["nu"].default != 0.001, f"{fn.__name__} still defaults nu=0.001"
    assert params["lmbda"].default != 1.0, f"{fn.__name__} still defaults lmbda=1.0"


# ---- (b) bare default == explicit canonical == pinned oracle --------------------

def test_bare_default_equals_explicit_canonical():
    """dist_rtwe with no nu/lmbda now hits the canonical point exactly."""
    D = _ground_cost(6)
    X, _ = _cohort(6, 32)
    for a, b in [(0, 3), (1, 7)]:
        bare = dist_rtwe(X[a], X[b], D)
        explicit = dist_rtwe(X[a], X[b], D, nu=RTWE_NU, lmbda=RTWE_LMBDA)
        assert bare == explicit  # identical code path -> bit-exact


def test_cells_match_pinned_oracle():
    oracle = _load_oracle()
    derived = _derive_cells()
    assert set(derived) == set(oracle)
    for name, val in derived.items():
        assert val == pytest.approx(oracle[name], rel=1e-9, abs=1e-12), name


# ---- (c) max_iters -> max_iter back-compat alias --------------------------------

def test_max_iter_alias_is_behaviour_identical():
    D = _ground_cost(6)
    X, _ = _cohort(6, 24)
    new = barycenter_dba_dtw(X, D, max_iter=4, random_state=0)
    with pytest.warns(DeprecationWarning):
        old = barycenter_dba_dtw(X, D, max_iters=4, random_state=0)
    np.testing.assert_array_equal(new, old)


def test_max_iter_signature_renamed():
    params = inspect.signature(barycenter_dba_dtw).parameters
    assert "max_iter" in params
    assert params["max_iter"].default == 30
    assert params["max_iters"].default is None  # deprecated alias, opt-in
