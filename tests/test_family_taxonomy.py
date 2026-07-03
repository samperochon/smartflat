"""Tests for the canonical discreteness-family taxonomy (Kickoff L).

Pins that ``barycenter_quality.{FAMILY, family_of}`` classify every quality-roster
registry key (so future notebooks import this map instead of re-copying the literal that
had drifted across 06i/06j/06k), and that suffix inference covers the FGW alpha-sweep and
the ``*_dg`` / ``*_cat`` lever keys. Dependency-free: the registry factories build closures
without importing ``ot`` / ``tslearn``.
"""

import numpy as np

from smartflat.features.symbolic_barycenter.baselines import (
    default_baseline_methods,
    discreteness_lever_methods,
    fgw_methods,
    msa_consensus_methods,
    softdtw_ssg_methods,
)
from smartflat.features.symbolic_barycenter.barycenter_quality import FAMILY, family_of

from _bary_helpers import _ground_cost


def _roster_keys():
    """Every key across the quality-comparison registries (no optional deps to construct)."""
    d_g = _ground_cost(6)
    keys = set()
    for reg in (
        default_baseline_methods(d_g),
        fgw_methods(d_g),
        softdtw_ssg_methods(d_g),
        msa_consensus_methods(d_g),
        discreteness_lever_methods(d_g),
    ):
        keys |= set(reg)
    return keys


def test_family_of_resolves_every_quality_roster_key():
    unresolved = {k for k in _roster_keys() if family_of(k) == '?'}
    assert not unresolved, f"family_of returned '?' for registry keys: {sorted(unresolved)}"


def test_family_labels_start_with_A_or_B():
    # every classified method is Family A or Family B (the two-family taxonomy)
    for method, label in FAMILY.items():
        assert label[0] in ('A', 'B'), (method, label)


def test_suffix_inference_for_unlisted_keys():
    # FGW alpha-sweep keys from build_fgw_registry (fgw_{enc}_a{alpha}) -> Family B
    assert family_of('fgw_onehot_a0.5') == 'B'
    assert family_of('fgw_mds_a0') == 'B'
    # lever suffixes on an otherwise-unlisted base
    assert family_of('some_new_method_cat') == 'B->cat (per-iter)'
    assert family_of('some_new_method_dg') == 'B (D_G decode)'
    # genuinely unknown
    assert family_of('totally_unknown') == '?'


def test_exact_map_takes_precedence_over_suffix():
    # fgw_onehot_dg is explicitly listed; must return its exact label, not a generic infer
    assert family_of('fgw_onehot_dg') == FAMILY['fgw_onehot_dg'] == 'B (D_G decode)'
