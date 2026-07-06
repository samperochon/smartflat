"""Guard for the additive ``_pairwise_subsets`` generalization (PAPER_TODO §2, Phase 0).

Two contracts: (1) on the SDS2 vocabulary the function is **byte-identical** to the
frozen three-comparison behaviour (so §15/§16 numbers cannot move), and (2) on any
other label vocabulary it yields all unordered class pairs (so the order/structure
harnesses run unchanged on generalization datasets).
"""
import numpy as np

from smartflat.features.symbolic_barycenter.evaluation import _pairwise_subsets


def test_sds2_labels_yield_the_three_frozen_comparisons():
    labels = np.array(['HEALTHY'] * 3 + ['RIL'] * 4 + ['TBI'] * 5, dtype=object)
    out = list(_pairwise_subsets(labels))
    assert [name for name, _, _ in out] == ['HEALTHY_vs_RIL', 'RIL_vs_TBI', 'CONTROL_vs_PATIENT']

    _, mask, y = out[0]                      # HEALTHY(0) vs RIL(1)
    assert mask.sum() == 7 and y.tolist() == [0, 0, 0, 1, 1, 1, 1]
    _, mask, y = out[1]                      # RIL(0) vs TBI(1)
    assert mask.sum() == 9 and y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    _, mask, y = out[2]                      # CONTROL(HEALTHY)=0 vs PATIENT(RIL,TBI)=1
    assert mask.sum() == 12 and y.tolist() == [0, 0, 0] + [1] * 9


def test_two_subset_still_takes_sds2_path():
    labels = np.array(['HEALTHY'] * 3 + ['RIL'] * 3, dtype=object)
    names = [n for n, _, _ in _pairwise_subsets(labels)]
    assert 'HEALTHY_vs_RIL' in names        # subset of SDS2 -> SDS2 branch fires


def test_generic_labels_yield_all_pairs():
    labels = np.array(['pour_milk', 'pour_coffee', 'fry_egg',
                       'pour_milk', 'fry_egg', 'pour_coffee'], dtype=object)
    out = list(_pairwise_subsets(labels))
    assert len(out) == 3                     # 3 classes -> 3 unordered pairs
    assert sorted(n for n, _, _ in out) == [
        'fry_egg_vs_pour_coffee', 'fry_egg_vs_pour_milk', 'pour_coffee_vs_pour_milk']
    for _, mask, y in out:
        assert mask.sum() == 4 and set(np.unique(y)) <= {0, 1}


def test_generic_ignores_nan_labels():
    labels = np.array(['a', 'b', 'a', 'b', np.nan], dtype=object)
    assert [n for n, _, _ in _pairwise_subsets(labels)] == ['a_vs_b']
