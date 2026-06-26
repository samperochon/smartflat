"""Unit tests for the G=28 vocabulary + ground-cost helpers (vocab.py).

Covers the pure functions (deterministic category coding, the thesis distance-matrix
transform, the delta ground cost, and the pyramid->category mapping). The data-driven
functions (add_category_columns / recompute_category_temporal_D_G) require the on-disk
gold dataframe and are exercised in the 06c notebook, not here.
"""

import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.vocab import (
    BACKGROUND_CODE,
    BACKGROUND_LABEL,
    build_category_codes,
    compute_distance_matrix,
    make_ground_cost,
    pyramid_labels_to_category,
)


@pytest.fixture
def D5():
    """5x5 symmetric raw distance matrix with zero diagonal."""
    rng = np.random.RandomState(7)
    D = rng.rand(5, 5)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


class TestBuildCategoryCodes:
    def test_background_is_zero(self):
        code, _ = build_category_codes(['-1', 'use mixer', 'recipe'])
        assert code[BACKGROUND_LABEL] == BACKGROUND_CODE == 0

    def test_sorted_named_codes(self):
        code, code_to_label = build_category_codes(['recipe', 'use mixer', '-1', 'butter'])
        # named categories get codes 1..K in sorted order
        assert code == {'-1': 0, 'butter': 1, 'recipe': 2, 'use mixer': 3}
        assert code_to_label == ['-1', 'butter', 'recipe', 'use mixer']

    def test_deterministic_regardless_of_input_order(self):
        a, _ = build_category_codes(['c', 'a', 'b', '-1'])
        b, _ = build_category_codes(['-1', 'b', 'a', 'c', 'a'])
        assert a == b

    def test_code_to_label_is_inverse(self):
        code, code_to_label = build_category_codes(['x', 'y', '-1'])
        for label, idx in code.items():
            assert code_to_label[idx] == label


class TestComputeDistanceMatrix:
    def test_shape_symmetry_zero_diag(self, D5):
        DG = compute_distance_matrix(D5, offset_value=0.3)
        assert DG.shape == D5.shape
        assert np.allclose(DG, DG.T)
        assert np.allclose(np.diag(DG), 0.0)

    def test_background_row_is_max(self, D5):
        DG = compute_distance_matrix(D5, offset_value=0.3)
        # background row/col (index 0) is the per-row/col max -> equal off-diagonal
        assert np.allclose(DG[0, 1:], DG[1:, 0])
        assert DG[0, 1:].max() == pytest.approx(DG.max())

    def test_normalized_to_unit_max(self, D5):
        DG = compute_distance_matrix(D5, offset_value=0.3)
        assert DG.max() == pytest.approx(1.0)
        assert DG.min() >= 0.0

    def test_faithful_variant_keeps_background_self_distance(self, D5):
        DG = compute_distance_matrix(D5, offset_value=0.3, zero_diagonal=False)
        # thesis leaves D[0,0] at the background-row max (not zeroed)
        assert DG[0, 0] > 0.0


class TestMakeGroundCost:
    def test_offsets_off_diagonal_only(self, D5):
        base = compute_distance_matrix(D5, offset_value=0.3)
        MG = make_ground_cost(base, 0.1)
        assert np.allclose(np.diag(MG), 0.0)
        off = ~np.eye(base.shape[0], dtype=bool)
        assert np.allclose(MG[off], base[off] + 0.1)

    def test_zero_delta_is_identity_offdiag(self, D5):
        base = compute_distance_matrix(D5, offset_value=0.3)
        MG = make_ground_cost(base, 0.0)
        assert np.allclose(MG, base)


class TestPyramidLabelsToCategory:
    def test_known_mapping(self):
        # 'G47' is a G_opt-space prototype mapped to 'merge butter chocolate'
        out = pyramid_labels_to_category(['G47', '-1', 'ZZZ_unknown'])
        assert out[0] == 'merge butter chocolate'
        assert out[1] == BACKGROUND_LABEL
        assert out[2] == BACKGROUND_LABEL  # unmapped -> background

    def test_length_preserved(self):
        seq = ['G47', 'R401', '-1', 'K81']
        assert len(pyramid_labels_to_category(seq)) == len(seq)
