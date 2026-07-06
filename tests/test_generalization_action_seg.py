"""Tests for the action-segmentation loader (PAPER_TODO §2.1, Phase 2).

Uses a tiny in-repo fixture in the standard MS-TCN layout (mapping.txt + groundTruth/*.txt)
so the parser — run-length encoding, background→0 remap, ground-cost build — is verified
without downloading the real datasets.
"""
import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.generalization import action_segmentation as A


def _write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    # background ('background') is deliberately NOT at id 0, to exercise the remap.
    (root / 'mapping.txt').write_text("0 take\n1 background\n2 pour\n")
    gt = root / 'groundTruth'
    gt.mkdir()
    # frame-level labels; consecutive repeats collapse under RLE.
    (gt / 'S1_Cheese_C1.txt').write_text("\n".join(
        ['take', 'take', 'pour', 'pour', 'pour', 'background']) + "\n")
    (gt / 'S2_Coffee_C1.txt').write_text("\n".join(
        ['background', 'take', 'take', 'background']) + "\n")
    return root


def test_load_action_seg_rle_remap_and_labels(tmp_path):
    root = _write_fixture(tmp_path / 'gtea')
    meta, X, labels, G = A.load_action_seg('gtea', root=str(root))

    assert G == 3
    # background 'background' remapped from id 1 -> 0; take 0 -> 1; pour stays 2.
    # S1: take take pour pour pour background -> [1,1,2,2,2,0] -> RLE [1,2,0]
    # S2: background take take background      -> [0,1,1,0]     -> RLE [0,1,0]
    assert [x.tolist() for x in X] == [[1, 2, 0], [0, 1, 0]]
    assert labels.tolist() == ['Cheese', 'Coffee']
    assert meta['n_segments'].tolist() == [3, 3]
    assert list(meta['video']) == ['S1_Cheese_C1.txt', 'S2_Coffee_C1.txt']


def test_min_len_filters_short_sequences(tmp_path):
    root = _write_fixture(tmp_path / 'gtea')
    # a single-segment (all-same) video is dropped at min_len=2
    (root / 'groundTruth' / 'S3_Cheese_C1.txt').write_text("take\ntake\ntake\n")
    _, X, labels, _ = A.load_action_seg('gtea', root=str(root), min_len=2)
    assert 'S3_Cheese_C1.txt' not in ''.join(labels)     # dropped
    assert len(X) == 2


def test_ground_cost_shapes_both_kinds(tmp_path):
    root = _write_fixture(tmp_path / 'gtea')
    _, X, _, G = A.load_action_seg('gtea', root=str(root))
    for kind in ('uniform', 'cooccurrence'):
        D = A.build_action_seg_ground_cost('gtea', X=X, kind=kind)
        assert D.shape == (G, G)
        assert np.allclose(np.diag(D), 0.0)
        assert np.allclose(D, D.T)
        assert (D >= 0).all()


def test_unknown_dataset_and_missing_root(tmp_path):
    with pytest.raises(KeyError):
        A.load_action_seg('not_a_dataset', root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        A.load_action_seg('gtea', root=str(tmp_path / 'nonexistent'))
