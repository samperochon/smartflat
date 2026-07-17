"""Tests for the BPI-2012 loader (PAPER_TODO §2.2, Phase 3).

Uses a tiny inline XES fixture (namespaced, gzipped) so the parser — namespace stripping,
terminal-outcome labelling, COMPLETE-lifecycle filtering, the minimal leakage guard, and the
ground-cost build — is verified without downloading the 3.3 MB real log.
"""
import gzip
import io

import numpy as np
import pytest

from smartflat.features.symbolic_barycenter.generalization import bpi as B


# 4 traces: approved (drops a W_ SCHEDULE sub-event), declined (kept), cancelled (1 event
# after the guard -> dropped at min_len=2), and one with no terminal outcome (dropped).
_XES = """<?xml version="1.0" encoding="UTF-8"?>
<log xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case_approved"/>
    <event><string key="concept:name" value="A_SUBMITTED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
    <event><string key="concept:name" value="W_Check"/><string key="lifecycle:transition" value="SCHEDULE"/></event>
    <event><string key="concept:name" value="W_Check"/><string key="lifecycle:transition" value="COMPLETE"/></event>
    <event><string key="concept:name" value="A_APPROVED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
  </trace>
  <trace>
    <string key="concept:name" value="case_declined"/>
    <event><string key="concept:name" value="A_SUBMITTED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
    <event><string key="concept:name" value="A_PARTLYSUBMITTED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
    <event><string key="concept:name" value="A_DECLINED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
  </trace>
  <trace>
    <string key="concept:name" value="case_short"/>
    <event><string key="concept:name" value="A_SUBMITTED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
    <event><string key="concept:name" value="A_CANCELLED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
  </trace>
  <trace>
    <string key="concept:name" value="case_no_outcome"/>
    <event><string key="concept:name" value="A_SUBMITTED"/><string key="lifecycle:transition" value="COMPLETE"/></event>
  </trace>
</log>
"""


def _write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(root / 'BPI_Challenge_2012.xes.gz'), 'wb') as fh:
        fh.write(_XES.encode('utf-8'))
    return root


def test_parse_xes_traces_namespaced():
    traces = B.parse_xes_traces(io.BytesIO(_XES.encode('utf-8')))
    assert len(traces) == 4
    case_id, events = traces[0]
    assert case_id == 'case_approved'
    assert events == [('A_SUBMITTED', 'COMPLETE'), ('W_Check', 'SCHEDULE'),
                      ('W_Check', 'COMPLETE'), ('A_APPROVED', 'COMPLETE')]


def test_load_bpi_guard_lifecycle_and_labels(tmp_path):
    root = _write_fixture(tmp_path / 'bpi2012')
    meta, X, labels, G = B.load_bpi('bpi2012', root=str(root))

    # kept: case_approved, case_declined (case_short -> 1 event after guard; case_no_outcome).
    # vocab (guarded, sorted): A_PARTLYSUBMITTED=1, A_SUBMITTED=2, W_Check=3 -> G = V+1 = 4.
    assert G == 4
    # approved: [A_SUBMITTED, W_Check] (W_Check SCHEDULE dropped, A_APPROVED guarded) -> [2, 3]
    # declined: [A_SUBMITTED, A_PARTLYSUBMITTED] (A_DECLINED guarded)                 -> [2, 1]
    assert [x.tolist() for x in X] == [[2, 3], [2, 1]]
    assert labels.tolist() == ['A_APPROVED', 'A_DECLINED']
    assert list(meta['case_id']) == ['case_approved', 'case_declined']
    assert meta['n_events'].tolist() == [2, 2]


def test_leakage_guard_removes_outcome_activities(tmp_path):
    root = _write_fixture(tmp_path / 'bpi2012')
    _, X, _, _ = B.load_bpi('bpi2012', root=str(root))
    mapping = B.bpi_activity_mapping('bpi2012', root=str(root))
    # no outcome-defining activity survives in the vocabulary...
    assert not (set(mapping) & set(B.DATASETS['bpi2012']['outcome_markers']))
    # ...and id 0 is reserved for padding (every real symbol is >= 1).
    assert all((x >= 1).all() for x in X)


def test_ground_cost_shapes_both_kinds(tmp_path):
    root = _write_fixture(tmp_path / 'bpi2012')
    _, X, _, G = B.load_bpi('bpi2012', root=str(root))
    for kind in ('uniform', 'cooccurrence'):
        D = B.build_bpi_ground_cost('bpi2012', X=X, kind=kind)
        assert D.shape == (G, G)
        assert np.allclose(np.diag(D), 0.0)
        assert np.allclose(D, D.T)
        assert (D >= 0).all()


def test_unknown_dataset_and_missing_file(tmp_path):
    with pytest.raises(KeyError):
        B.load_bpi('not_a_dataset', root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        B.load_bpi('bpi2012', root=str(tmp_path / 'nonexistent'))
