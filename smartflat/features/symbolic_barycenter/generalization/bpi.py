"""BPI-Challenge-2012 event logs used as symbolic sequences (process-mining domain).

Each loan-application **trace** is a symbolic sequence of activity names — one shared
activity vocabulary across all traces, so the classes differ mainly in the *path/order*
taken rather than in *which* activities occur. That makes BPI-2012 a candidate for a
genuinely **frequency-controlled** order test, unlike the frequency-saturated coarse
action-segmentation datasets (Breakfast/GTEA/50Salads; see :mod:`.headroom`). PAPER_TODO §2.2.

Config-driven like :mod:`.action_segmentation` — adding a dataset is one :data:`DATASETS`
entry. Emits the standard ``(meta, X_symbolic, labels, G)`` + a :func:`build_bpi_ground_cost`,
consumed unchanged by :func:`..suite.run_generalization_suite`.

**Task frame (minimal leakage guard).** The outcome labels ``A_APPROVED`` / ``A_DECLINED`` /
``A_CANCELLED`` are themselves *activities inside each trace*; keeping them would make both
the frequency-headroom screen and the order-null trivially ``AUC=1.0`` by label definition.
So the label is read from the terminal outcome activity, and **only those outcome-defining
activities are removed** from the symbolic sequence. Events are filtered to the ``COMPLETE``
lifecycle transition (dropping ``W_`` ``SCHEDULE``/``START`` sub-events) and the symbol is the
bare ``concept:name`` activity — the process control-flow, not its lifecycle mechanics.

Symbol id **0 is reserved for padding/background** (the harness convention); real activities
are ``1..V`` so ``G == V+1`` and ``build_bpi_ground_cost(name, X=X)`` agrees with the loader's
``G`` (no ``len(mapping)`` vs ``max(observed)+1`` mismatch).
"""
import gzip
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..vocab import compute_distance_matrix


DATA_SUBDIR = ('generalization',)

# BPI-Challenge-2012 on 4TU.ResearchData (DOI 10.4121/uuid:3926db30-…, article 12689204).
# A single 3.3 MB gzip — a plain flag-guarded download (the host does not honour HTTP Range,
# but the file is tiny so resume machinery is unnecessary).
_BPI2012_URL = ('https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/'
                '3276db7f-8bee-4f2b-88ee-92dbffb5a893')

DATASETS = {
    'bpi2012': dict(url=_BPI2012_URL, filename='BPI_Challenge_2012.xes.gz',
                    outcome_markers=('A_APPROVED', 'A_DECLINED', 'A_CANCELLED'),
                    lifecycle_keep=('COMPLETE',), n_traces=13087, n_classes=3),
}


def default_root(name):
    from smartflat.utils.utils_io import get_data_root
    return os.path.join(get_data_root(), *DATA_SUBDIR, name)


def download_bpi(name, root=None, force=False, timeout=120, max_retries=4):
    """Fetch the ``.xes.gz`` for ``name`` to ``<root>/<filename>`` (download-on-demand).

    Flag-guarded (``<root>/.download_complete``) and idempotent: a completed download
    short-circuits unless ``force=True``; an already-present non-empty file is adopted as
    complete. One stdlib ``urllib`` GET with a small exponential-backoff retry loop (the file
    is 3.3 MB, so no paced Range fetching is needed). Returns the resolved ``root``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown BPI dataset {name!r}; known: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    root = root or default_root(name)
    os.makedirs(root, exist_ok=True)
    flag = os.path.join(root, '.download_complete')
    dest = os.path.join(root, cfg['filename'])
    if os.path.isfile(flag) and not force:
        return root
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        with open(flag, 'w') as fh:  # downloaded before, flag lost — adopt it
            fh.write('ok\n')
        return root
    import time
    import urllib.request  # lazy: keeps the loader importable / offline for the unit test
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(cfg['url'], timeout=timeout) as resp:
                data = resp.read()
            break
        except Exception:  # noqa: BLE001 — retry transient network failures
            if attempt < max_retries - 1:
                time.sleep(min(30, 2 ** attempt))
            else:
                raise
    tmp = dest + '.part'
    with open(tmp, 'wb') as out:
        out.write(data)
    os.replace(tmp, dest)  # atomic: a partial download never masquerades as complete
    with open(flag, 'w') as fh:
        fh.write('ok\n')
    print(f"[download_bpi] {name}: {len(data)} bytes -> {dest}")
    return root


def parse_xes_traces(fileobj):
    """Stream a XES log → ``list[(case_id, [(concept_name, lifecycle), ...])]`` in event order.

    Pure (no network, no path handling), memory-bounded via incremental ``iterparse`` with
    per-trace clearing, and namespace-agnostic (the ``{uri}tag`` prefix is stripped). Kept
    separate from :func:`load_bpi` so the unit test can drive it with an inline fixture.
    """
    traces = []
    context = ET.iterparse(fileobj, events=('start', 'end'))
    _, root = next(context)  # the <log> root element
    for ev, elem in context:
        if ev != 'end' or elem.tag.rsplit('}', 1)[-1] != 'trace':
            continue
        case_id, events = None, []
        for child in elem:
            ctag = child.tag.rsplit('}', 1)[-1]
            if ctag == 'event':
                cname = lifecycle = None
                for attr in child:
                    key = attr.get('key')
                    if key == 'concept:name':
                        cname = attr.get('value')
                    elif key == 'lifecycle:transition':
                        lifecycle = attr.get('value')
                events.append((cname, lifecycle))
            elif ctag == 'string' and child.get('key') == 'concept:name':
                case_id = child.get('value')
        traces.append((case_id, events))
        elem.clear()
        root.clear()  # drop already-processed traces so memory stays flat
    return traces


def _read_symbolic(name, root, min_len):
    """Parse → label → COMPLETE-filter → leakage-guard. Shared by load + mapping.

    Returns ``(case_ids, seqs_str, labels, counts)`` where ``seqs_str`` are ragged lists of
    activity-name strings (outcome markers already removed) and ``counts`` reports drops.
    """
    cfg = DATASETS[name]
    path = os.path.join(root, cfg['filename'])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — download the log for {name!r} first (download_bpi).")
    markers = set(cfg['outcome_markers'])
    keep = {lc.upper() for lc in cfg['lifecycle_keep']} if cfg['lifecycle_keep'] else None
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as fh:
        traces = parse_xes_traces(fh)

    case_ids, seqs_str, labels = [], [], []
    n_no_outcome = n_short = n_multi = 0
    for case_id, events in traces:
        present = [a for (a, _lc) in events if a in markers]
        if not present:
            n_no_outcome += 1
            continue
        if len(set(present)) > 1:
            n_multi += 1
        outcome = present[-1]  # terminal marker (mutually exclusive in practice)
        kept = [a for (a, lc) in events
                if keep is None or (lc or '').upper() in keep]
        guarded = [a for a in kept if a not in markers]  # minimal leakage guard
        if len(guarded) < min_len:
            n_short += 1
            continue
        case_ids.append(case_id)
        seqs_str.append(guarded)
        labels.append(outcome)
    counts = dict(n_traces=len(traces), n_kept=len(seqs_str),
                  n_no_outcome=n_no_outcome, n_short=n_short, n_multi=n_multi)
    return case_ids, seqs_str, labels, counts


def _build_mapping(seqs_str):
    """``{activity_name: id}`` with ids ``1..V`` — id 0 reserved for padding/background."""
    vocab = sorted({a for s in seqs_str for a in s})
    return {a: i + 1 for i, a in enumerate(vocab)}


def bpi_activity_mapping(name, root=None, min_len=2):
    """``{activity_name: symbol_id}`` (ids ``1..V``; 0 is padding). Inverts ``X`` for labels."""
    if name not in DATASETS:
        raise KeyError(f"unknown BPI dataset {name!r}; known: {sorted(DATASETS)}")
    _, seqs_str, _, _ = _read_symbolic(name, root or default_root(name), min_len)
    return _build_mapping(seqs_str)


def load_bpi(name, root=None, upsample_to=None, min_len=2):
    """Load BPI-2012 as symbolic activity sequences, one per loan-application trace.

    Returns ``(meta, X_symbolic, labels, G)``: ``X_symbolic`` a ragged ``list`` of int
    activity sequences (or an ``(n, upsample_to)`` array if ``upsample_to`` is set), ``labels``
    the terminal outcome (``A_APPROVED``/``A_DECLINED``/``A_CANCELLED``), ``G`` = alphabet size
    (incl. padding 0). Outcome-defining activities are removed from ``X`` (minimal leakage
    guard); events are filtered to the ``COMPLETE`` lifecycle transition.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown BPI dataset {name!r}; known: {sorted(DATASETS)}")
    root = root or default_root(name)
    case_ids, seqs_str, labels_list, counts = _read_symbolic(name, root, min_len)
    if not seqs_str:
        raise RuntimeError(f"no traces survived parsing/labelling for {name!r} ({counts})")
    mapping = _build_mapping(seqs_str)
    G = len(mapping) + 1  # +1 for the reserved padding symbol 0
    X = [np.array([mapping[a] for a in s], dtype=int) for s in seqs_str]
    labels = np.array(labels_list, dtype=object)
    meta = pd.DataFrame({'case_id': case_ids, 'outcome': labels_list,
                         'n_events': [len(x) for x in X]})
    if counts['n_no_outcome'] or counts['n_multi']:
        print(f"[load_bpi] {name}: {counts['n_kept']}/{counts['n_traces']} traces kept; "
              f"dropped {counts['n_no_outcome']} without a terminal outcome, "
              f"{counts['n_short']} shorter than min_len={min_len}; "
              f"{counts['n_multi']} had >1 outcome marker (label=last).")
    if upsample_to is not None:
        from smartflat.utils.utils import upsample_sequence
        X = np.vstack([upsample_sequence(s, upsample_to) for s in X]).astype(int)
    return meta, X, labels, G


def build_bpi_ground_cost(name, X=None, kind='cooccurrence',
                          method='max_rows_cols_pre', offset_value=0.3):
    """``(G, G)`` rTWE ground cost for the BPI activity alphabet.

    ``kind='cooccurrence'`` (default): a data-driven raw dissimilarity ``1 − normalized
    adjacency co-occurrence`` (activities that frequently abut in traces are closer), pushed
    through the shared :func:`vocab.compute_distance_matrix`. ``kind='uniform'``: the unit
    discrete metric. Symbol 0 is pure padding, so ``max_rows_cols_pre`` correctly pins it as
    background. Mirrors :func:`..action_segmentation.build_action_seg_ground_cost`.
    """
    if X is None:
        _, X, _, G = load_bpi(name)
    else:
        G = int(max(int(np.max(np.asarray(s))) for s in X)) + 1
    if kind == 'uniform':
        raw = np.ones((G, G)) - np.eye(G)
    elif kind == 'cooccurrence':
        C = np.zeros((G, G), dtype=np.float64)
        for s in X:
            s = np.asarray(s, dtype=int)
            for a, b in zip(s[:-1], s[1:]):
                C[a, b] += 1
                C[b, a] += 1
        C = C / C.max() if C.max() > 0 else C
        raw = 1.0 - C
        np.fill_diagonal(raw, 0.0)
    else:
        raise ValueError(f"kind must be 'cooccurrence' or 'uniform', got {kind!r}")
    return compute_distance_matrix(raw, method=method, offset_value=offset_value)
