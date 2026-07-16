"""Action-segmentation datasets used **directly**: ground-truth action labels as symbols.

The frame-level GT of a temporal action-segmentation dataset already *is* a validated
symbolic sequence — no VideoMAE→cluster→prototype symbolization needed (unlike SDS2).
Run-length-encoding the per-frame labels gives a symbolic **segment** sequence; the
barycenter of a class's sequences is a **prototypical execution** of that activity.

One **config-driven parser** serves the standard MS-TCN layout shared by Breakfast /
50Salads / GTEA and easily extended — adding a dataset is one :data:`DATASETS` entry:

    <root>/<name>/mapping.txt          # "<id> <action_name>" per line
    <root>/<name>/groundTruth/*.txt    # one action name per frame, one file per video
    <root>/<name>/splits/*.bundle      # (optional) train/test splits

PAPER_TODO §2.1. Emits the standard ``(meta, X_symbolic, labels, G)`` + a
:func:`build_action_seg_ground_cost`, consumed by :func:`..suite.run_generalization_suite`.
The heavy I3D ``features/`` of the MS-TCN release are **not** needed and never read.
"""
import glob
import os

import numpy as np
import pandas as pd

from ..vocab import compute_distance_matrix


# --- per-dataset activity-label extraction from the GT filename stem ----------------

def _breakfast_activity(stem):
    """``'P16_cam01_P16_cereals'`` -> ``'cereals'`` (the 10 Breakfast activities)."""
    return stem.split('_')[-1]


def _gtea_activity(stem):
    """``'S1_Cheese_C1'`` -> ``'Cheese'`` (the 7 GTEA activities)."""
    parts = stem.split('_')
    return parts[1] if len(parts) > 2 else stem


def _const_activity(name):
    return lambda stem: name


# --- full filename-stem parsers: subject / camera view / trial / activity ------------
# The activity_fn above is all load_action_seg needs; these expose the *rest* of what the
# stem encodes, which is what identifies a distinct physical **execution** (see
# :func:`dedup_by_execution`) and what supplies alternative labellings (:func:`trial_labels`).

def _breakfast_parse(stem):
    """``'P16_cam01_P16_cereals'`` -> subject ``P16``, view ``cam01``, activity ``cereals``.

    All 1712 stems have exactly four ``_``-separated fields, the third a redundant repeat
    of the subject. 52 subjects (P03-P54) x 10 activities x up to 5 simultaneous views.
    """
    parts = stem.split('_')
    return {'subject': parts[0], 'view': parts[1] if len(parts) > 2 else None,
            'trial': None, 'activity': parts[-1]}


def _50salads_parse(stem):
    """``'rgb-01-1'`` -> subject ``01``, trial ``1``: 25 subjects x 2 attempts, one recipe."""
    parts = stem.split('-')
    return {'subject': parts[1] if len(parts) > 1 else stem, 'view': None,
            'trial': parts[2] if len(parts) > 2 else None, 'activity': 'salad'}


def _gtea_parse(stem):
    """``'S1_Cheese_C1'`` -> subject ``S1``, trial ``C1``, activity ``Cheese``."""
    parts = stem.split('_')
    return {'subject': parts[0], 'view': None,
            'trial': parts[2] if len(parts) > 2 else None,
            'activity': _gtea_activity(stem)}


DATASETS = {
    # background labels are remapped to symbol 0 (the harness's reserved background).
    'breakfast': dict(background=('SIL',), activity_fn=_breakfast_activity,
                      parse_fn=_breakfast_parse, n_videos=1712, n_classes=10),
    '50salads':  dict(background=('action_start', 'action_end'),
                      activity_fn=_const_activity('salad'), parse_fn=_50salads_parse,
                      n_videos=50, n_classes=1),
    'gtea':      dict(background=('background',), activity_fn=_gtea_activity,
                      parse_fn=_gtea_parse, n_videos=28, n_classes=7),
}

# Breakfast records each execution from several cameras at once; when more than one view
# of the same execution survives, keep the highest-priority one. Order is by prevalence
# (cam01 433 files, webcam01 365, webcam02 338, stereo01 304, cam02 272) so the retained
# subset is as large and as homogeneous as possible.
VIEW_PRIORITY = ('cam01', 'webcam01', 'webcam02', 'cam02', 'stereo01')

# Where the GT lives by default (git-ignored, download-on-demand). Overridable via ``root``.
DATA_SUBDIR = ('generalization',)


def default_root(name):
    from smartflat.utils.utils_io import get_data_root
    return os.path.join(get_data_root(), *DATA_SUBDIR, name)


# The canonical MS-TCN release: one 30 GB Zenodo zip bundling ``data/<name>/{features,
# groundTruth,mapping.txt,splits}`` for breakfast/50salads/gtea. Only the tiny GT text
# (a few MB) is pulled via HTTP Range — the heavy I3D ``features/`` are never downloaded.
ZENODO_DATA_URL = 'https://zenodo.org/api/records/3625992/files/data.zip/content'


def download_action_seg(name, root=None, force=False, pace=0.5, max_retries=8):
    """Fetch just ``mapping.txt`` + ``groundTruth/*.txt`` for ``name`` via HTTP Range.

    Uses :mod:`remotezip` to selectively pull the GT members from :data:`ZENODO_DATA_URL`
    (Zenodo honours ``Accept-Ranges: bytes``) — never the 30 GB features. Extracts with the
    ``data/<name>/`` prefix stripped so files land at ``<root>/mapping.txt`` and
    ``<root>/groundTruth/*.txt``, exactly where :func:`load_action_seg` reads them.

    One Range GET per file, so a large dataset (Breakfast ≈ 1712 files) is many requests:
    ``pace`` seconds between them keeps under Zenodo's rate limit, and a 429 triggers
    exponential backoff (up to ``max_retries``). **Resumable** — already-downloaded files are
    skipped, so a re-run after a throttle continues where it left off. Idempotent: a
    ``<root>/.download_complete`` flag short-circuits once complete unless ``force=True``.
    Returns the resolved ``root``. Requires ``pip install remotezip``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown action-seg dataset {name!r}; known: {sorted(DATASETS)}")
    root = root or default_root(name)
    flag = os.path.join(root, '.download_complete')
    if os.path.isfile(flag) and not force:
        return root
    import time
    from remotezip import RemoteZip  # lazy: keeps the loader importable without remotezip
    prefix = f'data/{name}/'
    gt_prefix = prefix + 'groundTruth/'
    os.makedirs(os.path.join(root, 'groundTruth'), exist_ok=True)
    with RemoteZip(ZENODO_DATA_URL) as rz:
        members = [m for m in rz.namelist()
                   if m == prefix + 'mapping.txt'
                   or (m.startswith(gt_prefix) and m.endswith('.txt'))]
        if not members:
            raise RuntimeError(
                f"no GT members matching {prefix!r} in {ZENODO_DATA_URL}")
        fetched = 0
        for m in members:
            dest = os.path.join(root, m[len(prefix):])  # strip 'data/<name>/'
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue  # resumable
            for attempt in range(max_retries):
                try:
                    data = rz.read(m)
                    break
                except Exception as e:  # noqa: BLE001 — retry Zenodo 429 throttling
                    if '429' in str(e) and attempt < max_retries - 1:
                        time.sleep(min(60, 2 ** attempt))
                    else:
                        raise
            with open(dest, 'wb') as out:
                out.write(data)
            fetched += 1
            if pace:
                time.sleep(pace)
    with open(flag, 'w') as fh:
        fh.write('ok\n')
    print(f"[download_action_seg] {name}: {fetched} new files -> {root}")
    return root


def read_mapping(path, background=()):
    """Read ``mapping.txt`` (``"<id> <name>"``) → ``{name: id}``, remapped so the
    background label(s) occupy id **0** (the harness's reserved background symbol)."""
    raw = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(maxsplit=1)
            raw[name] = int(idx)
    bg = next((b for b in background if b in raw), None)
    if bg is None or raw[bg] == 0:
        return raw
    # swap ids so the background symbol lands on 0
    zero_owner = next((n for n, i in raw.items() if i == 0), None)
    raw[zero_owner], raw[bg] = raw[bg], 0
    return raw


def _rle(codes):
    """Collapse consecutive identical frame labels into a segment sequence."""
    codes = np.asarray(codes, dtype=int)
    if codes.size == 0:
        return codes
    keep = np.concatenate(([True], codes[1:] != codes[:-1]))
    return codes[keep]


def load_action_seg(name, root=None, upsample_to=None, min_len=2):
    """Load an action-segmentation dataset as symbolic segment sequences.

    Returns ``(meta, X_symbolic, labels, G)``: ``X_symbolic`` a ragged ``list`` of int
    segment sequences (or an ``(n, upsample_to)`` array if ``upsample_to`` is set),
    ``labels`` the per-video activity class, ``G`` = alphabet size (incl. background 0).
    """
    if name not in DATASETS:
        raise KeyError(f"unknown action-seg dataset {name!r}; known: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    root = root or default_root(name)
    mapping_path = os.path.join(root, 'mapping.txt')
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            f"{mapping_path} not found — download the GT for {name!r} first "
            f"(see download_action_seg / module docstring).")
    mapping = read_mapping(mapping_path, cfg['background'])
    files = sorted(glob.glob(os.path.join(root, 'groundTruth', '*.txt')))
    if not files:
        raise FileNotFoundError(f"no groundTruth/*.txt under {root}")

    X, acts, vids = [], [], []
    for f in files:
        with open(f) as fh:
            frames = [ln.strip() for ln in fh if ln.strip()]
        codes = np.array([mapping[a] for a in frames], dtype=int)
        seq = _rle(codes)
        if len(seq) < min_len:
            continue
        X.append(seq)
        acts.append(cfg['activity_fn'](os.path.splitext(os.path.basename(f))[0]))
        vids.append(os.path.basename(f))

    G = len(mapping)
    labels = np.array(acts, dtype=object)
    meta = pd.DataFrame({'video': vids, 'activity': acts,
                         'n_segments': [len(x) for x in X]})
    if upsample_to is not None:
        from smartflat.utils.utils import upsample_sequence
        X = np.vstack([upsample_sequence(s, upsample_to) for s in X]).astype(int)
    return meta, X, labels, G


def build_action_seg_ground_cost(name, X=None, kind='cooccurrence',
                                 method='max_rows_cols_pre', offset_value=0.3):
    """``(G, G)`` rTWE ground cost for an action-seg alphabet.

    ``kind='cooccurrence'`` (default): a **data-driven** raw dissimilarity ``1 −
    normalized adjacency co-occurrence`` (symbols that frequently abut are closer — the
    action-seg analogue of SDS2's temporal ``D_G``), pushed through the shared
    :func:`vocab.compute_distance_matrix`. ``kind='uniform'``: the unit discrete metric
    (all symbols equidistant), for datasets with no natural symbol geometry.
    """
    if X is None:
        _, X, _, G = load_action_seg(name)
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


# --- post-hoc views over a loaded dataset --------------------------------------------
# Deliberately *not* folded into load_action_seg: notebooks 06m/06m2/06m3 are committed
# against its current output, so these take the already-loaded (meta, X, labels) instead
# of adding a behavioural flag to a function whose defaults must not move.

def parse_video_id(name, stem):
    """``(dataset, filename stem)`` -> ``{'subject', 'view', 'trial', 'activity'}``.

    ``stem`` is ``meta['video']`` **without** its ``.txt`` (``load_action_seg`` stores the
    basename *with* the extension). Fields the dataset does not encode are ``None``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown action-seg dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]['parse_fn'](stem)


def dedup_by_execution(meta, X, labels, name, *, view_priority=VIEW_PRIORITY):
    """Keep one sequence per distinct **execution** — i.e. per ``(subject, trial, activity)``.

    Breakfast films each execution with up to five cameras simultaneously, and the frame-
    level GT of those views is the *same* annotation: **98.8%** of cross-view sequences are
    byte-identical after RLE (symbol-set Jaccard median 1.000). So its 1712 "videos" are
    only **503** independent executions, ~3.4 near-duplicates each. The harness's
    :class:`RepeatedStratifiedKFold` is not group-aware, so leaving them in puts copies of
    the same execution in train *and* test — the classifier can recognise rather than
    generalise. Any per-sequence CV on Breakfast should run on the deduplicated set.

    A no-op for datasets whose stems encode no view (50Salads, GTEA): their ``(subject,
    trial, activity)`` keys are already unique. 50Salads' two trials are *distinct*
    executions and are kept — ``trial`` is part of the key.

    Returns ``(meta, X, labels, idx)`` — all filtered to ``idx`` (sorted positions into the
    originals), ``meta`` re-indexed.
    """
    rank = {v: i for i, v in enumerate(view_priority)}
    best = {}
    for i, video in enumerate(meta['video']):
        d = parse_video_id(name, os.path.splitext(video)[0])
        key = (d['subject'], d['trial'], d['activity'])
        r = rank.get(d['view'], len(view_priority))
        if key not in best or r < best[key][0]:
            best[key] = (r, i)
    idx = np.array(sorted(i for _, i in best.values()), dtype=int)
    return (meta.iloc[idx].reset_index(drop=True), [X[i] for i in idx],
            np.asarray(labels, dtype=object)[idx], idx)


def trial_labels(meta, name='50salads'):
    """Relabel by attempt number: ``'trial1'`` / ``'trial2'``.

    50Salads' 25 subjects each prepare the **same recipe twice**, so the two classes share
    a vocabulary *by construction* — no restriction needed to control frequency. That makes
    it a real-data **negative control** rather than a test: we have strong prior reason to
    believe the two attempts are exchangeable, so the order-null firing here would indicate
    a broken probe, not a discovery. (Its measured ``hist_auc`` is 0.459 — below chance,
    the signature of the same-subject pairing pulling predictions toward the wrong label.)

    The comparison is paired (every subject appears in *both* classes), which the harness's
    non-grouped CV does not model. That is conservative, not leaky: subject identity is
    exactly uninformative about the label, so it cannot be exploited to inflate AUC.
    """
    return np.array(
        ['trial' + str(parse_video_id(name, os.path.splitext(v)[0])['trial'])
         for v in meta['video']], dtype=object)
