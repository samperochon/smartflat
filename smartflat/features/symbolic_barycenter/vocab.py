"""G=28 semantic-category vocabulary + temporal ground-cost construction.

Faithful reproduction of the thesis-era pipeline (``archive-notebooks/
demo_rtwe_barycenter_averaging.ipynb`` / ``demo_symbolization_gold.ipynb``)
that the paper's headline G=28 result was computed on. It maps the raw K-space
prototype labels to the ~28 named cooking-action categories and builds the
category-level *temporal-occurrence* Wasserstein ground cost ``D_G`` used by the
rTWE barycenter.

The alphabet is the **27 named categories + 1 background symbol = 28**
(background = code 0). The recipe (verified end-to-end against the on-disk data):

    filtered_raw_embedding_labels (raw K idx, alphabet 647)
      --add_pyramid_labels--------> pyr_filtered_raw_embedding_labels ('G{g_opt}'/'K{g}'/'R{raw}'/'-1')
      --mapping_cluster_category--> category_labels (frame-level, 27 names + '-1')
      --update_segmentation(majority_voting_inner)--> cat_segm_embedding_labels (segment-smoothed)
      --integer encode ('-1'->0)--> int_cat_segm_embedding_labels (0..27)

``D_G`` = pairwise 1-D Wasserstein between the categories' cohort-level normalized
occurrence-time distributions (``compute_temporal_distance(temporal_distance=
'wasserstein-1')``), then ``compute_distance_matrix`` (+offset, /max, background
row/col = max).

Kept separate from :mod:`baselines.py` so the barycenter harness stays
representation-agnostic (it only consumes ``X_symbolic`` + ``D_G``).
"""

import numpy as np

from smartflat.constants_annotations_prototypes import mapping_cluster_category
from smartflat.features.symbolization.utils import (
    add_pyramid_labels,
    update_segmentation_from_embedding_labels,
)

BACKGROUND_LABEL = '-1'
BACKGROUND_CODE = 0

# Output columns of update_segmentation_from_embedding_labels, renamed with a
# 'cat_' prefix (mirrors the thesis notebook's category segmentation columns).
_CAT_SEG_COLS = [
    'cat_segm_embedding_labels', 'cat_segments_labels', 'cat_cpts', 'cat_n_cpts',
    'cat_segments_has_separations', 'cat_n_segmented', 'cat_segments_length',
    'cat_n_embed_changes', 'cat_percent_embed_changes', 'cat_sum_cpts_withdrawn',
    'cat_percent_cpts_withdrawn',
]


def pyramid_labels_to_category(pyr_seq):
    """Map a sequence of pyramid-label strings to named categories.

    Each pyramid label ('G47'/'K81'/'R12'/'-1') is looked up in
    ``mapping_cluster_category`` (deterministic ``[0]`` tie-break for the single
    ambiguous key 'K139'). Noise and any prototype with no named category map to
    ``BACKGROUND_LABEL`` ('-1').
    """
    return [
        mapping_cluster_category[i][0] if i in mapping_cluster_category else BACKGROUND_LABEL
        for i in pyr_seq
    ]


def build_category_codes(realized_labels):
    """Deterministic category -> int code: background -> 0, sorted names -> 1..K.

    Parameters
    ----------
    realized_labels : iterable of str
        Category-label strings present in the data (e.g. the union over all
        ``cat_segm_embedding_labels`` sequences).

    Returns
    -------
    (code, code_to_label) : (dict[str, int], list[str])
        ``code`` maps label -> int; ``code_to_label[i]`` is the label for code i.
        ``sorted`` of the named labels makes the encoding stable/reproducible.
    """
    named = sorted(set(realized_labels) - {BACKGROUND_LABEL})
    code = {BACKGROUND_LABEL: BACKGROUND_CODE}
    code.update({name: i + 1 for i, name in enumerate(named)})
    code_to_label = [BACKGROUND_LABEL] + named
    return code, code_to_label


def add_category_columns(df, config_name, annotator_id, round_number, code=None):
    """Build the G=28 category sequence columns on ``df`` (thesis pipeline).

    Adds ``pyr_filtered_raw_embedding_labels`` (via :func:`add_pyramid_labels`),
    ``category_labels`` (frame-level names), the segment-smoothed ``cat_*``
    columns (via :func:`update_segmentation_from_embedding_labels` with
    ``majority_voting_inner``, keeping noise), and the integer-encoded
    ``int_cat_segm_embedding_labels`` / ``int_cat_segments_labels``.

    Parameters
    ----------
    df : pandas.DataFrame
        Symbolization dataframe with ``filtered_raw_embedding_labels`` +
        ``raw_cpts`` + ``raw_embedding_labels``.
    config_name, annotator_id, round_number :
        Identify the round-8 reduction mappings / annotation CSVs used by
        :func:`add_pyramid_labels`.
    code : dict[str, int], optional
        Reuse an existing label -> int mapping (e.g. for a held-out split). If
        ``None``, it is derived from the realized labels via
        :func:`build_category_codes`.

    Returns
    -------
    (df, code, code_to_label)
    """
    df = add_pyramid_labels(df, config_name, annotator_id, round_number)
    df['category_labels'] = df['pyr_filtered_raw_embedding_labels'].apply(
        pyramid_labels_to_category
    )
    res = df.apply(
        update_segmentation_from_embedding_labels, axis=1,
        temporal_segmentation_col='raw_cpts', embedding_labels_col='category_labels',
        combine_func_name='majority_voting_inner', filter_noise_labels=False, verbose=False,
    )
    res.columns = _CAT_SEG_COLS
    for c in _CAT_SEG_COLS:
        df[c] = res[c]

    if code is None:
        realized = np.unique(np.hstack(df['cat_segm_embedding_labels'].values)).tolist()
        code, code_to_label = build_category_codes(realized)
    else:
        code_to_label = [None] * len(code)
        for label, idx in code.items():
            code_to_label[idx] = label

    df['int_cat_segm_embedding_labels'] = df['cat_segm_embedding_labels'].apply(
        lambda x: np.array([code[i] for i in x], dtype=int)
    )
    df['int_cat_segments_labels'] = df['cat_segments_labels'].apply(
        lambda x: np.array([code[i] for i in x], dtype=int)
    )
    return df, code, code_to_label


def recompute_category_temporal_D_G(
    df, config_name='SymbolicSourceInferenceGoldConfig', annotator_id='samperochon',
    round_number=8, embedding_labels_col='int_cat_segm_embedding_labels',
    segments_labels_col='int_cat_segments_labels', temporal_segmentations_col='cat_cpts',
    suffix='_cat_repro', overwrite=False,
):
    """Recompute the category-level temporal Wasserstein-1 ground cost (the thesis D_G).

    Wraps :func:`compute_temporal_distance` (``temporal_distance='wasserstein-1'``)
    on the integer-encoded category column so the resulting ``(G, G)`` matrix is
    indexed by the integer category codes (row ``c`` == code ``c``). The matrix is
    cached under ``{DATA_ROOT}/experiments/.../round_{n}/`` by the underlying
    function; pass ``overwrite=True`` to force a fresh computation.

    Returns the **raw** (untransformed) symmetric ``(G, G)`` matrix; apply
    :func:`compute_distance_matrix` to get the rTWE ground cost.
    """
    from smartflat.features.symbolization.co_clustering import compute_temporal_distance
    D = compute_temporal_distance(
        df=df, annotator_id=annotator_id, round_number=round_number, config_name=config_name,
        temporal_distance='wasserstein-1', input_space='K_space',
        embedding_labels_col=embedding_labels_col, segments_labels_col=segments_labels_col,
        temporal_segmentations_col=temporal_segmentations_col, suffix=suffix,
        overwrite_gw_distances=overwrite,
    )
    return np.asarray(D, dtype=np.float64)


def compute_distance_matrix(D, method='max_rows_cols_pre', offset_value=0.3, zero_diagonal=True):
    """Transform a raw temporal distance matrix into the rTWE ground cost (thesis recipe).

    Adds ``offset_value`` to every entry, normalizes by the max, zeros the
    diagonal, and (``method='max_rows_cols_pre'``) sets the background row/column
    (index 0) to the per-row/column max so background is maximally far from every
    action.

    Parameters
    ----------
    D : np.ndarray of shape (G, G)
        Raw symmetric temporal Wasserstein matrix.
    method : str
        ``'max_rows_cols_pre'`` (thesis) or ``'none'`` (offset + normalize only).
    offset_value : float
        Constant floor added before normalization (thesis used 0.3).
    zero_diagonal : bool
        If True, force a final zero diagonal (so ``D[0, 0] = 0`` too) — needed for
        a valid histogram-OT ground cost and harmless for rTWE. The verbatim
        thesis transform leaves ``D[0, 0]`` at the background-row max; pass
        ``zero_diagonal=False`` for that exact variant.
    """
    D = np.asarray(D, dtype=np.float64).copy()
    D = D + offset_value
    mx = D.max()
    if mx > 0:
        D = D / mx
    if method == 'max_rows_cols_pre':
        np.fill_diagonal(D, 0.0)
        D[0, :] = np.max(D, axis=1)
        D[:, 0] = np.max(D, axis=0)
    if zero_diagonal:
        np.fill_diagonal(D, 0.0)
    return np.ascontiguousarray(D)


def make_ground_cost(D_base, delta):
    """Add ``delta`` to all off-diagonal entries and zero the diagonal.

    The NB06 (G=77) delta-offset mechanism; used to sweep ``delta`` over an
    already-built base ground cost during hyperparameter search.
    """
    D = np.asarray(D_base, dtype=np.float64).copy()
    n = D.shape[0]
    D = D + delta * (1.0 - np.eye(n))
    np.fill_diagonal(D, 0.0)
    return np.ascontiguousarray(D)
