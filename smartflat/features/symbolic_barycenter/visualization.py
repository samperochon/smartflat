"""Alignment path, cost matrix, and distance distribution visualization (Ch. 6).

Provides visualization functions for the symbolic barycenter pipeline:
- Pairwise TWE distance heatmaps and KDE distributions by group
- Interactive violin plots of intra/inter-group distances
- Chronogram alignment with colored arrows for operation types
  (match, deletion, insertion, mismatch)
- rTWE cost decomposition plots
- Signal chronogram displays
"""

from collections import Counter
from itertools import product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

try:
    from aeon.distances import twe_alignment_path
    HAS_AEON = True
except ImportError:
    HAS_AEON = False

from smartflat.engine.distances._eshape_dtw import eshape_dtw_alignment_path
from smartflat.engine.distances._rtwe import rtwe_alignment_path
from smartflat.utils.utils_coding import blue, green, red
from smartflat.utils.utils_visualization import get_base_colors, get_cmap


def _get_segments(labels):
    """Identify contiguous segments in a label array (local copy to avoid circular import)."""
    import numpy as _np
    labels = _np.array(labels)
    change = _np.where(labels[1:] != labels[:-1])[0] + 1
    start_idxs = _np.r_[0, change]
    end_idxs = _np.r_[change, len(labels)]
    values = labels[start_idxs]
    return list(zip(start_idxs, end_idxs, values))


def plot_pairwise_twe_distances_by_group(D, df, covar_col):
    """Plot pairwise TWE distance distributions and heatmap by group.

    Displays two plots: (1) KDE curves of intra-group pairwise distances
    for each level of ``covar_col``, and (2) a heatmap of the full
    distance matrix sorted by group.

    Parameters
    ----------
    D : np.ndarray of shape (n, n)
        Symmetric pairwise TWE distance matrix.
    df : pd.DataFrame
        Metadata with one row per sample, containing ``covar_col``.
    covar_col : str
        Column name for grouping (e.g., 'pathologie', 'group').
    """
    labels = df[covar_col].values
    unique_labels = sorted(np.unique(labels))

    if covar_col != 'N':

        row_idxs, col_idxs = np.tril_indices(len(labels), k=-1)

        plt.figure(figsize=(12, 4))
        for label in unique_labels:
            group_idx = np.where(labels == label)[0]
            group_size = len(group_idx)

            mask = np.isin(row_idxs, group_idx) & np.isin(col_idxs, group_idx)

            dists = D[row_idxs[mask], col_idxs[mask]]
            if len(dists) > 1:
                sns.kdeplot(dists, fill=True, label=f'{label} (n={group_size})', linewidth=2)

        plt.xlabel("Pairwise Time Warp Edit Distance")
        plt.ylabel("Density")
        plt.title(f"Intra-group Time Warp Edit Distance Distributions by {covar_col}")
        plt.legend(title=f"{covar_col}")
        plt.tight_layout()
        plt.show()

    # Heatmap of distance matrix
    labels = df[covar_col].values
    sorted_idx = np.argsort(labels)
    sorted_labels = labels[sorted_idx]
    D_sorted = D

    fig, ax = plt.subplots(figsize=(15, 8))
    cax = ax.imshow(D_sorted, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Time Warp Edit Distance")

    if covar_col != 'N':
        _, counts = np.unique(sorted_labels, return_counts=True)
        boundaries = np.cumsum(counts)[:-1]
        for b in boundaries:
            ax.axhline(b - 0.5, color='k', linewidth=2)
            ax.axvline(b - 0.5, color='k', linewidth=2)

    boundaries = np.arange(D_sorted.shape[0])
    for b in boundaries:
        ax.axhline(b - 0.5, color='k', linewidth=0.1)
        ax.axvline(b - 0.5, color='k', linewidth=0.1)

    ax.set_title(
        f"Time Warp Edit Pairwise Distance Matrix (sorted by {covar_col})\n"
        f"{unique_labels[:6]}"
    )
    ax.set_xlabel("Participants Index (Sorted)")
    ax.set_ylabel("Participants Index (Sorted)")
    plt.tight_layout()
    plt.show()


def plot_distance_violin(df, D, covar_col, figsize=(1000, 600)):
    """Interactive violin plot of intra- and inter-group TWE distances.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata with one row per sample, containing ``covar_col``.
    D : np.ndarray of shape (n, n)
        Symmetric pairwise TWE distance matrix.
    covar_col : str
        Column name for grouping.
    figsize : tuple of int
        (width, height) in pixels for the Plotly figure.
    """
    tab20_colors = [plt.cm.tab20(i) for i in range(20)]
    tab20_hex = [
        '#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3])
        for rgba in tab20_colors
    ]

    labels = df[covar_col].values
    row_idxs, col_idxs = np.tril_indices(len(labels), k=-1)
    unique_labels = np.unique(labels)

    data = []
    for label1, label2 in product(unique_labels, repeat=2):
        group1_idx = np.where(labels == label1)[0]
        group2_idx = np.where(labels == label2)[0]
        if label1 == label2:
            mask = np.isin(row_idxs, group1_idx) & np.isin(col_idxs, group2_idx)
            pair_type = 'intra'
        else:
            mask = (
                (np.isin(row_idxs, group1_idx) & np.isin(col_idxs, group2_idx))
                | (np.isin(row_idxs, group2_idx) & np.isin(col_idxs, group1_idx))
            )
            pair_type = 'inter'
        dists = D[row_idxs[mask], col_idxs[mask]]
        for d in dists:
            data.append({
                'distance': d,
                'group_pair': f'{label1}-{label2}',
                'pair_type': pair_type,
                'selector': f'{pair_type} | {label1}-{label2}',
            })
    df_plot = pd.DataFrame(data)

    fig = px.violin(
        df_plot, x="distance", y="group_pair", color="selector",
        box=True, points="all", hover_data=df_plot.columns,
        color_discrete_sequence=tab20_hex,
    )
    fig.update_layout(
        title="TWE Distance Distributions by Group Pairs",
        width=figsize[0], height=figsize[1],
        legend_title_text='Type | Group Pair',
    )
    fig.show()


# Coarse grouping of fine-grained alignment operations into the three paper
# classes used in `chapter_6_examples_matching` (Match / Background / Mismatch).
# "Addition" steps are kept as their own bucket so they count toward the Match%
# denominator (matching the thesis-era `format_title`), even though they share
# the Mismatch arrow colour.
_COARSE_CASE_MAP = {
    "Stable match A and B longer": "Match",
    "Stable match A longer": "Match",
    "Stable match B longer": "Match",
    "Match from A": "Match",
    "Match from B": "Match",
    "Match from A and B": "Match",
    "Stable background match": "Background",
    "Stable background A longer": "Background",
    "Stable background B longer": "Background",
    "Addition A": "Addition",
    "Addition B": "Addition",
    "Addition A and B": "Addition",
    "Stable mismatch": "Mismatch",
    "Stable mismatch A longer": "Mismatch",
    "Stable mismatch B longer": "Mismatch",
}


def coarse_case_counts(case_counter):
    """Collapse fine-grained alignment operations into coarse paper categories.

    Returns a ``Counter`` over {'Match', 'Background', 'Addition', 'Mismatch'}.
    """
    coarse = Counter()
    for fine_label, count in case_counter.items():
        coarse[_COARSE_CASE_MAP.get(fine_label, "Other")] += count
    return coarse


def match_fraction(case_counter):
    """Fraction of aligned steps that are genuine (non-background) matches."""
    coarse = coarse_case_counts(case_counter)
    total = sum(coarse.values())
    if total == 0:
        return 0.0
    return coarse.get("Match", 0) / total


def _format_match_title(case_counter):
    return f"(Match: {100 * match_fraction(case_counter):.1f}%)"


def _draw_alignment_extras(arrow_colors, cost_matrix, path):
    """Standalone-mode companion figure: cost-matrix heatmap (if given) + legend.

    Only used when ``plot_chronogram_alignment`` creates its own figure; in grid
    mode the caller draws the shared legend instead.
    """
    legend_items = [
        ("Stable background match", arrow_colors[5]),
        ("Match", arrow_colors[4]),
        ("Mismatch", arrow_colors[6]),
    ]
    handles = [mpatches.Patch(color=color, label=label) for label, color in legend_items]
    if cost_matrix is not None:
        fig, (ax_h, ax_l) = plt.subplots(
            1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [10, 15]},
        )
        ax_h.imshow(cost_matrix, cmap='coolwarm', aspect='auto')
        ax_h.set_title("Cost matrix with alignment path")
        ax_h.invert_yaxis()
        ax_h.set_xlabel("y")
        ax_h.set_ylabel("x")
        ax_l.axis('off')
        ax_l.legend(handles=handles, loc='center', frameon=False, fontsize=12,
                    title='Alignment legend', title_fontsize=14)
    else:
        fig, ax_l = plt.subplots(figsize=(5, 2))
        ax_l.axis('off')
        ax_l.legend(handles=handles, loc='center', ncol=3, frameon=False, fontsize=12,
                    title='Alignment legend', title_fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_chronogram_alignment(
    x, y,
    paths=None,
    cost_path=None,
    cost_matrix=None,
    nu=0.001,
    step_sequ=1,
    t_max=None,
    lmbda=1.0,
    window=None,
    precomputed_distances=None,
    method='twe',
    title='',
    cmap=None,
    norm=None,
    ax=None,
    do_plot=True,
    background_values=(-1, -2),
    verbose=False,
):
    """Visualize the alignment path between two symbolic chronograms.

    Draws ``x`` (top) and ``y`` (bottom) as colored chronogram strips and
    connects aligned positions with arrows colored by operation type, collapsed
    to the three thesis-era paper classes: Match (green), Stable-background match
    (light-green) and Mismatch/addition (red). Restored from the
    ``demo_rtwe_barycenter_averaging`` archive notebook, with added ``ax`` support
    so panels can be tiled into the ``chapter_6_examples_matching`` grid (see
    :func:`plot_matching_grid`).

    Parameters
    ----------
    x, y : array-like
        Symbolic sequences, 1D or shape ``(1, n)``.
    paths : list of (i, j), optional
        Pre-computed alignment path. If None, it is computed via ``method``.
    cost_path : list of float, optional
        Per-step alignment cost; when given, arrow thickness scales with it.
    cost_matrix : np.ndarray, optional
        When given in standalone mode, a second figure shows the cost matrix.
    method : {'twe', 'rtwe'}
        Distance used to compute ``paths`` when it is None.
    ax : matplotlib axis, optional
        Draw into this axis (grid mode). When None a new figure is created.
    do_plot : bool
        When False, only tally operations and return the counter (no drawing).
    background_values : tuple
        Symbol values treated as background/noise (light-green class).

    Returns
    -------
    collections.Counter
        Counts of fine-grained alignment operations. Use
        :func:`match_fraction` / :func:`coarse_case_counts` to summarize.
    """
    # Resolve the alignment path. Convention (unchanged): paths[0] is the path,
    # whether it came from a (path, distance) tuple or a user-supplied list.
    if paths is None and method == 'rtwe':
        paths = rtwe_alignment_path(
            x, y, nu=nu, lmbda=lmbda, window=window,
            precomputed_distances=precomputed_distances,
        )
    elif paths is None and method == 'twe':
        if not HAS_AEON:
            raise ImportError("aeon is required for method='twe'. Install with: pip install aeon")
        paths = twe_alignment_path(x, y, nu=nu, lmbda=lmbda, window=window)
    else:
        paths = [paths]

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n_x, n_y = len(x), len(y)
    if t_max is None:
        t_max = max(n_x, n_y)

    tab20 = plt.cm.get_cmap('tab20')
    arrow_colors = [tab20(i) for i in range(20)]

    created_fig = False
    if do_plot:
        all_labels = np.unique(np.concatenate((x, y)).astype(int))
        if cmap is None:
            cmap = get_cmap(all_labels)
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 6))
            created_fig = True
        ax.set_xlim(0, t_max)
        ax.set_ylim(-1.5, 1.5)
        ax.imshow(x[None, :t_max], extent=[0, n_x, 0.8, 1.5], cmap=cmap, norm=norm, aspect="auto")
        ax.imshow(y[None, :t_max], extent=[0, n_y, -1.5, -0.8], cmap=cmap, norm=norm, aspect="auto")
        ax.axhline(0.8, color='black', lw=1, ls='--')
        ax.axhline(-0.8, color='black', lw=1, ls='--')

    # Arrow thickness: scale by cost when available, else a constant width.
    if cost_path is not None and len(cost_path):
        c_min, c_max = min(cost_path), max(cost_path)

        def _lw(k):
            c = cost_path[k] if k < len(cost_path) else cost_path[-1]
            return 1 + (c - c_min) / (c_max - c_min + 1e-8) * 9
    else:
        def _lw(k):
            return 3

    case_counter = Counter()
    path = paths[0]
    for k in range(1, len(path)):
        i_prev, j_prev = path[k - 1]
        i, j = path[k]
        _i, _ip = i * step_sequ, i_prev * step_sequ
        _j, _jp = j * step_sequ, j_prev * step_sequ
        if _i >= t_max or _j >= t_max:
            continue
        if _i >= n_x or _j >= n_y or _ip >= n_x or _jp >= n_y:
            continue
        start = (_i + 0.5, 0.8)
        end = (_j + 0.5, -0.8)
        xi, xip = x[_i], x[_ip]
        yj, yjp = y[_j], y[_jp]

        # Diagonal move: both sequences advance.
        if i == i_prev + 1 and j == j_prev + 1:
            if xip == yjp:
                if xi == yj:
                    if xi in background_values:
                        color, operation = arrow_colors[5], "Stable background match"
                    else:
                        color, operation = arrow_colors[4], "Stable match A and B longer"
                else:
                    color = arrow_colors[6]
                    if xi == xip and yj != yjp:
                        operation = "Addition B"
                    elif yj == yjp and xi != xip:
                        operation = "Addition A"
                    else:
                        operation = "Addition A and B"
            else:
                if xi == yj:
                    color = arrow_colors[4]
                    if xip == xi and yjp != yj:
                        operation = "Match from B"
                    elif xip != xi and yjp == yj:
                        operation = "Match from A"
                    else:
                        operation = "Match from A and B"
                else:
                    color = arrow_colors[6]
                    if xip == xi and yjp == yj:
                        operation = "Stable mismatch"
                    elif xip != xi and yjp == yj:
                        operation = "Addition A"
                    elif xip == xi and yjp != yj:
                        operation = "Addition B"
                    else:
                        operation = "Addition A and B"
        # Horizontal move: y advances, x stays.
        elif i == i_prev and j == j_prev + 1:
            if xi == yj:
                if xi in background_values:
                    color, operation = arrow_colors[5], "Stable background B longer"
                else:
                    color, operation = arrow_colors[4], "Stable match B longer"
            else:
                color = arrow_colors[6]
                operation = "Stable mismatch B longer" if yj == yjp else "Addition B"
        # Vertical move: x advances, y stays.
        elif i == i_prev + 1 and j == j_prev:
            if xi == yj:
                if xi in background_values:
                    color, operation = arrow_colors[5], "Stable background A longer"
                else:
                    color, operation = arrow_colors[4], "Stable match A longer"
            else:
                color = arrow_colors[6]
                operation = "Stable mismatch A longer" if xi == xip else "Addition A"
        else:
            raise ValueError("Unknown move type (not diagonal, horizontal, or vertical)")

        case_counter[operation] += 1
        if verbose:
            print(f"{operation}: x {xip}->{xi}, y {yjp}->{yj}")
        if do_plot:
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=_lw(k), alpha=0.9),
            )

    if do_plot:
        ax.set_title(
            f"Optimal alignment\n{title} {_format_match_title(case_counter)}",
            fontsize=14, weight='bold',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if created_fig:
            plt.tight_layout()
            plt.show()
            _draw_alignment_extras(arrow_colors, cost_matrix, path)

    return case_counter


def _categorical_cmap_norm(all_labels):
    """Build a categorical ``(cmap, norm, value_to_color)`` over integer symbols.

    Colors come from :func:`get_base_colors` over the sorted unique symbols (the
    same scheme as :func:`plot_chronogames`), and symbols are mapped by VALUE via a
    ``BoundaryNorm`` so colors stay identical across panels and the two strips.
    """
    vals = np.sort(np.unique(np.asarray(all_labels).astype(int)))
    colors = get_base_colors(len(vals))
    value_to_color = {int(v): colors[i] for i, v in enumerate(vals)}
    cmap = ListedColormap(colors)
    if len(vals) == 1:
        boundaries = np.array([vals[0] - 0.5, vals[0] + 0.5])
    else:
        mids = (vals[:-1] + vals[1:]) / 2.0
        boundaries = np.concatenate([[vals[0] - 0.5], mids, [vals[-1] + 0.5]])
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, value_to_color


def compute_alignment_path(method, x, y, D_G=None, nu=0.001, lmbda=1.0, window=None):
    """Return ``(path, distance)`` for one symbolic pair under the named technique.

    Parameters
    ----------
    method : {'twe', 'rtwe', 'eshape'}
        Alignment technique. ``'rtwe'`` and ``'eshape'`` use the Wasserstein ground
        cost ``D_G``; ``'twe'`` is stock aeon TWE on the raw symbol indices.
    x, y : array-like
        Symbolic sequences (1D or ``(1, n)``).
    D_G : np.ndarray, optional
        Prototype ground-cost matrix (required for ``'rtwe'``/``'eshape'``).
    """
    x2 = np.asarray(x, dtype=np.float64).reshape(1, -1)
    y2 = np.asarray(y, dtype=np.float64).reshape(1, -1)
    if method == 'twe':
        if not HAS_AEON:
            raise ImportError("aeon is required for method='twe'. Install with: pip install aeon")
        return twe_alignment_path(x2, y2, nu=nu, lmbda=lmbda, window=window)
    if method == 'rtwe':
        return rtwe_alignment_path(
            x2, y2, precomputed_distances=D_G, window=window, nu=nu, lmbda=lmbda)
    if method == 'eshape':
        return eshape_dtw_alignment_path(
            x2, y2, window=window, nu=nu, lmbda=lmbda, precomputed_distances=D_G)
    raise ValueError(f"Unknown method {method!r}; expected 'twe', 'rtwe' or 'eshape'.")


def plot_matching_grid(
    panels,
    n_cols=2,
    all_labels=None,
    code_to_label=None,
    background_values=(-1, -2),
    figsize=None,
    suptitle='',
    savepath=None,
):
    """Tile alignment panels into a grid with shared symbol + alignment legends.

    Reproduces the thesis-era ``chapter_6_examples_matching`` figure: each panel is
    one :func:`plot_chronogram_alignment` (top/bottom chronograms with Match /
    Stable-background match / Mismatch arrows and a per-panel ``Match %`` title). A
    single categorical colormap is shared across every panel so symbol colors are
    consistent; the notebook precomputes each panel's alignment path (e.g. via
    :func:`compute_alignment_path`) so this helper stays distance-agnostic.

    Parameters
    ----------
    panels : list of dict
        Each panel: ``{'x', 'y', 'path', 'label'}`` plus optional
        ``'t_max'`` / ``'background_values'``. ``'path'`` is a precomputed
        alignment path (list of ``(i, j)``); ``'label'`` is the title suffix
        (e.g. ``'rTWE distance=123.4'``) to which ``(Match: X%)`` is appended.
    n_cols : int
        Number of columns in the grid.
    all_labels : array-like, optional
        Global symbol set for the shared colormap. Defaults to the union of all
        symbols across the panels.
    code_to_label : dict, optional
        ``symbol value -> human label`` for the right-hand symbol legend.
    savepath : str, optional
        If given, the figure is saved (``dpi=150``, tight bbox).
    """
    n = len(panels)
    n_rows = int(np.ceil(n / n_cols))
    if all_labels is None:
        all_labels = np.concatenate(
            [np.asarray(p['x']).ravel() for p in panels]
            + [np.asarray(p['y']).ravel() for p in panels]
        )
    all_labels = np.sort(np.unique(np.asarray(all_labels).astype(int)))
    cmap, norm, value_to_color = _categorical_cmap_norm(all_labels)

    if figsize is None:
        figsize = (11 * n_cols, 2.4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, n_cols)
        plot_chronogram_alignment(
            panel['x'], panel['y'], paths=panel['path'], ax=axes[r][c],
            cmap=cmap, norm=norm, do_plot=True,
            background_values=panel.get('background_values', background_values),
            t_max=panel.get('t_max'), title=panel.get('label', ''),
        )
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    # Alignment legend (bottom): the three paper classes.
    tab20 = plt.cm.get_cmap('tab20')
    align_items = [
        ("Match", tab20(4)),
        ("Stable background match", tab20(5)),
        ("Mismatch", tab20(6)),
    ]
    align_handles = [mpatches.Patch(color=col, label=lab) for lab, col in align_items]
    fig.legend(handles=align_handles, loc='lower center', ncol=3, frameon=False,
               fontsize=12, title='Alignment legend', bbox_to_anchor=(0.5, -0.02))

    # Symbol legend (right), if a label map is provided.
    if code_to_label is not None:
        sym_handles = [
            mpatches.Patch(color=value_to_color[v], label=code_to_label.get(v, str(v)))
            for v in all_labels if v in value_to_color
        ]
        fig.legend(handles=sym_handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=8, title='Symbols')

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, weight='bold')
    right = 0.86 if code_to_label is not None else 1.0
    fig.tight_layout(rect=[0, 0.04, right, 0.97 if suptitle else 1.0])
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig


def plot_cohort_barycenters(
    group_sequences,
    groups=None,
    all_labels=None,
    mask_background=False,
    background_value=0,
    code_to_label=None,
    title='',
    figsize=None,
    savepath=None,
):
    """Stack per-cohort symbolic sequences as chronogram strips sharing one colormap.

    Reproduces the thesis-era cohort barycenter figures (``chapter_6_bg`` /
    ``chapter_6_non_bg``): one block per diagnosis group, every block sharing a
    single categorical symbol colormap so colors are comparable across cohorts —
    the consistency the ad-hoc ``imshow(..., cmap='tab20')`` cells lacked.

    Parameters
    ----------
    group_sequences : dict[str, np.ndarray]
        ``group -> sequences``. A 1D array is a single barycenter (one row); a 2D
        array ``(n_seqs, T)`` is stacked (one row per sequence). All sequences are
        assumed equal length (already upsampled/padded upstream).
    groups : list[str], optional
        Plot order; defaults to ``group_sequences`` insertion order.
    all_labels : array-like, optional
        Global symbol set for the shared colormap (defaults to the union over all
        groups).
    mask_background : bool
        If True, ``background_value`` cells are masked (drawn white) — the
        ``non_bg`` variant.
    code_to_label : dict, optional
        ``symbol value -> human label`` for the right-hand symbol legend.
    savepath : str, optional
        If given, save the figure (dpi=150, tight bbox).
    """
    if groups is None:
        groups = list(group_sequences.keys())
    mats = {}
    for g in groups:
        arr = np.asarray(group_sequences[g], dtype=float)
        mats[g] = arr[None, :] if arr.ndim == 1 else arr

    if all_labels is None:
        all_labels = np.concatenate([m.ravel() for m in mats.values()])
    all_labels = np.asarray(all_labels, dtype=float)
    all_labels = all_labels[~np.isnan(all_labels)]
    cmap, norm, value_to_color = _categorical_cmap_norm(all_labels)
    cmap = ListedColormap(list(cmap.colors))  # copy so set_bad does not mutate a shared cmap
    cmap.set_bad('white')

    n = len(groups)
    height_ratios = [mats[g].shape[0] for g in groups]
    if figsize is None:
        total_rows = sum(height_ratios)
        figsize = (20, max(1.2 * n, 0.04 * total_rows + 0.5 * n))
    fig, axes = plt.subplots(
        n, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
    )
    if n == 1:
        axes = [axes]
    for ax, g in zip(axes, groups):
        data = mats[g].copy()
        if mask_background:
            data[data == background_value] = np.nan
        ax.imshow(data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_yticks([])
        ax.set_ylabel(g, fontweight='bold', fontsize=11,
                      rotation=0, ha='right', va='center')
    axes[-1].set_xlabel('Time (symbols)', fontsize=10)

    if code_to_label is not None:
        present = np.sort(np.unique(all_labels.astype(int)))
        sym_handles = [
            mpatches.Patch(color=value_to_color[int(v)], label=code_to_label.get(int(v), str(int(v))))
            for v in present if int(v) in value_to_color
        ]
        fig.legend(handles=sym_handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=8, title='Symbols')
    if title:
        fig.suptitle(title, fontweight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0, 0.9 if code_to_label is not None else 1.0, 1.0])
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig


def plot_rtwe(
    match_same_costs,
    match_previous_costs,
    del_x_costs,
    del_y_costs,
    total_costs,
    option_paths,
    title='RTWE Alignment Costs',
    cmap=None,
):
    """Plot rTWE alignment cost decomposition with operation segments.

    Displays match, deletion, and total costs over alignment steps,
    with colored background segments indicating the chosen operation
    (Del x, Del y, or Match) at each step.

    Parameters
    ----------
    match_same_costs : list of float
        Match cost between current columns at each step.
    match_previous_costs : list of float
        Match cost between previous columns at each step.
    del_x_costs : list of float
        Deletion cost in x at each step.
    del_y_costs : list of float
        Deletion cost in y at each step.
    total_costs : list of float
        Total accumulated cost at each step.
    option_paths : list of int
        Chosen operation index (0=Del x, 1=Del y, 2=Match) at each step.
    title : str
        Plot title.
    cmap : colormap, optional
        Unused, kept for API consistency.
    """
    op_labels = {0: "Del x", 1: "Del y", 2: "Match"}
    op_colors = {"Del x": "lightblue", "Del y": "lightcoral", "Match": "lightgreen"}
    operations_handles = [Patch(color=c, label=l) for l, c in op_colors.items()]

    path_labels = np.array([op_labels[k] for k in option_paths])

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    plt.plot(match_same_costs, label='Match Same Costs', marker='o', linewidth=3, alpha=0.7)
    plt.plot(match_previous_costs, label='Match Previous Costs', marker='o', alpha=0.7)
    plt.plot(del_x_costs, label='Del X Costs', marker='o', alpha=0.7)
    plt.plot(del_y_costs, label='Del Y Costs', marker='o', alpha=0.7)
    plt.plot(total_costs, label='Total Costs', marker='o', linewidth=2, color='black', alpha=0.8)

    segments = _get_segments(path_labels)

    plt.title('RTWE Alignment Costs', fontsize=18, fontweight='bold')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Cost / Ratio', fontsize=14)

    # Shaded segments
    for start, end, label in segments:
        ax.fill_between(
            np.arange(start, end),
            -10, -3,
            color=op_colors[label],
            step='pre',
            alpha=0.8,
        )

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        existing_handles + operations_handles,
        existing_labels + list(op_colors.keys()),
        bbox_to_anchor=(1.01, 1), fontsize=12,
    )
    plt.grid(True, linestyle='--', alpha=0.5)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_signals(x, y, title='Dyad Chronograms', cmap=None, t_max=3000):
    """Display side-by-side chronograms for two symbolic sequences.

    Parameters
    ----------
    x, y : np.ndarray of shape (n_channels, n_timesteps)
        Symbolic sequences to display.
    title : str
        Plot title suffix.
    cmap : colormap or None
        Colormap for the chronogram heatmaps.
    t_max : int
        Maximum number of timesteps to display.
    """
    fig, axs = plt.subplots(2, 1, figsize=(18, 5), constrained_layout=True)
    im0 = axs[0].imshow(x[:, :t_max], aspect='auto', cmap=cmap)
    axs[0].set_title("Chronogram of x", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Label dimension", fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    cbar0 = plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.04, pad=0.02)
    cbar0.set_label('Label', fontsize=12)

    im1 = axs[1].imshow(y[:, :t_max], aspect='auto', cmap=cmap)
    axs[1].set_title("Chronogram of y", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("Label dimension", fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    cbar1 = plt.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.04, pad=0.02)
    cbar1.set_label('Label', fontsize=12)

    plt.suptitle("Chronograms for Sampled Subjects", fontsize=16, fontweight='bold')
    plt.show()


def plot_shapes_signal(new_x, new_b, cmap, step_sequ=1, title=''):
    """Plot the shapes of the signals new_x and new_b.

    Displays a 2x2 grid: full sequences (first 15 minutes) on the left,
    subsampled shapes on the right.

    Parameters
    ----------
    new_x, new_b : np.ndarray of shape (n_channels, n_timesteps)
        Symbolic sequences to display.
    cmap : colormap
        Colormap for the heatmaps.
    step_sequ : int
        Subsampling step for the "sampled shape" panels.
    title : str
        Plot title suffix.
    """
    indices = sorted(set(
        j for i in range(1, new_x.shape[1], step_sequ)
        for j in range(1, new_b.shape[1], step_sequ)
    ))
    indices = [j for j in indices if j < new_x.shape[1]]

    fig, axs = plt.subplots(2, 2, figsize=(50, 10), constrained_layout=True)
    fig.suptitle(f"Descriptor outputs\n{title}", fontsize=14, fontweight='bold')

    axs[0, 0].imshow(new_x[:, :1500], cmap=cmap, aspect='auto')
    axs[0, 0].set_title("S1: First 15 minutes", fontsize=12)
    axs[0, 0].set_ylabel("Transformed dims")

    axs[1, 0].imshow(new_b[:, :1500], cmap=cmap, aspect='auto')
    axs[1, 0].set_title("S2: First 15 minutes", fontsize=12)
    axs[1, 0].set_ylabel("Transformed dims")

    axs[0, 1].imshow(new_x[:, indices], cmap=cmap, aspect='auto')
    axs[0, 1].set_title("S1: Full Sampled Shape", fontsize=12)

    axs[1, 1].imshow(new_b[:, indices], cmap=cmap, aspect='auto')
    axs[1, 1].set_title("S2: Full Sampled Shape", fontsize=12)

    for ax in axs.flat:
        im = ax.images[0]
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.015, pad=0.04)

    plt.show()


def audit_scatter(x, y, xlabel='', ylabel='', title='', ax=None, color='tab:blue',
                  annotate_corr=True, savepath=None):
    """Scatter with optional Pearson/Spearman annotation and despined axes.

    Small shared-styling wrapper for the NB06d audit scatter plots
    (e.g. ``p_match`` vs symbol-frequency overlap).
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.scatter(x, y, s=14, alpha=0.5, color=color, edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight='bold')
    if annotate_corr:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 2:
            from scipy.stats import pearsonr, spearmanr
            r, _ = pearsonr(x[m], y[m])
            rho, _ = spearmanr(x[m], y[m])
            ax.annotate(
                f"Pearson r={r:.2f}\nSpearman ρ={rho:.2f}",
                xy=(0.03, 0.97), xycoords='axes fraction', va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85),
            )
    sns.despine(ax=ax)
    if created:
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig


def audit_line_vs_lambda(lam_grid, ys, labels=None, ylabel='', title='', ax=None,
                         selected_lambda=None, baseline=None, baseline_label='baseline',
                         symlog=True, linthresh=1e-3, marker='o', savepath=None):
    """Line(s) vs the edit penalty ``λ`` (symlog x), with optional selected-λ
    marker and a horizontal baseline.

    Shared-styling wrapper for the NB06d λ-sweep audits (edit fraction, AUC).
    ``ys`` may be a single series or a list of series (with matching ``labels``).
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure
    lam = np.asarray(lam_grid, dtype=float)
    ys_list = ys if (isinstance(ys, (list, tuple)) and np.ndim(ys[0]) > 0) else [ys]
    labels = labels if labels is not None else [None] * len(ys_list)
    for series, lab in zip(ys_list, labels):
        ax.plot(lam, np.asarray(series, dtype=float), marker=marker, label=lab)
    if symlog:
        ax.set_xscale('symlog', linthresh=linthresh)
    if selected_lambda is not None:
        ax.axvline(selected_lambda, color='crimson', ls='--', lw=1.2,
                   label=f'selected λ={selected_lambda:g}')
    if baseline is not None:
        ax.axhline(baseline, color='0.4', ls=':', lw=1.4, label=baseline_label)
    ax.set_xlabel('Edit penalty λ')
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight='bold')
    if any(l is not None for l in labels) or selected_lambda is not None or baseline is not None:
        ax.legend(fontsize=9)
    sns.despine(ax=ax)
    if created:
        fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches='tight')
    return fig
