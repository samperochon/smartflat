"""Alignment path, cost matrix, and distance distribution visualization (Ch. 6).

Provides visualization functions for the symbolic barycenter pipeline:
- Pairwise TWE distance heatmaps and KDE distributions by group
- Interactive violin plots of intra/inter-group distances
- Chronogram alignment with colored arrows for operation types
  (match, deletion, insertion, mismatch)
- rTWE cost decomposition plots
- Signal chronogram displays
"""

from itertools import product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from aeon.distances import twe_alignment_path
from aeon.distances._rtwe import rtwe_alignment_path
from matplotlib.patches import Patch

from smartflat.features.symbolic_barycenter.main import get_segments
from smartflat.utils.utils_coding import blue, green, red
from smartflat.utils.utils_visualization import get_cmap


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
    method='etwe',
    title='',
    cmap=None,
    verbose=False,
):
    """Visualize alignment path between two symbolic chronograms.

    Draws two chronograms (x on top, y on bottom) with colored arrows
    indicating the alignment operations (match, insertion, deletion,
    mismatch) from an Edit-Shape DTW or rTWE alignment path.

    Parameters
    ----------
    x, y : np.ndarray of shape (1, n_timesteps)
        Symbolic sequences to align.
    paths : list of list of tuple, optional
        Pre-computed alignment path. If None, computed via ``method``.
    cost_path : list of float, optional
        Cost at each alignment step (used for arrow thickness).
    cost_matrix : np.ndarray, optional
        If provided, a second plot shows the cost matrix heatmap.
    nu : float
        Stiffness parameter for TWE/rTWE.
    step_sequ : int
        Step size for subsampling columns.
    t_max : int or None
        Maximum timesteps to display.
    lmbda : float
        Edit penalty for TWE/rTWE.
    window : float or None
        Warping window constraint.
    precomputed_distances : np.ndarray or None
        Precomputed pairwise symbol distances.
    method : str
        Alignment method: 'rtwe' or 'twe'.
    title : str
        Title suffix for plots.
    cmap : colormap or None
        Colormap for chronograms. If None, auto-generated from labels.
    verbose : bool
        If True, print operation details for each alignment step.
    """
    if paths is None and method == 'rtwe':
        paths = rtwe_alignment_path(
            x, y, nu=nu, lmbda=lmbda, window=window,
            precomputed_distances=precomputed_distances,
        )
    elif paths is None and method == 'twe':
        paths = twe_alignment_path(x, y, nu=nu, lmbda=lmbda, window=window)
    else:
        paths = [paths]
    n_x, n_y = len(x.ravel()), len(y.ravel())
    max_len = max(n_x, n_y)

    if t_max is None:
        t_max = max_len

    # Build color mapping
    all_labels = np.unique(np.concatenate((x.ravel(), y.ravel())).astype(int))
    print(f'Label range: [{all_labels.min()}-{all_labels.max()}] ({len(all_labels)} unique labels)')
    if cmap is None:
        cmap = get_cmap(all_labels)

    # Get tab20 colormap
    tab20 = plt.cm.get_cmap('tab20')
    arrow_colors = [tab20(i) for i in range(20)]

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1.5, 1.5)

    # Draw chronograms
    ax.imshow(x[None, :t_max], extent=[0, n_x, 0.8, 1.5], cmap=cmap, aspect="auto")
    ax.imshow(y[None, :t_max], extent=[0, n_y, -1.5, -0.8], cmap=cmap, aspect="auto")
    ax.axhline(0.8, color='black', lw=1, ls='--')
    ax.axhline(-0.8, color='black', lw=1, ls='--')

    # Arrow thickness scaling from cost
    if cost_path is not None:
        min_lw, max_lw = 1, 10
        min_cost, max_cost = min(cost_path), max(cost_path)
    norm = lambda c: (c - min_cost) / (max_cost - min_cost + 1e-8)

    for k in range(1, len(paths[0])):

        i_prev, j_prev = paths[0][k - 1]
        i, j = paths[0][k]

        _i = i * step_sequ
        _j = j * step_sequ
        _ip = i_prev * step_sequ
        _jp = j_prev * step_sequ

        if _i >= t_max or _j >= t_max:
            continue

        start = (_i + 0.5, 0.8)
        end = (_j + 0.5, -0.8)

        xi, xip = x[0, _i], x[0, _ip]
        yj, yjp = y[0, _j], y[0, _jp]

        # Diagonal move: match or substitution (both i and j incremented)
        if i == i_prev + 1 and j == j_prev + 1:
            if xip == yjp:

                if xi == yj:

                    if xi in [-1, -2]:
                        color = arrow_colors[5]
                        operation = "Stable noise match"
                        if verbose:
                            green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")
                    else:
                        color = arrow_colors[4]
                        operation = "Stable match A and B longer"
                        if verbose:
                            green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj:

                    if xi == xip and yj != yjp:
                        color = arrow_colors[3]
                        operation = "Addition for B"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif yj == yjp and xi != xip:
                        color = arrow_colors[3]
                        operation = "Addition for A"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xi != xip and yj != yjp:
                        color = arrow_colors[2]
                        operation = "Addition for A and B"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

            elif xip != yjp:

                if xi == yj:

                    if xip == xi and yjp != yj:
                        color = arrow_colors[4]
                        operation = "Match from B"
                        if verbose:
                            green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xip != xi and yjp == yj:
                        color = arrow_colors[4]
                        operation = "Match from A"
                        if verbose:
                            green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xip != xi and yjp == yj:
                        color = arrow_colors[4]
                        operation = "Match from A and B"
                        if verbose:
                            green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj:

                    if xip == xi and yjp == yj:
                        color = arrow_colors[7]
                        operation = "Stable mismatch"
                        if verbose:
                            red(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xip != xi and yjp == yj:
                        color = arrow_colors[3]
                        operation = "Addition from A"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xip == xi and yjp != yj:
                        color = arrow_colors[3]
                        operation = "Addition from B"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    elif xip != xi and yjp != yj:
                        color = arrow_colors[2]
                        operation = "Addition from A and B"
                        if verbose:
                            blue(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                    else:
                        red(f" x: {xip}->{xi}, y: {yjp}->{yj}")
                        raise ValueError(
                            "Unknown case for diagonal move: match or substitution"
                        )

            else:
                raise ValueError(
                    "Unknown case for diagonal move: match or substitution"
                )

        # Vertical move: x stays, y moves
        elif i == i_prev and j == j_prev + 1:
            if xi == yj:

                if xi in [-1, -2]:
                    color = arrow_colors[5]
                    operation = "Stable noise B longer"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")
                else:
                    color = arrow_colors[4]
                    operation = "Stable match B longer"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

            elif xi != yj:

                if xi == yj and yj == yjp:
                    raise ValueError(
                        "This case should not happen: xi == yj and yj == yjp"
                    )

                elif xi == yj and yj != yjp:
                    color = arrow_colors[4]
                    operation = "Match from B"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj and yj == yjp:
                    color = arrow_colors[7]
                    operation = "Stable mismatch B longer"
                    if verbose:
                        red(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj and yj != yjp:
                    color = arrow_colors[3]
                    operation = "Stable mismatch addition B"
                    if verbose:
                        red(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")
                else:
                    green(f" x: {xip}->{xi}, y: {yjp}->{yj}")
                    raise ValueError(
                        "Unknown case for horizontal move: x moves, y stays"
                    )

            else:
                raise ValueError(
                    "Unknown case for horizontal move: x moves, y stays"
                )

        # Horizontal move: x moves, y stays
        elif i == i_prev + 1 and j == j_prev:
            if xi == yj:

                if xi in [-1, -2]:
                    color = arrow_colors[5]
                    operation = "Stable noise A longer"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")
                else:
                    color = arrow_colors[4]
                    operation = "Stable match A longer"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

            elif xi != yj:

                if xi == yj and yj == yjp:
                    raise ValueError(
                        "This case should not happen: xi == yj and yj == yjp"
                    )

                elif xi == yj and xi != xip:
                    color = arrow_colors[4]
                    operation = "Match from A"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj and xi == xip:
                    color = arrow_colors[7]
                    operation = "Stable mismatch A longer"
                    if verbose:
                        green(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")

                elif xi != yj and xi != xip:
                    color = arrow_colors[3]
                    operation = "Stable mismatch addition A"
                    if verbose:
                        red(f"{operation}: x: {xip}->{xi}, y: {yjp}->{yj}")
                else:
                    green(f" x: {xip}->{xi}, y: {yjp}->{yj}")
                    raise ValueError(
                        "Unknown case for horizontal move: x moves, y stays"
                    )

            else:
                raise ValueError(
                    "Unknown case for horizontal move: x moves, y stays"
                )

        else:
            print(f"Unknown case for move: i: {i}, j: {j}, i_prev: {i_prev}, j_prev: {j_prev}")
            red(f" x: {xip}->{xi}, y: {yjp}->{yj}")
            raise ValueError(
                "Unknown case for move: neither diagonal, horizontal nor vertical"
            )

        cost = cost_path[k]
        lw = min_lw + norm(cost) * (max_lw - min_lw)

        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=0.9),
        )

    ax.set_title(
        f"Chronogram Alignment with TWE (Arrows Colored by Type)\n{title}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # Second plot: cost matrix heatmap with legend
    if cost_matrix is None:
        return
    fig, (ax_heatmap, ax_legend) = plt.subplots(
        1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [10, 15]},
    )

    ax_heatmap.imshow(cost_matrix, cmap='coolwarm', aspect='auto')
    ax_heatmap.set_title("rTWE Cost Matrix with Alignment Path")
    ax_heatmap.invert_yaxis()
    ax_heatmap.set_xlabel("y ")
    ax_heatmap.set_ylabel("x ")

    ax_legend.axis('off')
    legend_items = [
        ("Stable noise match", arrow_colors[5]),
        ("Match", arrow_colors[4]),
        ("Single Addition", arrow_colors[3]),
        ("Double Addition", arrow_colors[2]),
        ("Stable mismatch", arrow_colors[7]),
    ]
    handles = [mpatches.Patch(color=color, label=label) for label, color in legend_items]
    ax_legend.legend(
        handles=handles,
        loc='center',
        ncol=1,
        frameon=False,
        fontsize=12,
        title='Alignment Operation Legend',
        title_fontsize=14,
    )

    plt.tight_layout()
    plt.show()


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

    segments = get_segments(path_labels)

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

    # TODO: `info` was a notebook global — needs to be parameterized or removed
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
