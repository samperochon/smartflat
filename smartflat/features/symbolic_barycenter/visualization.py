
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.linalg
import seaborn as sns
import torch
import umap.umap_ as umap
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.datasets import load_classification
from aeon.distances import (
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    twe_alignment_path,
    twe_cost_matrix,
    twe_pairwise_distance,
)
from aeon.distances._alignment_paths import (
    _add_inf_to_out_of_bounds_cost_matrix,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._eshape_dtw import eshape_dtw_cost_matrix
from aeon.distances._rtwe import (
    rtwe_alignment_path,
    rtwe_alignment_path_with_costs,
    rtwe_distance,
)
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrow, Patch, Rectangle
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel



from smartflat.configs.loader import import_config
from smartflat.features.symbolization.co_clustering import (
    append_foreground_background_distance,
    clustering_prototypes_space,
    compute_distance_matrix,
    compute_multimodal_matrices,
    get_prototypes_mapping,
    plot_clustering_metrics,
    prototypes_clustering,
)
from smartflat.features.symbolization.inference_reduction_prototypes_distances import (
    agglomerate_distance_matrix,
    plot_distances_matrix_tuples,
    reduce_similarity_matrix,
)
from smartflat.features.symbolization.main import (
    build_clusterdf,
    define_symbolization,
    filter_prototypes_per_prevalence,
)
from smartflat.features.symbolization.temporal_distributions_estimation import (
    compute_gw_costs_and_transport_maps,
    estimate_kde_model,
    fit_kde_per_cluster,
    infer_kde_support,
    plot_clusters_kde_from_model,
    plot_gw_temporal_distance,
    plot_kde_from_model,
    prepare_row_timestamps_dataset,
    sort_kde_by_mean_timestamp,
    visualize_paiwise_gw,
)
from smartflat.features.symbolization.utils import (
    compute_centroid_distances,
    compute_cluster_probabilities,
    compute_sample_entropy,
    compute_threshold_per_cluster_index,
    get_prototypes,
    propagate_segment_labels_to_embeddings,
    qc_sanity_check_filters,
    reduce_segments_labels,
    reduce_similarity_matrix_per_cluster_type,
    retrieve_segmentation_costs,
    update_segmentation_from_embedding_labels,
)
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.features.symbolization.visualization import (
    plot_agreement_arrays,
    plot_binary_arrays,
    plot_centroid_cluster_distances_distributions,
    plot_centroid_cluster_distances_distributions_sbj,
    plot_cluster_prevalence,
    plot_cluster_prevalence_subplots,
    plot_distances_distribution_subplots,
    plot_frame_votes,
    plot_labels_counts_per_diagnosis,
    plot_labels_distributions_subplots,
    plot_latent_space_with_labels,
    plot_latent_space_with_labels_all,
    plot_sbj_cluster_distances_distribution_subplots,
    plot_segments_costs,
    plot_symbolization_variables,
    plot_symbols_frequency_subplots,
)
from smartflat.utils.utils import (
    get_upsampled_labels,
    pad_sequence_with_zeros,
    upsample_sequence,
)
from smartflat.utils.utils_dataset import (
    add_cum_sum_col,
    check_train_data,
    compute_matrix_stats,
    normalize_data,
)
from smartflat.utils.utils_io import fetch_qualification_mapping
from smartflat.utils.utils_visualization import (
    dynamic_row_plot_func,
    get_base_colors,
    get_cmap,
    plot_chronogames,
    plot_gram,
    plot_labels_2D_encoding,
    plot_per_cat_x_cont_y_distributions,
    plot_qualification_mapping,
)


def plot_pairwise_twe_distances_by_group(D, df, covar_col):

    labels = df[covar_col].values
    unique_labels = sorted(np.unique(labels))
    
    if covar_col != 'N':

        row_idxs, col_idxs = np.tril_indices(len(labels), k=-1)

        plt.figure(figsize=(12, 4))
        for label in unique_labels:
            # Get indices of samples in this group
            group_idx = np.where(labels == label)[0]
            group_size = len(group_idx)

            # Create a mask for pairwise comparisons within this group
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


    # Get group labels and indices sorted by class
    labels = df[covar_col].values
    
    sorted_idx = np.argsort(labels)
    sorted_labels = labels[sorted_idx]
    D_sorted = D#[sorted_idx][:, sorted_idx]


    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    cax = ax.imshow(D_sorted, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Time Warp Edit Distance")

    # Add lines to mark class boundaries
    if covar_col != 'N':
        _, counts = np.unique(sorted_labels, return_counts=True)
        boundaries = np.cumsum(counts)[:-1]
        for b in boundaries:
            ax.axhline(b - 0.5, color='k', linewidth=2)
            ax.axvline(b - 0.5, color='k', linewidth=2)

    boundaries =np.arange(D_sorted.shape[0])
    for b in boundaries:
        ax.axhline(b - 0.5, color='k', linewidth=0.1)
        ax.axvline(b - 0.5, color='k', linewidth=0.1)
        
        
    ax.set_title(f"Time Warp Edit Pairwise Distance Matrix (sorted by {covar_col})\n{unique_labels[:6]}")
    ax.set_xlabel("Participants Index (Sorted)")
    ax.set_ylabel("Participants Index (Sorted)")
    plt.tight_layout()
    plt.show()

def plot_distance_violin(df, D, covar_col, figsize=(1000, 600)):
    
    tab20_colors = [plt.cm.tab20(i) for i in range(20)]
    tab20_hex = ['#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3]) for rgba in tab20_colors]

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
            mask = (np.isin(row_idxs, group1_idx) & np.isin(col_idxs, group2_idx)) | \
                   (np.isin(row_idxs, group2_idx) & np.isin(col_idxs, group1_idx))
            pair_type = 'inter'
        dists = D[row_idxs[mask], col_idxs[mask]]
        for d in dists:
            data.append({
                'distance': d,
                'group_pair': f'{label1}-{label2}',
                'pair_type': pair_type,
                'selector': f'{pair_type} | {label1}-{label2}'
            })
    df_plot = pd.DataFrame(data)

    #fig = px.violin(df_plot, y="distance", x="group_pair", color="selector",
    fig = px.violin(df_plot, x="distance", y="group_pair", color="selector",
                    box=True, points="all", hover_data=df_plot.columns,
                    color_discrete_sequence=tab20_hex)
    fig.update_layout(
        title="TWE Distance Distributions by Group Pairs",
        width=figsize[0], height=figsize[1],
        legend_title_text='Type | Group Pair'
    )
    fig.show()

def plot_chronogram_alignment(x, y, paths=None, cost_path=None, cost_matrix=None, nu=0.001, step_sequ=1, t_max=None, lmbda=1.0, window=None, precomputed_distances=None, method='etwe', title='', cmap=None, verbose=False):
    
    # Stable mismatch/substitution
    # This is a case where both x and y have moved, but they are not matching
    # This can happen when both sequences have different symbols at the current position
    # or when one sequence has a symbol that is not present in the other sequence
                
    # Prepare data
    #x = np.asarray(x)
    #y = np.asarray(y)
    if paths is None and method == 'rtwe':
        paths = rtwe_alignment_path(x, y, nu=nu, lmbda=lmbda, window=window, precomputed_distances=precomputed_distances)
    elif paths is None and method == 'twe':
        paths = twe_alignment_path(x, y, nu=nu, lmbda=lmbda, window=window)
    else:
        paths = [paths]
    n_x, n_y = len(x.ravel()), len(x.ravel())
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
    ax.set_ylim(-1.5, 1.5); #ax.axis('off')

    # Draw chronograms
    ax.imshow(x[None, :t_max], extent=[0, n_x, 0.8, 1.5], cmap=cmap, aspect="auto")
    ax.imshow(y[None, :t_max], extent=[0, n_y, -1.5, -0.8], cmap=cmap, aspect="auto")
    ax.axhline(0.8, color='black', lw=1, ls='--')
    ax.axhline(-0.8, color='black', lw=1, ls='--')

    # Draw arrows
    if cost_path is not None:
        min_lw, max_lw = 1, 10  # reasonable bounds
        min_cost, max_cost = min(cost_path), max(cost_path)
    # Normalize costs to [0, 1]
    norm = lambda c: (c - min_cost) / (max_cost - min_cost + 1e-8)


    for k in range(1, len(paths[0])):

        i_prev, j_prev = paths[0][k - 1]
        i, j = paths[0][k]
        
        _i = i * step_sequ; _j = j * step_sequ
        _ip = i_prev * step_sequ; _jp = j_prev * step_sequ
        
        #print(f'Processing path step {k}: i={i}, j={j}, i_prev={i_prev}, j_prev={j_prev}, _i={_i}, _j={_j}, _ip={_ip}, _jp={_jp}')
        
        if _i >= t_max or _j >= t_max:
            continue
        
        start = (_i + 0.5, 0.8)
        end = (_j + 0.5, -0.8)
        
        #print(f'Processing path step {k}: i={i}, j={j}, i_prev={i_prev}, j_prev={j_prev}, x={x}, y={y}')
        xi, xip = x[0, _i], x[0, _ip]
        yj, yjp = y[0, _j], y[0, _jp]
        
        
        
        #print(f'i: {i}, j: {j}, i_prev: {i_prev}, j_prev: {j_prev}, xi: {xi}, xip: {xip}, yj: {yj}, yjp: {yjp}')
        #print(f'Shapes - xi: {xi.shape}, xip: {xip.shape}, yj: {yj.shape}, yjp: {yjp.shape}')

        # Diagonal move: match or substitution (both i and j incremented)
        if i == i_prev + 1 and j == j_prev + 1:
            #print(f'Diagonal move: i={i}, j={j}, i_prev={i_prev}, j_prev={j_prev}')
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
                        raise ValueError("Unknown case for diagonal move: match or substitution (both i and j incremented)")

                                    
            else:
                raise ValueError("Unknown case for diagonal move: match or substitution (both i and j incremented)")
            
        # Vertical move: x stays, y moves              
        elif i == i_prev  and j == j_prev +1:  # Horizontal move: x stays, y moves
            #print(f'Vertical move: i={i}, j={j}, i_prev={i_prev}, j_prev={j_prev}')
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
                    
                    raise ValueError("This case should not happen: xi == yj and yj == yjp")
                    
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
                    raise ValueError("Unknown case for horizontal move: x moves, y stays")
                    
            else:
                raise ValueError("Unknown case for horizontal move: x moves, y stays")
            
        # Horizontal move: x moves, y stays
        elif i == i_prev + 1 and j == j_prev:  # Vertical move: x moves, y stays
            #print(f'Horizontal move: i={i}, j={j}, i_prev={i_prev}, j_prev={j_prev}')
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
                    
                    raise ValueError("This case should not happen: xi == yj and yj == yjp")
                    
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
                    raise ValueError("Unknown case for horizontal move: x moves, y stays")
                    
            else:
                raise ValueError("Unknown case for horizontal move: x moves, y stays")
            
                                
        else:
            print(f"Unknown case for move: i: {i}, j: {j}, i_prev: {i_prev}, j_prev: {j_prev}")
            red(f" x: {xip}->{xi}, y: {yjp}->{yj}")
            raise ValueError("Unknown case for move: neither diagonal, horizontal nor vertical")
        
        
        cost = cost_path[k]
        lw = min_lw + norm(cost) * (max_lw - min_lw)
    
        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=0.9)
        )
    # ax.spines['left'].set_visible(False)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    ax.set_title(f"Chronogram Alignment with TWE (Arrows Colored by Type)\n{title}", fontsize=14)
    plt.tight_layout()
    plt.show()

        
    # # Define the mapping of operation names to colors (same as in your logic)
    if cost_matrix is  None:
        return 
    fig, (ax_heatmap, ax_legend) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [10, 15]})

    # --- Heatmap with alignment path ---
    ax_heatmap.imshow(cost_matrix, cmap='coolwarm',  aspect='auto')

    ax_heatmap.set_title("rTWE Cost Matrix with Alignment Path")
    ax_heatmap.invert_yaxis()

    #path_x, path_y = zip(*paths[0])
    # ax_heatmap.plot(np.array(path_y) + 0.5, np.array(path_x) + 0.5, color="red", lw=1.5)
    #ax_heatmap.plot([path_y for path_x, path_y in paths[0]], [path_x for path_x, path_y in paths[0]], color='cyan', lw=1)

    ax_heatmap.set_xlabel("y ")
    ax_heatmap.set_ylabel("x ")

    # if x is not None and y is not None:
    #     ax_heatmap.set_xticks(np.arange(len(y)))
    #     ax_heatmap.set_yticks(np.arange(len(x)))

    # --- Legend panel ---
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
        title_fontsize=14
    )

    plt.tight_layout()
    plt.show()

def plot_rtwe(match_same_costs, match_previous_costs, del_x_costs, del_y_costs, total_costs, option_paths, title='RTWE Alignment Costs', cmap=None):

    
    
    op_labels = {0: "Del x", 1: "Del y", 2: "Match"}; op_labels_r = {v: k for k, v in op_labels.items()}
    op_colors = {"Del x": "lightblue", "Del y": "lightcoral", "Match": "lightgreen"}
    operations_handles = [Patch(color=c, label=l) for l, c in op_colors.items()]

    # Convert list of (i, j) alignment positions and option_paths
    path_labels = np.array([op_labels[k] for k in option_paths])
    color_map = dict(zip(op_labels.keys(), plt.cm.tab10.colors[:3]))

    # First figure: alignment costs
    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    plt.plot(match_same_costs, label='Match Same Costs', marker='o', linewidth=3, alpha=0.7)
    plt.plot(match_previous_costs, label='Match Previous Costs', marker='o', alpha=0.7)
    plt.plot(del_x_costs, label='Del X Costs', marker='o', alpha=0.7)
    plt.plot(del_y_costs, label='Del Y Costs', marker='o', alpha=0.7)
    plt.plot(total_costs, label='Total Costs', marker='o', linewidth=2, color='black', alpha=0.8)

    segments = get_segments(path_labels)
    for start, end, label in segments:
        ax.fill_between([start, end], -10, -3, color=op_colors[label], step='pre', alpha=0.8)


    plt.title('RTWE Alignment Costs', fontsize=18, fontweight='bold')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Cost / Ratio', fontsize=14)

    # Shaded segments without redundant labels
    for start, end, label in segments:
        ax.fill_between(
            np.arange(start, end),
            -10, -3,
            color=op_colors[label],
            step='pre',
            alpha=0.8
        )

    # Add clean, predefined legend
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(existing_handles + operations_handles, existing_labels + list(op_colors.keys()), bbox_to_anchor=(1.01, 1), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add chronogram


    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_signals(x, y, title='Dyad Chronograms', cmap=None, t_max=3000):

    # Plot chronogram for x
    fig, axs = plt.subplots(2, 1, figsize=(18, 5), constrained_layout=True)
    im0 = axs[0].imshow(x[:, :t_max], aspect='auto', cmap=cmap)
    axs[0].set_title("Chronogram of x", fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Time", fontsize=12)
    axs[0].set_ylabel("Label dimension", fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    cbar0 = plt.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.04, pad=0.02)
    cbar0.set_label('Label', fontsize=12)

    # Plot chronogram for y
    im1 = axs[1].imshow(y[:, :t_max], aspect='auto', cmap=cmap)
    axs[1].set_title("Chronogram of y", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Time", fontsize=12)
    axs[1].set_ylabel("Label dimension", fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    cbar1 = plt.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.04, pad=0.02)
    cbar1.set_label('Label', fontsize=12)

    plt.suptitle(f"Chronograms for Sampled Subjects\n{info}", fontsize=16, fontweight='bold')
    plt.show()

def plot_shapes_signal(new_x, new_b, cmap, step_sequ=1, title=''):
    """
    Plot the shapes of the signals new_x and new_b.
    """

    # # Collect unique indices from the loop
    indices = sorted(set(j for i in range(1, new_x.shape[1], step_sequ)
                        for j in range(1, new_b.shape[1], step_sequ)))

    # Optional: make sure indices are within bounds
    indices = [j for j in indices if j < new_x.shape[1]]


    fig, axs = plt.subplots(2, 2, figsize=(50, 10), constrained_layout=True)
    fig.suptitle(f"Descriptor outputs\n{title}", fontsize=14, fontweight='bold')

    # Plot full sequence
    axs[0, 0].imshow(new_x[:, :1500], cmap=cmap, aspect='auto')
    axs[0, 0].set_title("S₁: First 15 minutes", fontsize=12)
    axs[0, 0].set_ylabel("Transformed dims")

    axs[1, 0].imshow(new_b[:, :1500], cmap=cmap, aspect='auto')
    axs[1, 0].set_title("S₂: First 15 minutes", fontsize=12)
    axs[1, 0].set_ylabel("Transformed dims")

    # Plot sampled sequence
    axs[0, 1].imshow(new_x[:, indices], cmap=cmap, aspect='auto')
    axs[0, 1].set_title("S₁: Full Sampled Shape", fontsize=12)

    axs[1, 1].imshow(new_b[:, indices], cmap=cmap, aspect='auto')
    axs[1, 1].set_title("S₂: Full Sampled Shape", fontsize=12)

    # Optional: add colorbars
    for ax in axs.flat:
        im = ax.images[0]
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.015, pad=0.04)

    plt.show()
    
    
