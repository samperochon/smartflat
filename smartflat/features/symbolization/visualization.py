"""Visualization functions for the symbolization pipeline.

Provides chronograms, distance distributions, label distributions,
latent space plots (UMAP/PCA), dendrogram inspection, OT graphs,
and symbol frequency charts for the recursive prototyping and
symbolization results (Ch. 5).

Used primarily by notebooks:
    - ``demo_symbolization_gold.ipynb``
    - ``demo_prototypes.ipynb``
    - ``demo_recursive_procedure.ipynb``

External dependencies:
    - plotly (interactive visualizations)
    - umap (dimensionality reduction for latent space plots)
    - decord (video frame extraction)
"""

import argparse
import json
import logging
import os
import random
import socket
import sys
import time
from collections import Counter, defaultdict

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.spatial.distance import squareform
from scipy.stats import gamma, gaussian_kde, kstest, norm, poisson, skew
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


from decord import bridge
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from umap import UMAP

from smartflat.configs.loader import import_config
from smartflat.constants import progress_cols
from smartflat.datasets.filter import filter_progress_cols
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import add_pca, add_umap, use_light_dataset
from smartflat.engine.builders import compute_metrics
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.utils.utils import pairwise, upsample_sequence
from smartflat.utils.utils_coding import green
from smartflat.utils.utils_dataset import collapse_cluster_stats, normalize_data
from smartflat.utils.utils_io import (
    fetch_output_path,
    fetch_root_dir,
    get_api_root,
    get_data_root,
    get_host_name,
    get_video_loader,
    load_df,
    parse_flag,
    parse_identifier,
    save_df,
)
from smartflat.utils.utils_visualization import (
    dynamic_row_plot_func,
    get_colors,
    plot_chronogames,
    plot_gram,
    plot_labels_2D_encoding,
    plot_per_cat_x_cont_y_distributions,
)

bridge.set_bridge('native')


def plot_frame_votes(frame_votes_list):
    """
    Plots a concatenated image of frame votes arrays.

    Parameters:
    - frame_votes_list: List of frame votes arrays (each array corresponds to one video).
    """
    # Concatenate all frame votes arrays vertically
    concatenated_votes = np.vstack(frame_votes_list)

    # Plot the concatenated frame votes arrays
    plt.figure(figsize=(35, 2))
    plt.imshow(
        concatenated_votes.transpose(),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    plt.title("Frame-level number of label votes", fontsize=18, weight="bold")
    plt.ylabel("Label", weight="bold")
    plt.xlabel("Frames", weight="bold")
    plt.colorbar(label="Number of Votes")
    plt.show()

def plot_binary_arrays(agreement_arrays,  title=''):
    """
    Plots a concatenated image of binary agreement arrays.

    Parameters:
    - agreement_arrays: List of binary agreement arrays (each array corresponds to one video).
    """
    # Plot the concatenated binary arrays
    plt.figure(figsize=(35, 2))
    plt.imshow(
        agreement_arrays[None, :], aspect="auto", cmap="gray", interpolation="nearest"
    )
    plt.title(
       title,
        fontsize=18,
        weight="bold",
    )
    plt.xlabel("Time", weight="bold")
    plt.ylabel("", weight="bold")
    plt.colorbar(label="Negative (0) / Positive (1)")
    plt.show()
    
def plot_agreement_arrays(agreement_arrays):
    """
    Plots a concatenated image of binary agreement arrays.

    Parameters:
    - agreement_arrays: List of binary agreement arrays (each array corresponds to one video).
    """
    # Plot the concatenated binary arrays
    plt.figure(figsize=(35, 2))
    plt.imshow(
        agreement_arrays[None, :], aspect="auto", cmap="gray", interpolation="nearest"
    )
    plt.title(
        "Frame-level label agreement \nAgreement (White) / Disagreement (black)",
        fontsize=18,
        weight="bold",
    )
    plt.xlabel("Frames", weight="bold")
    plt.ylabel("Segments", weight="bold")
    plt.colorbar(label= "Disagreement (0) / Agreement (1) ")
    plt.show()

def x(row, savefig_folder=None):

    video_loader = get_video_loader()
    video_path = row.video_path
    vr = video_loader(video_path)

    frame_labels = row.frame_labels
    represented_clusters, counts = np.unique(frame_labels, return_counts=True)
    sorted_clusters = [x for _, x in sorted(zip(-counts, represented_clusters))]
    n_clusters = len(sorted_clusters)
        
    L, l, C = 224, 398, 3
    n_frames_to_plot = 6  # frames per cluster

    # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
    tiled_frames = np.zeros((n_clusters * L, n_frames_to_plot * l, C), dtype=np.uint8)

    # Loop through each sorted cluster and fill the array
    for cluster_idx in range(n_clusters):
        cluster_frame_indexes = np.argwhere(frame_labels == sorted_clusters[cluster_idx]).flatten()
        cluster_frame_indexes = cluster_frame_indexes[cluster_frame_indexes < len(vr)]
        sampled_frames = sorted(random.sample(list(cluster_frame_indexes), min(len(cluster_frame_indexes), n_frames_to_plot, len(vr))))

        # Place each sampled frame in the array
        for i, frame_idx in enumerate(sampled_frames):
            frame = np.array(vr[frame_idx])
            try:
                frame = cv2.resize(frame, (l, L))  # resize to the target size
            except:
                print(f'Error with frame {frame_idx} of shape {frame.shape}')
                print(frame)
                frame = frame[:l, :L]
                
            tiled_frames[cluster_idx * L:(cluster_idx + 1) * L, i * l:(i + 1) * l, :] = frame
    
    # Plot the result
    plt.figure(figsize=(30, n_clusters * 3))
    plt.title('Participant: {} - Modality: {}'.format(row.participant_id, row.modality), weight='bold', fontsize=20)
    plt.imshow(tiled_frames);plt.axis('off')

    # Add cluster index and prevalence annotations
    for cluster_idx in range(n_clusters):
        cluster_frame_indexes = np.argwhere(frame_labels == sorted_clusters[cluster_idx]).flatten()
        prevalence = len(cluster_frame_indexes) / len(frame_labels)  # Calculate prevalence
        plt.text(-50, cluster_idx * L + L // 2, f'Cluster {sorted_clusters[cluster_idx]}\nProportion: {prevalence:.2%}', 
                 fontsize=18, ha='right', va='center', color='black')
    if savefig_folder is not None:
        figure_path = os.path.join(savefig_folder, row.task_name + '_' + row.participant_id + '_' + row.modality + "_clusters.svg")
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path, bbox_inches='tight')
        print(f'Saved figure to {figure_path}')
    plt.show()
    
    return tiled_frames

def plot_latent_space_with_labels(df, embed_col='umap_embed', label_col='embedding_labels'):
    """Plot UMAP embeddings with color-coding based on `embed_labels`."""
    
    # Get unique labels and define colormap
    #Y = np.vstack(df[label_col].apply(np.array).to_list())
    #unique_labels = np.unique(Y); 
    unique_labels = np.unique(np.concatenate(df[label_col].tolist()))
    n_labels = len(unique_labels)
    
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    cmap = ListedColormap(colors[:n_labels]) 
    norm = plt.Normalize(vmin=0, vmax=n_labels - 1)
    #colors = cmap(norm(Y))
    
    #colors = [cmap(i / (n_labels - 1)) for i in range(n_labels)]  # Map labels to distinct colors


    if df.identifier.nunique() == 1:
        fig, axes = plt.subplots(1, 1, figsize=(4, 4)); axes=[axes]
        fig.suptitle(f'{df.participant_id.iloc[0]} - {df.modality.iloc[0]}\n{embed_col}', fontsize=20)
        
    else:
        fig, axes = plt.subplots(4, 4, figsize=(20, 12))
        axes = axes.flatten()

    # Iterate over each row in the dataframe to plot each participant's embeddings
    for ax, (idx, row) in zip(axes, df.sample(25).iterrows()):
        
        latent_coords = row[embed_col]  # N_i x 2 matrix
        labels = row[label_col]         # N_i labels
        colors = cmap(norm(labels))
        # Generate consistent colors based on labels
        #color_map = {label: colors[i] for i, label in enumerate(sorted(set(labels)))}
        #mapped_colors = [color_map[label] for label in labels]

        # Plot the UMAP coordinates for the participant in the respective axis
        #print(mapped_colors.shape, latent_coords.shape, labels.shape)
        scatter = ax.scatter(latent_coords[:, 0], latent_coords[:, 1], s=80, c=colors, alpha=.3, edgecolors='k', linewidth=0.1)
        
        # Create a legend based on the unique labels
        # Create legend handles manually by mapping labels to their corresponding colors

        # Add legend
        unique_labels = sorted(set(labels))
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label), markerfacecolor=colors[label], markersize=10, linestyle='') for label in unique_labels]
        ax.legend(handles=handles, title=label_col)
        if df.identifier.nunique() != 1:

            ax.set_title(f'{row.participant_id} - {row.modality}')
    
        ax.axis('off'); ax.set_xlabel('x1'); ax.set_ylabel('x2')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    

    return 

def plot_latent_space_with_labels_all(df, embed_col='umap_embed', label_col='embedding_labels'):
    """Plot UMAP embeddings with color-coding based on `embed_labels`."""
    
    # Get unique labels and define colormap
    if label_col == 'participant_id':
        all_labels = np.concatenate(df[label_col].tolist())
        
        
    all_labels = np.concatenate(df[label_col].tolist())

    unique_labels = np.unique(all_labels)
    n_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', n_labels)
    colors = [cmap(i / (n_labels - 1)) for i in range(n_labels)]  # Map labels to distinct colors

    norm = plt.Normalize(vmin=0, vmax=n_labels - 1)


    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle(f'{df.participant_id.nunique()} particpants - {embed_col} - {label_col}', fontsize=20)


    all_embeddings = np.vstack(df[embed_col].tolist())
    all_embeddings = np.nan_to_num(all_embeddings)
    
    # Generate consistent colors based on labels
    color_map = {label: colors[i] for i, label in enumerate(sorted(set(all_labels)))}
    mapped_colors = [color_map[label] for label in all_labels]

    # Plot the UMAP coordinates for the participant in the respective axis
    #print(mapped_colors.shape, latent_coords.shape, labels.shape)
    scatter = ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], s=80, c=mapped_colors, alpha=.7, edgecolors='k', linewidth=0.1)
    
    # Create a legend based on the unique labels
    # Create legend handles manually by mapping labels to their corresponding colors

    # Add legend
    unique_labels = sorted(set(all_labels))
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label), markerfacecolor=color_map[label], markersize=10, linestyle='') for label in unique_labels]
    ax.legend(handles=handles, title=label_col)

    ax.axis('off')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    
    return 

def plot_latent_space_with_labels_3d(df, identifier=None, n_sbj = 2, embed_col='umap_embed', label_col='embedding_labels', n_neighbors=15, min_dist=0.4, plot_timing=False):

    if identifier is None:
        sdf = df.iloc[:n_sbj]
    else:
        sdf = df[df['identifier'] == identifier]
    
    for i, row in sdf.iterrows():
        row = row.to_frame().T
        if embed_col not in row.columns:
            if 'umap' in embed_col:
                row[embed_col] = add_umap(row,  n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, embed_col='X')
            elif 'pca' in embed_col:
                row[embed_col] = add_pca(row,  n_components=3, embed_col='X')
                
        # Create a new DataFrame with flattened embeddings and corresponding labels
        flattened_data = []
        for _, row in row.iterrows():
            coords = row[embed_col]  # N_i x 3 array
            labels = row[label_col]  # N_i labels
            for t, (coord, label) in enumerate(zip(coords, labels)):
                flattened_data.append({
                    'x': coord[0],
                    'y': coord[1],
                    'z': coord[2],
                    'label': str(label),
                    't': 100* t / len(coords),
                    'participant': row.participant_id,
                    'modality': row.modality
                })

        plot_df = pd.DataFrame(flattened_data)

        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
        unique_clusters = sorted(plot_df.label.unique())
        color_map = {str(label): f'rgb{tuple(int(c * 255) for c in colors[i])}' for i, label in enumerate(unique_clusters)}


        # Create a 3D scatter plot using Plotly
        fig = px.scatter_3d(
            plot_df,
            x='x',
            y='y',
            z='z',
            color='t' if plot_timing else 'label',
            color_discrete_map=color_map if not plot_timing else None,
            color_continuous_scale='Viridis' if plot_timing else None,
            symbol='modality',
            hover_data=['participant', 'modality'],
            title=f"3D UMAP Embedding - {embed_col}",
            width=1000,  
            height=1000
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.show()

    return 

def plot_segments_costs(row, p_start=0, p_end=1, n_start=None, n_end=None):
    # Convert percentages to indices

    if type(row) == pd.DataFrame:
        row = row.iloc[0]
        
    if 'segments_start' not in row.keys():
        row["segments_start"], row["segments_end"] = row["cpts"][:-1], row["cpts"][1:]

    def closest_index(series, target):
        return np.abs(series - target).argmin()

    if n_start is None:
        start_idx = closest_index(row['segments_start'], int(p_start * row.N))
        end_idx = closest_index(row['segments_end'], int(p_end * row.N))
    else:
        start_idx = None
        end_idx = None

    plt.figure(figsize=(20, 2))
    for i, (segment_start, segment_end, cost) in  enumerate(zip(row['segments_start'][start_idx:end_idx], row['segments_end'][start_idx:end_idx], row['segments_fit_cost'][start_idx:end_idx])):

        # Plot cost for 'segments_fit_cost'
        plt.hlines(
            y=cost, 
            xmin=segment_start, 
            xmax=segment_end, 
            color=plt.cm.tab10(0), label='Reduced segments fitting cost' if i == 0 else "", alpha=1, linewidth=4
        )
        plt.axvline(x=segment_end, color='r', linestyle='-', alpha=0.5)
    
        
    if 'segments_start_raw' in row.index:
        start_idx = closest_index(row['segments_start_raw'], int(p_start * row.N))
        end_idx = closest_index(row['segments_end_raw'], int(p_end * row.N))

        for i, (segment_start, segment_end, cost) in  enumerate(zip(row['segments_start_raw'][start_idx:end_idx], row['segments_end_raw'][start_idx:end_idx], row['segments_fit_cost_raw'][start_idx:end_idx])):
            # Plot cost for 'segments_fit_cost'
            plt.hlines(
                y=cost, 
                xmin=segment_start, 
                xmax=segment_end, 
                color=plt.cm.tab10(1), label='Raw segments fitting costs' if i == 0 else "", alpha=1, linewidth=4
            )
            
    if n_start is not None:
        plt.xlim(n_start, n_end)
        
    plt.xlabel('Time / Cpts')
    plt.ylabel('Cost Value')
    plt.title('Segment Costs between Start and End Points')
    plt.legend()
    plt.show()
    return 

def plot_labels_counts_per_diagnosis(df, column='embedding_labels', plot_samples=True, do_normalize=True, title=''):
    
    # Pool counts for both groups
    def pool_counts(label_counts):
        pooled_counts = defaultdict(int)
        for counts in label_counts:
            for label, count in counts.items():
                pooled_counts[label] += count
        return pooled_counts
    
    def get_colors(labels):
        return [cmap(norm(label)) for label in labels]    # Assuming your DataFrame is named df
    
    df['label_counts'] = df[column].apply(lambda x: dict(Counter(x)))

    
    # Assume df is your DataFrame with 'label_counts' and 'groupe'
    label_counts_list = df['label_counts'].tolist()

    # Split data based on 'groupe'
    df_group0 = df[df['group'] == 0]['label_counts']
    df_group1 = df[df['group'] == 1]['label_counts']

    global_counts_group0 = pool_counts(df_group0)
    global_counts_group1 = pool_counts(df_group1)

    # Extract sorted label-value pairs for both groups
    labels_0, values_0 = zip(*sorted(global_counts_group0.items()))
    labels_1, values_1 = zip(*sorted(global_counts_group1.items()))

    # Ensure both groups have the same labels
    labels = sorted(set(labels_0) | set(labels_1))
    values_0 = np.array([global_counts_group0.get(label, 0) for label in labels])
    values_1 = np.array([global_counts_group1.get(label, 0) for label in labels])

    # Normalize to plot densities
    if do_normalize:
        values_0_density = values_0 / values_0.sum()
        values_1_density = values_1 / values_1.sum()
    else:
        values_0_density = values_0
        values_1_density = values_1


    # Color Mapping Logic
    n_labels = len(labels)
    cmap = plt.cm.get_cmap('tab20', n_labels)
    norm = plt.Normalize(vmin=0, vmax=n_labels - 1)

    # Plot global density for both groups
    x = np.arange(len(labels))  # Label positions
    width = 0.4  # Bar width

    plt.figure(figsize=(12, 3))
    plt.bar(x - width / 2, values_0_density, width=width, color=get_colors(labels), alpha=0.6, label='Group 0')
    plt.bar(x + width / 2, values_1_density, width=width, color=get_colors(labels), alpha=1.0, label='Group 1')

    title = title+'\n'+column#f"Densité d'apparition des {replace_counts_col} par groupe diagnostique"
    # Formatting
    plt.title(title, weight='bold')
    plt.xlabel('Cluster label')
    plt.ylabel('Density' if do_normalize else 'Total count ($N_S$)')
    plt.xticks(x, labels)
    plt.legend()
    plt.show()
    
    if plot_samples:
        
         # Plot individual distributions (first 20 examples)
        fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, counts in enumerate(label_counts_list[:25]):
            labels, values = zip(*sorted(counts.items()))
            axes[i].bar(labels, values, color=get_colors(labels))
            axes[i].set_title(f'Example {i + 1}')
            axes[i].set_xlabel('Labels')
            axes[i].set_ylabel('Density' if do_normalize else 'Total count ($N_S$)')

        plt.tight_layout()
        plt.show()

def plot_centroid_cluster_distances_distributions(df, labels_col='embedding_labels', cluster_colors=None, subset_clusters=None, distance_col='cluster_dist', clustering_config_name=None, title=''):
    """
    Plot the distribution of cluster_dist for each cluster value in the dataframe.

    Args:
    - df (pd.DataFrame): DataFrame with 'cluster_dist' and cluster labels.
    - labels_col (str): Column name containing cluster labels.
    - cluster_colors (list): List of colors for each cluster.

    Returns:
    - None: Displays the plots.
    """
    assert len(df) > 0
    assert df.task_name.nunique() == 1
    assert df.modality.nunique() == 1
    
    task_name = df.task_name.iloc[0]
    modality = df.modality.iloc[0]
    
    if subset_clusters is None:
        cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
        n_clusters = len(cluster_values)
    else:
        cluster_values = subset_clusters
        n_clusters = len(cluster_values)
        
    # Create subplots
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 3 * n_rows), constrained_layout=True)
    fig.suptitle(f'Cluster Distances Distribution \n{title}', fontsize=20, weight='bold')
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, cluster in enumerate(cluster_values):
        ax = axes[i]
            
        
        # Extract distances for the current cluster
        cluster_distances = np.hstack(df.apply(lambda x: [d for i, d in enumerate(x[distance_col]) if x[labels_col][i] == cluster] , axis=1))

        # Compute statistics
        mean_val = np.mean(cluster_distances)
        q1_val = np.percentile(cluster_distances, 25)
        q3_val = np.percentile(cluster_distances, 75)
        
        # Compute statistics
        std_val = np.std(cluster_distances)
        threshold = mean_val + 2 * std_val  # Compute threshold
        skewness_val = skew(cluster_distances)

        #print(f"Threshold (mean + 2 * std): {threshold}")
        #print('Skewness:', skewness_val)




        # # Test for Gaussian, Poisson, and Gamma distributions
        # gaussian_fit = norm.fit(cluster_distances)
        # #poisson_fit = np.mean(cluster_distances)  # Poisson parameter is the mean
        # gamma_fit = gamma.fit(cluster_distances)

        # # Perform goodness-of-fit tests (Kolmogorov-Smirnov test)
        # ks_gaussian = kstest(cluster_distances, 'norm', args=gaussian_fit)
        # #ks_poisson = kstest(cluster_distances, 'poisson', args=(poisson_fit,))
        # ks_gamma = kstest(cluster_distances, 'gamma', args=gamma_fit)

        # # Build the test results for the title
        # test_results = [
        #     f'Gaussian Test p-value: {ks_gaussian.pvalue:.3e}',
        #    # f'Poisson Test p-value: {ks_poisson.pvalue:.3e}',
        #     f'Gamma Test p-value: {ks_gamma.pvalue:.3e}'
        # ]

        # # Determine the best fit based on the smallest p-value
        # best_fit = min((ks_gaussian, 'Gaussian', gaussian_fit),
        #             # (ks_poisson, 'Poisson', poisson_fit),
        #             (ks_gamma, 'Gamma', gamma_fit),
        #             key=lambda x: x[0].pvalue)

        # distribution_name = best_fit[1]
        # #return skewness_val, gaussian_fit, poisson_fit, gamma_fit, ks_gaussian, ks_poisson, ks_gamma, distribution_name, best_fit, distribution_name
        
        # # Width of each histogram bin
        # counts, bin_edges = np.histogram(cluster_distances, bins=70)
        # bin_width = bin_edges[1] - bin_edges[0]  

        # # Plot the parametric estimation
        # x_vals = np.linspace(min(cluster_distances), max(cluster_distances), 1000)
        # if distribution_name == 'Gaussian':
        #     pdf_vals = norm.pdf(x_vals, *gaussian_fit)
        #     ax.plot(x_vals, pdf_vals * len(cluster_distances) * bin_width, color='purple', alpha=0.5,
        #             label=f'Gaussian Fit\nMean: {gaussian_fit[0]:.2f}, Std: {gaussian_fit[1]:.2f}')
        # # elif distribution_name == 'Poisson':
        # #     pmf_vals = poisson.pmf(np.round(x_vals), poisson_fit)
        # #     ax.plot(x_vals, pmf_vals * len(cluster_distances), color='purple', alpha=0.5,
        # #             label=f'Poisson Fit\nλ: {poisson_fit:.2f}')
        # elif distribution_name == 'Gamma':
        #     pdf_vals = gamma.pdf(x_vals, *gamma_fit)
        #     ax.plot(x_vals, pdf_vals * len(cluster_distances) * bin_width, color='purple', alpha=0.5,
        #             label=f'Gamma Fit\nShape: {gamma_fit[0]:.2f}, Scale: {gamma_fit[2]:.2f}')

        # Plot the histogram
        ax.hist(cluster_distances, bins='fd', color=cluster_colors[i] if cluster_colors is not None else None,
                alpha=0.7, edgecolor='black')
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(mean_val + 1 * std_val, color='red', linestyle='--', linewidth=2, label=f'Mean+STD: {mean_val+std_val:.2f}')
        ax.axvline(mean_val + 2 * std_val, color='red', linestyle='--', linewidth=3, label=f'Mean+STD: {mean_val+2*std_val:.2f}')

        ax.axvline(q1_val, color='blue', linestyle='--', label=f'Q1: {q1_val:.2f}')
        ax.axvline(q3_val, color='green', linestyle='--', label=f'Q3: {q3_val:.2f}')

        # Title with skewness and test results
        ax.set_title(f'Cluster {cluster} - N={len(cluster_distances)}\n'
                    f'Skewness: {skewness_val:.2f}\n'
                    f'Mean: {mean_val:.2f}, Std: {std_val:.2f}\n'
                    , fontsize=12)

        # Legend
        ax.legend(fontsize=10)
        ax.set_xlim([-0.03, 1.03])

    # Remove unused subplots
    for j in range(len(cluster_values), len(axes)):
        fig.delaxes(axes[j])
    
    if clustering_config_name is not None:
        config = import_config(clustering_config_name)
        
        output_figure_path = os.path.join(get_data_root(), 'outputs', config.experiment_name, task_name, modality, f'K_{n_clusters}', 'artifacts')
        os.makedirs(output_figure_path, exist_ok=True)

        plt.savefig(os.path.join(output_figure_path, f'{distance_col}_distribution_K_{n_clusters}.svg'), bbox_inches = 'tight', dpi=80)

    plt.show()
        
def plot_centroid_cluster_distances_distributions_sbj(df, labels_col, cluster_colors, distance_col='cluster_dist', n_sbj=5):
    """
    Plot the distribution of cluster_dist for each cluster value in the dataframe.

    Args:
    - df (pd.DataFrame): DataFrame with 'cluster_dist' and cluster labels.
    - labels_col (str): Column name containing cluster labels.
    - cluster_colors (list): List of colors for each cluster.

    Returns:
    - None: Displays the plots.
    """
    
    participant_ids = df['participant_id'].unique()[:n_sbj]
     
    sdf = df[df['participant_id'].isin(participant_ids)]
     
    cluster_values = sorted(np.unique(np.hstack(sdf[labels_col].to_list())))
    n_clusters = len(cluster_values)

    # Create subplots
    n_cols = 2
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows), constrained_layout=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, cluster in enumerate(cluster_values):
        ax = axes[i]
        # Extract distances for the current cluster
        cluster_distances = np.hstack(sdf.apply(lambda x: [d for i, d in enumerate(x[distance_col]) if x['embedding_labels'][i] == cluster] , axis=1))

        # Compute statistics
        mean_val = np.mean(cluster_distances)
        q1_val = np.percentile(cluster_distances, 25)
        q3_val = np.percentile(cluster_distances, 75)

        # Assign unique colors per participant_id
       
        participant_colors = {pid: plt.cm.tab20(i % 20) for i, pid in enumerate(participant_ids)}

        # Plot the histogram with participant-specific colors
        for pid in participant_ids:
            
            
            participant_distances = sdf[sdf['participant_id'] == pid].apply(lambda x: [d for i, d in enumerate(x[distance_col]) if x['embedding_labels'][i] == cluster] , axis=1)

            ax.hist(
                participant_distances,
                bins=70,
                color=participant_colors[pid],
                alpha=0.3,
                edgecolor='black',
                label=f"Participant {pid}"
            )

        # Add legend to identify participants
        ax.legend(title="Participant ID")
                
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(q1_val, color='blue', linestyle='--', label=f'Q1: {q1_val:.2f}')
        ax.axvline(q3_val, color='green', linestyle='--', label=f'Q3: {q3_val:.2f}')

        # Title and legend
        ax.set_title(f'Cluster {cluster}', fontsize=12)

    # Remove unused subplots
    for j in range(len(cluster_values), len(axes)):
        fig.delaxes(axes[j])

    plt.show()
    
def plot_cluster_prevalence(df, segments_labels_col='segments_labels', title='', figsize=(20, 3), verbose=True):
    """
    Plot the prevalence of each cluster index across participants.
    """
    
    assert (df.task_name.nunique() == 1) & (df.modality.nunique() == 1)
    assert df.participant_id.nunique() == len(df) 
    
    long_df = pd.DataFrame(
    [
        {"cluster_index": k, "n_appearance": v, "participant_id": row["participant_id"], "group": row["group"]}
        for _, row in df.iterrows()
        for k, v in dict(Counter(row[segments_labels_col])).items()
    ]
    )

    cluster_participants = long_df.groupby('cluster_index')['participant_id'].nunique()
    total_participants = long_df['participant_id'].nunique()

    # Calculate proportions
    prevalence = (cluster_participants / total_participants).reset_index(name='prevalence').sort_values(by='prevalence', ascending=False)
    
    # Ensure the x-axis order is preserved
    prevalence['cluster_index'] = pd.Categorical(prevalence['cluster_index'], categories=prevalence['cluster_index'], ordered=True)

    unique_clusters = sorted(prevalence.cluster_index.unique())
    
    if verbose:
        cmap = get_colors(unique_clusters)
        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
        cmap = colors[:prevalence.cluster_index.nunique()]
        
        plt.figure(figsize=figsize)
        #sns.barplot(data=prevalence, x='cluster_index', y='prevalence', palette=cmap, legend=False)
        sns.barplot(data=prevalence, x='cluster_index', y='prevalence', color='lightgrey', edgecolor='black', legend=False)

        plt.title(f"Prevalence of each cluster across participants\n{title}", fontsize=14)
        plt.xlabel("Cluster Index", fontsize=12)
        plt.ylabel("Proportion of participants", fontsize=12)
        plt.xticks(rotation=45)
        plt.axhline(1, linestyle='--', color='r')
        plt.ylim([0, 1.05])
        plt.tight_layout()
        plt.show()
    return list(prevalence['cluster_index'].values)

def plot_clusters(df, config_name="ClusteringDeploymentConfig", cpts_cols='reduced_cpts_frames', labels_col='reduced_segments_labels', n_sbj = 5, n_frame_per_sbj = 5, suffix='', sampling_mode='random_within_different_segments', smallest=True, show=False, figure_folder=None, verbose=False):
    """
        Here we plot a figure for each cluster represented across the `frame_labels_col` columns (arrays of different length), 
        with `n_sbj` rows and `n_frame_per_sbj` columns representing for each subejcts `n_frame_per_sbj` randomly sampled frames across the cluster index.
        
        1) Find the total number of clusters 
        
        2) For each cluster value (between 0 and P):
            i) Sample `n_subj` random participant_id and iterate over subjects
            ii) Find `n_frame_per_sbj` indexes (without replacement) of segments of the query label
            iii) Draw an index between the `cpts` of that segment to select a frame number 
        
        
    
    """
    
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()
    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1)
    
    # 1) Find total number of clusters and get colors
    cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)
    colors = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors)
    ) * 20
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])

    
    if sampling_mode=='global_centroid_proximity':
            
        results = find_extreme_embeddings(df, labels_col=labels_col, n_closest=n_frame_per_sbj, n_frames_per_subject=n_frame_per_sbj, smallest=smallest)



    for cluster_value in cluster_values:
        
        
        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ ".png")
            if os.path.exists(figure_path):
                print(f'Figure {figure_path} already exists. Skipping')
                continue
            
        
        
        fig, ax = plt.subplots(figsize=(30, n_sbj * 3))
        plt.title('Cluster value: {}'.format(cluster_value), weight='bold', fontsize=20)
    
            
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        tiled_frames = np.zeros((n_sbj * L, n_frame_per_sbj * l, C), dtype=np.uint8)
        #print(f'Final tile of shape {tiled_frames.shape}')
        n_plotted=-1
        n_tent = 0
        while (n_plotted < n_sbj - 1) and (n_tent < 10):
                
            row = df.sample(1).iloc[0]        
                
            try:
                vr = video_loader(row.video_path)
            except:
                print(f'Error loading {row.video_path}. Continue')
                import subprocess
                subprocess.run(['open', os.path.dirname(row.video_path)])
                subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
                continue
            
        
            if sampling_mode=='random_within_different_segments':
                
                labels = row[labels_col]
                cluster_indexes = np.argwhere(labels == cluster_value).flatten()   
                
                if len(cluster_indexes) == 0:
                    #print('No cluster found. Continue')
                    n_tent+=1
                    continue

                else:
                    #print(f'Found {len(cluster_indexes)} cluster segments for {cluster_value}')         
                    n_plotted+=1                        
                        
                sampled_segments_idx = sorted(random.sample(list(cluster_indexes), min(len(cluster_indexes), n_frame_per_sbj)))
                cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in  sampled_segments_idx]
                sampled_frames = [int(start + (end-start) * np.random.sample()) for start, end in  cpts_tuples]
                sampled_frames = [min(max(0, frame_idx), len(vr) - 1) for frame_idx in sampled_frames]
                
                sampled_frames_info = [f'frame_idx={frame_idx}' for frame_idx in sampled_frames]
                
                if verbose:
                    print(f'Exploring label {cluster_value}')
                    print(f'Segments labels: {labels}')
                    print(f'Cpts start-end: {cpts_tuples}')
                    print(f'Final sampled frames: {sampled_frames}')

                # Compute cluster prevalence
                prevalence_sum = np.sum([end - start for (start, end) in cpts_tuples])
                norm = row[cpts_cols][-1]
                
                prevalence = prevalence_sum / norm
                plt.text(-50, n_plotted * L + L // 2, f'{row.participant_id}\nProportion: {prevalence:.2%}', 
                        fontsize=18, ha='right', va='center', color='black')
            
            elif sampling_mode=='centroid_proximity':
                
                labels = row[labels_col]
                cluster_indexes = np.argwhere(labels == cluster_value).flatten()   
                
                if len(cluster_indexes) == 0:
                    #print('No cluster found. Continue')
                    n_tent+=1
                    continue

                else:
                    #print(f'Found {len(cluster_indexes)} cluster segments for {cluster_value}')         
                    n_plotted+=1
                        
                
                assert 'embedding_labels' in labels_col
                
                
                labels = row[labels_col]

                # Get 'cluster_dist' values and their corresponding indices
                cluster_distances = row['cluster_dist']  # Assuming it's an array-like structure
                
                if smallest:
                    # Get indices of n smallest cluster_dist values
                    sorted_indices = np.argsort(cluster_distances)
                    filtered_cluster_indexes = np.intersect1d(cluster_indexes, sorted_indices)[:n_frame_per_sbj]
                    
                else:
                    # Get indices of n largest cluster_dist values
                    sorted_indices = np.argsort(cluster_distances)
                    filtered_cluster_indexes = np.intersect1d(cluster_indexes, sorted_indices)[-n_frame_per_sbj:]

                # Select the corresponding cluster indices
                #print(f'Cluster indexes: {cluster_indexes}')
                #print(f'Sorted indices: {sorted_indices}')
                #print(f'Filtered cluster indexes: {filtered_cluster_indexes}')
                
                # Find frames for each cluster index
                
                sampled_frames = [int( idx / row.N * len(vr)) for idx in filtered_cluster_indexes]
                sampled_frames_info = [f't_n={idx / row.N:.2f}, d={cluster_distances[idx]:.2f}' for idx in filtered_cluster_indexes]
                #print('Sampled_frames: ', sampled_frames)
            
            
            # Place each sampled frame in the array
            for i, frame_idx in enumerate(sampled_frames):
                
                try:
                    frame = np.array(vr[frame_idx].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                    
                tiled_frames[n_plotted * L:(n_plotted + 1) * L, i * l:(i + 1) * l, :] = frame
                
                # Add sampled frame number at the top-right corner
                text_x = i * l + l - 10  # Slight offset from the right
                text_y = n_plotted * L + 5  # Slight offset from the top
                ax.text(text_x, text_y, str(sampled_frames_info[i]), color='red', fontsize=10, ha='right', va='top')
                #if i == 0:
                #    ax.text(text_x-100, text_y, f'{row.participant_id}', color='black', fontsize=10, ha='right', va='top')
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    (i * l, n_plotted * L), l, L,
                    linewidth=3,
                    edgecolor=cluster_colors(cluster_value),
                    facecolor='none'
                )
                ax.add_patch(rect)
        
        
        # Plot the result
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ ".png")
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path)
            #print(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
            

    return 


def plot_participant_clusters(df, labels_col='segments_labels', cpts_cols='cpts', n_rows=10, n_max_segments=10, max_length=500, prefix='', suffix='', n_sbj=10, verbose=False, show=True, figure_folder=None):
    

    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1) and (df.n_cluster.nunique() == 1)
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()

    n_cluster = df.n_cluster.iloc[0]
    df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                            .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))



    # 1) Find total number of clusters and get colors
    cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)
    colors = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors) + 
        list(plt.cm.tab20c.colors) + 
        list(plt.cm.tab20c.colors)
    )* 20
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])
    
    print('n_max_segments:', n_max_segments, 'max_length:', max_length, 'n_cluster:', n_cluster)
    if n_max_segments >  max_length /  n_cluster:
        n_max_segments = int(min(n_max_segments, max_length/n_cluster))
    elif n_max_segments < max_length/n_cluster:
        n_max_segments = int(max(n_max_segments, max_length/n_cluster))
    print('n_max_segments:', n_max_segments)
    df['cluster_dist_cluster'] = collapse_cluster_stats(df, labels_col='embedding_labels', per_segment_feature='cluster_dist')

    for _, row in df.sample(n_sbj).iterrows():
        
        labels = row[labels_col]
        #sbj_cluster_values = sorted(np.unique(labels)); 
        sbj_cluster_values = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False).index
        sbj_prevalence = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False)[0].astype(str).to_list()
        n_clusters = len(sbj_cluster_values)
        print(f'Subjects shows {n_clusters} clusters')
        if figure_folder is not None:
            figure_path = os.path.join(figure_folder, prefix+row.identifier + suffix+ ".png")
            if os.path.exists(figure_path):
                print(f'Figure {figure_path} already exists. Skipping')
                continue
                

        try:
            vr = video_loader(row.video_path)
        except:
            print(f'Error loading {row.video_path}. Continue')
            #import subprocess
            #subprocess.run(['open', os.path.dirname(row.video_path)])
            #subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
            continue
        
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(f'Action segmentation prediction (per increasing prototypes occurence prevalence)\nId: {row.task_name} - {row.modality} - {row.participant_id} - K={row.n_cluster} (|S|={len(labels)})\n' + '-'.join(sbj_prevalence), weight='bold', y=1.02,fontsize=20)
        
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        
        n_plots =  min(max_length, len(labels), n_max_segments * n_cluster )
        tiled_frames = np.zeros([n_rows * L, (n_plots // n_rows+1) * l, C], dtype=np.uint8)
        #print(f'Initial tile of shape {tiled_frames.shape}')
        n_plotted = 0
        for _, cluster_value in enumerate(sbj_cluster_values):
            
            cluster_indexes = np.argwhere(labels == cluster_value).flatten(); n_s = len(cluster_indexes) 
            sampled_segments_idx = sorted(random.sample(list(cluster_indexes), min(len(cluster_indexes), n_max_segments)))
            cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in sampled_segments_idx]
            sampled_embeddings = [(start, end, int(start + (end-start) * np.random.sample())) for  start, end in  cpts_tuples]
            
            sampled_embeddings = sampled_embeddings[:n_max_segments]
            
            try:
                std_d_c = np.std(row['cluster_dist_cluster'][cluster_value])
                range_c = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) 
                range_c_n = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) / np.max(row['cluster_dist_cluster'][cluster_value])
                dn = ((row.cluster_dist[embedding_idx] - np.min(row.cluster_dist_cluster[cluster_value])) / range_c)
            except:
                std_d_c = np.nan
                range_c = np.nan
                range_c_n = np.nan
                dn = np.nan

            if verbose:
                print(f'Exploring label {cluster_value}')
                print(f'Segments labels: {labels}')
                print(f'Sampled segments idx: {sampled_segments_idx}')
                print(f'Cpts start-end: {cpts_tuples}')
                print(f'Final sampled frames: {sampled_embeddings}')

        
            #print(f'Final tile of shape {tiled_frames.shape}')
            for i_seg , (start, end, embedding_idx) in enumerate(sampled_embeddings):
                
                sampled_frame = int( embedding_idx / row.N * len(vr))
                if i_seg == 0:
                     #sampled_frame_info = f'C={cluster_value} (x{n_s}) ({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m)t={100*embedding_idx / row.N:.1f}% d={10*row.cluster_dist[embedding_idx]:.1f}'#\nt_n={100*embedding_idx / row.N:.1f}  '
                     sampled_frame_info = f'C={cluster_value} (x{n_s}) sig_c={100*std_d_c:.1f} 10*r_c={10*range_c:.1f}'
                else:
                    sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) - d={100*dn:.1f}'

                #sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) d={100*(row.cluster_dist[embedding_idx] / range_c):.1f}%'
                    
                    
                    
                # Place each sampled frame in the array
                try:
                    frame = np.array(vr[sampled_frame].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    print(n_plotted, 'Error loading frame', sampled_frame)
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                        
                
                #print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                tiled_frames[(n_plotted % n_rows) * L:(n_plotted % n_rows + 1) * L, (n_plotted // n_rows) * l:(n_plotted // n_rows +1) * l, :] = frame
                
                
                # Add sampled frame number at the top-right corner
                text_x = n_plotted // n_rows * l  +l - 3  # Slight offset from the right
                text_y = (n_plotted % n_rows) * L + 5  # Slight offset from the top
                #print('text_y=', text_y,'text_x=', text_x,  sampled_frame_info)
                #ax.text(text_x, text_y, sampled_frame_info, color='k', fontsize=5, ha='right', va='top')
                ax.text(text_x, text_y, sampled_frame_info, 
                        color='r' if i_seg==0 else 'k', fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                # if (n_plotted % 5) == 0:
                #     ax.text(text_x-50, text_y, f'{row.participant_id}', color='black', fontsize=20, ha='right', va='top')
                
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    ((n_plotted // n_rows) * l, (n_plotted % n_rows) * L), l, L,
                    linewidth=4,
                    edgecolor=cluster_colors(unique_clusters.tolist().index(cluster_value)),
                    facecolor='none'
                )
                
                ax.add_patch(rect)
                n_plotted += 1
                
        
        tiled_frames = tiled_frames[0:n_rows * L, 0:((n_plotted-1) // n_rows+1) * l, :]
        #print(f'Final tile of shape {tiled_frames.shape}')
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, prefix+row.identifier + '_K_' + str(cluster_value) + suffix+ ".png")
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path)
            green(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    return

def plot_participant_clusters_chronological(df, labels_col='segments_labels', cpts_cols='cpts', n_rows=10,  max_length=500, prefix='', suffix='', n_sbj=10, verbose=False, show=True, figure_folder=None):

    
    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1) and (df.n_cluster.nunique() == 1)
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()

    n_cluster = df.n_cluster.iloc[0]
    df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                            .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))


    print('cpts_cols:', cpts_cols, 'labels_col', labels_col,  'n_rows', n_rows, 'n_sbj', n_sbj, 'max_length:', max_length, 'n_cluster:', n_cluster)

    # 1) Find total number of clusters and get colors
    cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)
    colors = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors) + 
        list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors) 
        ) * 20
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])

    df['cluster_dist_cluster'] = collapse_cluster_stats(df, labels_col='embedding_labels', per_segment_feature='cluster_dist')

    for _, row in df.sample(n_sbj).iterrows():
        
        labels = row[labels_col]
        #sbj_cluster_values = sorted(np.unique(labels)); 
        sbj_cluster_values = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False).index
        sbj_prevalence = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False)[0].astype(str).to_list()
        n_clusters = len(sbj_cluster_values)
        if figure_folder is not None:
            figure_path = os.path.join(figure_folder, prefix+row.identifier + suffix+ ".png")
            if os.path.exists(figure_path):
                print(f'Figure {figure_path} already exists. Skipping')
                continue
                
        try:
            vr = video_loader(row.video_path)
        except:
            print(f'Error loading {row.video_path}. Continue')
            #import subprocess
            #subprocess.run(['open', os.path.dirname(row.video_path)])
            #subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
            continue
        
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(f'Chronological action segmentation\n{row.task_name} - {row.modality} - {row.participant_id} - K={row.n_cluster} (|S|={len(labels)})\n' + '-'.join(sbj_prevalence), weight='bold', y=1.02,fontsize=20)
        
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        
        n_plots =  min(len(labels), max_length)
        tiled_frames = np.zeros([n_rows * L, (n_plots // n_rows+1) * l, C], dtype=np.uint8)
        #print(f'Initial tile of shape {tiled_frames.shape}')
        n_plotted = 0
        for cluster_indexes, cluster_value  in enumerate(labels):
            
            cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in [cluster_indexes]]
            sampled_embeddings = [(start, end, int(start + (end-start) * np.random.sample())) for  start, end in  cpts_tuples]
            
            
            try:
                std_d_c = np.std(row['cluster_dist_cluster'][cluster_value])
                range_c = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) 
                range_c_n = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) / np.max(row['cluster_dist_cluster'][cluster_value])
            except:
                std_d_c = np.nan
                range_c = np.nan
                range_c_n = np.nan

            if verbose:
                print(f'Exploring label {cluster_value}')
                print(f'Segments labels: {labels}')
                print(f'Cpts start-end: {cpts_tuples}')
                print(f'Final sampled frames: {sampled_embeddings}')

        
            #print(f'Final tile of shape {tiled_frames.shape}')
            for i_seg , (start, end, embedding_idx) in enumerate(sampled_embeddings):
                
                sampled_frame = int( embedding_idx / row.N * len(vr))
                sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) {cluster_indexes}/{len(labels)}'

                #sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) d={100*(row.cluster_dist[embedding_idx] / range_c):.1f}%'
                
                    
                # Place each sampled frame in the array
                try:
                    frame = np.array(vr[sampled_frame].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    print(n_plotted, 'Error loading frame', sampled_frame)
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                        
                
                try:
                    #print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                    tiled_frames[(n_plotted % n_rows) * L:(n_plotted % n_rows + 1) * L, (n_plotted // n_rows) * l:(n_plotted // n_rows +1) * l, :] = frame
                except:
                    print('Error placing frame', frame.shape, tiled_frames.shape)
                    print(f'Tile of shape {tiled_frames.shape}')
                    print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                    
                # Add sampled frame number at the top-right corner
                text_x = n_plotted // n_rows * l  +l - 3  # Slight offset from the right
                text_y = (n_plotted % n_rows) * L + 5  # Slight offset from the top
                #print('text_y=', text_y,'text_x=', text_x,  sampled_frame_info)
                #ax.text(text_x, text_y, sampplot_cluster_prevalenceled_frame_info, color='k', fontsize=5, ha='right', va='top')
                ax.text(text_x, text_y, sampled_frame_info, 
                        color=cluster_colors(unique_clusters.tolist().index(cluster_value)), #'r' if i_seg==0 else 'k', 
                        fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                # if (n_plotted % 5) == 0:
                #     ax.text(text_x-50, text_y, f'{row.participant_id}', color='black', fontsize=20, ha='right', va='top')
                
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    ((n_plotted // n_rows) * l, (n_plotted % n_rows) * L), l, L,
                    linewidth=4,
                    edgecolor=cluster_colors(unique_clusters.tolist().index(cluster_value)),
                    facecolor='none'
                )
                
                ax.add_patch(rect)
                n_plotted += 1
                
        
        tiled_frames = tiled_frames[0:n_rows * L, 0:((n_plotted-1) // n_rows+1) * l, :]
        #print(f'Final tile of shape {tiled_frames.shape}')
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, prefix+row.identifier + '_K_' + str(cluster_value) + suffix+ ".png")
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path)
            green(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
    return


def plot_cluster_boxplots_hue(df, labels_col='segments_labels', x='cluster_index', y='n_appearance', subset_clusters=None, hue=None, ax=None, figsize=(20, 3)):
    
    """
    Plots boxplots of number of occurrences (n_appearance) per cluster (cluster_index)
    with group as hue and consistent colors for clusters.
    
    Parameters:
        data (DataFrame): Long-form dataframe with `cluster_index`, `n_appearance`, and `group` columns.
        x (str): Column name for x-axis (default: 'cluster_index').
        y (str): Column name for y-axis (default: 'n_appearance').
        hue (str): Column name for hue (default: 'group').
        figsize (tuple): Size of the figure (default: (14, 7)).
    """
    
    
    assert (df.task_name.nunique() == 1) & (df.modality.nunique() == 1)
    assert (df.participant_id.nunique() == len(df)) or (hue == 'participant_id')

    n_cluster = df[labels_col].apply(max).max() +1 
    
    if y == 'n_occurences':

        df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                                .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))
        
        if 'pathologie' in df.columns:
            long_df = pd.DataFrame([{"cluster_index": k, y: v, "participant_id": row["participant_id"], "group": row["group"], "pathologie": row["pathologie"]}
                                        for _, row in df.iterrows()
                                        for k, v in row['cluster_counts'].items()])
        else:
            long_df = pd.DataFrame([{"cluster_index": k, y: v, "participant_id": row["participant_id"], "group": row["group"]}
                                        for _, row in df.iterrows()
                                        for k, v in row['cluster_counts'].items()])
        
    else:
        long_df_list = [] 
        for _, row in df.iterrows():
            
            if isinstance(row[y], dict):
                for cluster_index, value_list in row[y].items():
                    for value in value_list:
                        long_df_list.append({"cluster_index": cluster_index, y: value, "participant_id": row["participant_id"], "group": row["group"], "pathologie": row["pathologie"]})

        long_df = pd.DataFrame(long_df_list)

    
    # Generate consistent colors for clusters
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    unique_clusters = sorted(long_df[x].unique())
    cluster_colors = colors[:len(unique_clusters)]
    
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if subset_clusters is not None:
        long_df = long_df[long_df[x].isin(subset_clusters)]
        long_df[x] = pd.Categorical(long_df[x], categories=subset_clusters, ordered=True)

    
    # sns.boxplot(
    #     data=long_df,
    #     x=x,
    #     y=y,
    #     palette=cluster_colors, #"muted",
    #     showfliers=False,
    #     ax=ax
    # )
    sns.boxplot(
        data=long_df,
        x=x,
        y=y,
        hue=x,  # Explicitly assign the x variable to hue
        #palette=cluster_colors,
        showfliers=False,
        legend=False,  # Ensures the legend is disabled
        ax=ax
    )
    

    # Overlay consistent colors for clusters
    # for patch, cluster in zip(ax.artists, unique_clusters):
    #     patch.set_facecolor(cluster_palette[cluster])

    # Formatting
    ax.set_title(f"x={x}, y={y}\n{df.task_name.iloc[0]} {df.modality.iloc[0]} N={len(df)}", fontsize=14)
    ax.set_xlabel("Cluster Index", fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    #plt.legend(title="Group", loc='upper right')
    
def plot_cluster_boxplots_hue_samples(df, labels_col='segments_labels', x='cluster_index', y='n_appearance', hue=None, subset_clusters=None):
        
        
    assert (df.task_name.nunique() == 1) & (df.modality.nunique() == 1)
    assert (df.participant_id.nunique() == len(df)) or (hue == 'participant_id')

    n_cluster = df[labels_col].apply(max).max() +1 
    
    if y == 'n_occurences':

        df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                                .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))
        
        long_df = pd.DataFrame([{"cluster_index": k, y: v, "participant_id": row["participant_id"], "group": row["group"], "pathologie": row["pathologie"]}
                                    for _, row in df.iterrows()
                                    for k, v in row['cluster_counts'].items()])
        
    else:
        long_df_list = [] 
        for _, row in df.iterrows():
            
            if isinstance(row[y], dict):
                for cluster_index, value_list in row[y].items():
                    for value in value_list:
                        long_df_list.append({"cluster_index": cluster_index, y: value, "participant_id": row["participant_id"], "group": row["group"], "pathologie": row["pathologie"]})

        long_df = pd.DataFrame(long_df_list)


    if subset_clusters is not None:
        long_df = long_df[long_df[x].isin(subset_clusters)]
    
    # Generate consistent colors for clusters
    unique_clusters = sorted(long_df[x].unique())
    cluster_colors = get_colors(unique_clusters)
    cluster_palette = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
    
    # Plot individual distributions (first 20 examples)
    fig, axes = plt.subplots(5, 5, figsize=(20, 8), sharex=True, sharey=False)
    fig.suptitle(f"y={y}\n{df.task_name.iloc[0]} {df.modality.iloc[0]} N={len(df)}", fontsize=14)
    axes = axes.flatten()

    for i, participant_id in enumerate(long_df.participant_id.unique()[:25]):
        
        sdf = long_df[long_df['participant_id'] == participant_id]  
        
        if sdf[y].isna().mean() == 1:
            continue 
        
        if y == 'n_occurences':
            cluster_indexes = sdf.cluster_index.to_list()
            label = sdf[y].to_list()
            axes[i].bar(cluster_indexes, label, color=get_colors(cluster_indexes))
            
        else:
            try:
                sns.boxplot(
                        data=sdf,
                        x=x,
                        y=y,
                        #hue=hue,
                        #palette="muted",
                        color='lightgrey',
                        edgecolor='black',
                        #dodge=True,
                        ax = axes[i],
                        showfliers=False
                    )
            except:
                print('Faled')
                print('len(sdf)', len(sdf), participant_id)
                

        axes[i].set_title(f'{participant_id}-{df.modality.iloc[0]}')
        axes[i].set_xlabel('Cluster Index')
        axes[i].set_ylabel(y)

    plt.tight_layout()
    plt.show()

    return 

# Function to plot labels distributions in subplots
def plot_labels_distributions_subplots(df, clustering_config_names, labels_col='embedding_labels', segments_labels_col='segments_labels', clusterdf=None, figsize=(25, 12)):

    num_plots = len(clustering_config_names)
    if num_plots > 1:
        num_cols = min(2, num_plots)
        num_rows = (num_plots + num_cols - 1) // num_cols
        
        # num_rows = 4  # Fixed number of rows
        # num_cols = (num_plots + num_rows - 1) // num_rows  # Calculate the number of columns

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        axes = [axes]
        
    for i, clustering_config_name in enumerate(clustering_config_names):
        results = df[df['symbolization_config_name'] == clustering_config_name].copy()
        labels = np.hstack(results[labels_col])
        

        if clusterdf is None:
            from smartflat.features.symbolization.main import build_clusterdf
            clusterdf = build_clusterdf(results, embeddings_label_col=labels_col, segments_labels_col=segments_labels_col, qualification_mapping=None)

        #print('label support:', len(np.unique(labels)))
        #print('clusterdf shape:', clusterdf.shape)
        
        if clusterdf is not None and 'cluster_type' in clusterdf.columns:
            
            if clusterdf.cluster_index.max() > 500:
                bins=200
            else:
                bins = np.arange(0, clusterdf.cluster_index.max() + 1)
        
            # Map cluster_index to cluster_type, then to colors
            cluster_types = clusterdf['cluster_type'].unique()
            colors = dict(zip(cluster_types, plt.get_cmap('tab10').colors[:len(cluster_types)]))

            # Create a dictionary mapping each label to its color based on cluster_type
            label_to_type = clusterdf.set_index('cluster_index')['cluster_type'].to_dict()
            label_to_color = {label: colors[label_to_type[label]] for label in label_to_type}
            # Plot labels by cluster_type
            for k, cluster_type in enumerate(cluster_types):
                cluster_labels = [label for label in labels if label_to_type.get(label) == cluster_type]
                color = plt.get_cmap('tab20').colors[k]
                if cluster_labels:
                    #print(f'Plotting {len(cluster_labels)} labels for cluster type {cluster_type}')
                    axes[i].hist(cluster_labels, bins=bins, color=color, 
                                edgecolor='black', label=cluster_type, alpha=0.7)
                    axes[i].legend()
        else:
            print('No cluster_type available. Plotting all labels in grey.')
            # Plot all labels in grey if no cluster_type is available
            axes[i].hist(labels, bins=200, color='lightgrey', edgecolor='black')
        
        # Assign colors based on cluster type of each label
        #label_colors = [label_to_color.get(label, 'grey') for label in labels]

        #axes[i].hist(labels, bins=200, color=label_colors, edgecolor='black')

        axes[i].set_title(f'Embedding Label Distribution {labels_col}\n{clustering_config_name}: N={len(results)} K={len(np.unique(labels))}')
        axes[i].set_xlabel('Labels')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



# Function to plot distances distribution in subplots
def plot_distances_distribution_subplots(df, clustering_config_names, distance_col='cluster_dist', radius_symbolization=0.3, cluster_type=None, title='', figsize=(25, 12)):
    num_plots = len(clustering_config_names)
    # num_cols = min(3, num_plots)
    # num_rows = (num_plots + num_cols - 1) // num_cols
    
    num_rows = 3  # Fixed number of rows
    num_cols = (num_plots + num_rows - 1) // num_rows  # Calculate the number of columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, clustering_config_name in enumerate(clustering_config_names):
        results = df[df['symbolization_config_name'] == clustering_config_name].copy()
        distances = np.hstack(results[distance_col])
        
        if 'opt_cluster_type' in results.columns and cluster_type is not None:
            opt_cluster_type = np.hstack(results['opt_cluster_type'])
            distances = np.array(distances)[np.array(opt_cluster_type) == cluster_type]
        
        axes[i].hist(distances, bins='fd', color='lightgrey', edgecolor='black')
        axes[i].set_xlim([-0.05, 1.05])
        axes[i].set_title(f'Cluster Distances Distribution\n{distance_col}: {clustering_config_name} (ct={cluster_type}) N={len(results)}\n{title}')
        if radius_symbolization is not None:
            axes[i].axvline(x=radius_symbolization, color='r', linestyle='--', label='Radius Symbolization', linewidth=4)
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot symbols frequency in subplots
def plot_symbols_frequency_subplots(df, clustering_config_names, figsize=(25, 12)):
    num_plots = len(clustering_config_names)
    # num_cols = min(3, num_plots)
    # num_rows = (num_plots + num_cols - 1) // num_cols
    
    num_rows = 3  # Fixed number of rows
    num_cols = (num_plots + num_rows - 1) // num_rows  # Calculate the number of columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=False)
    axes = axes.flatten()

    for i, clustering_config_name in enumerate(clustering_config_names):
        results = df[df['symbolization_config_name'] == clustering_config_name].copy()
        results['n_segments'] = results.segments_labels.apply(len)
        results['symbols_freq'] = results['n_segments'] / results['duration']
        axes[i].hist(results['symbols_freq'], bins=50, color='lightgrey', edgecolor='black')
        axes[i].set_title(f'Distribution of symbol frequency over time (symbol/minutes)\n{clustering_config_name} N={len(results)}')
        axes[i].set_xlabel('[symbol/minutes]')
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to plot subject cluster sampling sequdistances distribution in subplots
def plot_sbj_cluster_distances_distribution_subplots(df, clustering_config_names, figsize=(25, 12)):
    num_plots = len(clustering_config_names)
    # num_cols = min(3, num_plots)
    # num_rows = (num_plots + num_cols - 1) // num_cols
    
    num_rows = 3  # Fixed number of rows
    num_cols = (num_plots + num_rows - 1) // num_rows  # Calculate the number of columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, clustering_config_name in enumerate(clustering_config_names):
        results = df[df['symbolization_config_name'] == clustering_config_name].copy()
        results['cluster_dist_mean'] = results.cluster_dist.apply(lambda x: np.mean(x))
        axes[i].hist(results['cluster_dist_mean'], bins=50, color='lightgrey', edgecolor='black')
        axes[i].set_title(f'Distance distribution across subjects\n{clustering_config_name} N={len(results)}')
        axes[i].set_xlabel('Distance distribution across subjects')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_symbolization_variables(df, figsize=(25, 12)):
    
    config_names = df['annotator_id'].unique()
    num_plots = len(config_names)
    rounds = sorted(df['round_number'].unique())
    palette = sns.color_palette("tab10", n_colors=len(rounds))

    if num_plots > 1:
        num_rows = num_plots
        num_cols = 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 4 * num_rows), sharex=True, sharey=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=False)
        axes = [axes]

    fig.suptitle('Scatter Plot: Number (left) and Percentage (right) of Embed Changes vs Percent Cpts Withdrawn', fontsize=16)

    for i, clustering_config_name in enumerate(config_names):
        result = df[df['symbolization_config_name'] == clustering_config_name].copy()
        ax0, ax1 = axes[i]  # works for both cases now

        for j, r in enumerate(rounds):
            round_df = result[result['round_number'] == r]
            ax0.scatter(round_df["percent_embed_changes"], round_df['sum_cpts_withdrawn'],
                        label=f'Round {r}', alpha=0.6, edgecolor='k', color=palette[j])
            ax1.scatter(round_df["percent_embed_changes"], round_df['percent_cpts_withdrawn'],
                        label=f'Round {r}', alpha=0.6, edgecolor='k', color=palette[j])

        ax0.set_title(f'{clustering_config_name}')
        ax0.set_xlabel('Percent Embed Changes')
        ax0.set_ylabel('Number Cpts Withdrawn')

        ax1.set_title(f'{clustering_config_name}')
        ax1.set_xlabel('Percent Embed Changes')
        ax1.set_ylabel('Percent Cpts Withdrawn')

        if i == 0 or num_plots == 1:
            ax0.legend(title="Round", loc="upper right")
            ax1.legend(title="Round", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
        
        
    
# Function to plot cluster prevalence in subplots
def plot_cluster_prevalence_subplots(df, clustering_config_names, figsize=(25, 10), verbose=True):
    num_plots = len(clustering_config_names)
    # num_cols = min(3, num_plots)
    # num_rows = (num_plots + num_cols - 1) // num_cols
    
    num_rows = 3  # Fixed number of rows
    num_cols = (num_plots + num_rows - 1) // num_rows  # Calculate the number of columns

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, clustering_config_name in enumerate(clustering_config_names):
        results = df[df['symbolization_config_name'] == clustering_config_name].copy()
        long_df = pd.DataFrame(
            [
                {"cluster_index": k, "n_appearance": v, "participant_id": row["participant_id"], "group": row["group"]}
                for _, row in results.iterrows()
                for k, v in dict(Counter(row['segments_labels'])).items()
            ]
        )

        cluster_participants = long_df.groupby('cluster_index')['participant_id'].nunique()
        total_participants = long_df['participant_id'].nunique()

        # Calculate proportions
        prevalence = (cluster_participants / total_participants).reset_index(name='prevalence').sort_values(by='prevalence', ascending=False)

        # Ensure the x-axis order is preserved
        prevalence['cluster_index'] = pd.Categorical(prevalence['cluster_index'], categories=prevalence['cluster_index'], ordered=True)

        unique_clusters = sorted(prevalence.cluster_index.unique())

        if verbose:
            cmap = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
            cmap = cmap[:prevalence.cluster_index.nunique()]

            sns.barplot(data=prevalence, x='cluster_index', y='prevalence', color='lightgrey', edgecolor='black', ax=axes[i])
            axes[i].set_title(f"Prevalence of each cluster across participants\n{clustering_config_name}", fontsize=14)
            axes[i].set_xlabel("Cluster Index", fontsize=12)
            axes[i].set_ylabel("Proportion of participants", fontsize=12)
            axes[i].axhline(1, linestyle='--', color='r')
            axes[i].set_ylim([0, 1.05])
            axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



# Hierarchical custering
def compute_inconsistency_across_depth(linkage_matrix, centroids, output_folder=None, suffix=''):
    
    depths = [2, 3, 4, 5, 6, 7, 8, 10]

    # Prepare 2D embedding
    embedding = UMAP(n_components=2).fit_transform(centroids)

    cluster_counts = []

    # Plot 1: Dendrograms with actual cut merge distances
    plt.figure(figsize=(25, 8))
    for i, d in enumerate(depths):
        I = sch.inconsistent(linkage_matrix, d)
        incons = I[:, -1]
        threshold = np.percentile(incons, 90)
        labels = sch.fcluster(linkage_matrix, t=threshold, criterion='inconsistent', depth=d)
        cluster_counts.append(len(np.unique(labels)))

        # Compute actual merge distances in the linkage matrix
        # Filter only the merges above the threshold
        merges = linkage_matrix[:, 2]
        cut_distance = np.min(merges[np.where(incons > threshold)])

        plt.subplot(2, len(depths)//2, i+1)
        sch.dendrogram(linkage_matrix, no_labels=True, color_threshold=cut_distance)
        plt.axhline(y=cut_distance, color='red', linestyle='--')
        plt.title(f'depth={d}\nk={len(np.unique(labels))}', fontsize=8)
    plt.tight_layout()

    if output_folder is not None:
        figpath = os.path.join(output_folder, f'inconsistency_per_depth_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
        
        

    # Plot 2: Cluster count vs. depth
    plt.figure(figsize=(25, 8))
    plt.plot(depths, cluster_counts, marker='o')
    plt.xlabel('depth')
    plt.ylabel('n_clusters from inconsistency cut')
    plt.title('Cluster count vs. inconsistency depth')
    plt.grid(True)
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'n_cluster_per_depth_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
    plt.show()

    # Plot 3: Embedding with colored clusters
    fig, axs = plt.subplots(2, len(depths)//2, figsize=(25, 10))
    axs = axs.flatten()
    for i, d in enumerate(depths):
        I = sch.inconsistent(linkage_matrix, d)
        threshold = np.percentile(I[:, -1], 90)
        labels = sch.fcluster(linkage_matrix, t=threshold, criterion='inconsistent', depth=d)

        axs[i].scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=10)
        axs[i].set_title(f'depth={d}, k={len(np.unique(labels))}', fontsize=8)
        axs[i].axis('off')

    plt.tight_layout()
    
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'umap_embedding_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
    
    plt.show()

def compute_inconsistency_across_thresholds(linkage_matrix, centroids, depth=6, output_folder=None, suffix=''):

    # Prepare 2D embedding
    embedding = UMAP(n_components=2).fit_transform(centroids)

    I = sch.inconsistent(linkage_matrix, depth)
    coeffs = I[:, -1]
    percentiles = np.arange(60, 100, 2)
    cluster_counts = []

    # Plot 1: Dendrograms with colored cuts per threshold
    plt.figure(figsize=(25, 8))
    for i, p in enumerate(percentiles):
        threshold = np.percentile(coeffs, p)
        labels = sch.fcluster(linkage_matrix, t=threshold, criterion='inconsistent', depth=depth)
        cluster_counts.append(len(np.unique(labels)))

        # Actual merge distance at this threshold
        merges = linkage_matrix[:, 2]
        cut_distance = np.min(merges[np.where(coeffs > threshold)])

        plt.subplot(2, len(percentiles)//2, i+1)
        sch.dendrogram(linkage_matrix, no_labels=True, color_threshold=cut_distance)
        plt.axhline(y=cut_distance, color='red', linestyle='--')
        plt.title(f'p={p}%, k={len(np.unique(labels))}', fontsize=8)
    plt.tight_layout()
    
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'n_cluster_per_threshold_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
        
    # Plot 2: Cluster count vs. threshold percentile
    plt.figure(figsize=(25, 6))
    plt.plot(percentiles, cluster_counts, marker='o')
    plt.xlabel('Inconsistency threshold percentile')
    plt.ylabel('n_clusters from inconsistency cut')
    plt.title(f'Cluster count vs. inconsistency threshold (depth={depth})')
    plt.grid(True)
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'inconsistency_per_depth_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
    
    plt.show()

    # Plot 3: Embedding with clusters for each threshold
    fig, axs = plt.subplots(2, len(percentiles)//2, figsize=(25, 10))
    axs = axs.flatten()
    for i, p in enumerate(percentiles):
        threshold = np.percentile(coeffs, p)
        labels = sch.fcluster(linkage_matrix, t=threshold, criterion='inconsistent', depth=depth)

        axs[i].scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=10)
        axs[i].set_title(f'p={p}%, k={len(np.unique(labels))}', fontsize=8)
        axs[i].axis('off')

    plt.tight_layout()
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'umap_per_percentiles_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
    
    plt.show()
    
def explore_inconsistency_depth_threshold(linkage_matrix, centroids, output_folder=None, suffix=''):
    depths = [3, 4, 5, 6, 7]
    percentiles = np.arange(60, 100, 2)

    records = []

    for depth in depths:
        I = sch.inconsistent(linkage_matrix, depth)
        coeffs = I[:, -1]

        for p in percentiles:
            threshold = np.percentile(coeffs, p)
            labels = sch.fcluster(linkage_matrix, t=threshold, criterion='inconsistent', depth=depth)
            n_clusters = len(np.unique(labels))
            records.append({
                'depth': depth,
                'percentile': p,
                'n_clusters': n_clusters
            })

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(25, 8))

    sns.lineplot(data=df, x='depth', y='n_clusters', hue='percentile', marker='o', palette='viridis', ax=axes[0])
    axes[0].set(title='Clusters vs. Inconsistency\nColored by Threshold Percentile', xlabel='Depth', ylabel='Number of Clusters')
    axes[0].grid(True)

    sns.lineplot(data=df, x='percentile', y='n_clusters', hue='depth', marker='o', palette='viridis', ax=axes[1])
    axes[1].set(title='Clusters vs. Ic Threshold\nColored by Depth', xlabel='Percentile Threshold', ylabel='Number of Clusters')
    axes[1].grid(True)

    plt.tight_layout()
    if output_folder is not None:
        figpath = os.path.join(output_folder, f'inconsistency_per_depth_percentiles_D_{centroids.shape[0]}_{suffix}.png')
        plt.savefig(figpath, bbox_inches='tight', dpi=100)
        print(f"Saved figure to {figpath}")
    plt.show()
    return df


### OT plots

def plot_trajectory(T, title='Trajectory', T_high=30, do_normalize=True, figsize=(25, 12), add_lowest=False, cmap='Blues', xlim=None):

    if do_normalize:
        T = T / np.abs(np.max(T))
        
    # Get the top 5 row indices for each column
    top5_indices = np.argsort(T, axis=0)[-5:]

    # Create mask
    mask = np.zeros_like(T, dtype=bool)
    np.put_along_axis(mask, top5_indices, True, axis=0)

    # Apply mask
    T_highlight = np.where(mask, T, -30)  # Dim non-masked components
    
    
    # Get the top 5 and bottom 5 row indices for each column
    top5_indices = np.argsort(T, axis=0)[-5:]     # Largest 5


    mask_high = np.zeros_like(T, dtype=bool)
    np.put_along_axis(mask_high, top5_indices, True, axis=0)   # Top 5 mask
    T_highlight = np.where(mask_high, T_high, T)    # Top 5 → +30

    if add_lowest:
        bottom5_indices = np.argsort(T, axis=0)[:5]   # Smallest 5
        mask_low = np.zeros_like(T, dtype=bool)
        np.put_along_axis(mask_low, bottom5_indices, True, axis=0) # Bottom 5 mask
        T_highlight = np.where(mask_low, -T_high, T_highlight)  # Bottom 5 → -30

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # Display the final highlighted matrix with a colormap
    cmap = plt.get_cmap("coolwarm" if add_lowest else cmap)  # Blue for low, red for high
    im = ax.imshow(T_highlight, aspect='auto', cmap=cmap, vmin=-30, vmax=30)

    if add_lowest:
        # Add a colorbar with a custom colormap
        cbar = plt.colorbar(im, ticks=[-30, 0, 30])
        cbar.ax.set_yticklabels(['-30', '0', '30'])
        plt.title('Top 5 (+30 Red) and Bottom 5 (-30 Blue) projections coordinates\n'+title)
    else:
        plt.title('Top 5 (+30 Red)  projections coordinates\n'+title)
        plt.colorbar(im)
        
    if xlim is not None:
        plt.xlim(xlim)       
    plt.show()

def plot_ot_graphs(X_proj, T, labels, alpha=None, beta=None, suptitle=''):
    

    # Create a 5x2 subplot layout to accommodate the new plots
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 20))
    fig.suptitle(suptitle, fontsize=16, weight='bold')
    
    # Plot 1: Distribution of (max-) Normalized Optimal Transport Map Values
    axes[0, 0].hist((T / T.max()).flatten(), bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of (max-) Normalized Optimal Transport Map Values', fontsize=12, weight='bold')
    axes[0, 0].set_xlabel('Normalized Transport Map Values', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Solving the OT compared to projections
    axes[0, 1].hist((np.unique(T) / T.max()).flatten(), bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Unique transport map value the OT compared to 1D-coordinates (subspace projection)', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('Normalized Transport Map Values', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Distribution of Projected Embedding Values
    axes[1, 0].hist(X_proj.flatten(), bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Distribution of Projected Embedding Values\n(Transport map on the prototypes or projection calibrated by distance to the orth prototypes basis)', fontsize=12, weight='bold')
    axes[1, 0].set_xlabel('Projected Values', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    if alpha is not  None:

        # Plot 4: Sample weight (alpha) for solving the Earth Movers distance
        axes[1, 1].hist(alpha, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title(f'Sample weight (alpha) for solving the Earth Movers distance (N={len(np.min(M, axis=1))})', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Minimum Distance', fontsize=10)
        axes[1, 1].set_ylabel('Frequency', fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)

        # Plot 5: Prototypes weight (beta) for solving the Earth Movers distance
        axes[2, 0].hist(beta, bins=200, color='skyblue', edgecolor='black', alpha=0.7)
        axes[2, 0].set_title(f'Prototypes weight (beta) for solving the Earth Movers distance (N={len(np.min(M, axis=0))})', fontsize=12, weight='bold')
        axes[2, 0].set_xlabel('Minimum Distance', fontsize=10)
        axes[2, 0].set_ylabel('Frequency', fontsize=10)
        axes[2, 0].grid(True, linestyle='--', alpha=0.7)

        # Plot 6: Distribution of Beta Values
        axes[2, 1].scatter(np.arange(beta.shape[0]), beta, color='skyblue', edgecolor='black', alpha=0.7)
        axes[2, 1].set_ylim([0, 0.002])
        axes[2, 1].set_title('Distribution of Beta Values', fontsize=12, weight='bold')
        axes[2, 1].set_xlabel('Index', fontsize=10)
        axes[2, 1].set_ylabel('Beta Value', fontsize=10)
        axes[2, 1].grid(True, linestyle='--', alpha=0.7)

        # Plot 7: Distribution of Alpha Values
        axes[3, 0].scatter(np.arange(alpha.shape[0]), alpha, color='skyblue', edgecolor='black', alpha=0.7)
        axes[3, 0].set_ylim([0, 0.00025])
        axes[3, 0].set_title('Distribution of Alpha Values', fontsize=12, weight='bold')
        axes[3, 0].set_xlabel('Index', fontsize=10)
        axes[3, 0].set_ylabel('Alpha Value', fontsize=10)
        axes[3, 0].grid(True, linestyle='--', alpha=0.7)    
        

    else:
        # Delete axes
        fig.delaxes(axes[1, 1])
        fig.delaxes(axes[2, 0])
        fig.delaxes(axes[2, 1])
        fig.delaxes(axes[3, 0])
        
        
    # Plot 8: Distribution of Labels
    axes[3, 1].hist(labels, bins=np.arange(-1, np.max(labels)+1), color='skyblue', edgecolor='black', alpha=0.7)
    axes[3, 1].set_title(f'Distribution of Labels (N={len(labels)})', fontsize=12, weight='bold')
    axes[3, 1].set_xlabel('Labels', fontsize=10)
    axes[3, 1].set_ylabel('Frequency', fontsize=10)
    axes[3, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot 9: Distribution of X_proj Embedding Values with weights_priors
    # axes[4, 0].hist(X_proj.flatten(), bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    # axes[4, 0].set_title(f'Distribution of X_proj Embedding Values', fontsize=12, weight='bold')
    # axes[4, 0].set_xlabel('Projected Values', fontsize=10)
    # axes[4, 0].set_ylabel('Frequency', fontsize=10)
    # axes[4, 0].grid(True, linestyle='--', alpha=0.7)
    fig.delaxes(axes[4, 0])

    # Plot 10: Distribution of M Values with weights_priors
    axes[4, 1].hist(M.flatten(), bins=200, color='skyblue', edgecolor='black', alpha=0.7)
    axes[4, 1].set_title(f'Distribution of M Values', fontsize=12, weight='bold')
    axes[4, 1].set_xlabel('Projected Values', fontsize=10)
    axes[4, 1].set_ylabel('Frequency', fontsize=10)
    axes[4, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def visualize_cohort_symbolic_representation(df, covariate = 'n_cluster', labels_col='embedding_labels', figsize=(25, 14), upsampling="interpolation"):


    df.sort_values(['duration'], inplace=True)
    fig, axes = plt.subplots(nrows=len(df[covariate].unique()) // 2, ncols=4, figsize=figsize)#(25, 8 * len(df.n_cluster.unique()) // 5))
    axes = axes.flatten()

    for ax, covar in zip(axes, sorted(df[covariate].unique())):
        row = df[df[covariate] == covar]
            
        if len(row) == 0:
            print(f'No data for covariate {covar}. Continue')
            continue

        
        plot_chronogames(row, labels_col=labels_col, time_calibration='embeddings', 
                         title=f'{covar}', figsize=(25, 14), upsampling=upsampling, ax=ax)

    # remove extra axes
    for i in range(len(df[covariate].unique()), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()
    


### Statistics and results 

def summarize_metrics(df, group_vars = ['input_space', 'kernel_name', 'multimodal_method', 'alpha', 'linkage_method']):
    raise DeprecationWarning("This function is deprecated.")
    metrics = [
        'silhouette_score_baseline', 'silhouette_score_M', 'silhouette_score_T', 'silhouette_score_X', 
        'davies_bouldin_baseline', 'davies_bouldin_M', 'davies_bouldin_T', 'davies_bouldin_X', 
        'compactness_separation_baseline', 'compactness_separation_M', 'compactness_separation_T', 'compactness_separation_X', 
        'mmd_baseline', 'mmd_M', 'mmd_T', 'mmd_X', 
        'modularity_baseline', 'modularity_M', 'modularity_T', 'modularity_X',
        'mean_ic', 'median_ic'
    ]

    agg_dict = {
            col: (
                'min' if any(k in col for k in ['davies', 'mmd'])
                else 'max'
            )
            for col in metrics
            if col not in group_vars and col != 'cluster_type'
        }
    
    
    return {
        ct: (
            df[df['cluster_type'] == ct]
            .groupby(group_vars)
            [metrics]
            .agg(agg_dict)
        )
        for ct in df['cluster_type'].unique()
    }
    