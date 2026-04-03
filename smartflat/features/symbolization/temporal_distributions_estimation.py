"""KDE-based temporal distribution estimation and Wasserstein distance computation.

Estimates when each prototype appears in video sequences using kernel density
estimation (KDE), then computes Wasserstein and Gromov-Wasserstein distances
between temporal profiles. Used for multi-view HAC consolidation
(Ch. 5, Section 5.3).

Prerequisites:
    - Symbolization DataFrame from ``utils_dataset.get_experiments_dataframe``.
    - Cluster labels assigned by inference or main pipeline.

Main entry points:
    - ``estimate_kde_model()``: Fit KDE per prototype temporal distribution.
    - ``compute_wasserstein_distance()``: 1D Wasserstein between KDEs.
    - ``compute_gw_costs_and_transport_maps()``: Gromov-Wasserstein distance.
    - ``plot_gw_temporal_distance()``: Visualize temporal distance matrices.

External dependencies:
    - ot (POT library for optimal transport)
    - scipy.stats (gaussian_kde)
    - statsmodels (KDEUnivariate)
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.linalg
import seaborn as sns
import torch
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib.colors import BoundaryNorm, ListedColormap
from ot.gromov import gromov_wasserstein, gromov_wasserstein2
from scipy.integrate._quadrature import simps
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, binary_opening, label
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, squareform
from scipy.stats import gaussian_kde, rv_continuous, wasserstein_distance
from sklearn.metrics.pairwise import pairwise_kernels
from statsmodels.nonparametric.kde import KDEUnivariate
from tqdm import tqdm

from smartflat.configs.loader import import_config
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.utils.utils import pairwise
from smartflat.utils.utils_coding import green
from smartflat.utils.utils_visualization import get_base_colors

# 1) Kernel Density estimation (KDE) of the temporal distributions
# 2) Gromov-Wasserstein (GW) distance computation between the KDEs of different clusters
# 3) Visualization of the KDEs and GW distances


# Global (module-level) variables for the multiprocessing workers

def prepare_row_timestamps_dataset(mdf, inverse_mode='exp', alpha=0.5):
    """Compute the timestamps, weights, and inverse prevalence weighting dictionnaries dataset for each K clusters."""
    
    unique_clusters = mdf['cluster_timetamps_list'].apply(lambda x: list(x.keys())).explode().unique()
    #assert len(unique_clusters) == mdf['K'].iloc[0] + 1, f"Number of unique clusters {len(unique_clusters)} does not match the number of clusters in the dataframe {mdf['K'].iloc[0]}+1"
    # Step 1: Compute weights
    cluster_row_weights = mdf['cluster_timetamps_list'].apply(
        lambda clusters: {cluster: len(timestamps) for cluster, timestamps in clusters.items()}
    )
    cluster_totals = {cluster: sum(weights.get(cluster, 0) for weights in cluster_row_weights) for cluster in unique_clusters}

    cluster_row_weights = cluster_row_weights.apply(
        lambda row_weights: {cluster: row_weights.get(cluster, 0) / (cluster_totals[cluster] + 1e-6) for cluster in unique_clusters}
    )

    # Step 2: Aggregate weighted timestamps
    n_rows = len(mdf)
    timestamps_dict = {cluster: [] for cluster in unique_clusters}
    w_timestamps_dict = {cluster: [] for cluster in unique_clusters}
    iw_timestamps_dict = {cluster: [] for cluster in unique_clusters}
    prevalence = {cluster: 0 for cluster in unique_clusters}

    for row_index, clusters in enumerate(mdf['cluster_timetamps_list']):
        row_cluster_weights = cluster_row_weights.iloc[row_index]
        for cluster, timestamps in clusters.items():
            w = row_cluster_weights[cluster]
            if inverse_mode == 'exp':
                inv_w = (1 / (w + 1e-6)) ** alpha
            elif inverse_mode == 'log':
                inv_w = -np.log(w + 1e-6)
            else:
                inv_w = 1.0  # fallback

            if len(timestamps) > 0:
                timestamps_dict[cluster].extend(timestamps)
                w_timestamps_dict[cluster].extend([w] * len(timestamps))
                iw_timestamps_dict[cluster].extend([inv_w] * len(timestamps))
                prevalence[cluster] += 1

    prevalence = {cluster: (count / n_rows) * 100 for cluster, count in prevalence.items()}

    return timestamps_dict, w_timestamps_dict, iw_timestamps_dict, prevalence


def estimate_kde_model(config_name, df=None, annotator_id='samperochon', round_number=0, input_space='K_space', embedding_labels_col=None, segments_labels_col=None, temporal_segmentations_col=None, verbose=False):

    """
    Estimate the KDE model for each cluster in the dataframe.
    Args:
        config_name (str): The name of the configuration to use.
        embedding_labels_col (str): The column name for the embedding labels.
        segments_labels_col (str): The column name for the segments labels.
        temporal_segmentations_col (str): The column name for the temporal segmentations.
    Returns:
        kde_model (dict): A dictionary containing the KDE model for each cluster."""
        
    from smartflat.features.symbolization.co_clustering import get_prototypes_mapping

    config =  import_config(config_name)
    gridsize = config.gridsize
    use_segments_timetamps = config.use_segments_timetamps
    timestamps_weighting_scheme = config.timestamps_weighting_scheme
    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    cluster_types = ['foreground'] #clustering_config.model_params['cluster_types'] + 
    if df is None:
        df = get_experiments_dataframe(experiment_config_name=config_name,  annotator_id=annotator_id, round_number=round_number, return_symbolization=True, verbose=verbose) 
    
    if embedding_labels_col is None:
        if input_space == 'K_space': 
            #print('Using inference-based K prototypes')
            embedding_labels_col = 'filtered_raw_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'cpts' # 'symb_cpts' # 'processed_symb_cpts
            
        
        elif input_space == 'G_space': 
            #print('Using reduced G prototypes')
                
            embedding_labels_col = 'opt_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'opt_segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'opt_cpts' # 'symb_cpts' # 'processed_symb_cpts

        elif input_space == 'G_opt_space': 
            #print('Using reduced G prototypes')
                
            embedding_labels_col = 'G_opt_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'G_opt_segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'G_opt_cpts' # 'symb_cpts' # 'processed_symb_cpts

    #CODE: assert df[embedding_labels_col].apply(len).sum() == df['N'].sum(), f"Number of timestamps in the {embedding_labels_col} does not match the number of embedding in the video duration length column {'N'}"

    n_lost = df[embedding_labels_col].apply(len).sum() - df['N_raw'].sum()
    #print(f'Number of dropped (edges) embedding: N={n_lost} - remaining N={df[embedding_labels_col].apply(len).sum()}')
    #print('Supports of embedding and segments labels:')
    #print((embedding_labels_col, segments_labels_col), len(np.unique(np.hstack(df[embedding_labels_col]))), len(np.unique(np.hstack(df[segments_labels_col]))))


    t0 = time.time()    
    #print(f'Preparing timestamps datasets with columns {embedding_labels_col}, {segments_labels_col}, {temporal_segmentations_col}')
    # Create for each row the dictionnary keyed by segments_labels and values the list of timestamps normalized per the actual sequence length
    if use_segments_timetamps:
        # Use the midle-timestamps of the segments to compute the centroids KDE
        df['cpts_percentiles'] = df.apply(lambda x: [cpts / x['N']  for cpts in x[temporal_segmentations_col]], axis=1)
        df['segments_timestamps_percentiles'] = df.cpts_percentiles.apply(lambda x: [(s + e) / 2 for s, e in pairwise(x)])
        df['cluster_timetamps_list'] = df.apply(lambda row:  {cv: list(np.array(row['segments_timestamps_percentiles'])[np.argwhere(row[segments_labels_col] == cv).flatten()]) for cv in np.unique(row[segments_labels_col])}, axis=1)
    else:
        # Create for each row the dictionnary keyed by segments_labels and values the list of timestamps normalized per the actual sequence length
        df['cluster_timetamps_list'] = df.apply(lambda row: {cv: np.argwhere(row[embedding_labels_col] == cv).flatten() / row['N'] for cv in np.unique(row[embedding_labels_col])}, axis=1)

    n_df=0
    for i, row in df.iterrows():
        t_dict = row.cluster_timetamps_list
        for cl_index, timestamps in t_dict.items():
            n_df+=len(timestamps)
    green(f"{annotator_id}-round: {round_number}.  Total number of timestamps:  {n_df} (N_raw={df.N_raw.sum()}, N={df.N.sum()} -> Keep={df.N.sum() / df.N_raw.sum():.2f} %")

    cluster_timestamps, timestamps_weights, timestamps_weights_inverse, cluster_prevalence = prepare_row_timestamps_dataset(df)
    #assert len(cluster_timestamps.keys()) == df['K'].iloc[0]+1, f"Number of clusters in the timestamps dictionary (+noise) {len(cluster_timestamps.keys())} does not match the number of clusters in the dataframe {df['K'].iloc[0]}+1"
    
    kde_models = {} 
    for cluster_type in cluster_types:
        
        # Deactivate cluster types to estimate KDE (always in foreground mode)
        
        #TODO: clean this 
        # if input_space == 'K_space': 
            
        #     if cluster_type == 'foreground':
        #         _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=True, input_space='K_space', return_cluster_types=True, verbose=False)
        #     else:
        #         _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=False, input_space='K_space', return_cluster_types=True, verbose=False)
        
        # elif input_space == 'G_space':
            
        #     if cluster_type == 'foreground':
        #        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=True, input_space='G_space', round_number=round_number, return_cluster_types=True, verbose=False)
        #     else:
        #        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=False, input_space='G_space',  round_number=round_number, return_cluster_types=True, verbose=False)
        
        # elif input_space == 'G_opt_space':
            
        #     if cluster_type == 'foreground':
        #        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=True, input_space='G_opt_space',  round_number=round_number, return_cluster_types=True, verbose=False)
        #     else:
        #        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=False, input_space='G_opt_space',  round_number=round_number, return_cluster_types=True, verbose=False)

        if timestamps_weighting_scheme == 'direct':
            #kde_model = fit_kde_per_cluster(df, cluster_timestamps, timestamps_weights, mapping_index_cluster_type, cluster_types=[cluster_type], gridsize=gridsize)
            kde_model = fit_kde_per_cluster(df, cluster_timestamps, timestamps_weights, cluster_types=[cluster_type], gridsize=gridsize)
        elif timestamps_weighting_scheme == 'inverse_exp':            
            #kde_model = fit_kde_per_cluster(df, cluster_timestamps, timestamps_weights_inverse, mapping_index_cluster_type, cluster_types=[cluster_type], gridsize=gridsize)
            kde_model = fit_kde_per_cluster(df, cluster_timestamps, timestamps_weights_inverse, cluster_types=[cluster_type], gridsize=gridsize)
        kde_model = sort_kde_by_mean_timestamp(kde_model)
        kde_models[cluster_type] =  kde_model
        
    #cluster_order = list(kde_model.keys())
    #print(f'{annotator_id}-round: {round_number}\nFitted KDE models in {time.time() - t0:.2f} seconds')
    #print('--------------------------------------------------')
    if verbose:
        for cluster_type, kde_model in kde_models.items():
            green(f'Cluster type: {cluster_type} has {len(kde_model)} fitted KDE models')
        
    return kde_models


CLUSTERS = None
TIMESTAMPS = None
WEIGHTS = None
def init_worker(clusters, timestamps, weights):
    global CLUSTERS, TIMESTAMPS, WEIGHTS
    CLUSTERS = clusters
    TIMESTAMPS = timestamps
    WEIGHTS = weights
    
def pairwise_wasserstein_distance_mp(indices):
    i, j = indices

    t1 = np.asarray(TIMESTAMPS[CLUSTERS[i]], dtype=np.float64).ravel()
    w1 = np.asarray(WEIGHTS[CLUSTERS[i]], dtype=np.float64).ravel()
    t2 = np.asarray(TIMESTAMPS[CLUSTERS[j]], dtype=np.float64).ravel()
    w2 = np.asarray(WEIGHTS[CLUSTERS[j]], dtype=np.float64).ravel()

    # Normalize weights
    w1 /= w1.sum() + 1e-12
    w2 /= w2.sum() + 1e-12

    # Sort timestamps for EMD-1D
    idx1 = np.argsort(t1)
    idx2 = np.argsort(t2)
    t1, w1 = t1[idx1], w1[idx1]
    t2, w2 = t2[idx2], w2[idx2]

    # Compute W1 distance using POT
    w1_dist = ot.emd2_1d(t1, t2, a=w1, b=w2, metric='sqeuclidean', p=1.0)
    
    return i, j, w1_dist


def compute_wasserstein_distance(config_name, cluster_type, df=None, annotator_id='samperochon', round_number=0, input_space='K_space', embedding_labels_col=None, segments_labels_col=None, temporal_segmentations_col=None,  verbose=False, n_processes=5):
    
    from smartflat.features.symbolization.co_clustering import get_prototypes_mapping
    


    def multiprocessing_distance_computation(clusters, cluster_timestamps_ct, timestamps_weights_ct, n_processes=30):
        K = len(clusters)
        D = np.zeros((K, K))

        index_pairs = [(i, j) for i in range(K) for j in range(i, K)]

        with mp.Pool(
            processes=n_processes or mp.cpu_count(),
            initializer=init_worker,
            initargs=(clusters, cluster_timestamps_ct, timestamps_weights_ct)
        ) as pool:
            for i, j, dist in tqdm(pool.imap_unordered(pairwise_wasserstein_distance_mp, index_pairs), total=len(index_pairs)):                
                D[i, j] = D[j, i] = dist

        np.fill_diagonal(D, 0)
        D[np.isnan(D)] = np.inf
        return D


    def pairwise_wasserstein_distance(i, j, return_transport=False):
        """
        Compute Wasserstein-1 distance between cluster i and j using POT's
        1D specialized solvers: emd2_1d (distance) and optionally emd_1d (transport map).
        """
        # Extract and normalize timestamps & weights
        t1 = np.asarray(cluster_timestamps_ct[clusters[i]], dtype=np.float64).ravel()
        w1 = np.asarray(timestamps_weights_ct[clusters[i]], dtype=np.float64).ravel()
        t2 = np.asarray(cluster_timestamps_ct[clusters[j]], dtype=np.float64).ravel()
        w2 = np.asarray(timestamps_weights_ct[clusters[j]], dtype=np.float64).ravel()

        w1 /= w1.sum() + 1e-12
        w2 /= w2.sum() + 1e-12

        # Sort timestamps
        idx1 = np.argsort(t1)
        idx2 = np.argsort(t2)

        xa, a = t1[idx1], w1[idx1]
        xb, b = t2[idx2], w2[idx2]

        # Compute Wasserstein-1 distance
        w1_dist = ot.emd2_1d(xa, xb, a=a, b=b, metric='sqeuclidean')
        
        if return_transport:
            T = ot.emd_1d(t1, t2, a=w1, b=w2)
            return (w1_dist, T)
        else:
            return w1_dist
    
                
    config =  import_config(config_name)
    gridsize = config.gridsize
    use_segments_timetamps = config.use_segments_timetamps
    timestamps_weighting_scheme = config.timestamps_weighting_scheme
    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    if df is None:
        df = get_experiments_dataframe(experiment_config_name=config_name,  annotator_id=annotator_id, round_number=round_number, return_symbolization=True) 

    if embedding_labels_col is None:
        if input_space == 'K_space': 
            print('Using inference-based K prototypes (filtered_raw_embedding_labels)')
            embedding_labels_col = 'filtered_raw_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'cpts' # 'symb_cpts' # 'processed_symb_cpts
            
        elif input_space == 'G_space': 
            print('Using reduced G prototypes')
            embedding_labels_col = 'opt_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'opt_segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'opt_cpts' # 'symb_cpts' # 'processed_symb_cpts
        
        elif input_space == 'G_opt_space': 
            print('Using doubled reduced G prototypes')
            embedding_labels_col = 'G_opt_embedding_labels' #'symb_labels' #'processed_symb_labels'
            segments_labels_col = 'G_opt_segments_labels' # 'symb_segments_labels' # 'processed_symb_segments_labels'
            temporal_segmentations_col = 'G_opt_cpts' # 'symb_cpts' # 'processed_symb_cpts

    assert df[embedding_labels_col].apply(len).sum() == df['N'].sum(), f"Number of timestamps in the {embedding_labels_col} does not match the number of embedding in the video duration length column {'N'}"

    n_lost = df[embedding_labels_col].apply(len).sum() - df['N_raw'].sum()
    #print(f'Number of dropped (edges) embedding: N={n_lost} - remaining N={df[embedding_labels_col].apply(len).sum()}')
    #print('Supports of embedding and segments labels:')
    #print((embedding_labels_col, segments_labels_col), len(np.unique(np.hstack(df[embedding_labels_col]))), len(np.unique(np.hstack(df[segments_labels_col]))))


    t0 = time.time()    
    #print(f'Preparing timestamps datasets with columns {embedding_labels_col}, {segments_labels_col}, {temporal_segmentations_col}')
    # Create for each row the dictionnary keyed by segments_labels and values the list of timestamps normalized per the actual sequence length
    if use_segments_timetamps:
        # Use the midle-timestamps of the segments to compute the centroids KDE
        df['cpts_percentiles'] = df.apply(lambda x: [cpts / x['N']  for cpts in x[temporal_segmentations_col]], axis=1)
        df['segments_timestamps_percentiles'] = df.cpts_percentiles.apply(lambda x: [(s + e) / 2 for s, e in pairwise(x)])
        df['cluster_timetamps_list'] = df.apply(lambda row:  {cv: list(np.array(row['segments_timestamps_percentiles'])[np.argwhere(row[segments_labels_col] == cv).flatten()]) for cv in np.unique(row[segments_labels_col])}, axis=1)
    else:
        # Create for each row the dictionnary keyed by segments_labels and values the list of timestamps normalized per the actual sequence length
        df['cluster_timetamps_list'] = df.apply(lambda row: {cv: np.argwhere(row[embedding_labels_col] == cv).flatten() / row['N'] for cv in np.unique(row[embedding_labels_col])}, axis=1)

    n_df=0
    for i, row in df.iterrows():
        t_dict = row.cluster_timetamps_list
        for cl_index, timestamps in t_dict.items():
            n_df+=len(timestamps)
    green(f"{annotator_id}-round: {round_number}.  Total number of timestamps:  {n_df} (N_raw={df.N_raw.sum()}, N={df.N.sum()} -> Keep={df.N.sum() / df.N_raw.sum():.2f} %")

    cluster_timestamps, timestamps_weights, timestamps_weights_inverse, cluster_prevalence = prepare_row_timestamps_dataset(df)
        
    # if use_K_space:
        
    #     if cluster_type == 'foreground':
    #         _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=True, return_cluster_types=True, verbose=False)
    #     else:
    #         _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=False, return_cluster_types=True, verbose=False)
    
    # else:
    #     if cluster_type == 'foreground':
    #         mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=True,  round_number=round_number, return_cluster_types=True, verbose=False)
    #     else:
    #         mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, is_foreground=False,  round_number=round_number, return_cluster_types=True, verbose=False)
    
    #cluster_timestamps_ct = {cluster: timestamps for cluster, timestamps in cluster_timestamps.items() if (mapping_index_cluster_type[cluster] == cluster_type) & (len(timestamps) > 3)}
    cluster_timestamps_ct = {cluster: timestamps for cluster, timestamps in cluster_timestamps.items() if  (len(timestamps) > 3)}

    
    if timestamps_weighting_scheme == 'direct':
        timestamps_weights_ct = {cluster: timestamps_weights[cluster] for cluster in cluster_timestamps_ct.keys()}
    elif timestamps_weighting_scheme == 'inverse_exp':            
        timestamps_weights_ct = {cluster: timestamps_weights_inverse[cluster] for cluster in cluster_timestamps_ct.keys()}
        
    clusters = sorted(list(cluster_timestamps_ct.keys()))
    if -1 in clusters:
        clusters.remove(-1)  # Remove noise cluster if present
    #clusters = sorted([c for c in clusters if mapping_index_cluster_type[c] ==
    print(f'Computing Wasserstein distance and transport maps for {len(clusters)} clusters')
    
    
    # Test to visualize distribution and transport maps
    n_pairs = 5
    pairs = [(i, j) for i in range(len(clusters)) for j in range(i, len(clusters))]
    sampled_pairs = random.sample(pairs, n_pairs)

    # Plot results for sampled pairs
    for i, j in sampled_pairs:


        dist, T = pairwise_wasserstein_distance(i, j, return_transport=True)


        print(f'Wasserstein distance between clusters {clusters[i]} and {clusters[j]}: {dist:.2e}')
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))

        # Normalize weights
        xi = np.array(cluster_timestamps_ct[clusters[i]])
        xj = np.array(cluster_timestamps_ct[clusters[j]])
        wi = np.array(timestamps_weights_ct[clusters[i]])
        wj = np.array(timestamps_weights_ct[clusters[j]])
                    

        # Plot histogram for Cluster i
        axs[0].hist(xi, bins=np.arange(0, 1, 0.01), weights=wi, alpha=0.8, label=f'Cluster {clusters[i]}', color='tab:blue')
        axs[0].set_title(f'Cluster {clusters[i]}')
        axs[0].set_xlabel("Normalized timestamp")
        axs[0].set_ylabel("Density")
        axs[0].legend()

        # Plot histogram for Cluster j
        axs[1].hist(xj, bins=np.arange(0, 1, 0.01), weights=wj, alpha=0.8, label=f'Cluster {clusters[j]}', color='tab:brown')
        axs[1].set_title(f'Cluster {clusters[j]}')
        axs[1].set_xlabel("Normalized timestamp")
        axs[1].legend()

                    
        # Transport map
        im = axs[2].imshow(T, cmap='Blues', origin='lower', aspect='auto', interpolation='nearest')
        axs[2].set_title('Transport Plan')
        axs[2].set_xlabel(f'Cluster {clusters[j]} bins')
        axs[2].set_ylabel(f'Cluster {clusters[i]} bins')
        axs[2].set_facecolor('white')
        fig.colorbar(im, ax=axs[2], label='Transport Mass')


        plt.suptitle(f"Wasserstein (Earth Mover's) Distance = {int(dist*100)}", fontsize=14)
        plt.tight_layout()
        plt.show()
        

    D = multiprocessing_distance_computation(clusters, cluster_timestamps_ct, timestamps_weights_ct, n_processes=n_processes)
    print(f'Number of nans in the distance matrix: {np.sum(np.isnan(D))}')
    D[np.isnan(D)] = np.inf
    np.fill_diagonal(D, 0)
    
    
    if verbose:
        print(f'Wasserstein distance computed in {time.time() - t0:.2f} seconds')
        plt.figure(figsize=(15, 8))
        plt.imshow(D, cmap='coolwarm', aspect='auto')
        plt.title(f'Wasserstein Distance Matrix for {cluster_type} clusters (K={len(clusters)})')
    
    return D


class DiscreteKDE(rv_continuous):
    def __init__(self, support, density):
        super().__init__(a=0, b=1)
        self.support_ = support  
        self.density_ = density
        area = simps(density, support)
        #print(f"[DiscreteKDE] PDF integral over support: {area:.4f}")
        self._pdf_interp = interp1d(support, density, bounds_error=False, fill_value=0)
        self._cdf_interp = interp1d(support, np.cumsum(density) / np.sum(density), bounds_error=False, fill_value=(0,1))

    def _pdf(self, x):
        return self._pdf_interp(x)

    def _cdf(self, x):
        return self._cdf_interp(x)
    
    

def fit_kde_per_cluster(df, cluster_timestamps, timestamps_weights, mapping_index_cluster_type=None, cluster_types=None, gridsize=64):
    """Fit KDE models for each cluster using the provided timestamps and weights.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        cluster_timestamps (dict): A dictionary containing the timestamps for each cluster.
        timestamps_weights (dict): A dictionary containing the weights for each cluster.
        gridsize (int): The number of points to use for the KDE support.
    Returns:
        kde_model (dict): A dictionary containing the KDE model for each cluster.
    """
    
    # Fit KDE models for each cluster
    kde_model = {}
    #print('/!\ once the opt labels are offset to 0 (not currently 1), we can remove the cluster-1')
    #print(f'Mapping meta-cluster: cluster type: {list(mapping_index_cluster_type.keys())}')
    for cluster, timestamps in cluster_timestamps.items():
        if len(timestamps) < 2:
            continue
        
        #if cluster_types is not None and mapping_index_cluster_type[cluster-1] not in cluster_types:
        
        if mapping_index_cluster_type is not None:
            if cluster_types is not None and mapping_index_cluster_type[cluster] not in cluster_types:

                continue
        
        # TODO: once the opt labels are offset to 0 (not currently 1), we can remove this
        
        
        weights = np.array(timestamps_weights[cluster])
        timestamps = np.array(timestamps)
        #print(f'Fitting KDE for cluster {cluster} - N={len(timestamps)} - weights sum={np.sum(weights):.2f} - min={np.min(timestamps):.2f} - max={np.max(timestamps):.2f}')
        kde = KDEUnivariate(timestamps)
        
        #plt.hist(timestamps, bins=20)
        #print(np.min(timestamps), np.max(timestamps))
        
        #kde.fit(weights=weights, fft=True, bw='silverman')  # Or specify bandwidth manually

        kde.fit(weights=weights, fft=False, bw='silverman', kernel='gau', gridsize=gridsize, cut=3, clip=(-0.5, 1.5))
        
        #mask = (kde.support >= 0) & (kde.support <= 1)
        kde.support = kde.support#[mask]
        kde.density = kde.density#[mask]
        #print(f'Mean mask: {np.mean(mask)}')
        #print(f'Shape of kde.support: {kde.support.shape} - {kde.support.min()} - {kde.support.max()}')
        #print(f'Shape of kde.density: {kde.density.shape} - {kde.density.min()} - {kde.density.max()}')
        dist = DiscreteKDE(kde.support, kde.density)
        
        kde_model[cluster] = dist

    kde_model = sort_kde_by_mean_timestamp(kde_model)

    return kde_model

# def sort_kde_by_mean_timestamp(kde_model):
#     sorted_items = sorted(
#         kde_model.items(),
#         key=lambda kv: np.average(kv[1].support, weights=kv[1].density)
#     )
#     return {k: v for k, v in sorted_items}

# def sort_kde_by_mean_timestamp(dist_model, by='mean'):
#     """Sort the KDE models by their estimated mean timestamp."""
    
#     if by == 'mean':
#         sorted_items = sorted(
#             dist_model.items(),
#             key=lambda kv: simps(kv[1]._pdf_interp(kv[1].support_) * kv[1].support_, kv[1].support_)
#         )
#     elif by == 'std':
#         sorted_items = 

#     return {k: v for k, v in sorted_items}

def sort_kde_by_mean_timestamp(dist_model, by='mean'):
    """Sort the KDE models by their estimated mean or std timestamp."""
    
    def kde_mean(kde):
        pdf_vals = kde._pdf_interp(kde.support_)
        return simps(pdf_vals * kde.support_, kde.support_)

    def kde_std(kde, mean):
        pdf_vals = kde._pdf_interp(kde.support_)
        var = simps(((kde.support_ - mean) ** 2) * pdf_vals, kde.support_)
        return np.sqrt(var)

    if by == 'mean':
        sorted_items = sorted(
            dist_model.items(),
            key=lambda kv: kde_mean(kv[1])
        )
    elif by == 'std':
        sorted_items = sorted(
            dist_model.items(),
            key=lambda kv: kde_std(kv[1], kde_mean(kv[1]))
        )
    else:
        raise ValueError("`by` must be 'mean' or 'std'")

    return {k: v for k, v in sorted_items}

def infer_kde_support(kde):
    """Infer the support and density of the KDE model."""
    
    # Use the support and density from the KDE model (Gromov-Wasserstein being robust to non-overlapping support)
    x = kde.support_
    y = kde.pdf(x)
    y /= (simps(y, x) + 1e-7)  # Normalize

    return x, y

# 2) Gromov-Wasserstein (GW) distance computation between the KDEs of different clusters

def compute_gw_costs_and_transport_maps(kde_model, compute_similarity=False, gridsize=64, temporal_distance='gw', kernel_name='linear', temperature_tau=0.1, n_clt=None, loss_name='kl_loss', pre_computed_D=None, ordered_index=True, verbose=False):
    """Compute the Gromov-Wasserstein distance and transport maps between clusters.
    Args:
        kde_model (dict): A dictionary containing the KDE model for each cluster.
        gridsize (int): The number of points to use for the KDE support.
        temporal_distance (str): The method to use for computing the temporal distance ('gw' or 'linear_gw').
        sign_eigs (bool): Whether to use signed eigenvalues for the linear GW transport.
        n_clt (int): The number of clusters to consider.
        loss_name (str): The loss function to use for the GW distance.
        ordered_index (bool): Whether to order the clusters by their index.
        verbose (bool): Whether to print verbose output.
    Returns:
        D (ndarray): The Gromov-Wasserstein distance matrix.
        transport_maps (dict): A dictionary containing the transport maps between clusters.
        logs (dict): A dictionary containing the logs of the GW distance computation.
    """
    
    # Compute the Gromov-Wasserstein distance and transport maps between clusters
    if ordered_index:
        clusters = sorted(list(kde_model.keys()))
    else:
        clusters = list(kde_model.keys())
    if n_clt is not None or verbose:
        clusters = clusters[:n_clt]
        
    
    def process_pair(i, j):
        ci, cj = clusters[i], clusters[j]
        Xs, ys = histograms[ci]
        Xt, yt = histograms[cj]
        if temporal_distance == 'gw':
            gw_dist, T, log = compute_gw_distance_and_transport_map(Xs, ys, Xt, yt, loss_name=loss_name)
        elif temporal_distance == 'linear_gw':
            gw_dist, T, log = compute_linear_gw_transport(Xs, ys, Xt, yt)
        else:
            raise ValueError(f"Unknown method: {temporal_distance}")
        
        return (i, j, gw_dist, T, log)

    
    if pre_computed_D is None:
        
        kernel_name = 'pre_computing'
        
        histograms = {c: infer_kde_support(kde_model[c]) for c in clusters}

        D = np.zeros((len(clusters), len(clusters)))
        transport_maps = {}
        logs = {}
        print(f'Computing Gromov-Wasserstein distance and transport maps for {len(clusters)} clusters')
        pairs = [(i, j) for i in range(len(clusters)) for j in range(i, len(clusters))]
        results = Parallel(n_jobs=-1)(delayed(process_pair)(i, j) for i, j in tqdm(pairs, total=len(pairs)))
                
        for i, j, gw_dist, T, log in results:
            
                        
            if verbose and (i == 0) and (j==1):
                
                ci, cj = clusters[i], clusters[j]
                Xs, ys = histograms[ci]
                Xt, yt = histograms[cj]
                        
                cost = log['cost']
                last_loss = log['loss'][-1]
                
                fig, axs = plt.subplots(1, 3, figsize=(18, 4))
                axs[0].plot(Xs, ys, label=f'Cluster {ci}')
                axs[1].plot(Xt, yt, label=f'Cluster {cj}')
                
                im = axs[2].imshow(T, aspect='auto', cmap='coolwarm', origin='lower')
                fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

                axs[0].set_title(f'Cluster {ci}')
                axs[1].set_title(f'Cluster {cj}')
                axs[2].set_title('Transport Map')
                
                axs[0].legend()
                axs[1].legend()
                
                plt.suptitle(f'GW² Distance = {gw_dist:.2e} Cost = {cost:.2e} Loss[-1] = {last_loss:.2e}\nloss={loss_name}', fontsize=14)
                plt.tight_layout()
                plt.show()
                
            
            D[i, j] = D[j, i] = np.sqrt(gw_dist)
            ci, cj = clusters[i], clusters[j]
            transport_maps[(ci, cj)] = csr_matrix(T)
            logs[(ci, cj)] = log

        D[np.isnan(D)] = np.inf
        np.fill_diagonal(D, 0)
            
            
    else:
        D = pre_computed_D.copy()
        transport_maps = np.nan
        logs = np.nan
          
    if kernel_name == 'pre_computing':
        pass
        
    elif kernel_name in ['cosine', 'euclidean', 'linear']:
        D = np.sqrt(D)
        
    elif kernel_name == 'gaussian_rbf':

        sigma_rbf = np.median(D[~np.eye(D.shape[0], dtype=bool)])
        #print(f'Temporal RBF kernel sigma (estimated median): {sigma_rbf:.4f} - {temperature_tau:.4f}')      
        D = np.exp( - D / (2 * temperature_tau * sigma_rbf**2))
        
        if not compute_similarity:
            # Step 2: convert similarity to distance
            # Note: D is a similarity matrix, we need to convert it to a distance matrix
            D = 1 - D
            #D = (D - D.min()) / (D.max() - D.min())            
            np.fill_diagonal(D, 0)    
            print('RBF: Converted similarity to distance matrix')  
        # Step 3: recover approximate squared distances using inverse of kernel formula
        # Note: clip for numerical stability (log(0) → -inf)
        #condensed_distances = np.sqrt(-2 * sigma_rbf**2 * np.log(np.clip(K, 1e-10, 1)))
    
    else: 
        raise ValueError(f"Unknown kernel name: {kernel_name}")
    
    return D, transport_maps, logs


def compute_linear_gw_transport(Xs, ys, Xt, yt, sign_eigs=None):
    """Compute the linear Gromov-Wasserstein transport map between two distributions."""
    
    linear_gw = ot.da.LinearGWTransport(log=True, sign_eigs=sign_eigs)
    linear_gw.fit(Xs=Xs.reshape(-1, 1), ys=ys,  Xt=Xt.reshape(-1, 1), yt=yt)
    est_Xt = linear_gw.transform(Xs.reshape(-1, 1))
    
    lin_gw_error = np.sqrt(np.linalg.norm(est_Xt - Xt.reshape(-1, 1), ord=2))
    #print('A, B, A1:', linear_gw.A_, linear_gw.B_, linear_gw.A1_, )
    return lin_gw_error, est_Xt, linear_gw.log_

def compute_gw_distance_and_transport_map(Xs, ys, Xt, yt, loss_name='kl_loss'):
    """Compute the Gromov-Wasserstein distance and transport map between two distributions."""
    # Compute the cost matrices
    #C1 = ot.utils.dist(Xs.reshape(-1, 1), Xs.reshape(-1, 1))
    C1 = ot.utils.dist(Xs.reshape(-1, 1))
    C2 = ot.utils.dist(Xt.reshape(-1, 1))
    p = ys / np.sum(ys)
    q = yt / np.sum(yt)
    
    assert np.isfinite(p).all() and np.isfinite(q).all(), "p or q contains NaNs"
    assert np.all(p >= 0) and np.all(q >= 0), "p or q contains negatives"

    T = ot.gromov.gromov_wasserstein(C1, C2, p, q, loss_name, verbose=False)
    gw_dist, log = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_name, log=True, verbose=False)
    
    return gw_dist, T, log



# 3) Visualization of the KDEs and GW distances

def plot_cluster_timestamps(mdf, cluster_timestamps, timestamps_weights, cluster_prevalence, n_clusters, unique_clusters=None, figpath=None, n_cols=3):

    fig, axes = plt.subplots(n_clusters // n_cols + (n_clusters % n_cols > 0), n_cols, figsize=(25, n_clusters // n_cols * 3 + 3))
    axes = axes.flatten()

    if unique_clusters is None:
        unique_clusters = mdf['cluster_timetamps_list'].apply(lambda x: list(x.keys())).explode().unique()
    
    for i, cluster_value in enumerate(unique_clusters):
        if i >= len(axes):  # Skip excess subplots
            break

        ax = axes[i]
        timestamps = cluster_timestamps[cluster_value]
        weights = timestamps_weights[cluster_value]

        if timestamps:  # Plot only if there is data
            scatter = ax.scatter(
                np.arange(len(timestamps)), timestamps, 
                #c=weights,
                cmap='viridis', s=20, alpha=0.8
            )
            # Add colorbar for weights
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Weight')
        
    
        ax.set_title(
                f'Cluster {cluster_value}\nSamples: {len(timestamps)}, '
                #f'Weight Sum: {np.sum(weights):.2f}'
                f'Prev: {cluster_prevalence[cluster_value]:.2f}%'
        )
        ax.set_xlabel('Index')
        ax.set_ylabel('Timestamps')

    # Hide excess subplots
    for ax in axes[n_clusters:]:
        ax.axis('off')

    plt.tight_layout()
    #plt.suptitle("Aggregated Timestamps and Weights per Cluster", y=1.02)
    plt.suptitle(f"Weighted timestamps per prototype index\n{mdf.task_name.iloc[0]} {mdf.modality.iloc[0]} N rows:{len(mdf)} - K={n_clusters}", y=1.02)

    if figpath is not None:
        plt.savefig(figpath, dpi=90, bbox_inches='tight')
    plt.show()

def plot_gw_temporal_distance(D, title='Gromov-Wasserstein Distance Matrix'):
    
    plt.figure(figsize=(20, 5))
    plt.hist(D.flatten(), bins=256, color='tab:blue', alpha=0.7)
    plt.title('Histogram\n'+title)
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(D, cmap='viridis')
    plt.title('Histogram\n'+title)
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_clusters_kde_from_model(mdf, kde_model, cluster_timestamps, cluster_prevalence, n_clusters, title='', figpath=None, n_cols=6, mapping_name=None):
    
    unique_clusters = list(kde_model.keys())[:n_clusters]
    
    edges_color = (0.88, 0.93, 0.97, 1)  # Pale Blue
    noise_color = (0, 0, 0, 1) # Black

    colors = get_base_colors(n_colors=n_clusters, verbose=False)

    color_space =unique_clusters
    color_mapping = {label: colors[i] for i, label in enumerate(color_space)}
    color_mapping[-1] = noise_color       # Noise            
    cmap = ListedColormap([color_mapping[label] for label in color_space])
    norm = plt.Normalize(vmin=np.min(color_space), vmax=np.max(color_space))
    
    fig, axes = plt.subplots(n_clusters // n_cols-1 , n_cols, figsize=(30, n_clusters // n_cols * 3 + 3))
    axes = axes.flatten()

    #unique_clusters = mdf['cluster_timetamps_list'].apply(lambda x: list(x.keys())).explode().unique()
    
    for i, cluster_value in enumerate(unique_clusters):
        if i >= len(axes):  # Skip excess subplots
            break
                
        ax = axes[i]
        kde = kde_model.get(cluster_value)
        timestamps = cluster_timestamps[cluster_value]
        if mapping_name is not None:
            _cluster_value = mapping_name.get(cluster_value, cluster_value)
        else:
            _cluster_value = cluster_value
        if kde is not None:
            plot_kde_from_model(ax, kde, _cluster_value, timestamps, cluster_prevalence[cluster_value], color=cmap(norm(cluster_value)), bins=100)
        else:
            ax.set_title(f'K={cluster_value} - N=0')
            ax.axis('off')
            
    for ax in axes[n_clusters:]:
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Kernel density estimate of normalized temporal distribution (labelled clusters)", y=1.02, weight='bold', fontsize=14)
    
    if figpath is not None:
        plt.savefig(figpath, dpi=90, bbox_inches='tight')
    plt.show()

def plot_kde_from_model(ax, kde, cluster_value, timestamps, cluster_prevalence, color='tab:blue', bins=100):
    x = kde.support_
    y = kde.density_

    # Histogram
    hist_y, bin_edges = np.histogram(timestamps, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Interpolate KDE on histogram bin centers
    kde_interp = np.interp(bin_centers, x, y)
    r2 = 1 - np.sum((hist_y - kde_interp)**2) / np.sum((hist_y - np.mean(hist_y))**2 + 1e-8)

    # Plot
    ax.plot(x, y, label='KDE', color=color)
    ax.hist(timestamps, bins=bins, density=True, alpha=0.4, color=color, label='Weighted Histogram')
    ax.fill_between(x, y, alpha=0.2, color=color)

    ax.set_title(
        f'Cluster: {cluster_value}\nSubject representation: {cluster_prevalence:.1f}%'
        f'\nNum samples: {len(timestamps)}',
        #f'R²: {r2:.3f}', 
        fontsize=10,
       # weight='bold'
    )
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)


def visualize_paiwise_gw(kde_model, T, idx1, idx2, gridsize=128, title='Gromov-Wasserstein Distance'):
    kde1 = kde_model[idx1]
    kde2 = kde_model[idx2]

    Xs, ys = infer_kde_support(kde1, gridsize=gridsize)
    Xt, yt = infer_kde_support(kde2, gridsize=gridsize)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(Xs, ys, label=f'Cluster {idx1}')
    axs[1].plot(Xt, yt, label=f'Cluster {idx2}')
    axs[2].imshow(T, cmap='viridis')
    axs[0].set_title(f'Cluster {idx1}')
    axs[1].set_title(f'Cluster {idx2}')
    axs[2].set_title('Transport Map')
    axs[0].legend(); axs[1].legend()
    plt.suptitle(f'{title}', fontsize=14)
    plt.show()

    return 



def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()
    
    
if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    #print('Process registration by loading vector labels and timestamps, meta-prototypes basis, radius per clusters, noise encoding,   using config_name: {}'.format(args.config_name))

    t1 = time.time()
    print(f'Time elapsed: {t1 - t0} seconds')
    
    sys.exit(0)
    