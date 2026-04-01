"""Hierarchical agglomerative clustering (HAC) for prototype consolidation.

Implements the HAC consolidation step (Ch. 5, Section 5.3): combines
latent-space cosine distance with Wasserstein temporal distance to cluster
K accumulated centroids into G final prototypes (vocabulary size ~55).
Silhouette score maximization selects optimal G.

Supports multiple linkage methods (Ward, complete, average) and
multi-modal distance kernels (multiplicative, additive).

Prerequisites:
    - Recursive prototyping completed (K source centroids accumulated).
    - Temporal distributions estimated via
      ``temporal_distributions_estimation``.

Main entry points:
    - ``compute_multimodal_matrices()``: Build multi-view distance matrices.
    - ``clustering_prototypes_space()``: Run HAC with silhouette optimization.
    - ``run_comparison_clusterings()``: Compare HAC vs alternatives (K-Means,
      DBSCAN, HDBSCAN, GMM, Spectral).

External dependencies:
    - fastcluster (efficient HAC linkage computation)
    - hdbscan (density-based clustering alternative)
"""

import argparse
import json
import os
import sys
import time

import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from hdbscan import HDBSCAN
from IPython.display import display
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from tqdm import tqdm

from smartflat.configs.loader import import_config
from smartflat.engine.builders import build_metrics, compute_metrics
from smartflat.features.symbolization.temporal_distributions_estimation import (
    compute_gw_costs_and_transport_maps,
    compute_wasserstein_distance,
    estimate_kde_model,
    plot_gw_temporal_distance,
)
from smartflat.features.symbolization.utils import (
    check_pairwise_distance_matrix,
    get_prototypes,
    reduce_centroid_vectors_per_cluster_type,
    reduce_similarity_matrix_per_cluster_type,
)
from smartflat.features.symbolization.visualization import (
    compute_inconsistency_across_depth,
    compute_inconsistency_across_thresholds,
    explore_inconsistency_depth_threshold,
)
from smartflat.metrics import plot_clustering_metrics
from smartflat.utils.utils_coding import green, purple, yellow
from smartflat.utils.utils_io import (
    fetch_qualification_mapping,
    get_data_root,
    load_df,
    save_df,
)


def compute_multimodal_matrices(config_name, annotator_id='samperochon', round_number=0, input_space='K_space', verbose=False):
    
    # Reduce prototypes redundancy as per the silhouette score (CS)
    config = import_config(config_name)
    gridsize = config.gridsize
    loss_name = config.loss_name
    temporal_distance = config.temporal_distance 
    multimodal_method = config.multimodal_method
    linkage_method = config.linkage_method
    alpha = config.alpha
    temperature_tau = config.temperature_tau; temperature_tau_K = temperature_tau['K_space'];temperature_tau_G = temperature_tau['G_space']
    temperature_x = config.temperature_x; temperature_x_K = temperature_x['K_space'];temperature_x_G = temperature_x['G_space']
    agg_fn_str = config.agg_fn_str #TODO: test with np.max
    agg_fn = np.mean if agg_fn_str == 'mean' else np.max if agg_fn_str == 'max' else np.median if agg_fn_str == 'median' else id

    distance_aggregate_col = config.distance_aggregate_col


    kernel_name = 'gaussian_rbf'#config.model_params['kernel_name']
    sigma_rbf = config.model_params['sigma_rbf']
    temperature = config.model_params['temperature']
    compute_similarity = True if kernel_name == 'gaussian_rbf' else False
    # if kernel_name == 'cosine':
    #     normalization = 'l2'
    # elif kernel_name in ['gaussian_rbf', 'euclidean']:
    #     normalization = 'Z-score'

    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(config.clustering_config_name)
    cluster_types = clustering_config.model_params['cluster_types']
    normalization = clustering_config.model_params['normalization']
    agg_fn = np.mean


    # Get prototypes mapping 
    qualification_mapping = fetch_qualification_mapping(verbose=False)
    cluster_types_size = {cluster_type: len(v) for cluster_type, v in qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'].items()}


    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(experiments_folder, exist_ok=True)    

    title = (f"D-Deployment kernel_name={kernel_name}, multimodal_method={multimodal_method}, normalization={normalization}, multimodal_method={multimodal_method}\n" + 
            f"temporal_distance={temporal_distance} (n_bins={gridsize}, loss_name={loss_name}, alpha={alpha}")


    green("--------------------------------------------------")

    green(title)
    green("--------------------------------------------------")


    # qualification_mapping = fetch_qualification_mapping()
    # results, best_df, mapping_prototypes_reduction, mapping_index_cluster_type, qualification_mapping = clustering_prototypes_space(clustering_config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', qualification_mapping=qualification_mapping)
    # clusterdf = build_clusterdf(df, annotator_id=annotator_id, round_number=round_number, qualification_mapping=qualification_mapping, embeddings_label_col='symb_labels', segments_labels_col='symb_segments_labels', segments_length_col='symb_segments_length')
    # opt_mapping_prototypes_df =  pd.DataFrame(mapping_prototypes_reduction, index=['projected_cluster_index']).transpose().reset_index(names=['original_cluster_index'])
    # mapping_prototypes_reduction_list = opt_mapping_prototypes_df.groupby(['projected_cluster_index']).original_cluster_index.agg(list).to_dict()#.apply(len)#.hist(bins=10)
    # mapping_prototypes_reduction_list[-2] = exluded_clusters
    # mapping_prototypes_reduction_list[-1] = []
    # clusterdf['source_index'] = clusterdf.cluster_index.map(mapping_prototypes_reduction_list)

    # 1) Estimate KDE model run comparison  between the expected prototypes index and the present data
    # 309: (288 task (2 missing), 20 exo (0 missing), then the 40 noise clusters are reduced to a single -2 cluster, 
    # and there is the -1 cluster for roughly misfitted data, 
    # so 288 + 20 + 1 + 1 = 310 expected clusters in total
    kde_models = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space)
    excluded_clusters = fetch_missing_temporal_prototypes(kde_models, annotator_id, round_number, config_name, clustering_config_name, use_K_space=True if input_space == 'K_space' else False)

    # 1) Compute temporal distance matrices across prototypes in the latent space before/after reduction per cluster types
    D_tf_pc = compute_temporal_distance(kde_models, annotator_id, round_number, config_name=config_name, kernel_name='pre_computing', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=None, overwrite_gw_distances=False)
    print('cdnjcdn', D_tf_pc.shape)
    print('fdfd', kde_models.keys())# 2) Compute spatiotemporal distance matrices across prototypes in the latent space per cluster types
    D_xf, D_tf, D_f  = compute_combined_distance(kde_models, 
                                                annotator_id, 
                                                round_number, 
                                                excluded_clusters, 
                                                config_name, 
                                                input_space=input_space, 
                                                    # D_te_pc=D_te_pc, 
                                                    # D_tt_pc=D_tt_pc,    
                                                    D_tf_pc=D_tf_pc,
                                                normalization=normalization, 
                                                kernel_name=kernel_name, 
                                                agg_fn=agg_fn, 
                                                gridsize=gridsize, 
                                                loss_name=loss_name, 
                                                temporal_distance=temporal_distance, 
                                                    multimodal_method=multimodal_method, 
                                                    alpha=alpha)

    # 3) Visulizations
    use_minmax = False


    distance_matrices = {
                        # K-latent space
                         "Foreground K-Latent Distance (D_xf)": max_normalize_distance(D_xf) if use_minmax else D_xf,
                        # "Task K-latent space (D_xt)": max_normalize_distance(D_xt) if use_minmax else D_xt,
                        # "Exo K-latent space (D_xe)": max_normalize_distance(D_xe) if use_minmax else D_xe,
                        
                        # "Foreground K-Temporal Distance\n(raw; D_tt_pc) ":  max_normalize_distance(D_tf) if use_minmax else D_tf,
                        # "Task K-Temporal Space\n(raw; D_tt_pc)": max_normalize_distance(D_tt_pc) if use_minmax else D_tt_pc,
                        # "Exo K-Temporal Space\n(raw; D_tf_pc)": max_normalize_distance(D_te_pc) if use_minmax else D_te_pc,
                        
                        "Foreground K-Temporal Distance\n(temp; D_tf) ":  max_normalize_distance(D_tf) if use_minmax else D_tf,
                        # "Task K-Temporal Distance\n(temp; D_tt) ":  max_normalize_distance(D_tt) if use_minmax else D_tt,
                        # "Exo K-Temporal Distance\n(temp; D_te) ":  max_normalize_distance(D_te) if use_minmax else D_te,
                        

                        f"Foreground K-Space Distance\n(space/time balance; D_f)": max_normalize_distance(D_f) if use_minmax else D_f,
                        # f"Task K-Space Distance\n(space/time balance; D_f)": max_normalize_distance(D_t) if use_minmax else D_t,
                        # f"Exo K-Space Distance\n(space/time balance; D_f)": max_normalize_distance(D_e) if use_minmax else D_e,
                        # f"Exo K-temporal Space (D_est_KEa) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KEa) if use_minmax else D_est_KEa,
                        # f"Task K-temporal Space (S_est_KT) {kernel_name}/{multimodal_method} (S)": max_normalize_distance(S_est_KT),


                        # # G-latent space
                        # "Task G-latent space (D_xtr)": max_normalize_distance(D_xtr),
                        # "Exo G-latent space (D_xer)": max_normalize_distance(D_xer),
                        # "Noise G-latent space (D_xnr)": max_normalize_distance(D_xnr),

                        # "Task G-temporal Space (D_ttr)": max_normalize_distance(D_t_est_GT),
                        # "Exo G-temporal Space (D_ter)": max_normalize_distance(D_t_est_GE),
                        # "placeholder_3": np.nan,

                        # f"Task G-temporal Space (D_est_GT) {kernel_name}/{multimodal_method}": max_normalize_distance(D_est_GT),
                        # f"Exo G-temporal Space (D_est_KE)  {kernel_name}/{multimodal_method}": max_normalize_distance(D_est_GE),
                        # f"Task G-temporal Space (S_est_KT) {kernel_name}/{multimodal_method}": max_normalize_distance(S_est_GT),
                    }

    
    
    
    # distance_matrices = {
    #     # K-latent space
    #     # "Task K-latent space (D_xt)": max_normalize_distance(D_xt),
    #     # "Exo K-latent space (D_xe)": max_normalize_distance(D_xe),
    #     # "Noise K-latent space (D_xn)": max_normalize_distance(D_xn),
        
    #     #"Task K-temporal Space (D_ttr)": max_normalize_distance(D_t_est_KT),
    #     #"Exo K-temporal Space (D_ter)": max_normalize_distance(D_t_est_KE),
    #     #"placeholder_2":  np.nan,
        
    #     #f"Task K-temporal Space (D_est_KT) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KT),
    #     #f"Exo K-temporal Space (D_est_KE) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KE),
    #     #f"Task K-temporal Space (S_est_KT) {kernel_name}/{multimodal_method} (S)": max_normalize_distance(S_est_KT),


    #     # G-latent space
    #     "Latent Space Distance - Foreground": max_normalize_distance(D_xf),
    #     "Latent Space Distance - Task": max_normalize_distance(D_xt),
    #     "Latent Space Distance - Exogeneous": max_normalize_distance(D_xe),
    #     "Latent Space Distance - Noise/Edges": max_normalize_distance(D_xn),

    #     "Temporal Space Distance - Foreground": max_normalize_distance(D_tf),
    #     "Temporal Space Distance - Task": max_normalize_distance(D_tt),
    #     "Temporal Space Distance - Exogeneous": max_normalize_distance(D_te),
    #     "placeholder_1": np.nan,

    #     f"Multimodal Space Distance - Foreground\n{kernel_name}/{multimodal_method}": max_normalize_distance(D_f),
    #     f"Multimodal Space Distance - Task\n{kernel_name}/{multimodal_method}": max_normalize_distance(D_t),
    #     f"Multimodal Space Distance - Exogeneous\n{kernel_name}/{multimodal_method}": max_normalize_distance(D_e),
    #     "placeholder_2": np.nan,
    #     #f"Task G-temporal Space (S_est_KT) {kernel_name}/{multimodal_method}": max_normalize_distance(S_est_GT),
    # }
    if verbose:
        from smartflat.features.symbolization.inference_reduction_prototypes_distances import (
            plot_distances_matrix_tuples,
        )
        plot_distances_matrix_tuples(distance_matrices, n_rows=4, n_cols=3, suptitle='Prototypes Distances Matrices per cluster-types\n'+title, figsize=(25, 15))
     
    return max_normalize_distance(D_xf), max_normalize_distance(D_tf),  max_normalize_distance(D_f)

def append_foreground_background_distance(D, method='mean', alpha=1.0):
    """
    Append a background-vs-all distance row and column to the distance matrix D.
    
    Parameters:
    - D: np.ndarray of shape (N, N), original distance matrix
    - method: str, one of ['min', 'mean', 'max', 'constant']
    - alpha: float, used only if method='constant'
    
    Returns:
    - D_aug: np.ndarray of shape (N+1, N+1)
    """
    if method == 'min':
        d = D.min(axis=1)
    elif method == 'mean':
        d = D.mean(axis=1)
    elif method == 'max':
        d = D.max(axis=1)
    elif method == 'constant':
        d = np.full(D.shape[0], alpha)
    else:
        raise ValueError(f"Unknown method '{method}'")
    if method != 'constant':
        print(f"Background-to-all vector ({method} cost per prototype):", d)
    
    D_aug = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
    D_aug[1:, 1:] = D
    D_aug[0, 1:] = d
    D_aug[1:, 0] = d
    return D_aug

def compute_combined_distance(kde_models, annotator_id, round_number, excluded_clusters, config_name,  input_space='K_space', D_tf_pc=None, normalization='l2', kernel_name='cosine', gridsize=128, loss_name='square_loss', temporal_distance='gw', multimodal_method='normalized_exp', alpha=.5, agg_fn=np.mean):
        
    config = import_config(config_name)
    temperature_x = config.temperature_x[input_space]
    temperature_tau = config.temperature_tau[input_space]
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    

    # 1) Compute temporal distance matrices across prototypes in the latent space per cluster types
    D_tf = compute_temporal_distance(kde_models, annotator_id, round_number, D_tf_pc=D_tf_pc,  config_name=config_name, kernel_name=kernel_name, loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=temperature_tau, overwrite_gw_distances=False)
     
    # 2) Compute latent distance matrices across prototypes in the latent space  per cluster types
    compute_similarity = True if kernel_name == 'gaussian_rbf' else False
    print('excluded_clusters', excluded_clusters)
    D_xf = compute_latent_distance(config_name, annotator_id, round_number, input_space=input_space, compute_similarity=compute_similarity, kernel_name=kernel_name, normalization=normalization, temperature_x=temperature_x, excluded_clusters=excluded_clusters, agg_fn=agg_fn)

    print(f'Shapes of matrices: D_xf={D_xf.shape}, D_tf={D_tf.shape}')
    # 3) Compute the mspatiotemporal distance matrices across prototypes in the latent space per cluster types
    #D_task = compute_distance_matrix(D_x_pc=D_xt, D_tau=D_tt, compute_similarity=False, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
    #D_exo = compute_distance_matrix(D_x_pc=D_xe, D_tau=D_te, compute_similarity=False, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
    D_foreground = compute_distance_matrix(D_x_pc=D_xf, D_tau=D_tf, compute_similarity=False, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
    #S = compute_distance_matrix(D_x_pc=D_x_pc, D_tau=D_tau, compute_similarity=True, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)

    # Post-processing: Normaization or inversion 
    # D_xt, D_tt, D_task = _post_process_multimodal_distance_matrix(D_xt, D_tt, D_task, kernel_name)
    # D_xe, D_te, D_exo = _post_process_multimodal_distance_matrix(D_xe, D_te, D_exo, kernel_name)
    # D_xf, D_tf, D_foreground = _post_process_multimodal_distance_matrix(D_xf, D_tf, D_foreground, kernel_name)

    # return D_xt, D_xe, D_xn, D_xf, D_tt, D_te, D_tf, D_task, D_exo, D_foreground

    #D_xt, D_tt, D_task = _post_process_multimodal_distance_matrix(D_xt, D_tt, D_task, kernel_name)
    #D_xe, D_te, D_exo = _post_process_multimodal_distance_matrix(D_xe, D_te, D_exo, kernel_name)
    D_xf, D_tf, D_foreground = _post_process_multimodal_distance_matrix(D_xf, D_tf, D_foreground, kernel_name)

    return D_xf, D_tf,  D_foreground





def _post_process_multimodal_distance_matrix(D_X, D_T, D_M, kernel_name):
    # comnpute similarity for each context
    if kernel_name == 'euclidean':
        S_X = 1 / (1 + D_X)
        S_X = (S_X - S_X.min()) / (S_X.max() - S_X.min())
        S_T = 1 / (1 + D_T)
        S_T = (S_T - S_T.min()) / (S_T.max() - S_T.min())
        
    elif kernel_name == 'cosine':
        S_X = 1 - D_X
        S_X = (S_X - S_X.min()) / (S_X.max() - S_X.min())
        S_T = 1 / (1 + D_T)
        S_T = (S_T - S_T.min()) / (S_T.max() - S_T.min())
        
    elif kernel_name == 'gaussian_rbf':        
        # Here for [G] spaces where the diagonal reflects within meta-cluster distances, we set
        # diagonal to 1 (D are transported as similarity matrices in the case of gaussian_rbf kernel)
        # np.fill_diagonal(D_M, 1)
        # np.fill_diagonal(D_X, 1)
        # np.fill_diagonal(D_T, 1)
        
        S_X = D_X.copy()
        S_T = D_T.copy()
        S_M = D_M.copy()
        
        D_X = 1 - D_X
        D_X = (D_X - D_X.min()) / (D_X.max() - D_X.min())
        D_T = 1 - D_T
        D_T = (D_T - D_T.min()) / (D_T.max() - D_T.min())
        
        D_M = 1 - D_M
        D_M = (D_M - D_M.min()) / (D_M.max() - D_M.min())


    #check_pairwise_distance_matrix(D_M)
    #check_pairwise_distance_matrix(D_X)
    #check_pairwise_distance_matrix(D_T)
    
    # plt.figure(figsize=(6, 5))
    # plt.imshow(D_M, cmap='coolwarm')
    # plt.title("D_M (Distance Matrix)")
    # plt.colorbar()
    # plt.show()

    
    return D_M, D_X, D_T
    
def compute_distance_matrix(X=None, D_x_pc=None, D_tau=None,  compute_similarity=False, multimodal_method='normalized_exp', kernel_name='cosine', temperature=1.,  alpha=0.5):
    
    
    # 1) We first compute the latent distance if not available (require sample X (centroids))
    if D_x_pc is None:
        
        assert X is not None
            
        # 1. Compute the pairwise distance matrix across centroids/samples
        if kernel_name == 'cosine':
            
            condensed_distances = pdist(X, metric='cosine')
            #print(f'Median cosine distance: {np.median(condensed_distances):.2f}')
            D_x = squareform(condensed_distances, checks=True) 

        elif kernel_name == 'euclidean':
            
            condensed_distances = pdist(X, metric='euclidean')
            #print(f'Median euclidean distance: {np.median(condensed_distances):.2f}')
            D_x = squareform(condensed_distances, checks=True) 
        
        elif kernel_name == 'gaussian_rbf':
            
            # from sklearn.metrics.pairwise import rbf_kernel
            # S2 = rbf_kernel(X, gamma=1 / (2 * temperature * sigma_rbf ** 2))
            
            # Step 1: compute squared Euclidean distances
            condensed_sq_dists = pdist(X, metric='sqeuclidean')
            #print(f'Computed sigma_rbf: {np.median(np.sqrt(condensed_sq_dists))}')
            sigma_rbf = np.median(np.sqrt(condensed_sq_dists))
            # Step 2: compute RBF kernel in condensed form
            condensed_distances = np.exp(-condensed_sq_dists / (2 * temperature * sigma_rbf ** 2))
            
            D_x = squareform(condensed_distances, checks=True) 
            np.fill_diagonal(D_x, 1)
        
    else:
        D_x = D_x_pc.copy()
        
    # 2) Then if a Temporal distance matrix is provided (or wihout using multimodal setting), we compute
    # the aggregated distance matrix 
    
    if (D_tau is None) or multimodal_method == 'without':
        
        # 1) We compute D as similarity/distance depending on the kernel if needed
        #print(f'[INFO]: No temporal distance provided, using only {kernel_name} kernel')
        D = D_x.copy()
    
        if compute_similarity and kernel_name == 'cosine':
            D = 1 - D
            print(f'Min/Max D_x Distance before minmax scaling (to [0-1]): min={np.min(D):.2f}, max={np.max(D):.2f}')
            
        
        elif compute_similarity and kernel_name == 'euclidean':
            
            D = 1 / (1 + D)
            p2, p98 = np.percentile(D, [0.05, 99.9])
            #D = np.clip(D, p2, p98)
            #D = (D - p2) / (p98 - p2)    
            #print(f'Min/Max D_tau Distance before minmax scaling (to [0-1]; p2, p98={(p2, p98)}): min={np.min(D):.2f}, max={np.max(D):.2f}')        
            #D = (D - D.min()) / (D.max() - D.min())    
            #np.fill_diagonal(D, 1)      
            
        elif not compute_similarity and kernel_name == 'gaussian_rbf':  
            #print(f'[INFO]: Inversing Gaussian RBF Kernel compute_similarity={compute_similarity}')
            D = 1 - D
            #p2, p98 = np.percentile(D, [0.05, 99.9])
            # D = np.clip(D, p2, p98)
            # D = (D - p2) / (p98 - p2)         
            #print(f'Min/Max D_tau Distance before minmax scaling (to [0-1]; p2, p98={p2:.2f}, {p98:.2f}): {np.min(D):.2f}, max={np.max(D):.2f}')        
            #D = (D - D.min()) / (D.max() - D.min())            
            #np.fill_diagonal(D, 0)      
            # Step 3: recover approximate squared distances using inverse of kernel formula
            # Note: clip for numerical stability (log(0) → -inf)
            #condensed_distances = np.sqrt(-2 * sigma_rbf**2 * np.log(np.clip(K, 1e-10, 1)))
        
    else:
        
        green(f'[INFO]: Combining distances using {kernel_name} kernel {multimodal_method} method and alpha={alpha}')
        
        # 2. Combine with temporal Gromov-Wasserstein distance        
            
        if (multimodal_method == 'normalized_exp') and (kernel_name in ['cosine', 'euclidean']):
                        
            #D_tau = D_tau / np.max(D_tau)
            #D_x = D_x / np.max(D_x)
            
            D = np.exp(- alpha * D_x - (1 - alpha) * D_tau)
            
            if not compute_similarity:
                D = 1 - D
                p2, p98 = np.percentile(D, [0.05, 99.9])
                # D = np.clip(D, p2, p98)
                # D = (D - p2) / (p98 - p2)
                
                #print(f'Distance {kernel_name} {multimodal_method} before minmax scaling (to [0-1]; p2, p98={(p2, p98)}): {np.min(D):.2f}, max={np.max(D):.2f}')
                #D = (D - D.min()) / (D.max() - D.min())
                print(f'Commenting minmax normalization of D in normalized_exp Co/Eucl, {D_tau.min()}, {D_tau.max()}, {D_x.min()}, {D_x.max()}, {D.min()}, {D.max()}')
                # Commenting minmax normalization of D in normalized_exp Co/Eucl, 0.0, 0.6696355569362913, 0.0, 1.4759863098103936, 0.0, 0.6337916913114281
        elif (multimodal_method == 'unnormalized_exp') and (kernel_name in ['cosine', 'euclidean']):
                        
            D = np.exp(- alpha * D_x - (1 - alpha) * D_tau)       
            
            if not compute_similarity:
                D = 1- D
                p2, p98 = np.percentile(D, [0.05, 99.9])
                # D = np.clip(D, p2, p98)
                # D = (D - p2) / (p98 - p2)
                
                #print(f'Distance {kernel_name} {multimodal_method} before minmax scaling (to [0-1]; p2, p98={(p2, p98)}): {np.min(D):.2f}, max={np.max(D):.2f}')
                #D = (D - D.min()) / (D.max() - D.min())
                print(f'Commenting minmax normalization of D in normalized_exp Co/Eucl, {D_tau.min()}, {D_tau.max()}, {D_x.min()}, {D_x.max()}, {D.min()}, {D.max()}')


            
        elif (multimodal_method == 'multiplicative') and (kernel_name in ['cosine', 'euclidean']):
            
            
            #D_tau = D_tau / np.max(D_tau)
            #D_x = D_x / np.max(D_x)
            
            # Weighting is perform geometrically in the exponent here, such as to preserve distance dimensions
            
        
            
            # We transform the distances to be reflective of ([0-1]-scaled) similarity 
            # per modality before combining them
            # red(f'BEFORE SIM Min/Max D_w Distance before minmax scaling (to [0-1]): {np.min(D_x):.2f}, max={np.max(D_x):.2f}')
            # if kernel_name == 'cosine':
            #     D_x = 1 - D_x
            #     # D_x = 1 / (1 + np.exp(-D_x))
                
            #     # Already scaled between 0 and 1
            #     #print(f'Min cosine similarity: {np.min(D_x):.2f}, max={np.max(D_x):.2f}')       
                
            # elif kernel_name == 'euclidean':
            #     # Inversion and Min-max scaling
            #     D_x = 1 / (1 + D_x)
            #     #p2, p98 = np.percentile(D_x, [0.05, 99.9])
            #     #D_x = np.clip(D_x, p2, p98)
            #     #D_x = (D_x - p2) / (p98 - p2)
            #     #print(f'Min euclidean Distance before minmax scaling (to [0-1]): {np.min(D_x):.2f}, max={np.max(D_x):.2f}')
            #     #D_x = (D_x - D_x.min()) / (D_x.max() - D_x.min())
            # red(f'BEFORE SIM Min/Max D_tau Distance before minmax scaling (to [0-1]): {np.min(D_tau):.2f}, max={np.max(D_tau):.2f}')
            # D_tau = 1 / (1 + D_tau)
            # # p2, p98 = np.percentile(D_tau, [0.05, 99.9])
            # # D_tau = np.clip(D_tau, p2, p98)
            # # D_tau = (D_tau - p2) / (p98 - p2)
            # red(f'Min/Max D_tau Distance before minmax scaling (to [0-1]): {np.min(D_tau):.2f}, max={np.max(D_tau):.2f}')
            # red(f'Min/Max D_x Distance before minmax scaling (to [0-1]): {np.min(D_x):.2f}, max={np.max(D_x):.2f}')

            # red(f'Commenting minmax normalization of D_tau and D_x in multiplicative Co/Eucl, {D_tau.min()}, {D_tau.max()}, {D_x.min()}, {D_x.max()}')
            purple(f'/!\ Commenting minmax normalization of D_tau and D_x in multiplicative Co/Eucl, {D_tau.min()}, {D_tau.max()}, {D_x.min()}, {D_x.max()}')
            D_tau = (D_tau - D_tau.min()) / (D_tau.max() - D_tau.min())
            #D_x = (D_x - D_x.min()) / (D_x.max() - D_x.min())
            
            
            # fi(20, 5);plt.hist(D_x.flatten(), bins=50); plt.title('D_x similarity distribution after minmax scaling'); plt.show()
            # fi(20, 5);plt.hist(D_tau.flatten(), bins=50); plt.title('D_tau similarity distribution after minmax scaling'); plt.show()
            
            # Multiplicative combination
            D = D_x ** (alpha) * D_tau ** (1 - alpha)
    
            
            if compute_similarity:
                # We convert back to distance
                D = 1 - D 

            # fi(20, 5);plt.hist(D.flatten(), bins=50); plt.title('Final distance distribution after minmax scaling'); plt.show()
                
        elif  (multimodal_method in ['normalized_exp', 'unnormalized_exp', 'multiplicative']) and (kernel_name == 'gaussian_rbf'):
        
            # Latent and temporal distance matrices are already computed using the exponential family 
            # already scaled to their median values and potential temperature
            #print('/!\ Rescaling D_x and D_tau ! ')
            #D_tau = D_tau / np.max(D_tau)
            #D_x = D_x / np.max(D_x)
            # print((D_x ** alpha)[:5, :5])
            # print((D_tau ** (1 - alpha))[:5, :5])
            D =  D_x ** alpha * D_tau ** (1 - alpha)
            # print((D)[:5, :5])
            #print(f'Computed Gaussian RBF distance matrix with alpha={alpha} and temperature={temperature}')
            
            if not compute_similarity:
                D = 1 - D 
                #p2, p98 = np.percentile(D, [0.05, 99.9])
                #D = np.clip(D, p2, p98)
                #D = (D - p2) / (p98 - p2)
                #print(f'Min Gaussian rbf Distance before minmax scaling (to [0-1]; p2, p98={(p2, p98)}): {np.min(D):.2f}, max={np.max(D):.2f}')
                #D = (D - D.min()) / (D.max() - D.min())
       
        elif  (multimodal_method  == 'temporal_weighting') and (kernel_name == 'gaussian_rbf'):
        
            # Latent and temporal distance matrices are already computed using the exponential family 
            # already scaled to their median values and potential temperature
            # print(D_tau[:5, :5])
            D_tau = D_tau / np.sum(D_tau)
            # D_x = D_x / np.max(D_x)
            # print(D_tau[:5, :5])
            # print(D_x[:5, :5])
            # print((D_tau ** (1 - alpha))[:5, :5])
            D =  D_x * D_tau
            # print((D)[:5, :5])
            #print(f'Computed Gaussian RBF distance matrix with alpha={alpha} and temperature={temperature}')
            
            if not compute_similarity:
                D = 1 - D 
                #p2, p98 = np.percentile(D, [0.05, 99.9])
                #D = np.clip(D, p2, p98)
                #D = (D - p2) / (p98 - p2)
                #print(f'Min Gaussian rbf Distance before minmax scaling (to [0-1]; p2, p98={(p2, p98)}): {np.min(D):.2f}, max={np.max(D):.2f}')
                #D = (D - D.min()) / (D.max() - D.min())
                print(f'Commenting minmax normalization of D_tau and D_x in temporal_weighting gaussian_rbf, {D_tau.min()}, {D_tau.max()}, {D_x.min()}, {D_x.max()}')


        elif (multimodal_method == 'additive'):
                
            D_tau = D_tau / np.max(D_tau)
            D_x = D_x / np.max(D_x)
            
            # Weighting is perform geometrically in the exponent here, such as to preserve distance dimensions
            D = alpha * D_x + (1 - alpha) *  D_tau 

    #blue(f'D_x: mean={np.mean(D_x):.2f}, median={np.median(D_x):.2f}, std={np.std(D_x):.2f}, min={np.min(D_x):.2f}, max={np.max(D_x):.2f} | D_tau: mean={np.mean(D_tau):.2f}, median={np.median(D_tau):.2f}, std={np.std(D_tau):.2f}, min={np.min(D_tau):.2f}, max={np.max(D_tau):.2f} | D: mean={np.mean(D):.2f}, median={np.median(D):.2f}, std={np.std(D):.2f}, min={np.min(D):.2f}, max={np.max(D):.2f}')           
    # Compute statistics for D_x, D_tau, and D
    #print(f'Statistics for D_x: mean={np.mean(D_x):.2f}, median={np.median(D_x):.2f}, std={np.std(D_x):.2f}, min={np.min(D_x):.2f}, max={np.max(D_x):.2f}')
    #print(f'Statistics for D_tau: mean={np.mean(D_tau):.2f}, median={np.median(D_tau):.2f}, std={np.std(D_tau):.2f}, min={np.min(D_tau):.2f}, max={np.max(D_tau):.2f}')
    #print(f'Statistics for D: mean={np.mean(D):.2f}, median={np.median(D):.2f}, std={np.std(D):.2f}, min={np.min(D):.2f}, max={np.max(D):.2f}')
    
    #blue(f'D: mean={np.mean(D):.2f}, median={np.median(D):.2f}, std={np.std(D):.2f}, min={np.min(D):.2f}, max={np.max(D):.2f}')           
    #print(f'Average assymetry: {np.mean(D - D.T):.2f}, median assymetry: {np.median(D - D.T):.2f}, std assymetry: {np.std(D - D.T):.2f}, min assymetry: {np.min(D - D.T):.2f}, max assymetry: {np.max(D - D.T):.2f}')
    D = (D + D.T) / 2
    #print(f'Average assymetry: {np.mean(D - D.T):.2f}, median assymetry: {np.median(D - D.T):.2f}, std assymetry: {np.std(D - D.T):.2f}, min assymetry: {np.min(D - D.T):.2f}, max assymetry: {np.max(D - D.T):.2f}')
    return D

def compute_reduced_centroids(config_name, annotator_id, round_number, normalization, excluded_clusters={}, agg_fn=np.mean, input_space='K_space', cluster_types=['foreground']):
        
    config = import_config(config_name)
    clustering_config_name = config.clustering_config_name
    
    # 1) In the original space, we gather prototypes and slice  present clusters per cluster type 
    mu = get_prototypes(clustering_config_name, annotator_id=annotator_id, round_number=round_number, cluster_types=cluster_types, normalization=normalization)
    if input_space == 'K_space': 
        
        
       

        # _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number,  is_foreground=True, return_cluster_types=True, input_space=input_space, verbose=False)
        # cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['foreground']}

        mu_f = mu#[cluster_indexes_remaining['foreground'], :]

        # _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number,  is_foreground=False, return_cluster_types=True, verbose=False)
        # cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['task-definitive', 'exo-definitive', 'Noise']}
        

        # mu_t = mu[cluster_indexes_remaining['task-definitive'], :]
        # mu_e = mu[cluster_indexes_remaining['exo-definitive'], :]
        # mu_n = mu[cluster_indexes_remaining['Noise'], :]

        # 2) In the reduced space and per cluster-type, we gather prototypes and compute an aggregate of the sub-clusters distances components, 
        # which notably provide a proxy for the within-cluster consistency and can be read in the distance matrices diagonal elements. 
        # 
    elif input_space == 'G_space': 

        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, is_foreground=True,  round_number=round_number, input_space='K_space', return_cluster_types=True, verbose=False)
        mu_f = reduce_centroid_vectors_per_cluster_type(mu, mapping_prototypes_reduction, mapping_index_cluster_type, agg_fn=agg_fn, excluded_clusters=excluded_clusters)
    
    elif input_space == 'G_opt_space': 

        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, is_foreground=True,  round_number=round_number, input_space='G_space', return_cluster_types=True, verbose=False)
        mu_f = reduce_centroid_vectors_per_cluster_type(mu, mapping_prototypes_reduction, mapping_index_cluster_type, agg_fn=agg_fn, excluded_clusters=excluded_clusters)

    
    return  mu_f

def compute_latent_distance(config_name, annotator_id, round_number, input_space, kernel_name, normalization, temperature_x, compute_similarity=False, excluded_clusters={}, agg_fn=np.mean):
    
    config = import_config(config_name)
    clustering_config_name = config.clustering_config_name
        
    # HERE excluded_clusters_G['foreground']
    # 1) In the original space, we gather prototypes and slice  present clusters per cluster type 
    if input_space == 'K_space' : 
        
        centroids = get_prototypes(clustering_config_name, annotator_id=annotator_id, round_number=round_number, cluster_types=['task-definitive', 'exo-definitive', 'Noise'], normalization=normalization)
        D_K = compute_distance_matrix(X=centroids, compute_similarity=compute_similarity, kernel_name=kernel_name, temperature=temperature_x)

        #_, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space,  #is_foreground=True, return_cluster_types=True, verbose=False)
        #cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['foreground']}
        D_xf = D_K#[cluster_indexes_remaining['foreground'], :][:, cluster_indexes_remaining['foreground']]

        # _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number,  is_foreground=False, return_cluster_types=True, verbose=False)
        # cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['task-definitive', 'exo-definitive', 'Noise']}
        
        # D_xt = D_K[cluster_indexes_remaining['task-definitive'], :][:, cluster_indexes_remaining['task-definitive']]
        # D_xe = D_K[cluster_indexes_remaining['exo-definitive'], :][:, cluster_indexes_remaining['exo-definitive']]
        # D_xn = D_K[cluster_indexes_remaining['Noise'], :][:, cluster_indexes_remaining['Noise']]
        #print(f"Computed distance matrices in K space for {config_name} with temperature {temperature_x} with use_K_space={use_K_space}")
    
        
        # 2) In the reduced space and per cluster-type, we gather prototypes and compute an aggregate of the sub-clusters distances components, 
        # which notably provide a proxy for the within-cluster consistency and can be read in the distance matrices diagonal elements. 
        # 
    elif input_space == 'G_space' : 
        
        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, round_number=round_number, input_space='K_space',  return_cluster_types=True, verbose=False)

        centroids = get_prototypes(clustering_config_name, annotator_id=annotator_id, round_number=round_number, cluster_types=['task-definitive', 'exo-definitive', 'Noise'], normalization=normalization)
        D_K = compute_distance_matrix(X=centroids, compute_similarity=compute_similarity, kernel_name=kernel_name, temperature=temperature_x)
        D_xt, D_xe, D_xn, D_xf = reduce_similarity_matrix_per_cluster_type(D_K, mapping_prototypes_reduction, mapping_index_cluster_type,  excluded_clusters=excluded_clusters, agg_fn=agg_fn)
   
    elif input_space == 'G_opt_space' : 
       
        mapping_prototypes_reduction, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, round_number=round_number, input_space='G_space',  return_cluster_types=True, verbose=False)

        centroids = get_prototypes(clustering_config_name, annotator_id=annotator_id, round_number=round_number, cluster_types=['task-definitive', 'exo-definitive', 'Noise'], normalization=normalization)
        D_K = compute_distance_matrix(X=centroids, compute_similarity=compute_similarity, kernel_name=kernel_name, temperature=temperature_x)
        D_xt, D_xe, D_xn, D_xf = reduce_similarity_matrix_per_cluster_type(D_K, mapping_prototypes_reduction, mapping_index_cluster_type,  excluded_clusters=excluded_clusters, agg_fn=agg_fn)
   

    return D_xf

def compute_temporal_distance(kde_models=None, df=None, annotator_id='samperochon', round_number=8, config_name='SymbolicSourceInferenceGoldConfig', loss_name='square_loss', gridsize=256, temporal_distance=None, kernel_name='linear',  D_tf_pc=None, per_cluster_types=False, input_space='K_space', embedding_labels_col=None, segments_labels_col=None, temporal_segmentations_col=None, temperature_tau=None, overwrite_gw_distances=False, ordered_index=True, suffix=''):
    """
    TODO: remove cluster type distinction 
    """
    config = import_config(config_name)
    experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(experiments_folder, exist_ok=True)    
    
    if temporal_distance is None:
        temporal_distance = config.temporal_distance 
    multimodal_method = config.multimodal_method
    alpha = config.alpha

    # Option A) Compute the raw Wasserstein distances if overwrite or not provided or missing
    # Option B) Load the raw Wasserstein distances if available
    # Option C) Use the pre-computed distances if available, and perform post processing operation (scaling, normalization, etc.)

    # Create foreground kde_models
    #kde_models_foreground = {**kde_models['task-definitive'], **kde_models['exo-definitive']} 
    #27 min for reduced (G) and 269.12 minutes for original (K=914) and 232.5 min fr K=906)
    D_tf_pc_path = os.path.join(experiments_folder, f'D_ttrr_pc_{temporal_distance}_{loss_name}_{gridsize}_{multimodal_method}_{alpha}_{input_space}{suffix}.npy')     
    

    if  (not os.path.exists(D_tf_pc_path) or overwrite_gw_distances) and D_tf_pc is None :
        #print(f"Pre-computing K={len(kde_models['foreground'])} Gromov-Wasserstein distances for temporal_distance={temporal_distance} and loss_name={loss_name} and temperature_tau={temperature_tau} - {input_space} - {D_tf_pc_path}...")
        
        
        if temporal_distance in ['linear_gw', 'gw']:
            # D_tt, _, _ = compute_gw_costs_and_transport_maps(kde_models['task-definitive'], pre_computed_D=None, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name='pre_computing', temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=True)
            # np.save(D_tt_pc_path, D_tt)
            # D_te, _, _ = compute_gw_costs_and_transport_maps(kde_models['exo-definitive'], pre_computed_D=None, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name='pre_computing', temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=True)
            # np.save(D_te_pc_path, D_te)
            D_tf, _, _ = compute_gw_costs_and_transport_maps(kde_models, pre_computed_D=None, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name='pre_computing', temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=True)
            np.save(D_tf_pc_path, D_tf)
        elif temporal_distance == 'wasserstein-1':
            # D_tt = compute_wasserstein_distance(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space, cluster_type='task-definitive', verbose=True)
            # np.save(D_tt_pc_path, D_tt)
            # D_te  = compute_wasserstein_distance(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space, cluster_type='exo-definitive',  verbose=True)
            # np.save(D_te_pc_path, D_te)
            D_tf = compute_wasserstein_distance(config_name, df=df, annotator_id=annotator_id, round_number=round_number, input_space=input_space, embedding_labels_col=embedding_labels_col, segments_labels_col=segments_labels_col, temporal_segmentations_col=temporal_segmentations_col, cluster_type='foreground', verbose=True)
            np.save(D_tf_pc_path, D_tf)
            

        print(f"Saved pre-computed distance matrices to {D_tf_pc_path}")
    elif os.path.exists(D_tf_pc_path) and D_tf_pc is None:
        
        # D_tt = np.load(D_tt_pc_path, allow_pickle=True)
        # D_te = np.load(D_te_pc_path, allow_pickle=True)
        D_tf = np.load(D_tf_pc_path, allow_pickle=True)
        
        print(f"Loaded pre-computed distance matrices from {D_tf_pc_path}")
        
    else:
        
        compute_similarity = True if kernel_name == 'gaussian_rbf' else False
        #D_tt, _, _ = compute_gw_costs_and_transport_maps(kde_models['task-definitive'], compute_similarity=compute_similarity, pre_computed_D=D_tt_pc, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name=kernel_name, temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=False)
        #D_te, _, _ = compute_gw_costs_and_transport_maps(kde_models['exo-definitive'], compute_similarity=compute_similarity, pre_computed_D=D_te_pc, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name=kernel_name, temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=False)
        D_tf, _, _ = compute_gw_costs_and_transport_maps(kde_models, compute_similarity=compute_similarity,  pre_computed_D=D_tf_pc, gridsize=gridsize, temporal_distance=temporal_distance, kernel_name=kernel_name, temperature_tau=temperature_tau, loss_name=loss_name, ordered_index=ordered_index, n_clt=None, verbose=False)

    return  D_tf


def fetch_missing_temporal_prototypes(kde_models, annotator_id, round_number, config_name, clustering_config_name, use_K_space=True, verbose=False):

    config = import_config(config_name)
    clustering_config_name = config.clustering_config_name
    
    excluded_clusters = {}
    if use_K_space: 
        
        config = import_config(config_name)
        clustering_config_name = config.clustering_config_name
        _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=True, input_space='K_space', return_cluster_types=True, verbose=False)

        cluster_type = 'foreground'
        n_missing = set([i for i, v in mapping_index_cluster_type.items() if v == cluster_type]).difference(set(kde_models[cluster_type].keys()))
        excluded_clusters[cluster_type] = list(n_missing)
        if len(n_missing) > 0 and cluster_type != 'Noise' and verbose:
            yellow(f"/!\ Missing prototypes for {cluster_type}: N={len(n_missing)} ({list(n_missing)[:5]}..., no timestmaps data found for this label column.")
            yellow('-> We remove these indexes from the distances matrices computation and ensure alignements across temporal and latent space matrices')

        _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=False, input_space='K_space', return_cluster_types=True, verbose=False)

        for cluster_type in ['task-definitive', 'exo-definitive', 'Noise']:
            n_missing = set([i for i, v in mapping_index_cluster_type.items() if v == cluster_type]).difference(set(kde_models['foreground'].keys()))
            excluded_clusters[cluster_type] = list(n_missing)
            if len(n_missing) > 0 and cluster_type != 'Noise' and verbose:
                yellow(f"/!\ Missing prototypes for {cluster_type}: N={len(n_missing)} ({list(n_missing)[:5]}..., no timestmaps data found for this label column.")
                yellow('-> We remove these indexes from the distances matrices computation and ensure alignements across temporal and latent space matrices')

    else:

        _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=True, input_space='G_space', return_cluster_types=True, verbose=False)
        cluster_type = 'foreground'
        n_missing = set([i for i, v in mapping_index_cluster_type.items() if v == cluster_type]).difference(set(kde_models[cluster_type].keys()))
        excluded_clusters[cluster_type] = list(n_missing)
        if len(n_missing) > 0 and cluster_type != 'Noise' and verbose:
            yellow(f"/!\ Missing prototypes for {cluster_type}: N={len(n_missing)} ({list(n_missing)[:5]}..., no timestmaps data found for this label column.")
            yellow('-> We remove these indexes from the distances matrices computation and ensure alignements across temporal and latent space matrices')

        _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=config_name, annotator_id=annotator_id, round_number=round_number, is_foreground=False, input_space='G_space', return_cluster_types=True, verbose=False)

        for cluster_type in ['task-definitive', 'exo-definitive', 'Noise']:
            n_missing = set([i for i, v in mapping_index_cluster_type.items() if v == cluster_type]).difference(set(kde_models['foreground'].keys()))
            excluded_clusters[cluster_type] = list(n_missing)
            if len(n_missing) > 0 and cluster_type != 'Noise' and verbose:
                yellow(f"/!\ Missing prototypes for {cluster_type}: N={len(n_missing)} ({list(n_missing)[:5]}..., no timestmaps data found for this label column.")
                yellow('-> We remove these indexes from the distances matrices computation and ensure alignements across temporal and latent space matrices')
                
    # excluded_clusters['foreground'] = excluded_clusters['task-definitive'] + excluded_clusters['exo-definitive']
    return excluded_clusters

def compute_combined_distance_across_temperatures(kde_models, annotator_id, round_number, excluded_clusters, config_name, temperatures=[0.1, 1, 10], temperatures_tau=[0.1, 1, 10], D_tt_pc=None,  D_te_pc=None,  D_tf_pc=None, normalization='l2', kernel_name='cosine', gridsize=128, loss_name='square_loss', temporal_distance='gw', multimodal_method='normalized_exp', alpha=.5, agg_fn=np.mean, task_type='task-definitive',  input_space='K_space', title='', temperature_x=None, temperature_tau=None, figpath=None, do_show=True):    
    
    if kernel_name in ['euclidean', 'cosine']:
        D_xf, D_tf, D_foreground = compute_combined_distance(kde_models, annotator_id, round_number,  excluded_clusters, config_name,  D_tf_pc=D_tf_pc, normalization=normalization, kernel_name=kernel_name, gridsize=gridsize, loss_name=loss_name, temporal_distance=temporal_distance, multimodal_method=multimodal_method, alpha=alpha, agg_fn=agg_fn, input_space=input_space)
        
        D_est, S_est, D_x_est, D_t_est  = D_foreground, D_foreground, D_xf, D_tf
        
        return D_est, S_est, D_x_est, D_t_est
    
    config = import_config(config_name)
    
    if temperature_x is None:
        temperature_x = config.temperature_x[input_space]
    if temperature_tau is None:
        temperature_tau = config.temperature_tau[input_space]
    
    if temperature_tau not in temperatures_tau:
        temperatures_tau = sorted([temperature_tau] + list(temperatures_tau))
        
    if temperature_x not in temperatures:
        temperatures = sorted([temperature_x] + list(temperatures))
        
    fig_img, axes_img = plt.subplots(len(temperatures_tau), len(temperatures), figsize=(20, 20), sharex=True, sharey=True)
    fig_hist, axes_hist = plt.subplots(len(temperatures_tau), len(temperatures), figsize=(20, 15), sharex=True, sharey=False)

    for i, temp_tau in enumerate(temperatures_tau):
        for j, temp in enumerate(temperatures):
            
            # 1) Compute temporal distance matrices across prototypes in the latent space before/after reduction per cluster types
            D_tf = compute_temporal_distance(kde_models,  annotator_id, round_number, D_tf_pc=D_tf_pc, config_name=config_name, kernel_name=kernel_name, loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=temp_tau, overwrite_gw_distances=False)
            # 2) Compute distance matrices across prototypes in the latent space before/after reduction per cluster types
            
            # For gaussian rbf we keep the similarity (exp(...)) variable
            compute_similarity = True if kernel_name == 'gaussian_rbf' else False
            D_xf = compute_latent_distance(config_name=config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space, compute_similarity=compute_similarity, kernel_name=kernel_name, normalization=normalization, temperature_x=temp, excluded_clusters=excluded_clusters, agg_fn=agg_fn)

            #print(f'Shape of D_x_pc: {D_x_pc.shape}, D_tau: {D_tau.shape} for {task_type} and temperature {temp} and {temp_tau}')
            #print('RBF: diag(D_x_pc)', np.diag(D_x_pc)[:10], 'diag(D_tau)', np.diag(D_tau)[:10])
            D = compute_distance_matrix(D_x_pc=D_xf, D_tau=D_tf, compute_similarity=False, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
            S = compute_distance_matrix(D_x_pc=D_xf, D_tau=D_tf, compute_similarity=True, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
            
            q75, q25 = np.percentile(D, [75, 25]); iqr = q75 - q25; std_D = np.std(D); mean_D = np.mean(D)
            figtitle = f" Tτ={temp_tau} Tx={temp}\niqr={iqr:.2f}, std={std_D:.2f}"
            im = axes_img[i, j].imshow(D, aspect='auto', cmap='coolwarm')
            fig_img.colorbar(im, ax=axes_img[i, j], fraction=0.046, pad=0.04)
            axes_img[i, j].set_title(figtitle)
            
            axes_hist[i, j].hist(D.flatten(), bins=50)
            axes_hist[i, j].set_title(figtitle)
            axes_hist[i, j].set_xlabel("Values")
            axes_hist[i, j].set_ylabel("Frequency")
            
            
            if (temp_tau == temperature_tau) and (temp == temperature_x):
                
                axes_img[i, j].set_title(figtitle, weight='bold')
                axes_hist[i, j].set_title(figtitle, weight='bold')
                
                D_est = D
                S_est = S
                D_x_est = D_xf
                D_t_est = D_tf

    fig_img.suptitle(f"{annotator_id}-round: {round_number} {task_type}\nDistance Matrices (D) across latent and temporal spaces temperatures\n"+title, y=1.02, fontsize=18)
    fig_hist.suptitle(f"{annotator_id}-round: {round_number} {task_type}\nHistograms of D across latent and temporal spaces temperatures\n"+title, y=1.02, fontsize=18)
    fig_img.tight_layout()
    fig_hist.tight_layout()
        
    if figpath is not None:
        fig_img.savefig(figpath, bbox_inches='tight', dpi=100)
        fig_hist.savefig(figpath.replace('combined_distance_matrix_task_', 'combined_distance_hist_task_'), bbox_inches='tight', dpi=100)
        print(f"Saved figure to e.g. {figpath}")
        
    if do_show:
        fig_img.show()
        fig_hist.show()
    else:
        plt.close()
    return D_est, S_est, D_x_est, D_t_est

def benchmark_over_alpha(kde_models, annotator_id, round_number, excluded_clusters, config_name, D_tf_pc=None, normalization='l2', kernel_name='cosine', gridsize=128, loss_name='square_loss', temporal_distance='gw', multimodal_method='normalized_exp', alpha=0.75, alpha_values=np.linspace(0, 1, 9), agg_fn=np.mean, task_type='task-definitive', input_space='K_space', title='', temperature_x=None, temperature_tau=None, figpath=None, do_show=True):
    
    assert alpha in alpha_values, f"Alpha {alpha} must be in the provided alpha_values: {alpha_values}"
    
    config = import_config(config_name)
    if temperature_x is None:
        temperature_x = config.temperature_x['K_space' if use_K_space else 'G_space']
    if temperature_tau is None:
        temperature_tau = config.temperature_tau['K_space' if use_K_space else 'G_space']

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig_hist, axes_hist = plt.subplots(3, 3, figsize=(20, 14), sharey=True)

    D_est = S_est = D_x_est = D_t_est = None

    D_tau = compute_temporal_distance(kde_models, annotator_id, round_number, D_tf_pc=D_tf_pc, config_name=config_name, kernel_name=kernel_name, loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=temperature_tau)
    
    compute_similarity = True if kernel_name == 'gaussian_rbf' else False
    D_x_pc  = compute_latent_distance(config_name=config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space, compute_similarity=compute_similarity, kernel_name=kernel_name, normalization=normalization, temperature_x=temperature_x, excluded_clusters=excluded_clusters, agg_fn=agg_fn)

    for idx, _alpha in enumerate(alpha_values):
        
        
        ax = axes[idx // 3, idx % 3]
        ax_hist = axes_hist[idx // 3, idx % 3]
        
        D = compute_distance_matrix(D_x_pc=D_x_pc, D_tau=D_tau, compute_similarity=False, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=_alpha)
        im = ax.imshow(D, aspect='auto', cmap='coolwarm')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


        ax_hist.hist(D.flatten(), bins=50)
        ax_hist.set_xlabel("Values")
        ax_hist.set_ylabel("Frequency")

        q75, q25 = np.percentile(D, [75, 25]); iqr = q75 - q25
        figtitle = f"Distance Matrix (D) for \ntemp_x={temperature_x}, temp_tau={temperature_tau} α={_alpha:.2f}\nmean={np.mean(D):.2f}, iqr={iqr:.2f}, std={np.std(D):.2f}" + title
    
        ax.set_title(figtitle)
        ax_hist.set_title(figtitle)

        if _alpha == alpha:
            print(f"Selected alpha={alpha} for temp_x={temperature_x}, temp_tau={temperature_tau}")
            
            ax.set_title(figtitle, weight='bold')
            ax_hist.set_title(figtitle, weight='bold')
            D_est = D
            S_est = compute_distance_matrix(D_x_pc=D_x_pc, D_tau=D_tau, compute_similarity=True, multimodal_method=multimodal_method, kernel_name=kernel_name, alpha=alpha)
            D_x_est = D_x_pc
            D_t_est = D_tau

    fig.suptitle(f"{annotator_id}-round: {round_number} {task_type}\nDistance Matrices for varying spatiotemporal balance (alpha)\n" + title, y=1.02, fontsize=16)
    fig_hist.suptitle(f"{annotator_id}-round: {round_number} {task_type}\nHistograms for varying spatiotemporal balance (alpha)\n"+title, y=1.02, fontsize=18)
    fig.tight_layout()
    fig_hist.tight_layout()

    if figpath:
        fig.savefig(figpath, bbox_inches='tight', dpi=100)
    if do_show:
        plt.show()
    else:
        plt.close()
    return D_est, S_est, D_x_est, D_t_est

def max_normalize_distance(x, do_normalize_distances=False): 
    return x / x.max() if do_normalize_distances else x

def get_prototypes_mapping(clustering_config_name, annotator_id='samperochon', round_number=0, is_foreground=True, input_space=None, return_cluster_types=False, verbose=False):
    
    config = import_config(clustering_config_name)
    
    # Get prototes mapping from the annotations
    if config.task_type == 'clustering':
        print(f'Using clusering type of config for get_prototypes_mapping {clustering_config_name}')
        qualification_mapping = fetch_qualification_mapping(verbose=False)
        
        if is_foreground:
            qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['foreground'] = qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['task-definitive'] + qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['exo-definitive']
            mapping_index_cluster_type = {cluster_idx: 'foreground' for  cluster_idx in  qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['foreground'] }
        else:
            mapping_index_cluster_type = {cluster_idx: cluster_type for  cluster_type, cluster_indexes in qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'].items() for cluster_idx in cluster_indexes}
        
        mapping_prototypes_reduction = None

    elif config.task_type == 'symbolization' :
        
        output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrix'); os.makedirs(output_folder, exist_ok=True)    
        best_results_path = os.path.join(output_folder, f'best_df_{input_space}.pkl')
        
                
        if os.path.isfile(best_results_path):    
            best_df = load_df(best_results_path)
        else:
            raise FileNotFoundError(f"Optimal results file not found: {best_results_path}.")
            
        mapping_prototypes_reduction = {}
        mapping_index_cluster_type = {}
        for cluster_type in ['foreground']: 
            mapping_prototypes_reduction.update(best_df.loc[cluster_type]['opt_index_mapping_per_cluster_type'])
            mapping_index_cluster_type.update(best_df.loc[cluster_type]['mapping_transformed_space_cluster_type'])



        
        #clustering_config_name = config.clustering_config_name
        # clustering_model_names = config.clustering_model_names
        # qualification_mapping = fetch_qualification_mapping(verbose=False)
        # results, best_df, mapping_prototypes_reduction, mapping_index_cluster_type, qualification_mapping = clustering_prototypes_space(symbolization_config_name=clustering_config_name, 
        #                                                                                                                                  annotator_id=annotator_id, 
        #                                                                                                                                  round_number=round_number,
        #                                                                                                                                  input_space=input_space,
        #                                                                                                                                 model_names=clustering_model_names,
        #                                                                                                                                 qualification_mapping=qualification_mapping, 
        #                                                                                                                                 overwrite=False,
        #                                                                                                                                 verbose=True)
        
        #mapping_index_cluster_type[-2] = 'Edge'
    
    else:
        raise ValueError(f"Unknown task type: {config.task_type}. Expected 'clustering' or 'symbolization'.")

    mapping_index_cluster_type[-1] = 'Noise'
    if return_cluster_types:
        return mapping_prototypes_reduction, mapping_index_cluster_type
    
    return mapping_prototypes_reduction


def hierarchical_prototypes_clustering(centroids, D_M, D_X, D_T, S_M, S_X, S_T, explored_K = [5, 50], X=None, model_name='hierarchical', linkage_method='complete', kernel_name='euclidean', temperature=None, depth=4, verbose=True, output_folder=None, suffix=None,  **kwargs):
    """
    
    Notes:
    - If X is provided, evaluation metrics are computed on the samples in X using the mapping provided by the clustering.
    Oterwise, the metrics are computed on the original prototypes samples
    
    """
    
    assert not (D_M is None and centroids is None)
    
    for D in [D_M, D_X, D_T]:
        tri_mean = np.mean(np.triu(D_M, k=1))
        diag_mean = np.mean(np.diag(D_M))
        if diag_mean > tri_mean * 1.2:  # adjust threshold as needed
            print(f"High diagonal mean: {diag_mean:.3f} vs upper triangle mean: {tri_mean:.3f} (shape={D_M.shape})")
        
        np.fill_diagonal(D, 0)
    
    
    if centroids is not None:
        explored_K = [n for n in explored_K if n < centroids.shape[0]]
    
    #explored_K = [-1] + explored_K
    # Build metrics, hyperparameters are not used if the similarity matrix is provided
    # TODO: xtend to account for the temperature_tau and thus the temporal domain in the evaluation 
    metrics = build_metrics(kernel_name=kernel_name, temperature=temperature)
    results = []; mean_ic = np.nan; median_ic = np.nan; percentiles = np.nan
    plot_linkage_matrix = None
    

    # Perform hierarchical clustering
    condensed_matrix = squareform(D_M, force='tovector', checks=True)
    linkage_matrix = sch.linkage(condensed_matrix, method=linkage_method, optimal_ordering=True)
        
    inconsistency_matrix = sch.inconsistent(linkage_matrix, depth)
    inconsistency_coefficients = inconsistency_matrix[:, -1]

    # Compute summary statistics
    mean_ic = np.mean(inconsistency_coefficients)
    median_ic = np.median(inconsistency_coefficients)
    std_ic = np.std(inconsistency_coefficients)
    
    #print(f"Mean inconsistency coefficient: {mean_ic:.2f}, Median: {median_ic:.2f}, Std: {std_ic:.2f} Th: {median_ic+std_ic:.2f}")
    print('explored_K', explored_K)
    for i, n_clusters in enumerate(tqdm(explored_K), start=1):
        green(f'Using hierarchical clustering with method={linkage_method} and depth={depth} and n_clusters={n_clusters} for {model_name} model')
        

        if n_clusters == -1: 
            t = np.percentile(inconsistency_coefficients, 0.7) 
            cluster_labels = sch.fcluster(linkage_matrix, t=t, criterion='inconsistent', depth=depth)
        
        else:                
            cluster_labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
            

        #print(f'Number of clusters: {len(np.unique(cluster_labels))} for n_clusters={n_clusters} and method={linkage_method}')
        # linkage_matrix = sch.linkage(condensed_distances, method=method)
        # cluster_labels = sch.fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
                    
        if verbose and (i == 1): # (i == len(explored_K) // 2):
            plot_linkage_matrix = linkage_matrix.copy()

        # Determine best cluster index permutations according to the optimization of the silhouette score across K values 
        perm = np.argsort(cluster_labels, kind='quicksort')
        labels = np.sort(cluster_labels, kind='quicksort') -1 # We force the meta labels to start at 0
        prototypes_cpts = np.argwhere(np.ediff1d(labels) != 0).flatten()
        assert np.min(labels) >= 0, f'Labels should be >= 0, got {np.min(labels)}'
        
        sigma_rbf = np.median(np.sqrt(D_M[np.triu_indices_from(D_M, k=1)]))

        A_permuted = D_M[perm][:, perm]
        
        # Compute metrics
        if X is None:
            print('centroi shapeds', centroids.shape)
            print('centroi D_M', D_M.shape)
            result = compute_metrics(X=centroids, labels=cluster_labels, D_M=D_M, D_X=D_X, D_T=D_T, S_M=S_M, S_X=S_X, S_T=S_T, kernel_name=kernel_name, metrics=metrics)
                    
        else:
            result = compute_metrics(X=X, labels=cluster_labels, D_M=D_M, D_X=D_X, D_T=D_T, S_M=S_M, S_X=S_X, S_T=S_T, kernel_name=kernel_name, metrics=metrics)             
                                
        result.update({'n_clusters': n_clusters, 
                        'exp_id': i,
                        'kernel_name': kernel_name,
                        'model_name': model_name,
                        'linkage_method': linkage_method,
                        'sigma_rbf': sigma_rbf,
                        #'temperature': temperature,
                        'cluster_id': np.arange(len(labels)).astype(int),
                        'perm': perm.astype(int), 
                        'labels': labels.astype(int), 
                        'prototypes_cpts': prototypes_cpts.astype(int), 
                        'mean_ic': mean_ic,
                        'median_ic': median_ic,
                        'percentiles_ic':percentiles,
                        })
            
        results.append(result)

        if verbose:
            
            if i == 1:
            
                n_plots = len(explored_K) 
                print(f'Number of plot for the distance matrix (len(explored_K)): {n_plots}')
                fig, axes = plt.subplots(n_plots // 5 +1, 5, figsize=(25, 15))
                axes = axes.flatten()
                print(f'Number of axis: {len(axes)}')
                
                # Plot the original matrix
                if X is None:
                
                    axes[0].imshow(D_M, cmap='coolwarm')
                    axes[0].set_title(f'Original Distance Matrix\nkernel_name={kernel_name}')
                    #print(f'Statitics of the oiginal matrix: mean dist={np.mean(D):.2f} median_dist={np.median(D):.2f} std_dist={np.std(D):.2f} min={np.min(D):.2f} max={np.max(D):.2f}') 
                
                else:
                    
                    axes[0].imshow(centroids @ X.T, aspect='auto',cmap='coolwarm')
                    axes[0].set_title('Original Mu @ X.T')

            if X is None:
                axes[i].imshow(A_permuted, cmap='coolwarm')
                axes[i].set_title(f'Permuted (k={n_clusters})')
            else:
                axes[i].imshow(centroids[perm] @ X.T, aspect='auto', cmap='coolwarm')
                axes[i].set_title(f'Permuted (k={n_clusters}) Mu_T @ X.T')


    if verbose and plot_linkage_matrix is not None:
        
        # remove addiitonal axis
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if output_folder is not None:
            figpath = os.path.join(output_folder, f'permuted_matrix_task_D_{D_M.shape[0]}_{suffix}.png')
            plt.savefig(figpath, bbox_inches='tight', dpi=100)
            print(f"Saved figure to {figpath}")
        plt.show()
        
        if verbose:
            # Plot the dendrogram and cluster cuts across depth (used in the inconsistency criteria)
            explore_inconsistency_depth_threshold(plot_linkage_matrix, centroids, output_folder=output_folder, suffix=suffix)
            compute_inconsistency_across_depth(plot_linkage_matrix, centroids, output_folder=output_folder, suffix=suffix)
            compute_inconsistency_across_thresholds(plot_linkage_matrix, centroids, depth=depth, output_folder=output_folder, suffix=suffix)
            # Plot middle-support dendogram
            
            plt.figure(figsize=(20, 5))
            plt.hist(plot_linkage_matrix[:, 2], bins='fd')
            plt.title("linkage_matrix distance")
            plt.xlabel("method-distance")
            plt.ylabel("f")
            plt.show()

        # compute cluster labels
        n_clusters_mid = explored_K[len(explored_K) // 2]
        cluster_labels = sch.fcluster(linkage_matrix, t=n_clusters_mid, criterion='maxclust')
        color_threshold = max(linkage_matrix[-(n_clusters - 1):, 2])  # height of the (n_clusters-1)th merge


        plt.figure(figsize=(20, 5))
        dendrogram = sch.dendrogram(plot_linkage_matrix, color_threshold=color_threshold)
        plt.title(f"Hierarchical Clustering Dendrogram K_mid={n_clusters_mid}\n{suffix}")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        
        if output_folder is not None:
            figpath = os.path.join(output_folder, f'dendogram_K_{n_clusters_mid}_{D_M.shape[0]}_{suffix}.png')
            plt.savefig(figpath, bbox_inches='tight', dpi=100)
            print(f"Saved figure to {figpath}")
            
        plt.show()

    return results


def prototypes_clustering(centroids, D_M, D_X, D_T, S_M, S_X, S_T, explored_K = [5, 50], X=None, model_name='hierarchical', kernel_name='euclidean', temperature=None, verbose=True, output_folder=None, suffix=None,  **kwargs):
    """
    
    Notes:
    - If X is provided, evaluation metrics are computed on the samples in X using the mapping provided by the clustering.
    Oterwise, the metrics are computed on the original prototypes samples
    
    """
    
    assert not (D_M is None and centroids is None)
    
    for D in [D_M, D_X, D_T]:
        tri_mean = np.mean(np.triu(D_M, k=1))
        diag_mean = np.mean(np.diag(D_M))
        if diag_mean > tri_mean * 1.2:  # adjust threshold as needed
            print(f"High diagonal mean: {diag_mean:.3f} vs upper triangle mean: {tri_mean:.3f} (shape={D_M.shape})")
        
        np.fill_diagonal(D, 0)
    
    
    if centroids is not None:
        explored_K = [n for n in explored_K if n < centroids.shape[0]]
    
    #explored_K = [-1] + explored_K
    # Build metrics, hyperparameters are not used if the similarity matrix is provided
    # TODO: xtend to account for the temperature_tau and thus the temporal domain in the evaluation 
    metrics = build_metrics(kernel_name=kernel_name, temperature=temperature)
    results = []
    num_experiments = 1
    mean_ic = np.nan; median_ic = np.nan; percentiles = np.nan
    plot_linkage_matrix = None
    
    for exp_id in range(num_experiments):
        
        if model_name in ['spectral', 'dbscan', 'hdbscan'] and exp_id > 0:
            continue
            
        for i, n_clusters in enumerate(tqdm(explored_K), start=1):
            

            # A) Gaussian Mixture Models clustering
            if model_name == 'gmm':  # Non-Bayesian Gaussian Mixture Model
                
                gmm = GaussianMixture(n_components=n_clusters, 
                                    covariance_type='full',
                                    tol=0.001,
                                    reg_covar=1e-06,
                                    max_iter=100, #5000,
                                    n_init=1, #3,
                                    init_params='k-means++',
                                    weights_init=None,
                                    means_init=None,
                                    precisions_init=None,
                                    random_state=None,
                                    warm_start=False, # if batch_size is not None else False,
                                    verbose=0,
                                    verbose_interval=10)
                
                cluster_labels = gmm.fit_predict(D_M)
            
            # D) Bayesian Gaussian Mixture Models clustering
            elif model_name == 'bayesian_gmm':  # Bayesian Gaussian Mixture Model
            
                bgmm = BayesianGaussianMixture(n_components=n_clusters,
                                        covariance_type='full',
                                        tol=0.001,
                                        reg_covar=1e-06,
                                        max_iter=500, #5000,
                                        n_init=1, #5,
                                        init_params='kmeans',
                                        weight_concentration_prior_type='dirichlet_process',
                                        weight_concentration_prior=None,
                                        mean_precision_prior=1.0,
                                        mean_prior=None,
                                        degrees_of_freedom_prior=None,
                                        covariance_prior=None,
                                        random_state=47,
                                        warm_start=False,# True if batch_size is not None else False,
                                        verbose=0,
                                        verbose_interval=10)
                
                cluster_labels = bgmm.fit_predict(D_M)
        


            elif model_name == 'spectral':
                
                
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    assign_labels='cluster_qr',
                    random_state=0
                )
                affinity = 1 - D_M / D_M.max()  
                cluster_labels = spectral.fit_predict(affinity)

            elif model_name == 'dbscan':
                dbscan = DBSCAN(
                    eps=0.15,
                    min_samples=2,
                    metric='precomputed'
                )
                cluster_labels = dbscan.fit_predict(D_M)

            elif model_name == 'hdbscan':
                hdb = HDBSCAN(
                    min_cluster_size=2,
                    eps=0.15,
                    metric='precomputed'
                )
                cluster_labels = hdb.fit_predict(D_M)
                        
            
            # Determine best cluster index permutations according to the optimization of the silhouette score across K values 
            perm = np.argsort(cluster_labels, kind='quicksort')
            labels = np.sort(cluster_labels, kind='quicksort') #-1 # We force the meta labels to start at 0
            print(f'Method {model_name} min labels: {np.min(labels)}, max labels: {np.max(labels)}')
            prototypes_cpts = np.argwhere(np.ediff1d(labels) != 0).flatten()
            #assert np.min(labels) >= 0, f'Labels should be >= 0, got {np.min(labels)}'
            
            sigma_rbf = np.median(np.sqrt(D_M[np.triu_indices_from(D_M, k=1)]))

            A_permuted = D_M[perm][:, perm]
            
            # Compute metrics
            if X is None:
                result = compute_metrics(X=centroids, labels=cluster_labels, D_M=D_M, D_X=D_X, D_T=D_T, S_M=S_M, S_X=S_X, S_T=S_T, kernel_name=kernel_name, temperature=temperature, metrics=metrics)
                        
            else:
                result = compute_metrics(X=X, labels=cluster_labels, D_M=D_M, D_X=D_X, D_T=D_T, S_M=S_M, S_X=S_X, S_T=S_T, kernel_name=kernel_name, temperature=temperature, metrics=metrics)             
                                    
            result.update({'n_clusters': n_clusters, 
                           'exp_id': exp_id,
                           'kernel_name': kernel_name,
                            'model_name': model_name,
                            'sigma_rbf': sigma_rbf,
                            #'temperature': temperature,
                            'linkage_method': np.nan,

                           'cluster_id': np.arange(len(labels)).astype(int),
                           'perm': perm.astype(int), 
                            'labels': labels.astype(int), 
                            'prototypes_cpts': prototypes_cpts.astype(int), 
                            'mean_ic': mean_ic,
                            'median_ic': median_ic,
                            'percentiles_ic':percentiles,
                            
                            
                            })
                
            results.append(result)
            
            if verbose and exp_id == 0:
                
                if i == 1:
                
                    n_plots = len(explored_K) 
                    print(f'Number of plot for the distance matrix (len(explored_K)): {n_plots}')
                    fig, axes = plt.subplots(n_plots // 5 +1, 5, figsize=(25, 15))
                    axes = axes.flatten()
                    print(f'Number of axis: {len(axes)}')
                    
                    # Plot the original matrix
                    if X is None:
                    
                        axes[0].imshow(D_M, cmap='coolwarm')
                        axes[0].set_title(f'Original Distance Matrix\nkernel_name={kernel_name}')
                        #print(f'Statitics of the oiginal matrix: mean dist={np.mean(D):.2f} median_dist={np.median(D):.2f} std_dist={np.std(D):.2f} min={np.min(D):.2f} max={np.max(D):.2f}') 
                    
                    else:
                        
                        axes[0].imshow(centroids @ X.T, aspect='auto',cmap='coolwarm')
                        axes[0].set_title('Original Mu @ X.T')

                if X is None:
                    axes[i].imshow(A_permuted, cmap='coolwarm')
                    axes[i].set_title(f'Permuted (k={n_clusters})')
                else:
                    axes[i].imshow(centroids[perm] @ X.T, aspect='auto', cmap='coolwarm')
                    axes[i].set_title(f'Permuted (k={n_clusters}) Mu_T @ X.T')

    if verbose:
        
        # remove addiitonal axis
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        if output_folder is not None:
            figpath = os.path.join(output_folder, f'permuted_matrix_task_D_{D_M.shape[0]}_{suffix}.png')
            plt.savefig(figpath, bbox_inches='tight', dpi=100)
            print(f"Saved figure to {figpath}")
        plt.show()
        

    return results

def clustering_prototypes_space(symbolization_config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', model_names=['hierarchical-ward'], annotator_id='samperochon', input_space=None, round_number=0, alpha=None, verbose=True, do_save=True, overwrite=False):
    """TODO: Docstring .
    
    Improve metrics selection (fix groupby and aggregation operation)
    
    """
    
    config = import_config(symbolization_config_name)
    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(config.clustering_config_name)

    # Types of prototypes used (task, exo, foreground)
    if input_space is None:
        input_space = 'K_space' if config.model_params['use_K_space'] else 'G_space'
        
    model_name = config.clustering_model_names[0] 
    multimodal_method = config.multimodal_method
    temporal_distance = config.temporal_distance
    linkage_method = config.linkage_method
    if alpha is None:
        alpha = config.alpha

    cluster_types = clustering_config.model_params['cluster_types']
    kernel_name = clustering_config.model_params['kernel_name']
    normalization = clustering_config.model_params['normalization']
    
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrix'); os.makedirs(output_folder, exist_ok=True)    
    results_path = os.path.join(output_folder, 'results_co_clustering_final_fg.pkl')
    results_path = os.path.join(output_folder, 'results_co_clustering_final_test_10092025_fixed.pkl')
    
    best_results_path = os.path.join(output_folder, f'best_df_{input_space}.pkl')


    if os.path.exists(results_path):# and not overwrite:
        
        results = load_df(results_path)
        green(f'Loaded results of clustering inn {results_path}')
        results.linkage_method.fillna('placeholder', inplace=True)
        results = results[results['model_name'] != 'dbscan']
        results = results[results['temporal_distance'] != 'gw']
        display(results.groupby(['input_space', 'temporal_distance', 'cluster_type', 'kernel_name', 'multimodal_method', 'model_name', 'linkage_method', 'alpha']).agg('size').to_frame().T)

        # 2) Set up original cluster index 
        results['original_cluster_index'] = results.perm.apply(sorted)
        # kde_models_K = estimate_kde_model(symbolization_config_name, annotator_id=annotator_id, round_number=round_number, use_K_space=True)
        # excluded_clusters = fetch_missing_temporal_prototypes(kde_models_K, annotator_id, round_number, symbolization_config_name, clustering_config_name, use_K_space=True)
        # _, mapping_index_cluster_type = get_prototypes_mapping(clustering_config_name=clustering_config_name, annotator_id=annotator_id, round_number=round_number, return_cluster_types=True, verbose=False)
        # cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['task-definitive', 'exo-definitive', 'foreground']}
        # results.loc[results['input_space'] == 'K_space', 'original_cluster_index'] =  results.cluster_type.map(cluster_indexes_remaining)

        if input_space == 'G_space':
        
            best_results_K_path = os.path.join(output_folder, f'best_df_K_space.pkl')
            best_df_K_space = load_df(best_results_K_path)
            green(f'Loading optimal results from K_space optimization: N_opt={best_df_K_space.n_clusters.iloc[0]}\n\t{best_results_K_path}')

            
            K_index_per_cluster_type = best_df_K_space.opt_index_mapping_per_cluster_type.apply(lambda x: sorted(list(set(x.values())))).to_dict()
            mapping_index_cluster_type_G = {idx: ct for ct, idx_list in K_index_per_cluster_type.items() for idx in idx_list}
        
            # kde_models_G = estimate_kde_model(symbolization_config_name, annotator_id=annotator_id, round_number=round_number, use_K_space=False)
            # excluded_clusters = fetch_missing_temporal_prototypes(kde_models_G, annotator_id, round_number, symbolization_config_name, clustering_config_name, use_K_space=False)
            # cluster_indexes_remaining = {ct: [k for k, ctt in mapping_index_cluster_type_G.items() if (ctt == ct) and k not in excluded_clusters[ct]] for ct in ['task-definitive', 'exo-definitive', 'foreground']}
            # results.loc[results['input_space'] == 'G_space', 'original_cluster_index'] =  results.cluster_type.map(cluster_indexes_remaining)

        
    else:
    
        print(f'File does not exist: {results_path}')
        print(f'Running clustering comparisons for: {clustering_config_name}')
        raise ValueError(f'Please run the clustering_prototypes_space function with overwrite=True to recompute the results or set overwrite=False to load existing results from {results_path}')
        results = run_comparison_clusterings(config_name=clustering_config_name,
                                             annotator_id=annotator_id,
                                             round_number=round_number,
                                             model_names = ['hierarchical-ward'], 
                                             metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation"],
                                             verbose=verbose)
        
        print('Results of clustering K-loop')
        display(results.head(2))
        
    
    if verbose:
        plot_clustering_metrics(results, hue='cluster_type')

    best_rows = {}
    for cluster_type in results.cluster_type.unique(): 
        green(f'Finding best clustering for annotator_id={annotator_id}, cluster_type={cluster_type}, temporal_distance={temporal_distance}, model_name={model_name}, linkage_method={linkage_method}, kernel_name={kernel_name}, multimodal_method={multimodal_method}, input_space={input_space},  alpha={alpha}')        
        sorted_df = results[(results.cluster_type == cluster_type) &
                            (results.input_space == input_space) &
                            (results.temporal_distance == temporal_distance) &
                            (results.model_name == model_name) &
                            (results.linkage_method == linkage_method) &
                            (results.kernel_name == kernel_name) &
                            (results.multimodal_method == multimodal_method) &  
                            (results.alpha == alpha)].sort_values('silhouette_score_M', ascending=False)
        best_rows[cluster_type] = sorted_df.reset_index().iloc[0]  # First row (best)

    # Convert to DataFrame for better visualization
    best_df = pd.DataFrame(best_rows).T  
    best_df.columns = ['_'.join(col) if (len(col) >1 and isinstance(col, tuple)) else col if (len(col) ==1 and isinstance(col, tuple))  else col for col in best_df.columns]
    best_df.rename(columns = {'original_cluster_index_first': 'original_cluster_index', 'labels_first': 'labels' , 'perm_first': 'perm', 'n_clusters_': 'n_clusters', 'model_name_': 'model_name'}, inplace=True)
    if verbose:
        green(f'Best clustering for each cluster types for model_name={model_name}, kernel_name={kernel_name}, multimodal_method={multimodal_method}, input_space={input_space}, linkage_method={linkage_method}, alpha={alpha}')
        display(best_df)
        
    # Find the correct global mapping across transformed labels across cluster types 
    cluster_types_size = best_df['n_clusters'].to_dict()
    cluster_types_offset = {}; n=0
    for cluster_type in cluster_types_size.keys():
        cluster_types_offset[cluster_type] = n
        n += cluster_types_size[cluster_type]
    
    best_df['opt_index_mapping_per_cluster_type'] = best_df.apply(lambda x: {original_index: transformed_index + cluster_types_offset[x.cluster_type]  for (original_index, transformed_index) in zip(np.array(x.original_cluster_index)[np.array(x.perm)], x.labels)}, axis=1)
    best_df['mapping_transformed_space_cluster_type'] = best_df.apply(lambda x: {ti: x.cluster_type for ti in x.opt_index_mapping_per_cluster_type.values()}, axis=1)
    best_df.set_index(['cluster_type'], inplace=True)
    save_df(best_df, best_results_path)
    green(f'Saved optimal clustering results to {best_results_path}\n\tN_opt ({input_space})={best_df.n_clusters.iloc[0]}')

    mapping_prototypes_reduction = {}
    for cluster_type in results.cluster_type.unique(): 
        mapping_prototypes_reduction.update(best_df.loc[cluster_type]['opt_index_mapping_per_cluster_type'])

    mapping_index_cluster_type = {}
    for cluster_type in results.cluster_type.unique(): 
        mapping_index_cluster_type.update(best_df.loc[cluster_type]['mapping_transformed_space_cluster_type'])

    #sprint('Adding noise and edge clusters types mapping')
    mapping_index_cluster_type[-1] = 'Noise'
   
    K_total = len(np.unique([v for v in mapping_prototypes_reduction.values()]))
    
    if verbose:
        print(f'{annotator_id}-round: {round_number}: Total number of clusters after reduction for each cluster types (using offset indexes for each cluster type, and sourcing from all in the qualification_mapping, i.e.e visual annotations): {K_total}')

    if do_save and (not os.path.exists(results_path) or overwrite):
        print('Saving results and mapping_prototypes_reduction')
        dict_save_path= os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id,f'round_{round_number}', f'mapping_prototypes_reduction_{input_space}.json')
        mapping_prototypes_reduction = {int(k): int(v) for k, v in mapping_prototypes_reduction.items()}
        with open(dict_save_path, 'w') as f:
            json.dump(mapping_prototypes_reduction, f)
            
        save_df(results, results_path)
        green(f'Saved results to {results_path}')
        results_path = os.path.join(output_folder, f'{clustering_config_name}_prototypes_results_optimal_clustering_per_types.pkl')
        save_df(best_df, results_path)
        green(f'Saved best_df to {results_path}')

    return results, best_df, mapping_prototypes_reduction, mapping_index_cluster_type

def run_comparison_clusterings(config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', annotator_id='samperochon', round_number=0, model_names = ['hierarchical-clustering'], metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation"], verbose=True):
    """TODO: turn this into a K-searching method and create from the notebook the experiment version (seperate experiemnts vs deployment setups)"""
    # 'hierarchical-centroid', 'spectral', 'gmm', 'bayesian_gmm']

    # Reduce prototypes redundancy as per the silhouette score (CS)
    config = import_config(config_name)
    kernel_name = config.model_params['kernel_name']
    sigma_rbf = config.model_params['sigma_rbf']
    temperature = config.model_params['temperature']
    loss_name = config.loss_name
    method_name = config.method_name
    temporal_distance = config.temporal_distance 
    gridsize = config.gridsize

    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(config.clustering_config_name)
    
    cluster_types = clustering_config.model_params['cluster_types']
    normalization = clustering_config.model_params['normalization']

    qualification_mapping = fetch_qualification_mapping(verbose=False)
    cluster_types_size = {cluster_type: len(v) for cluster_type, v in qualification_mapping[clustering_config_name].items()}
    explored_K = {'task-definitive': np.arange(15, cluster_types_size['task-definitive'], 25),
                    'exo-definitive': np.arange(15,  cluster_types_size['exo-definitive'], 5),
                    'Noise': np.arange(15, cluster_types_size['Noise'], 25)
    }
    
    results = []
    for cluster_type in cluster_types:
        for model_name in model_names: 
            centroids = get_prototypes(clustering_config_name, cluster_types=cluster_type)
            
            D_tau, transport_maps, logs = compute_gw_costs_and_transport_maps(kde_model, gridsize=gridsize, temporal_distance=temporal_distance, loss_name=loss_name, n_clt=None, verbose=True)
            
            plot_gw_temporal_distance(D_tau, title=title)    

            title = (f"Testing temporal_distance: {temporal_distance}, loss_name: {loss_name}, gridsize={gridsize}" + 
                     f"model_name={model_name}, kernel_name={kernel_name}, normalization={normalization}, temperature={temperature}")
            result = prototypes_clustering(centroids, 
                                           annotator_id=annotator_id,
                                           round_number=round_number,
                                            explored_K = explored_K[cluster_type],
                                            model_name=model_name, 
                                            normalization=normalization,
                                            kernel_name=kernel_name,
                                            sigma_rbf=sigma_rbf,
                                            temperature=temperature,
                                            verbose=verbose)
            
            [res.update({'annotator_id': annotator_id, 'round_number': round_number, 'cluster_type': cluster_type, 'model_name': model_name}) for res in result]

            results.extend(result)

    results = pd.DataFrame(results)
    results['original_cluster_index'] = results.cluster_type.map(qualification_mapping[clustering_config_name])
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()
      
if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    print('Process registration by loading vector labels and timestamps, meta-prototypes basis, radius per clusters, noise encoding,   using config_name: {}'.format(args.config_name))
    t1 = time.time()
    print(f'Time elapsed: {t1 - t0} seconds')
    
    sys.exit(0)
        