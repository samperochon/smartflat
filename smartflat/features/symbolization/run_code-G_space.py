
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import binary_closing, binary_opening, label



import random
import time
from collections import Counter

import aeon

#import umap.umap_ as umap
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import ot
import scipy.cluster.hierarchy as sch
import scipy.linalg
import seaborn as sns
import torch
from decord import bridge
from IPython.display import clear_output, display
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_kernels

import api
from smartflat.annotation_smartflat import add_ground_truth_labels
from smartflat.configs.loader import import_config
from smartflat.constants import available_modality, gaze_features, progress_cols
from smartflat.datasets.dataset_gaze import compute_segments_gaze_features, populate_gaze_data
from smartflat.datasets.filter import filter_progress_cols
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import (
    add_covar_label,
    add_pca,
    add_pca_3d_grouped,
    add_umap,
    load_embedding_dimensions,
    load_embeddings,
    use_light_dataset,
)
from smartflat.engine.builders import build_model, compute_metrics
from smartflat.engine.clustering import predict_with_centroids
from smartflat.features.symbolization.main import (
    build_clusterdf,
    clustering_prototypes_space,
    define_symbolization,
    filter_prototypes_per_prevalence,
)
from smartflat.features.symbolization.main_prototypes_annotation import main
from smartflat.features.symbolization.utils import (
    compute_centroid_distances,
    compute_cluster_probabilities,
    compute_sample_entropy,
    compute_threshold_per_cluster_index,
    get_prototypes,
    propagate_segment_labels_to_embeddings,
    qc_sanity_check_filters,
    reduce_segments_labels,
    retrieve_segmentation_costs,
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
from smartflat.metrics import plot_clustering_metrics
from smartflat.utils.utils import pairwise, upsample_sequence
from smartflat.utils.utils_coding import green, purple
from smartflat.utils.utils_dataset import (
    add_cum_sum_col,
    check_train_data,
    collapse_cluster_stats,
    compute_matrix_stats,
    normalize_data,
    quantize_signal,
    sample_uniform_rows_by_col,
)
from smartflat.utils.utils_io import (
    fetch_has_gaze,
    fetch_qualification_mapping,
    get_data_root,
    get_video_loader,
    load_df,
    save_df,
)
from smartflat.utils.utils_visualization import (
    dynamic_row_plot_func,
    get_base_colors,
    plot_chronogames,
    plot_distance_evolution,
    plot_gram,
    plot_labels_2D_encoding,
    plot_per_cat_x_cont_y_distributions,
    plot_qualification_mapping,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()


import time
from itertools import product

import matplotlib.pyplot as plt
import ot
import scipy.linalg
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels, rbf_kernel

from smartflat.configs.loader import import_config
from smartflat.features.symbolization.co_clustering import (
    benchmark_over_alpha,
    clustering_prototypes_space,
    compute_reduced_centroids,
    compute_combined_distance,
    compute_combined_distance_across_temperatures,
    compute_distance_matrix,
    compute_latent_distance,
    compute_multimodal_matrices,
    compute_temporal_distance,
    fetch_missing_temporal_prototypes,
    get_prototypes_mapping,
    hierarchical_prototypes_clustering,
    max_normalize_distance,
    plot_clustering_metrics,
    prototypes_clustering,
)
from smartflat.features.symbolization.inference_reduction_prototypes_distances import (
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
    check_pairwise_distance_matrix,
    get_prototypes,
    reduce_similarity_matrix_per_cluster_type,
)
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.utils.utils_dataset import (
    add_cum_sum_col,
    compute_matrix_stats,
    normalize_data,
)
from smartflat.utils.utils_io import fetch_qualification_mapping
from smartflat.utils.utils_visualization import (
    dynamic_row_plot_func,
    plot_chronogames,
    plot_gram,
    plot_labels_2D_encoding,
    plot_per_cat_x_cont_y_distributions,
    plot_qualification_mapping,
)
from multiprocessing import Pool
from itertools import chain

if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    use_multiprocessing = True
    annotator_id = 'samperochon'
    round_number = 8

    config_name = 'SymbolicSourceInferenceGoldConfig' #'SymbolicSourceInferenceGoldConfig'#'SymbolicSourceCosineCroppedInferenceConfig'
    model_names = ['hierarchical-clustering']
    #model_names = ['spectral', 'hdbscan', 'gmm', 'bayesian_gmm']

    linkage_methods = ['complete']#, 'average', 'single']
    metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation", "modularity"]

    input_space = 'K_space'
    use_K_space = True if input_space == 'K_space' else False
    verbose=True
    overwrite_gw_distances = False
    do_normalize_distances = True
    overwrite_results = False
    check_results = False

    # ----------

    # Reduce prototypes redundancy as per the silhouette score (CS)
    config = import_config(config_name)
    gridsize = config.gridsize
    loss_name = config.loss_name
    temporal_distance = config.temporal_distance 
    temperature_tau = config.temperature_tau; temperature_tau_K = temperature_tau['K_space'];temperature_tau_G = temperature_tau['G_space']
    temperature_x = config.temperature_x; temperature_x_K = temperature_x['K_space'];temperature_x_G = temperature_x['G_space']
    agg_fn_str = config.agg_fn_str #TODO: test with np.max
    agg_fn = np.mean if agg_fn_str == 'mean' else np.max if agg_fn_str == 'max' else np.median if agg_fn_str == 'median' else id

    distance_aggregate_col = config.distance_aggregate_col
    multimodal_method = config.multimodal_method
    alpha = config.alpha


    #kernel_name = config.model_params['kernel_name']
    # sigma_rbf = config.model_params['sigma_rbf']
    # temperature = config.model_params['temperature']

    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(config.clustering_config_name)
    cluster_types = clustering_config.model_params['cluster_types']
    normalization = clustering_config.model_params['normalization']
    kernel_name = clustering_config.model_params['kernel_name']

    # Get prototypes mapping 
    qualification_mapping = fetch_qualification_mapping(verbose=False)
    cluster_types_size = {cluster_type: len(v) for cluster_type, v in qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'].items()}

    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrix'); os.makedirs(output_folder, exist_ok=True)    
    experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrix'); os.makedirs(experiments_folder, exist_ok=True)    
    results_path = os.path.join(output_folder, "results_co_clustering_final_fg.pkl")

    results_path = os.path.join(output_folder, "results_co_clustering_final_test_10092025_fixed.pkl")
    #results_path = os.path.join(output_folder, "results_co_clustering_final.pkl")

    # Covariates
    kernel_names = ['cosine'] #'gaussian_rbf']
    multimodal_methods = ['multiplicative',  'without'] # unnormalized_exp
    alpha_benchmark = [0, 0.62, 0.75, 0.88]#, 1] # Temporary 
    #method_names = ['temporal_similarity', 'knn', 'temporal']
    loss_names =  ['square_loss']#, 'kl_loss']
    alpha_values =np.linspace(0, 1, 9).round(2); print(f'Alpha values: {alpha_values}')
    # RBF Kernel 
    explored_temperatures = {'K_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                            'temperatures':[ 1, 5, 10, 50, 100, 1000]},
                            }
    explored_temperatures = {'K_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                        'temperatures':[ 1, 5, 10, 50, 100, 1000]},
                            'G_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                        'temperatures':[ 1, 5, 10, 50, 100, 1000]}
                        }

    kde_models_K = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space)
    excluded_clusters_K = fetch_missing_temporal_prototypes(kde_models_K, annotator_id, round_number, config_name, clustering_config_name, use_K_space=True)

    if not use_K_space:
        kde_models_G = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space)
        excluded_clusters_G = fetch_missing_temporal_prototypes(kde_models_G, annotator_id, round_number, config_name, clustering_config_name, use_K_space=False)
    else:
        kde_models_G = None
        excluded_clusters_G = None


    explored_K = {'K_space': {'foreground': np.unique(list(np.arange(15, len(kde_models_K['foreground']), 30)) ), #+ list(np.arange(75, 250, 5)) + list(np.arange(170, 185, 1)) ),
                            # 'task-definitive': np.arange(15, len(kde_models_K['foreground']), 35),
                            #         'exo-definitive': np.arange(15,  len(kde_models_K['exo-definitive']), 5),
                                #      'Noise': np.arange(15, len(kde_models_K['Noise']), 5)
                                    }
    }
    explored_K = {'K_space': {'foreground': np.unique(list(np.arange(173, 183, 1)) )}
    }

    # explored_K = {'K_space': {'foreground': np.arange(170, 185, 1),
    #                         'task-definitive': np.arange(15, len(kde_models_K['foreground']), 35),
    #                                 'exo-definitive': np.arange(15,  len(kde_models_K['exo-definitive']), 5),
    #                                 'Noise': np.arange(15, len(kde_models_K['Noise']), 5)}
    # }
    if not use_K_space:
        explored_K.update({
                # 'G_space': {'foreground': np.arange(50, len(kde_models_G['foreground'])-15, 1)}})
                        'G_space': {'foreground': np.arange(50, 80, 1)}})




    # TEST other clustering approaches 
    #model_names = ['spectral', 'dbscan', 'hdbscan', 'gmm', 'bayesian_gmm']


    # 1) Compute temporal distance matrices across prototypes in the latent space before/after reduction per cluster types
    D_tf_pc = compute_temporal_distance(kde_models_K,  annotator_id, round_number, config_name=config_name, kernel_name='pre_computing', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=None, overwrite_gw_distances=overwrite_gw_distances)
    if not use_K_space:
        D_tfr_pc = compute_temporal_distance(kde_models_G,  annotator_id, round_number, config_name=config_name, kernel_name='pre_computng', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, input_space=input_space, temperature_tau=None, overwrite_gw_distances=overwrite_gw_distances)
    else:
        D_tfr_pc = None

    # Load existing results if available
    if os.path.exists(results_path) and not overwrite_results:
        existing_df = load_df(results_path)
        results = existing_df.to_dict('records')
    else:
        existing_df = pd.DataFrame(columns=['model_name', 'annotator_id', 'round_number',  
                                                'n_exp', 'input_space',   'alpha', 'n_clusters', 
                                            'linkage_method', 
                                            'kernel_name', 'multimodal_method', 'loss_name', 'normalization', 'temperature_x', 'temperature_tau', 'temporal_distance', 'gridsize', 'agg_fn_str', 'cluster_type'] + metrics, index=[-1])
        results = []
        
        
    
    print(f'Len of existing results: {len(existing_df)}')
    n_exp=0
    # Test loop to print experiments to perform n_exp=0
    for model_name in model_names:
        for loss_name in loss_names:
            for multimodal_method in multimodal_methods:    
                
                for kernel_name in kernel_names:
                    
                    # Already tested when multimodal_method == '_exp'
                    if kernel_name == 'gaussian_rbf' and multimodal_method == 'multiplicative':
                        continue
                    
                    
                    n_exp += 1
                    title = (f"E-{n_exp} use_K_space={use_K_space} model_name={model_name}  kernel_name={kernel_name}, multimodal_method={multimodal_method}, normalization={normalization}, multimodal_method={multimodal_method}\n" + 
                            f"temporal_distance={temporal_distance} (n_bins={gridsize}, loss_name={loss_name}, alpha={alpha}")
                    
                    
                    mask = (
                        (existing_df.model_name == model_name) &
                        (existing_df.input_space == input_space) &
                        (existing_df.loss_name == loss_name) &
                        (existing_df.multimodal_method == multimodal_method) &
                        (existing_df.temporal_distance == temporal_distance) &
                        (existing_df.kernel_name == kernel_name)
                    )
                    rows = existing_df[mask]

                    if model_name == 'hierarchical-clustering':
                        # Check completeness
                        expected = set(product(alpha_benchmark, linkage_methods))
                        observed = set(zip(rows.alpha, rows.linkage_method))
                        missing = expected - observed
                        all_done = not missing
                    else:
                        
                        existing_alphas = set(rows.alpha)
                        all_done = all(lm in alpha_benchmark for lm in existing_alphas)

                        missing = set(alpha_benchmark) - existing_alphas


                    # Skip or drop if done
                    if all_done and mask.any():
                        
                        if not overwrite_results:
                            'HERE'
                            print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name}")
                            continue
                        else:
                            existing_df = existing_df[~mask]
                    else:
                        print(f'Missing experiments for {model_name}, {loss_name}, {multimodal_method}, {kernel_name}: {missing}')
                                        
                                    
                    #title = f"Exp n={n_exp}\tTesting temporal_distance={temporal_distance}\tloss_name={loss_name}\tgridsize={gridsize}\t(ordered_index={ordered_index})\tmodel_name={model_name}\tkernel_name={kernel_name}\tnormalization={normalization}\ttemperature={temperature}\tmultimodal_method={multimodal_method}\talpha={alpha}\ttemperature_tau={temperature_tau}"
                    green("\n\n\n\n--------------------------------------------------")
                    green(title)
                    green("----------------------------------------------------------")
                    
                    for _alpha in alpha_benchmark:
                        
                        
                        mask = (
                        (existing_df.input_space == input_space) &
                        (existing_df.model_name == model_name) &
                        (existing_df.loss_name == loss_name) &
                        (existing_df.multimodal_method == multimodal_method) &
                        (existing_df.kernel_name == kernel_name) &
                        (existing_df.temporal_distance == temporal_distance) &
                        (existing_df.alpha == _alpha)
                        )
                        rows = existing_df[mask]


                        if model_name == 'hierarchical-clustering':
                            # Check completeness
                            # Check completeness
                            existing_linkage_method = set(rows.linkage_method)
                            all_linkage_done = all(lm in existing_linkage_method for lm in linkage_methods)
                            missing = set(linkage_methods) - existing_linkage_method

                        else:
                            
                            all_linkage_done = mask.any()
                            missing = []
                        


                        print(f"All done: {all_linkage_done} for {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}")
                        if all_linkage_done and mask.any():
                            if not overwrite_results:
                                print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}")
                                continue
                            else:
                                existing_df = existing_df[~mask]
                                            
                        else:
                            print(f'Missing experiments for {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}: {missing}')

                        green(f'1) Fitting prototypes clustering for {kernel_name} kernel, {multimodal_method} method, {loss_name} loss, {normalization} normalization')
                        green(f'Parameters: alpha={alpha} temperature_x={temperature_x_K}, temperature_tau={temperature_tau_K}, gridsize={gridsize}, temporal_distance={temporal_distance}, agg_fn={agg_fn_str}')
                        
                        
                        for linkage_method in linkage_methods:
                            
                            mask = (
                                    (existing_df.input_space == input_space) &
                                    (existing_df.model_name == model_name) &
                                    (existing_df.loss_name == loss_name) &
                                    (existing_df.multimodal_method == multimodal_method) &
                                    (existing_df.kernel_name == kernel_name) &
                                    (existing_df.alpha == alpha) &
                                    (existing_df.temporal_distance == temporal_distance) &
                                    (existing_df.linkage_method == linkage_method) 
                                    )

                            if mask.any():
                                if not overwrite_results:
                                    print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha} and linkage_method={linkage_method}")
                                    continue
                                else:
                                    existing_df = existing_df[~mask]
                                            
                            
                            purple('2) Running Hierarchical clustering')
                            
                            green(f'Saved intermediate exp {n_exp} results to {results_path}')


    print(f'Len of existing results: {len(existing_df)}')
    n_exp=0; start_time = time.time()
    for model_name in model_names:
        for loss_name in loss_names:
            for multimodal_method in multimodal_methods:    
                
                for kernel_name in kernel_names:
                    
                    # Already tested when multimodal_method == '_exp'
                    if kernel_name == 'gaussian_rbf' and multimodal_method == 'multiplicative':
                        continue
                    
                    
                    n_exp += 1
                    title = (f"E-{n_exp} input_space={input_space} kernel_name={kernel_name}, multimodal_method={multimodal_method}, normalization={normalization}, multimodal_method={multimodal_method}\n" + 
                            f"temporal_distance={temporal_distance} (n_bins={gridsize}, loss_name={loss_name}, alpha={alpha}")
                    
                        
                    mask = (
                        (existing_df.model_name == model_name) &
                        (existing_df.input_space == input_space) &
                        (existing_df.loss_name == loss_name) &
                        (existing_df.multimodal_method == multimodal_method) &
                        (existing_df.temporal_distance == temporal_distance) &
                        (existing_df.kernel_name == kernel_name)
                    )
                    rows = existing_df[mask]

                    if model_name == 'hierarchical-clustering':
                        # Check completeness
                        expected = set(product(alpha_benchmark, linkage_methods))
                        observed = set(zip(rows.alpha, rows.linkage_method))
                        missing = expected - observed
                        all_done = not missing
                    else:
                        
                        existing_alphas = set(rows.alpha)
                        all_done = all(lm in alpha_benchmark for lm in existing_alphas)

                        missing = set(alpha_benchmark) - existing_alphas


                    # Skip or drop if done
                    if check_results and all_done and mask.any():
                        
                        if not overwrite_results:
                            'HERE'
                            print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name}")
                            continue
                        else:
                            existing_df = existing_df[~mask]
                    else:
                        print(f'Missing experiments for {model_name}, {loss_name}, {multimodal_method}, {kernel_name}: {missing}')
                                        
                                            
                    
                    #title = f"Exp n={n_exp}\tTesting temporal_distance={temporal_distance}\tloss_name={loss_name}\tgridsize={gridsize}\t(ordered_index={ordered_index})\tmodel_name={model_name}\tkernel_name={kernel_name}\tnormalization={normalization}\ttemperature={temperature}\tmultimodal_method={multimodal_method}\talpha={alpha}\ttemperature_tau={temperature_tau}"
                    green("\n\n\n\n--------------------------------------------------")
                    green(title)
                    green("----------------------------------------------------------")
                    
                    # 1) Load centroid vectors per cluster types
                    mu_kf = compute_reduced_centroids(config_name, annotator_id, round_number=round_number,  normalization=normalization, input_space=input_space, excluded_clusters=excluded_clusters_K if use_K_space else excluded_clusters_G, agg_fn=agg_fn)

                    # 2) Compute distance matrices across prototypes in the latent space before/after reduction per cluster types
                    compute_similarity = True if kernel_name == 'gaussian_rbf' else False
                    D_xf = compute_latent_distance(config_name, annotator_id, round_number, compute_similarity=compute_similarity, input_space=input_space, kernel_name=kernel_name, normalization=normalization, temperature_x=temperature_x_K if use_K_space else temperature_x_G, excluded_clusters=excluded_clusters_K if use_K_space else excluded_clusters_G, agg_fn=agg_fn)



                    # 3) Experiment: Estimate optimal temperature parameters for spatio-temporal distance (maximizing IQR)
                    # figpath = os.path.join(output_folder, f'combined_distance_matrix_task_K_{n_exp}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}.png')
                    # D_est_KT, S_est_KT, D_x_est_KT, D_t_est_KT = compute_combined_distance_across_temperatures(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name, explored_temperatures['K_space']['temperatures'], explored_temperatures['K_space']['temperatures_tau'], 
                    #                                                                                             D_tt_pc=D_tt_pc,  D_te_pc=D_te_pc,  D_tf_pc=D_tf_pc, task_type='task-definitive', use_K_space=True, normalization=normalization, kernel_name=kernel_name, agg_fn=agg_fn, 
                    #                                                                                             gridsize=gridsize, loss_name=loss_name, temporal_distance=temporal_distance, temperature_tau=temperature_tau_K, temperature_x=temperature_x_K, 
                    #                                                                                             multimodal_method=multimodal_method, alpha=0.5, title=title, figpath=figpath, do_show=verbose)

                    # figpath = os.path.join(output_folder, f'combined_distance_matrix_exo_K_{n_exp}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}.png')
                    # D_est_KE, S_est_KE, D_x_est_KE, D_t_est_KE = compute_combined_distance_across_temperatures(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name, explored_temperatures['K_space']['temperatures'], explored_temperatures['K_space']['temperatures_tau'], 
                    #                                                                                             D_tt_pc=D_tt_pc,  D_te_pc=D_te_pc,  D_tf_pc=D_tf_pc, task_type='exo-definitive', use_K_space=True, normalization=normalization, kernel_name=kernel_name, agg_fn=agg_fn, 
                    #                                                                                             gridsize=gridsize, loss_name=loss_name, temporal_distance=temporal_distance,  temperature_tau=temperature_tau_K, temperature_x=temperature_x_K, 
                    #                                                                                             multimodal_method=multimodal_method, alpha=0.5, title=title, figpath=figpath, do_show=verbose)

                    if use_K_space:
                        
                        figpath = os.path.join(output_folder, f'combined_distance_matrix_foreground_K_{n_exp}_{model_name}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}.png')
                        D_est_KF, S_est_KF, D_x_est_KF, D_t_est_KF = compute_combined_distance_across_temperatures(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name, explored_temperatures['K_space']['temperatures'], explored_temperatures['K_space']['temperatures_tau'], 
                                                                                                                        D_tf_pc=D_tf_pc, task_type='foreground', input_space=input_space, normalization=normalization, kernel_name=kernel_name, agg_fn=agg_fn, 
                                                                                                                    gridsize=gridsize, loss_name=loss_name, temporal_distance=temporal_distance,  temperature_tau=temperature_tau_K, temperature_x=temperature_x_K, 
                                                                                                                    multimodal_method=multimodal_method, alpha=0.5, title=title, figpath=figpath, do_show=verbose)
                    else:

                        # G-space
                        figpath = os.path.join(output_folder, f'combined_distance_matrix_foreground_G_{n_exp}_{model_name}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_G}_{temperature_tau_G}.png')
                        D_est_KF, S_est_KF, D_x_est_KF, D_t_est_KF = compute_combined_distance_across_temperatures(kde_models_G, annotator_id, round_number, excluded_clusters_G, config_name, explored_temperatures['G_space']['temperatures'], explored_temperatures['G_space']['temperatures_tau'],
                                                                                                                        D_tf_pc=D_tfr_pc, task_type='foreground', input_space=input_space, normalization=normalization, kernel_name=kernel_name, agg_fn=agg_fn, 
                                                                                                                    gridsize=gridsize, loss_name=loss_name, temporal_distance=temporal_distance, temperature_tau=temperature_tau_G, temperature_x=temperature_x_G, 
                                                                                                                    multimodal_method=multimodal_method, alpha=0.5, title=title, figpath=figpath, do_show=verbose)


                    # 4) Experiment: Estimate optimal alpha parameters for balancing spatio-temporal distance
                    # figpath = os.path.join(output_folder, f'alpha_combined_distance_matrix_task_alpha_temp_opt_K_{n_exp}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}_{alpha}.png')
                    # D_est_KTa, S_est_KTa, D_x_est_KTa, D_t_est_KTa = benchmark_over_alpha(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name,  temperature_x=temperature_x_K, temperature_tau=temperature_tau_K,
                    #                                                                     D_tt_pc=D_tt_pc, D_te_pc=D_te_pc, D_tf_pc=D_tf_pc, normalization=normalization, kernel_name=kernel_name, gridsize=gridsize, 
                    #                                                                     loss_name=loss_name, temporal_distance=temporal_distance, multimodal_method=multimodal_method, alpha_values=alpha_values, agg_fn=agg_fn, 
                    #                                                                     task_type='task-definitive', use_K_space=True, title='', figpath=None, do_show=True)

                    # figpath = os.path.join(output_folder,  f'alpha_combined_distance_matrix_exo_alpha_temp_opt_K_{n_exp}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}_{alpha}.png')
                    # D_est_KEa, S_est_KEa, D_x_est_KEa, D_t_est_KEa = benchmark_over_alpha(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name,  temperature_x=temperature_x_K, temperature_tau=temperature_tau_K,
                    #                                                                     D_tt_pc=D_tt_pc, D_te_pc=D_te_pc, D_tf_pc=D_tf_pc, normalization=normalization, kernel_name=kernel_name, gridsize=gridsize, 
                    #                                                                     loss_name=loss_name, temporal_distance=temporal_distance, multimodal_method=multimodal_method, alpha_values=alpha_values, agg_fn=agg_fn, 
                    #                                                                     task_type='exo-definitive', use_K_space=True, title='', figpath=None, do_show=True)

                    for alpha in alpha_benchmark:
                        
                        mask = (
                        (existing_df.model_name == model_name) &
                        (existing_df.input_space == input_space) &
                        (existing_df.loss_name == loss_name) &
                        (existing_df.multimodal_method == multimodal_method) &
                        (existing_df.kernel_name == kernel_name) &
                        (existing_df.temporal_distance == temporal_distance) &
                        (existing_df.alpha == alpha)
                        )
                        rows = existing_df[mask]


                        if model_name == 'hierarchical-clustering':
                            # Check completeness
                            # Check completeness
                            existing_linkage_method = set(rows.linkage_method)
                            all_linkage_done = all(lm in existing_linkage_method for lm in linkage_methods)
                            missing = set(linkage_methods) - existing_linkage_method

                        else:
                            
                            all_linkage_done = mask.any()
                            missing = []
                        


                        print(f"All done: {all_linkage_done} for {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}")
                        if check_results and  all_linkage_done and mask.any():
                            if not overwrite_results:
                                print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}")
                                continue
                            else:
                                existing_df = existing_df[~mask]
                                            
                        else:
                            print(f'Missing experiments for {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha}')
                            print('Missing: ', missing)
                                                
                        if use_K_space:
                            
                            figpath = os.path.join(output_folder,  f'alpha_combined_distance_matrix_for_alpha_temp_opt_K_{n_exp}_{model_name}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}_{alpha}.png')
                            D_est_KFa, S_est_KFa, D_x_est_KFa, D_t_est_KFa = benchmark_over_alpha(kde_models_K, annotator_id, round_number, excluded_clusters_K, config_name, temperature_x=temperature_x_K, temperature_tau=temperature_tau_K,
                                                                                                D_tf_pc=D_tf_pc, normalization=normalization, kernel_name=kernel_name, gridsize=gridsize, alpha=alpha, 
                                                                                                loss_name=loss_name, temporal_distance=temporal_distance, multimodal_method=multimodal_method, alpha_values=alpha_values, agg_fn=agg_fn, 
                                                                                                task_type='foreground', input_space=input_space, title='', figpath=figpath, do_show=True)
                        else:
                            figpath = os.path.join(output_folder, f'alpha_combined_distance_matrix_for_alpha_temp_opt_G_{n_exp}_{model_name}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_G}_{temperature_tau_G}_{alpha}.png')
                            D_est_KFa, S_est_KFa, D_x_est_KFa, D_t_est_KFa  = benchmark_over_alpha(kde_models_G, annotator_id, round_number, excluded_clusters_G, config_name, temperature_x=temperature_x_K, temperature_tau=temperature_tau_K,
                                                                                                                        D_tf_pc=D_tfr_pc, normalization=normalization, kernel_name=kernel_name, gridsize=gridsize, alpha=alpha, 
                                                                                                                        loss_name=loss_name, temporal_distance=temporal_distance, multimodal_method=multimodal_method, alpha_values=alpha_values, agg_fn=agg_fn, 
                                                                                                                        task_type='foreground', input_space=input_space, title='', figpath=figpath, do_show=True)



                        # 4) Visulizations
                        use_minmax = False
                        # 4) Visulizations
                        distance_matrices = {
                            # K-latent space
                            "Foreground K-Latent Distance (D_xf)": max_normalize_distance(D_xf) if use_minmax else D_xf,
                            # "Task K-latent space (D_xt)": max_normalize_distance(D_xt) if use_minmax else D_xt,
                            # "Exo K-latent space (D_xe)": max_normalize_distance(D_xe) if use_minmax else D_xe,
                            
                            # "Foreground K-Temporal Distance (raw) (D_tt_pc) ":  max_normalize_distance(D_tt_pc) if use_minmax else D_tt_pc,
                            # "Task K-temporal Space (D_te_pc)": max_normalize_distance(D_te_pc) if use_minmax else D_te_pc,
                            # "Exo K-temporal Space (D_tf_pc)": max_normalize_distance(D_tf_pc) if use_minmax else D_tf_pc,
                            
                            "Foreground K-Temporal Distance (temp) (D_t_est_KFa) ":  max_normalize_distance(D_t_est_KFa) if use_minmax else D_t_est_KFa,
                            # "Task K-temporal Space (D_tt)": max_normalize_distance(D_t_est_KT)  if use_minmax else D_t_est_KT,
                            # "Exo K-temporal Space (D_te)": max_normalize_distance(D_t_est_KE)if use_minmax else D_t_est_KE,
                            
                            
                            f"Foreground K-Space Distance (joint scaling) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KF) if use_minmax else D_est_KF,
                            # f"Task K-temporal Space (D_est_KT) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KT) if use_minmax else D_est_KT,
                            # f"Exo K-temporal Space (D_est_KE) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KE) if use_minmax else D_est_KE,
                            
                            f"Foreground Final K-Space Distance (space/time balance) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KFa) if use_minmax else D_est_KFa,
                            # f"Task K-temporal Space (D_est_KTa) {kernel_name}/{multimodal_method} (D)": max_normalize_distance(D_est_KTa) if use_minmax else D_est_KTa,
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

                        figpath = os.path.join(output_folder, f'All_distance_matrix_{input_space}_{n_exp}_{kernel_name}_{multimodal_method}_{loss_name}_{normalization}_{temperature_x_K}_{temperature_tau_K}_{alpha}.png')
                        plot_distances_matrix_tuples(distance_matrices, n_rows=2, n_cols=2, suptitle=f'{annotator_id}-round: {round_number}\nPrototypes Distances Matrices before/after reduction mapping (K->G) per cluster-types\n'+title, figsize=(25, 15), figpath=figpath)
                        #plt.close()
                        green(f'Saved figure to {figpath}')
                        
                        # Save results


                        green(f'Fitting prototypes clustering for {kernel_name} kernel, {multimodal_method} method, {loss_name} loss, {normalization} normalization')
                        green(f'Parameters: alpha={alpha} temperature_x={temperature_x_K}, temperature_tau={temperature_tau_K}, gridsize={gridsize}, temporal_distance={temporal_distance}, agg_fn={agg_fn_str}')
                        for D_M, D_X, D_T, S_M, mu, cluster_type, temperature_x, temperature_tau in zip([D_est_KFa], #,  D_est_KT, D_est_KE], 
                                                                                                        [D_x_est_KF], #, D_x_est_KT, D_x_est_KE],
                                                                                                        [D_t_est_KF], #, D_t_est_KT, D_t_est_KE, ],
                                                                                                        [S_est_KF], #, S_est_KT, S_est_KE,], 
                                                                                                        [mu_kf], #, mu_kt, mu_ke], 
                                                                                                        ['foreground'], #, 'task-definitive', 'exo-definitive'],
                                                                                                        #['K_space'], #, 'K_space', 'K_space'],
                                                                                                        [temperature_x_K], #, temperature_x_K, temperature_x_K],
                                                                                                        [temperature_tau_K]): #, temperature_tau_K, temperature_tau_K]):


                            D_M = D_M.copy()
                            D_X = D_X.copy()
                            D_T = D_T.copy()
                            
                            print(f'Original elements of matrices (kernel={kernel_name}): {D_M[0, 0]}, {D_X[0, 0]}, {D_T[0, 0]}')

                            # comnpute similarity for each context
                            if kernel_name == 'euclidean':
                                
                                S_X = 1 / (1 + D_X)
                                S_X = (S_X - S_X.min()) / (S_X.max() - S_X.min())
                                S_T = 1 / (1 + D_T)
                                S_T = (S_T - S_T.min()) / (S_T.max() - S_T.min())
                                
                            elif kernel_name == 'cosine':
                                
                                S_X = 1 - D_X
                                #S_X = (S_X - S_X.min()) / (S_X.max() - S_X.min())
                                S_T = 1 - D_T
                                #S_T = (S_T - S_T.min()) / (S_T.max() - S_T.min())
                                
                            elif kernel_name == 'gaussian_rbf':        
                                # Here for [G] spaces where the diagonal reflects within meta-cluster distances, we set
                                # diagonal to 1 (D are transported as similarity matrices in the case of gaussian_rbf kernel)
                                # np.fill_diagonal(D_M, 1)
                                # np.fill_diagonal(D_X, 1)
                                # np.fill_diagonal(D_T, 1)
                                
                                S_X = D_X.copy()
                                S_T = D_T.copy()
                                
                                D_X = 1 - D_X
                                #D_X = (D_X - D_X.min()) / (D_X.max() - D_X.min())
                                D_T = 1 - D_T
                                #D_T = (D_T - D_T.min()) / (D_T.max() - D_T.min())
                                
                                S_M = 1 - D_M
                                #D_M = (D_M - D_M.min()) / (D_M.max() - D_M.min())
                            
                            print(f'Final elements of matrices (kernel={kernel_name}): {D_M[0, 0]}, {D_X[0, 0]}, {D_T[0, 0]}')

                            #check_pairwise_distance_matrix(D_M)
                            #check_pairwise_distance_matrix(D_X)
                            #check_pairwise_distance_matrix(D_T)
                            
                            for linkage_method in linkage_methods:
                                
                                mask = (
                                        (existing_df.input_space == input_space) &
                                        (existing_df.model_name == model_name) &
                                        (existing_df.loss_name == loss_name) &
                                        (existing_df.multimodal_method == multimodal_method) &
                                        (existing_df.kernel_name == kernel_name) &
                                        (existing_df.alpha == alpha) &
                                        (existing_df.temporal_distance == temporal_distance) &
                                        (existing_df.linkage_method == linkage_method) 
                                        )

                                if check_results and mask.any():
                                    if not overwrite_results:
                                        print(f"Skipping already completed experiment: {model_name}, {loss_name}, {multimodal_method}, {kernel_name} with alpha={alpha} and linkage_method={linkage_method}")
                                        continue
                                    else:
                                        existing_df = existing_df[~mask]
                                        
                                        

                                def run_clustering_for_K(K):
                                    result = hierarchical_prototypes_clustering(
                                        centroids=mu, 
                                        D_M=D_M,
                                        D_X=D_X,
                                        D_T=D_T,
                                        S_M=S_M,
                                        S_X=S_X,
                                        S_T=S_T,
                                        explored_K=[K],  # pass as single-element list or adapt inside the function
                                        model_name=model_name, 
                                        linkage_method=linkage_method,
                                        kernel_name=kernel_name,
                                        normalization=normalization,
                                        temperature=temperature_x,
                                        output_folder=output_folder,
                                        suffix=f"{n_exp}_{kernel_name}_{model_name}_{multimodal_method}_{loss_name}_{normalization}_{alpha}_{temperature_x}_{linkage_method}_{temperature_tau}_{cluster_type}_{input_space}_{temporal_distance}_K{K}",
                                        verbose=verbose
                                    )
                                    return result  # list of dicts

                                if use_multiprocessing: 
                                    K_list = explored_K[input_space][cluster_type]
                                    green(f'Exploring K values: {K_list}')

                                    with Pool(processes=30) as pool:
                                        all_results = pool.map(run_clustering_for_K, K_list)  # list of list of dicts

                                    # Flatten and enrich with common metadata
                                    up_dict = {
                                        'annotator_id': annotator_id, 
                                        'round_number': round_number, 
                                        'n_exp': n_exp, 
                                        'cluster_type': cluster_type, 
                                        'input_space': input_space, 
                                        'model_name': model_name, 
                                        'kernel_name': kernel_name, 
                                        'multimodal_method': multimodal_method,
                                        'loss_name': loss_name, 
                                        'normalization': normalization, 
                                        'temperature_x': temperature_x, 
                                        'temperature_tau': temperature_tau, 
                                        'alpha': alpha,
                                        'temporal_distance': temporal_distance, 
                                        'gridsize': gridsize, 
                                        'agg_fn': agg_fn_str
                                    }

                                    for result in chain.from_iterable(all_results):  # flatten
                                        result.update(up_dict)
                                        results.append(result)
                                                                                        
                                else:
                                    
                                
                                    result = hierarchical_prototypes_clustering(centroids=mu, 
                                                                                D_M=D_M,
                                                                                D_X=D_X,
                                                                                D_T=D_T,
                                                                                S_M=S_M,
                                                                                S_X=S_X,
                                                                                S_T=S_T,
                                                                                explored_K = explored_K[input_space][cluster_type],
                                                                                model_name=model_name, 
                                                                                linkage_method=linkage_method,
                                                                                kernel_name=kernel_name,
                                                                                normalization=normalization,
                                                                                temperature=temperature_x,
                                                                                output_folder=output_folder,
                                                                                suffix = f"{n_exp}_{kernel_name}_{model_name}_{multimodal_method}_{loss_name}_{normalization}_{alpha}_{temperature_x}_{linkage_method}_{temperature_tau}_{cluster_type}_{input_space}",
                                                                                verbose=verbose)
                                    up_dict = {'annotator_id': annotator_id, 'round_number': round_number, 'n_exp': n_exp, 'cluster_type': cluster_type, 'input_space': input_space, 'model_name': model_name, 'kernel_name': kernel_name, 'multimodal_method': multimodal_method,
                                            'loss_name': loss_name, 'normalization': normalization, 'temperature_x': temperature_x, 'temperature_tau': temperature_tau, 'alpha': alpha,
                                            'temporal_distance':temporal_distance, 'gridsize': gridsize, 'agg_fn': agg_fn_str}
                                    [res.update(up_dict) for res in result];results.extend(result)
                            
                                    result = prototypes_clustering(centroids=mu, 
                                                                                D_M=D_M,
                                                                                D_X=D_X,
                                                                                D_T=D_T,
                                                                                S_M=S_M,
                                                                                S_X=S_X,
                                                                                S_T=S_T,
                                                                                explored_K = explored_K[input_space][cluster_type],
                                                                                model_name=model_name, 
                                                                                kernel_name=kernel_name,
                                                                                normalization=normalization,
                                                                                temperature=temperature_x,
                                                                                output_folder=output_folder,
                                                                                suffix = f"{n_exp}_{kernel_name}_{model_name}_{multimodal_method}_{loss_name}_{normalization}_{alpha}_{temperature_x}_{linkage_method}_{temperature_tau}_{cluster_type}_{input_space}",
                                                                                verbose=verbose)
                                                                        

                                
                                print(f"Total computation for {cluster_type} ({input_space}): {time.time() - start_time:.4f} seconds")

                                resultdf =  pd.DataFrame([result])
                                if verbose:
                                    display(resultdf)  
                                    pass#plot_clustering_metrics(resultdf, hue='cluster_type')
                                
                                # Save intermediate results
                                interm_df_results = pd.DataFrame(results)
                                #interm_df_results['original_cluster_index'] = interm_df_results.cluster_type.map(qualification_mapping[clustering_config_name])
                                save_df(interm_df_results, results_path)
                                #interm_df_results.to_csv(results_path, index=False) 
                                print(f'Saved intermediate exp {n_exp} results to {results_path}')

                        print(f'Time elapsed: {time.time() - start_time:.2f} seconds')     









    # if __name__ == '__main__':
        
    #     args = parse_args()
    #     t0 = time.time()
        
        
            
    #     import time
    #     from collections import Counter, defaultdict

    #     import fastcluster
    #     import matplotlib.pyplot as plt
    #     import ot
    #     import scipy.cluster.hierarchy as sch
    #     import scipy.linalg
    #     import seaborn as sns
    #     import torch
    #     from IPython.display import display
    #     from matplotlib.colors import ListedColormap
    #     from scipy.optimize import linear_sum_assignment
    #     from scipy.spatial.distance import cdist, pdist, squareform
    #     from sklearn.cluster import SpectralClustering
    #     from sklearn.decomposition import PCA
    #     from sklearn.manifold import TSNE
    #     from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

    #     from smartflat.configs.loader import import_config
    #     from smartflat.engine.builders import build_metrics, compute_metrics

    #     # This experiment is only to test across with or without the latent space initialization
    #     from smartflat.features.symbolization.co_clustering import (
    #         clustering_prototypes_space,
    #         compute_reduced_centroids,
    #         compute_combined_distance,
    #         compute_combined_distance_across_temperatures,
    #         compute_distance_matrix,
    #         compute_latent_distance,
    #         compute_multimodal_matrices,
    #         compute_temporal_distance,
    #         fetch_missing_temporal_prototypes,
    #         get_prototypes_mapping,
    #         max_normalize_distance,
    #         plot_clustering_metrics,
    #         prototypes_clustering,
    #     )
    #     from smartflat.features.symbolization.inference_reduction_prototypes_distances import (
    #         aggomerate_distance_matrix,
    #         plot_distances_matrix_tuples,
    #         reduce_similarity_matrix,
    #     )
    #     from smartflat.features.symbolization.main import (
    #         build_clusterdf,
    #         define_symbolization,
    #         filter_prototypes_per_prevalence,
    #     )
    #     from smartflat.features.symbolization.temporal_distributions_estimation import (
    #         compute_gw_costs_and_transport_maps,
    #         estimate_kde_model,
    #         fit_kde_per_cluster,
    #         plot_clusters_kde_from_model,
    #         plot_gw_temporal_distance,
    #         plot_kde_from_model,
    #         prepare_row_timestamps_dataset,
    #         sort_kde_by_mean_timestamp,
    #         visualize_paiwise_gw,
    #     )
    #     from smartflat.features.symbolization.utils import (
    #         check_pairwise_distance_matrix,
    #         compute_centroid_distances,
    #         compute_cluster_probabilities,
    #         compute_sample_entropy,
    #         compute_threshold_per_cluster_index,
    #         get_prototypes,
    #         propagate_segment_labels_to_embeddings,
    #         qc_sanity_check_filters,
    #         reduce_centroid_vectors_per_cluster_type,
    #         reduce_segments_labels,
    #         reduce_similarity_matrix_per_cluster_type,
    #         retrieve_segmentation_costs,
    #         update_segmentation_from_embedding_labels,
    #     )
    #     from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
    #     from smartflat.features.symbolization.visualization import (
    #         compute_inconsistency_across_depth,
    #         compute_inconsistency_across_thresholds,
    #         explore_inconsistency_depth_threshold,
    #         plot_agreement_arrays,
    #         plot_binary_arrays,
    #         plot_centroid_cluster_distances_distributions,
    #         plot_centroid_cluster_distances_distributions_sbj,
    #         plot_cluster_prevalence,
    #         plot_cluster_prevalence_subplots,
    #         plot_distances_distribution_subplots,
    #         plot_frame_votes,
    #         plot_labels_counts_per_diagnosis,
    #         plot_labels_distributions_subplots,
    #         plot_latent_space_with_labels,
    #         plot_latent_space_with_labels_all,
    #         plot_sbj_cluster_distances_distribution_subplots,
    #         plot_segments_costs,
    #         plot_symbolization_variables,
    #         plot_symbols_frequency_subplots,
    #     )
    #     from smartflat.metrics import plot_clustering_metrics
    #     from smartflat.utils.utils_coding import *
    #     from smartflat.utils.utils_dataset import (
    #         check_train_data,
    #         compute_matrix_stats,
    #         normalize_data,
    #     )
    #     from smartflat.utils.utils_io import (
    #         fetch_qualification_mapping,
    #         get_data_root,
    #         load_df,
    #         save_df,
    #     )
    #     from smartflat.utils.utils_visualization import (
    #         dynamic_row_plot_func,
    #         plot_chronogames,
    #         plot_gram,
    #         plot_labels_2D_encoding,
    #         plot_per_cat_x_cont_y_distributions,
    #         plot_qualification_mapping,
    #     )

    #     # # This experiment is only to test across with or without the latent space initialization
    #     # config_name = 'SymbolicSourceCosineCroppedInferenceConfig'#'SymbolicSourceCosineCroppedInferenceConfig'
    #     # model_names = ['hierarchical-clustering']
    #     # metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation", "modularity"]
    #     # use_K_space = True
    #     # verbose=True
    #     #

    #     annotator_id = 'samperochon'
    #     round_number = 8

    #     config_name = 'SymbolicSourceInferenceGoldConfig'#'SymbolicSourceCosineCroppedInferenceConfig'
    #     model_names = ['hierarchical-clustering']
    #     metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation", "modularity"]
    #     use_K_space = True
    #     use_K_space_str = 'K_space' if use_K_space else 'G_space'
    #     verbose=True
    #     overwrite_gw_distances = False
    #     do_normalize_distances = True
    #     overwrite_results = True

    #     # ----------

    #     # Reduce prototypes redundancy as per the silhouette score (CS)
    #     config = import_config(config_name)
    #     gridsize = config.gridsize
    #     loss_name = config.loss_name
    #     temporal_distance = config.temporal_distance 
    #     temperature_tau = config.temperature_tau; temperature_tau_K = temperature_tau['K_space'];temperature_tau_G = temperature_tau['G_space']
    #     temperature_x = config.temperature_x; temperature_x_K = temperature_x['K_space'];temperature_x_G = temperature_x['G_space']
    #     agg_fn_str = config.agg_fn_str #TODO: test with np.max
    #     agg_fn = np.mean if agg_fn_str == 'mean' else np.max if agg_fn_str == 'max' else np.median if agg_fn_str == 'median' else id

    #     distance_aggregate_col = config.distance_aggregate_col
    #     multimodal_method = config.multimodal_method
    #     alpha = config.alpha


    #     #kernel_name = config.model_params['kernel_name']
    #     # sigma_rbf = config.model_params['sigma_rbf']
    #     # temperature = config.model_params['temperature']

    #     clustering_config_name = config.clustering_config_name
    #     clustering_config = import_config(config.clustering_config_name)
    #     cluster_types = clustering_config.model_params['cluster_types']
    #     normalization = clustering_config.model_params['normalization']
    #     kernel_name = clustering_config.model_params['kernel_name']

    #     # Get prototypes mapping 
    #     qualification_mapping = fetch_qualification_mapping(verbose=False)
    #     cluster_types_size = {cluster_type: len(v) for cluster_type, v in qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'].items()}

    #     output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, config.config_name); os.makedirs(output_folder, exist_ok=True)    
    #     experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, config.config_name); os.makedirs(experiments_folder, exist_ok=True)    
    #     results_path = os.path.join(output_folder, "results_co_clustering.csv")

    #     # Covariates
    #     kernel_names = ['euclidean', 'gaussian_rbf', 'cosine']
    #     multimodal_methods = ['normalized_exp', 'multiplicative', 'additive', 'without'] # unnormalized_exp
    #     #method_names = ['temporal_similarity', 'knn', 'temporal']
    #     loss_names =  ['square_loss']#, 'kl_loss']

    #     #loss_names = ['square_loss', 'kl_loss', 'mmd_loss']

    #     from smartflat.features.symbolization.temporal_distributions_estimation import (
    #         compute_gw_distance_and_transport_map,
    #         infer_kde_support,
    #     )
    #     kde_models_K = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, use_K_space=True)
    #     excluded_clusters_K = fetch_missing_temporal_prototypes(kde_models_K, annotator_id, round_number, config_name, clustering_config_name, use_K_space=True)


    #     explored_K = {'K_space': {'task-definitive': np.arange(15, len(kde_models_K['task-definitive']), 35),
    #                                     'exo-definitive': np.arange(15,  len(kde_models_K['exo-definitive']), 5),
    #                                     'Noise': np.arange(15, len(kde_models_K['Noise']), 5)},
                    
    #                 # 'G_space': {'task-definitive': np.arange(15, len(kde_models_G['task-definitive']), 35),
    #                 #                     'exo-definitive': np.arange(15,  len(kde_models_G['exo-definitive']), 5),
    #                 #                     'Noise': np.arange(15, len(kde_models_G['Noise']), 5)}
    #     }

    #     # RBF Kernel 
    #     explored_temperatures = {'K_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
    #                                             'temperatures':[ 1, 5, 10, 50, 100, 1000]},
    #                             'G_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
    #                                                 'temperatures':[ 1, 5, 10, 50, 100, 1000]
    #                             }}


    #     t0 = time.time()
    #     #kde_model_foreground = {**kde_models_G['task-definitive'], **kde_models_G['exo-definitive']}
    #     # 1) Compute temporal distance matrices across prototypes in the latent space before/after reduction per cluster types
    #     D_tt_pc, D_te_pc, _ = compute_temporal_distance(kde_models_K,  annotator_id, round_number, config_name=config_name, kernel_name='pre_computing', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, use_K_space=True, temperature_tau=None, overwrite_gw_distances=overwrite_gw_distances)

    #     t1 = time.time()
    #     print(f'Time elapsed for computing temporal distances: {t1 - t0} seconds')
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
