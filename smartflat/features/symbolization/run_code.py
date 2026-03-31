
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
from smartflat.utils.utils_coding import *
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
    aggomerate_distance_matrix,
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

if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()

    annotator_id = 'samperochon'
    round_number = 8

    config_name = 'SymbolicSourceInferenceGoldConfig'#'SymbolicSourceCosineCroppedInferenceConfig'

    # TEST other clustering approaches 
    model_names = ['spectral', 'dbscan', 'hdbscan', 'gmm', 'bayesian_gmm']

    model_names = ['hierarchical-clustering'] #['hierarchical-clustering']
    linkage_methods = ['complete', 'average', 'single']
    metrics = [ "silhouette_score", "davies_bouldin", "mmd", "compactness_separation", "modularity"]
    use_K_space = False
    input_space = 'G_space'
    verbose=True
    overwrite_gw_distances = False
    do_normalize_distances = True
    overwrite_results = False

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
    results_path = os.path.join(output_folder, "results_co_clustering_final.pkl")

    # Covariates
    kernel_names = ['cosine', 'gaussian_rbf', ]
    multimodal_methods = [ 'multiplicative',  'normalized_exp','without'] # unnormalized_exp
    alpha_benchmark = [0, 0.25, 0.5, 0.75, 1]
    #method_names = ['temporal_similarity', 'knn', 'temporal']
    loss_names =  ['square_loss']#, 'kl_loss']
    alpha_values = np.linspace(0, 1, 9); print(f'Alpha values: {alpha_values}')



    # qualification_mapping = fetch_qualification_mapping()
    # results, best_df, mapping_prototypes_reduction, mapping_index_cluster_type, qualification_mapping = clustering_prototypes_space(clustering_config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', qualification_mapping=qualification_mapping)
    # clusterdf = build_clusterdf(df, qualification_mapping=qualification_mapping, embeddings_label_col='symb_labels', segments_labels_col='symb_segments_labels', segments_length_col='symb_segments_length')
    # opt_mapping_prototypes_df =  pd.DataFrame(mapping_prototypes_reduction, index=['projected_cluster_index']).transpose().reset_index(names=['original_cluster_index'])
    # mapping_prototypes_reduction_list = opt_mapping_prototypes_df.groupby(['projected_cluster_index']).original_cluster_index.agg(list).to_dict()#.apply(len)#.hist(bins=10)
    # mapping_prototypes_reduction_list[-2] = exluded_clusters
    # mapping_prototypes_reduction_list[-1] = []
    # clusterdf['source_index'] = clusterdf.cluster_index.map(mapping_prototypes_reduction_list)

    # 1) Estimate KDE model r££un comparison  between the expected prototypes index and the present data

    # 309: (288 task (2 missing), 20 exo (0 missing), then the 40 noise clusters are reduced to a single -2 cluster, 
    # and there is the -1 cluster for roughly misfitted data, 
    # so 288 + 20 + 1 + 1 = 310 expected clusters in total


    kde_models_K = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, use_K_space=True)
    excluded_clusters_K = fetch_missing_temporal_prototypes(kde_models_K, annotator_id, round_number, config_name, clustering_config_name, use_K_space=True)
    kde_models_G = estimate_kde_model(config_name, annotator_id=annotator_id, round_number=round_number, use_K_space=False)
    excluded_clusters_G = fetch_missing_temporal_prototypes(kde_models_G, annotator_id, round_number, config_name, clustering_config_name, use_K_space=False)


    explored_K = {'K_space': {'foreground': np.arange(15, len(kde_models_K['foreground']), 35),
                            'task-definitive': np.arange(15, len(kde_models_K['foreground']), 35),
                                    'exo-definitive': np.arange(15,  len(kde_models_K['exo-definitive']), 5),
                                    'Noise': np.arange(15, len(kde_models_K['Noise']), 5)},
                
                'G_space': {'foreground': np.arange(15, len(kde_models_G['foreground']), 35),
                            'task-definitive': np.arange(15, len(kde_models_G['task-definitive']), 35),
                            'exo-definitive': np.arange(15,  len(kde_models_G['exo-definitive']), 5),
                            'Noise': np.arange(15, len(kde_models_G['Noise']), 5)}
    }

    # RBF Kernel 
    explored_temperatures = {'K_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                            'temperatures':[ 1, 5, 10, 50, 100, 1000]},
                            }
    explored_temperatures = {'K_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                        'temperatures':[ 1, 5, 10, 50, 100, 1000]},
                            'G_space': {'temperatures_tau': [1, 2.5, 5, 10, 50, 100],
                                        'temperatures':[ 1, 5, 10, 50, 100, 1000]}
                        }

    # explored_temperatures = {'K_space': {
    #                                         'temperatures':[ 0.1, 1, 10, 15, 20]},





    # 1) Compute temporal distance matrices across prototypes in the latent space before/after reduction per cluster types
    D_tt_pc, D_te_pc, D_tf_pc = compute_temporal_distance(kde_models_K,  annotator_id, round_number, config_name=config_name, kernel_name='pre_computing', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, use_K_space=True, temperature_tau=None, overwrite_gw_distances=overwrite_gw_distances)
    D_ttr_pc, D_ter_pc, D_tfr_pc = compute_temporal_distance(kde_models_G,  annotator_id, round_number, config_name=config_name, kernel_name='pre_computng', loss_name=loss_name, gridsize=gridsize, temporal_distance=temporal_distance, use_K_space=False, temperature_tau=None, overwrite_gw_distances=overwrite_gw_distances)


    # Load existing results if available
    if os.path.exists(results_path) and not overwrite_results:
        existing_df = load_df(results_path)
        results = existing_df.to_dict('records')
    else:
        existing_df = pd.DataFrame(columns=['model_name', 'annotator_id', 'round_number',  
                                                'n_exp', 'input_space',   'alpha', 'n_clusters', 
                                            'linkage_method', 
                                            'kernel_name', 'multimodal_method', 'loss_name', 'normalization', 'temperature_x', 'temperature_tau', 'temporal_distance', 'gridsize', 'agg_fn_str', 'cluster_type', 'input_space', ] + metrics)
        results = []

    print(f'Len of existing results: {len(existing_df)}')













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
        
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
