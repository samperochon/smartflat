"""Builder functions for the pipeline components (datasets, models, metrics, etc).

Once a builder function grows too much, this will be refactored into separate modules.
"""
import os
import sys
import time

#from cuml.mixture import GaussianMixture as CuMLGaussianMixture
try:
    import faiss
except:
    pass
import networkx as nx
import numpy as np
import ruptures as rpt
import torch
import torchmetrics
from scipy.sparse import csgraph
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import normalize

#from typing import Any, Callable, Dict, Literal, Optional




from smartflat.configs.loader import import_config

# finch_path = '/home/sam/FINCH-Clustering/TW-FINCH/python'
# sys.path.insert(0, finch_path)
try:
    from smartflat.contrib.twfinch import FINCH
except (ImportError, ModuleNotFoundError):
    FINCH = None
from smartflat.models.utils import CostCustom
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_dataset import normalize_data
from smartflat.utils.utils_io import fetch_qualification_mapping, get_data_root

# import torchmetrics

# import hcdatasets as hcd
# from healthssl.datasets.cifar10 import build_cifar10_loaders
# from healthssl.datasets.generic_hcd import build_hcdatasets_loaders
# from healthssl.engine.custom_metrics import SslMetrics

# from smartflat.models import EcgCNN

# from healthssl.objectives.contrastive import ContrastiveLoss
# from healthssl.objectives.custom_ssl_losses import InfoNceRegLoss, MultiContrastiveLoss, ContrastiveSupervisedLoss
# from healthssl.utils.utils_hds import download_model_artifacts

#
# main builders
#

def build_model(model_name: str, model_params: dict, verbose=False):
    """Model builder."""
    
    if model_name == 'KernelCPD':
        
        kernel=model_params['kernel']
        min_size=model_params['min_size']
        jump=model_params['jump']
        params = {}
        
        if kernel == 'rbf':
            params['gamma'] = model_params['gamma'] 
                
        model = rpt.KernelCPD(kernel=kernel, min_size=min_size, jump=jump, params=params)
        
    elif model_name == 'pelt':
        
        min_size=model_params['min_size']
        jump=model_params['jump']
        norm_used=model_params['norm_used']

        #model = rpt.Pelt(model=norm_used, custom_cost=CostCustom(), min_size=min_size, jump=jump)
        model = rpt.Pelt(model=norm_used, custom_cost=None, min_size=min_size, jump=jump)


    elif model_name == 'kmeans':
        
        n_clusters = model_params['n_clusters']
        algorithm_name = model_params['algorithm_name']
        #Default params
        model = KMeans(n_clusters=n_clusters, 
                        init='k-means++',
                        n_init=3, #Number of time the k-means algorithm will be run with different centroid seeds.TODO: change to 10 in deployment
                        max_iter=500,
                        tol=0.0001,
                        verbose=3,
                        copy_x=True,
                        algorithm=algorithm_name, 
                        random_state=4)
        
        
    elif model_name == 'batch-kmeans':
        
        n_clusters = model_params['n_clusters']
        #Default params
        model = MiniBatchKMeans(n_clusters=n_clusters, 
                                init='k-means++', 
                                batch_size=10000,
                                max_iter=500,
                                tol=0.0,
                                max_no_improvement=10,
                                init_size=100000,
                                n_init=10,
                                reassignment_ratio=0.1,
                                )

        
    elif model_name == 'faiss-kmeans-cosine':
        
        use_gpu = False
        
        n_clusters = model_params['n_clusters']
        n_iter = 250#model_params['n_iter']
        n_init = 2#model_params['n_init'],
        input_dimension = 1408 #model_params['input_dimension']



        model = faiss.Kmeans(d=input_dimension, k=n_clusters, niter=n_iter, nredo=n_init, spherical=True, seed=47, verbose=True)
        
        if use_gpu:
            # Initialize FAISS GPU resources
            res = faiss.StandardGpuResources()
            # TODO: add X_train_gpu = faiss.float_gpu_array_to_cpu(X_train)  # Direct GPU memory transfer
            index =  faiss.IndexFlatIP(input_dimension)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU

            # 3) Train KMeans on GPU
            model.index = gpu_index  # Assign GPU index to KMeans
        else:
            model.index = faiss.IndexFlatIP(input_dimension)  # Replace L2 with inner product
            
        model.cp.max_points_per_centroid = 20000
        
    elif model_name == 'faiss-kmeans-euclidean':
        
        n_clusters = model_params['n_clusters']
        n_iter = 250#model_params['n_iter']
        n_init = 2#model_params['n_init']
        input_dimension = 1408 #model_params['input_dimension']

        
        model = faiss.Kmeans(d=input_dimension, k=n_clusters, niter=n_iter, nredo=n_init, seed=47, verbose=True)
        print(model.cp.max_points_per_centroid)
        model.cp.max_points_per_centroid = 20000
                
    elif model_name == 'prototypes':
        
        # Here we concatenate the prototypes from different clustering rounds 
        modality = model_params['modality']
        task_name = model_params['task_name']
        normalization = model_params['normalization']
        annotator_id = model_params['annotator_id']
        round_number = model_params['round_number']
        
        # Desired set of cluster types forming the concatenated prototypes matrix
        input_clustering_config_names = model_params['input_clustering_config_names']
        if 'input_annotor_ids' in model_params.keys():
            input_annotor_ids = model_params['input_annotor_ids']
        else:
            input_annotor_ids = None
        input_clustering_config_names = model_params['input_clustering_config_names']
        qualification_cluster_types_list = model_params['qualification_cluster_types']
        round_numbers = model_params['round_numbers']
        
        model = build_cluster_specified_prototype_model(input_clustering_config_names=input_clustering_config_names, input_annotor_ids=input_annotor_ids, qualification_cluster_types_list=qualification_cluster_types_list, round_numbers=round_numbers,  normalization=normalization, annotator_id=annotator_id, round_number=round_number, verbose=True)
    
    elif model_name == 'w-prototypes':

        # Here we concatenate the prototypes from different clustering rounds 
        modality = model_params['modality']
        task_name = model_params['task_name']
        normalization = model_params['normalization']
        
        # Desired set of cluster types forming the concatenated prototypes matrix
        input_clustering_config_names = model_params['input_clustering_config_names']
        qualification_cluster_types_list = model_params['qualification_cluster_types']
        model = build_cluster_specified_prototype_model(input_clustering_config_names=input_clustering_config_names, input_annotor_ids=input_annotor_ids, qualification_cluster_types_list=qualification_cluster_types_list, normalization=normalization, annotator_id=annotator_id, round_number=round_number, verbose=True)
    
    
    elif model_name == 'gaussian_mixture':

                
        n_clusters = model_params['n_clusters']
        batch_size = model_params['batch_size']
        #Default params
        
        if torch.cuda.is_available():
        
            print("CUDA is available (/!\ but not used, not implemented)")
            
        model = GaussianMixture(n_components=n_clusters, 
                                    covariance_type='full',
                                    tol=0.001,
                                    reg_covar=1e-06,
                                    max_iter=100, #5000,
                                    n_init=1, #3,
                                    init_params='k-means++',
                                    weights_init=None,
                                    means_init=None,
                                    precisions_init=None,
                                    random_state=47,
                                    warm_start=True if batch_size is not None else False,
                                    verbose=1,
                                    verbose_interval=1)
        
    elif model_name == 'bayesian_gaussian_mixture':
        
        
        n_clusters = model_params['n_clusters']
        batch_size = model_params['batch_size']

        model = BayesianGaussianMixture(n_components=n_clusters,
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
                                        warm_start=True if batch_size is not None else False,
                                        verbose=0,
                                        verbose_interval=10)
            
    elif model_name == 'tw-finch':
        
        n_clusters = model_params['n_clusters']
        model = FinchClusteringModel(n_clusters=n_clusters, tw_finch=True)    
        
    else:
        raise ValueError(f"Model name '{model_name}' not found.")
    
    return model


def build_cluster_specified_prototype_model(input_clustering_config_names, qualification_cluster_types_list, round_numbers, normalization,  input_annotor_ids = None, annotator_id='samperochon', round_number=0, verbose=True):
    
    qualification_mapping = fetch_qualification_mapping(verbose=False)
    
    if input_annotor_ids is None:
        input_annotor_ids = [annotator_id] * len(input_clustering_config_names)
    
    centroids_list = []
    # For each clustering (ROUND >1), assume that each config has the concatenation of 
    # previous centroids and the new ones, and that the qualification mapping contain the mutual explusive 
    # partitioning for all types (i.e the sum of entries (cardinal) is equal to the definitive centroids from the previous
    # clustering, plus the new centrroids brought by a rfinement step of the algorithm))
    for annotator_id, input_clustering_config_name, qualification_cluster_types, round_number in zip(input_annotor_ids, input_clustering_config_names, qualification_cluster_types_list, round_numbers):
        print(f'Sampling prototypes from {input_clustering_config_name} for annotator {annotator_id} at round {round_number} with cluster types {qualification_cluster_types}')
        
        symb_config = import_config(input_clustering_config_name)
        config = import_config(symb_config.clustering_config_name)
        
        task_name =  'cuisine'#config.model_params['task_name'] TODO
        modality = 'Tobii' #config.model_params['modality']
        
        
        clusters_path = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', f'{task_name}_{modality}_cluster_centers.npy')
        #print(f'Loading centroids from {clusters_path} (.npy)')
        centroids = np.load(clusters_path).astype(np.float32)
        
        print(f'Loaded centroids from {clusters_path} (.npy): {centroids.shape} of dtype {centroids.dtype}')

        if len(qualification_cluster_types) == 0:
            print(f'Using all training prototypes from {input_clustering_config_name}')
            c1 = centroids

        elif input_clustering_config_name in qualification_mapping.keys() and annotator_id in qualification_mapping[input_clustering_config_name].keys() and f'round_{round_number}' in qualification_mapping[input_clustering_config_name][annotator_id].keys():             
            
            c1 = []
            for cluster_type in qualification_cluster_types:
                
                if cluster_type not in qualification_mapping[input_clustering_config_name][annotator_id][f'round_{round_number}'].keys():
                    #print(f'Cluster type {cluster_type} not found in qualification mapping for {input_clustering_config_name}')
                    continue
                #print('Sampling:', np.array(qualification_mapping[input_clustering_config_name][annotator_id][f'round_{round_number}'][cluster_type]))
                c1.append(centroids[np.array(qualification_mapping[input_clustering_config_name][annotator_id][f'round_{round_number}'][cluster_type])])
                #print('Collected centroids for cluster type:', cluster_type, 'shape:', c1[-1].shape)
                #print(np.array(qualification_mapping[input_clustering_config_name][cluster_type]))
                if verbose:
                    print(f'\t{cluster_type}: shape: {c1[-1].shape}')
                    
            c1 = np.concatenate(c1, axis=0)
            #print(f'Final sub-clustering centroids shape: {c1.shape}')
                        
        else:
            raise ValueError(f'Qualification mapping not found for {input_clustering_config_name}\n\t{annotator_id}\n\t{round_number}')
            
        if verbose:
            print(f'Centroids from {input_clustering_config_name} shape={c1.shape}')
        
        #print(f'Normm of the {np.linalg.norm(c1, axis=1).shape[0]} prototypes: {np.linalg.norm(c1, axis=1)}')
        green('/!\ Using identity normalization of per config centroids - inner loop')
        c1 = normalize_data(c1, normalization='identity')
        centroids_list.append(c1)
        
    model = np.concatenate(centroids_list, axis=0)
    model = normalize_data(model, normalization='identity')
    #green(f'/!\ Using {normalization} normalization of per config centroids - outer loop')
    red(f'/!\ Removing {normalization} normalization of per config centroids - outer loop')
    print(f'Final prototypes model shape: {model.shape}')
    return model 

def build_metrics(metric_names=None, **kwargs):
    """Metrics builder.
    """
    # if task_type == "temporal_segmentation":
    #     shared_args = {}
    #     return torchmetrics.MetricCollection(
    #         {
    #             "accuracy": torchmetrics.Accuracy(**shared_args),
    #             "precision_macro": torchmetrics.Precision(
    #                 average="macro", **shared_args
    #             ),
    #             "recall_macro": torchmetrics.Recall(average="macro", **shared_args),
    #             "f1_score_macro": torchmetrics.F1Score(**shared_args),
    #             "specificity": torchmetrics.Specificity(**shared_args),
    #             "average_precision": torchmetrics.AveragePrecision(**shared_args),
    #             "roc_auc": torchmetrics.AUROC(**shared_args),
    #             "recall_at_95_precision": torchmetrics.classification.BinaryRecallAtFixedPrecision(
    #                 min_precision=0.95, **shared_args
    #             ),
    #         }
    #     )


    metrics = {
                "silhouette_score_baseline": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: silhouette_score(X=X, labels=labels, metric='cosine', sample_size=None) if 2 <= len(set(labels)) < len(labels) else np.inf,
                "silhouette_score_M": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: silhouette_score(X=D_M, labels=labels, metric='precomputed', sample_size=None) if 2 <= len(set(labels)) < len(labels) else np.inf,
                "silhouette_score_T": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: silhouette_score(X=D_T, labels=labels, metric='precomputed', sample_size=None) if 2 <= len(set(labels)) < len(labels) else np.inf,
                "silhouette_score_X": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: silhouette_score(X=D_X, labels=labels, metric='precomputed', sample_size=None) if 2 <= len(set(labels)) < len(labels) else np.inf,


                "davies_bouldin_baseline": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: davies_bouldin_score(X, labels) if 2 <= len(set(labels)) < len(labels) else np.nan, 
                "davies_bouldin_M": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: kernel_db_score(D=D_M, labels=labels) if 2 <= len(set(labels)) < len(labels) else np.nan,
                "davies_bouldin_T": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: kernel_db_score(D=D_T, labels=labels) if 2 <= len(set(labels)) < len(labels) else np.nan,
                "davies_bouldin_X": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: kernel_db_score(X=X, labels=labels) if 2 <= len(set(labels)) < len(labels) else np.nan,

                "compactness_separation_baseline": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_cs(X=X, labels=labels, kernel_name='cosine'),
                "compactness_separation_M": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_cs(D=D_M, labels=labels),
                "compactness_separation_T": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_cs(D=D_T, labels=labels),
                "compactness_separation_X": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_cs(D=D_X, labels=labels),

                "mmd_baseline": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_mmd(X=X, labels=labels, S=None, kernel_name='cosine'),
                "mmd_M": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_mmd(X=None, labels=labels, S=S_M), 
                "mmd_T": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_mmd(X=None, labels=labels, S=S_T),
                "mmd_X": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_mmd(X=None, labels=labels, S=S_X),

                "modularity_baseline": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_modularity(X=X, labels=labels, S=None, temperature=kwargs.get('temperature', 1)),
                "modularity_M": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_modularity(X=None, labels=labels, S=S_M),
                "modularity_T": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_modularity(X=None, labels=labels, S=S_T),
                "modularity_X": lambda X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs: compute_modularity(X=None, labels=labels, S=S_X),
            }
    
    if metric_names is None:
        return metrics
    else:
        return {key: metric for key, metric in metrics.items() if key in metric_names}


def compute_metrics(X, labels, D_M, D_X, D_T, S_M, S_X, S_T, metrics=None, **kwargs):
    """Compute metrics given an adjacency matrix and cluster assignments."""
    
    if metrics is None:
        metrics = build_metrics(**kwargs)
        
    res = {key: np.nan for key in metrics.keys()}
    
    for metric_name, metric_fn in metrics.items():
        #start_time = time.time()
        #print(f"Computing {metric_name} with shapes D_M={D_M.shape}, X={X.shape}...")
        res[metric_name] = metric_fn(X, labels, D_M, D_X, D_T, S_M, S_X, S_T, **kwargs)
        #elapsed_time = time.time() - start_time
        #print(f"{metric_name}: {res[metric_name]:.4f} (computed in {elapsed_time:.2f} seconds)")

    return res

def compute_modularity(X=None, labels=None, S=None, temperature=1.0):
    """Computes modularity given an adjacency matrix and cluster assignments."""
    if S is None:
        
        condensed_sq_dists = pdist(X, metric='sqeuclidean')
        sigma_rbf = np.median(np.sqrt(condensed_sq_dists))
        # Step 2: compute RBF kernel in condensed form
        condensed_distances = np.exp(-condensed_sq_dists / (2 * temperature * sigma_rbf ** 2))
        S = squareform(condensed_distances, checks=True) 
        np.fill_diagonal(S, 1)

    # min-max scaling
    if  not np.all(S >= 0):
        
        print(f'Adjacency matrix has negative weights (Min/Max sim before scaling (to [0-1]): min={np.min(S):.2f}, max={np.max(S):.2f}')   
        S = (S - S.min()) / (S.max() - S.min())
    
    assert np.all(S >= 0), "RBF matrix has negative weights!"
    
    start = time.time()
    G = nx.from_numpy_array(S)
    communities = {i: l for i, l in enumerate(labels)}
    modularity = nx.community.quality.modularity(G, [{i for i in communities if communities[i] == c} for c in set(labels)])
    #print(f"Modularity={modularity:.6f} computed in {time.time() - start:.4f} seconds")
    assert -0.5 <= modularity <= 1, "Unexpected modularity value"
    
    return modularity

def compute_mmd(X=None, labels=None, S=None, return_list=False, kernel_name='cosine', gamma=None, temperature=1.):
    """ Compute MMD between clusters. If S (distance matrix) is provided, X is optional but only gaussian_rbf is supported."""
    
    # If kernel is gaussian rbf, we pre-compute sigma_rbf
    if S is None and kernel_name == 'gaussian_rbf':
        assert X is not None
        sigma_rbf = np.median(pdist(X, metric='euclidean'))
        gamma = 1 / (2 * temperature * sigma_rbf ** 2)

    clusters = [np.where(labels == label)[0] for label in np.unique(labels)]
    mmd_list = []; count = 0
    for j in range(len(clusters)):
        for k in range(j + 1, len(clusters)):
            if S is not None:
                mmd = pairwise_compute_mmd_indices(S, clusters[j], clusters[k])
            else:
                mmd = pairwise_compute_mmd(X[clusters[j]], X[clusters[k]], kernel_name=kernel_name, gamma=gamma)
            mmd_list.append(mmd); count += 1

    return mmd_list if return_list else (np.sum(mmd_list) / count if count > 0 else 0)

def pairwise_compute_mmd_indices(S, idx1, idx2):
    """Compute MMD using precomputed similarity matrix."""
    K_ii = S[np.ix_(idx1, idx1)]
    K_jj = S[np.ix_(idx2, idx2)]
    K_ij = S[np.ix_(idx1, idx2)]
    return np.mean(K_ii) + np.mean(K_jj) - 2 * np.mean(K_ij)

def pairwise_compute_mmd(C_i, C_j, kernel_name='cosine', gamma=None):
    """Compute the MMD between two clusters."""
    
    if kernel_name == 'euclidean':
        K_ii = pairwise_kernels(C_i, C_i, metric='linear')
        K_jj = pairwise_kernels(C_j, C_j, metric='linear')
        K_ij = pairwise_kernels(C_i, C_j, metric='linear')
        
    elif kernel_name == 'cosine':
        K_ii = pairwise_kernels(C_i, C_i, metric='cosine')
        K_jj = pairwise_kernels(C_j, C_j, metric='cosine')
        K_ij = pairwise_kernels(C_i, C_j, metric='cosine')
        
        
    elif kernel_name == 'gaussian_rbf':
        K_ii = pairwise_kernels(C_i, C_i, metric='rbf', gamma=gamma)
        K_jj = pairwise_kernels(C_j, C_j, metric='rbf', gamma=gamma)
        K_ij = pairwise_kernels(C_i, C_j, metric='rbf', gamma=gamma)

    mmd = (np.mean(K_ii) + np.mean(K_jj) - 2 * np.mean(K_ij))
    return mmd

def compute_cs(X=None, D=None, labels=None, kernel_name='cosine'):
    """Compute the compactness-separation metric from samples or a distance matrix."""
    if D is None:
        assert X is not None, "Either X or D must be provided."
        D = pairwise_distances(X, metric=kernel_name)
    
    labels = np.asarray(labels)
    intra = np.mean([D[np.ix_(labels == i, labels == i)].mean() for i in np.unique(labels)])
    inter = np.mean([D[np.ix_(labels == i, labels != i)].mean() for i in np.unique(labels)])
    return intra / inter if inter > 0 else np.inf


def kernel_db_score(X=None, D=None, labels=None, kernel_name='euclidean', temperature=1.0, gamma=None):
    """
    Compute Davies-Bouldin score using kernel-based distances.

    Parameters:
    - X: Data points (for distance calculation if D is not provided).
    - D: Precomputed distance matrix.
    - labels: Cluster labels for each data point.
    - kernel_name: Kernel to use ('cosine', 'euclidean', 'rbf').
    - temperature: Temperature parameter for RBF kernel.
    - gamma: Optional gamma parameter for RBF kernel (for Gaussian RBF).

    Returns:
    - Kernelized Davies-Bouldin index.
    """
    labels = np.asarray(labels)
    clusters = np.unique(labels)
    k = len(clusters)

    # If precomputed distance matrix is provided
    if D is not None:
        D = np.asarray(D)
        S = np.zeros(k)

        for i, ci in enumerate(clusters):
            idx_i = np.where(labels == ci)[0]
            if len(idx_i) == 1:
                S[i] = 0  # If cluster has only one element, set to 0
            else:
                S[i] = D[np.ix_(idx_i, idx_i)].mean()

        M = np.full((k, k), np.inf)
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters):
                if i != j:
                    idx_i = np.where(labels == ci)[0]
                    idx_j = np.where(labels == cj)[0]
                    M[i, j] = D[np.ix_(idx_i, idx_j)].mean()

    else:
        # Compute centroids for each cluster
        X = np.asarray(X)
        centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])

        # Define the distance function based on the kernel choice
        def dist(a, b):
            if kernel_name == 'cosine':
                a, b = normalize(a), normalize(b)
                return 1 - np.dot(a, b.T)
            elif kernel_name == 'gaussian_rbf':
                if gamma is None:
                    gamma = 1.0 / (2 * temperature ** 2)  # Default for Gaussian RBF
                d2 = cdist(a, b, 'sqeuclidean')
                return 1 - np.exp(-gamma * d2)  # Gaussian RBF
            return cdist(a, b, kernel_name)  # Euclidean or other metrics

        # Compute intra-cluster scatter (S)
        S = np.array([
            dist(X[labels == c], centroids[i].reshape(1, -1)).mean()
            for i, c in enumerate(clusters)
        ])

        # Compute inter-cluster distances (M) between centroids
        M = dist(centroids, centroids)
        np.fill_diagonal(M, np.inf)

    # Compute DB index
    R = (S[:, None] + S[None, :]) / M
    return np.mean(np.max(R, axis=1))


def build_loss(loss_name: str, loss_params: dict) -> torch.nn.Module:
    """Pytorch loss function builder."""
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss(**loss_params)

    if loss_name == "mse":
        return torch.nn.MSELoss(**loss_params)

    if loss_name == "bce_with_logits":
        return torch.nn.BCEWithLogitsLoss(**loss_params)

    if loss_name == "contrastive":
        return ContrastiveLoss(**loss_params)
    
    if loss_name == 'multi_contrastive':
        return MultiContrastiveLoss(**loss_params)

    if loss_name == 'contrastive_supervised':
        return ContrastiveSupervisedLoss(**loss_params)
        
    if loss_name == "byol":
        return ByolLoss(**loss_params)

    if loss_name == "barlow_twins":
        return BarlowTwinsLoss(**loss_params)

    if loss_name == "dino":
        return DinoLoss(**loss_params)

    if loss_name == "vic_reg":
        return VicRegLoss(**loss_params)

    if loss_name == "nce_reg":
        return InfoNceRegLoss(**loss_params)

    if loss_name == "cosine_embedding_loss":
        return torch.nn.CosineEmbeddingLoss(**loss_params)

    raise ValueError(f"Criterion name '{loss_name}' not found.")


def build_activation(activation_name: str, dim: int = -1):
    """Pytorch activation function builder."""
    if activation_name is None:
        return None

    if activation_name == "softmax":
        return torch.nn.Softmax(dim=dim)

    if activation_name == "sigmoid":
        return torch.sigmoid

    raise ValueError(f"Activation function name '{activation_name}' not found.")


def build_learning_schedule(optimizer, schedule_name: str, schedule_params: dict):
    if schedule_name == "exponential_lr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **schedule_params)

    if schedule_name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **schedule_params)

    if schedule_name == "step_lr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **schedule_params)

    if schedule_name == "constant_lr":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, **schedule_params)

    raise ValueError(f"Learning schedule name '{schedule_name}' not found.")


def build_optimizer(
    model, optimizer_name, optimizer_params: dict
) -> torch.optim.Optimizer:
    """Pytorch optimizer builder."""
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), **optimizer_params)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_params)

    raise ValueError(f"Optimizer name '{optimizer_name}' not found.")


def build_dataset(dataset_name: str, dataset_params: dict):
    """Dataloader builder."""
    if dataset_name == "dataset_cifar10":
        return build_cifar10_loaders(dataset_name=dataset_name, **dataset_params)

    if hcd.is_dataset_available(dataset_name):
        collate_fn = build_collate_fn(dataset_params)
        return build_hcdatasets_loaders(dataset_name=dataset_name, collate_fn=collate_fn, **dataset_params)

    raise ValueError(f"Dataset name '{dataset_name}' not found.")


#
# utility builders
#

def build_backbone(backbone_name: str, backbone_params: dict):
    if backbone_name == "efficientnet":
        return EfficientNet(**backbone_params)

    if backbone_name == "pggcnn":
        return PpgCNN(**backbone_params)

    if backbone_name == "mscnn":
        return MultiscaleCNN(**backbone_params)

    if backbone_name == "resnet":
        return resnet(**backbone_params)

    raise ValueError(f"Backbone name '{backbone_name}' not found.")

def build_collate_fn(dataset_params: dict):

    # Input is aggregated per 'aggregated_scale' days.
    # if 'aggregation_scale' in dataset_params.keys():

    #     # Input also consists in comparison between two time intervals
    #     if 'max_time_segment_comparison_per_time_period' in dataset_params.keys():
            
    #         collate_fn = comparison_batch_padding_collate
    #     else:
    #         collate_fn = batch_padding_collate
    # else:
    collate_fn = None

    return collate_fn



class FinchClusteringModel(object):

    """
    Perform FINCH clustering on input data.

    Parameters:
    - X: ndarray. Data to cluster (shape: [n_samples, n_features]).

    Returns:
    - c: ndarray. Cluster centroids.
    - num_clust: int. Number of clusters.
    - labels: ndarray. Cluster labels for each data point.
    """
    
    def __init__(self, n_clusters, verbose=True, tw_finch=True):
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.tw_finch = tw_finch
        
    def __call__(self, X):
        
        c, num_clust, labels = FINCH(X, req_clust=self.n_clusters, verbose=self.verbose, tw_finch=self.tw_finch)
        
        return c, num_clust, labels


