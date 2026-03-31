
import argparse
import os
import sys
import time

try:
    import faiss
except:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler



import random
import time
from collections import Counter

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

from smartflat.configs.loader import import_config
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import load_embedding_dimensions
from smartflat.engine.clustering import predict_with_centroids
from smartflat.features.symbolization.main import define_symbolization
from smartflat.features.symbolization.utils import gaussian_rbf_matrix, get_prototypes
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.features.symbolization.visualization import plot_ot_graphs, plot_trajectory
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_dataset import (
    add_cum_sum_col,
    check_train_data,
    compute_matrix_stats,
    normalize_data,
)
from smartflat.utils.utils_io import fetch_qualification_mapping, get_data_root
from smartflat.utils.utils_visualization import (
    plot_chronogames,
    plot_gram,
    plot_labels_2D_encoding,
)


def orthonormalize_embeddings(X=None, P=None):
    """
    Transforms embeddings X and prototypes P into a common orthonormalized space.

    Args:
        X (np.ndarray): Embedding matrix of shape (N, D).
        P (np.ndarray): Prototype matrix of shape (G, D).

    Returns:
        X_orth (np.ndarray): Transformed embeddings in the orthonormal basis.
        P_orth (np.ndarray): Orthonormalized prototypes.
    """
    # Step 1: Compute QR decomposition of P^T
    Q, R = np.linalg.qr(P.T)  # Q has shape (D, G), R has shape (G, G)
    
    # Step 2: Orthonormalized prototypes
    P_orth = Q.T  # Shape (G, D), now row vectors of P_orth are orthonormal
    
    if X is None:
        return None, P_orth
    
    # Step 3: Apply the same transformation to X to align it with the prototypes' space
    X_orth = X @ (Q @ Q.T)  # Shape (N, D), projecting X into the prototype basis

    return X_orth, P_orth

def project_onto_prototype_components(X, P, kernel_name='gaussian_rbf', orthonormalization=False, normalization=None, sigma_rbf=None, temperature=10, verbose=True):
    """
    Projects the embeddings X onto the orthogonal prototype components.

    Args:
        X (np.ndarray): Embedding matrix of shape (N, D).
        P (np.ndarray): Prototype matrix of shape (G, D).

    Returns:
        X_proj (np.ndarray): Projections of X onto the prototype basis (N, G).
        coords (np.ndarray): Coefficients in the prototype basis (N, G).
    """     
    
    if orthonormalization:
        print(f'Orthonormalization of the embeddings and the prototypes')

        # Ensure prototypes are orthonormalized
        Q, _ = np.linalg.qr(P.T)     # Q: (D, G)
        P = Q.T                  # Orthonormal prototypes (G, D)

        # Project X onto the orthogonal basis
        X_orth = X @ P.T         # Shape: (N, G)
        
        # Reconstruct the projected samples from the coordinates
        X = X_orth @ P      # Shape: (N, D)
    
    # else:
    #     pass#print(f'Using original embeddings and prototypes without orthonormalization')
        
        
    if normalization is not None:
        #print('Joint Z-normalization of {} samples and {} prototypes'.format(X.shape[0], P.shape[0]))
        X_P_concat = np.vstack([X, P])
        X_P_concat_norm = check_train_data(normalize_data(X_P_concat, normalization=normalization))
        X = X_P_concat_norm[:X.shape[0]]
        P = X_P_concat_norm[-P.shape[0]:]

        # X_proj = normalize_data(X_proj, normalization=normalization)
        # P_orth = normalize_data(P_orth, normalization=normalization)

    if kernel_name == 'gaussian_rbf':
        
        #gaussian_rbf = lambda x, mu: np.exp(-(np.sum((x - mu) ** 2) / (2 * sigma_rbf ** 2)))
        #distances = pairwise_distances(X_proj, P_orth, n_jobs=32, metric=gaussian_rbf)
        #distances = cdist(X_proj, P_orth, metric=gaussian_rbf)
        similarity = gaussian_rbf_matrix(X, P, sigma=sigma_rbf, temperature=temperature)

        sq_distances = -2 * temperature * sigma_rbf**2 * np.log(np.clip(similarity, 1e-10, 1.0))
        distances = np.sqrt(sq_distances)
        
        print(f' Min and max of the distances: {np.min(distances)}, {np.max(distances)}')
        labels = np.argmin(distances, axis=1)
        cluster_dist = np.min(distances, axis=1)
        
    elif kernel_name == 'euclidean':
        
        distances = pairwise_distances(X, P, metric='euclidean')
        
         # Compute 1st neireast neighborpartiitoning/Voronoi regions assignement and associated strength () and clusyer 
        labels = np.argmin(distances, axis=1)
        cluster_dist = np.min(distances, axis=1)
        
        
    elif kernel_name == 'cosine':
        
        # Compute 1st neireast neighborpartiitoning/Voronoi regions assignement and associated strength () and clusyer 
        # distances = pairwise_distances(X, P, metric='cosine')
        # labels = np.argmin(distances, axis=1)
        # cluster_dist = np.min(distances, axis=1)      
        X = X / np.linalg.norm(X, axis=1, keepdims=True) 
        P = P / np.linalg.norm(P, axis=1, keepdims=True)  # Ensure P is normalized
        distances = X @ P.T   # shape (n_queries, n_centroids)
        distances = 1 - distances
        cluster_dist = np.min(distances, axis=1)
        labels = np.argmin(distances, axis=1)

        # # Rebuild index with same dimension and spherical setting
        # index = faiss.IndexFlatIP(P.shape[1])  # IP = inner product = cosine sim (since vectors are L2-normalized)
        # faiss.normalize_L2(P)  # Redundant if saved from faiss.Kmeans with spherical=True, but safe
        # index.add(P)
        # faiss.normalize_L2(X)
        
        # distances, _ = index.search(X, P.shape[0])  # D.shape = (n_queries, n_centroids), contains cosine similarity
        # # Convert to cosine distance
        # distances = 1 - distances

        # # Search labels
        # cluster_dist, I = index.search(X, 1)
        # labels = I.flatten()
        # cluster_dist = 1 - cluster_dist.flatten()  # cosine distance

    #print('Matrix data shape: NxD={} Support of labels: {}'.format(distances.shape, len(np.unique(labels)))) # (N, D) in prototype subspace
    
    if verbose:
        
        fi(20, 8);plt.imshow(distances.T, aspect='auto', cmap='coolwarm');plt.title(f'Distance matrix (NxG={distances.shape})')
        plt.colorbar(); plt.show()

        fi(20, 4);_ = plt.hist(distances.flatten(), bins='fd');plt.title(f'Distance matrix with kernel={kernel_name}');plt.show()
        fi(20, 3);_ = plt.hist(np.median(distances, axis=0), bins='fd');plt.title(f'Median distance across samples (NxG={distances.shape}');plt.show()
        print('dsad', len(labels))
        fi(25, 3);_ = plt.hist(labels, bins=np.arange(-2, np.max(labels)+1));plt.title(f'labels with kernel={kernel_name} support={len(np.unique(labels))}');plt.show()

    return  distances, labels, cluster_dist

def compute_wasserstein_projection(X, P, weights_priors='uniform', temperature=5):
    """
    Computes the Wasserstein barycenter projection and assigns closest prototype labels.

    Args:
        X (np.ndarray): Embedding matrix (N, D).
        P (np.ndarray): Prototype matrix (G, D).

    Returns:
        X_proj (np.ndarray): Projected embeddings (N, D).
        labels (np.ndarray): Labels of the closest prototypes (N,).
        T (np.ndarray): Optimal transport matrix (N, G).
    """
    # Step 1: Compute squared Euclidean distances
    M = np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2) ** 2  # (N x G)
    if weights_priors == 'uniform':
        alpha = np.ones(M.shape[0]) / M.shape[0]   # Uniform weights for P
        beta = np.ones(P.shape[0]) / P.shape[0]   # Uniform weights for P
    elif weights_priors == 'distance-based':
        temperature = 0.5  # Lower temperature → sharper, higher → smoother
        alpha = np.exp(-np.min(M, axis=1)/M.max()/temperature); alpha = alpha /  (alpha.sum() + 1e-8)
        beta = np.ones(P.shape[0]) / P.shape[0]   # Uniform weights for P
    else:
        raise ValueError(f"Invalid weights_priors: {weights_priors}")
    
    # Compute optimal transport matrix
    T = ot.emd(alpha, beta, M) # (N, G)

    # Step 3: Barycentric projection
    X_proj = T @ P   # (N, D)

    # Assign closest prototypes based on the transport map
    #labels = np.argmax(T, axis=1)  # Closest prototype label for each sample -> This empirically lead to a uniform assignemnt across labels :-/
    labels = np.argmax(T, axis=1)
    #print('/!\ Chek if we should use labels = np.argmax(T, axis=1) instead of labels = np.argmax(T @ P (N,D), axis=1)  ')

    return M, X_proj, T, labels, alpha, beta

def inference_projection(row, prototypes, X=None, symbolization_config_name='SymbolicInferenceConfig', annotator_id='samperochon', round_number=0, sigma_rbf=None, normalization=None,  overwrite=True,  do_compute=False, verbose=False):
    
    """Predict participant labels using the closest prototypes after:
        1) Orthonormalization of the prototypes space, (applied to the empirical samples per basis transform) 
        2) Computation of the optimal transport map from the sample embeddings to the prototypes distributions 
        3) Assign labels for each samples using the Prototypes Wasserstein barycenter projection.
    
    Note: The transport map for each participant is stored with the labels and projected embeddings.
    Samples projections across participants cannot be performed as the transport map is not shared across participants.
    
    Does not account for temporal segmentation results. 
    
    """

    symbolic_config = import_config(symbolization_config_name)

    clustering_config_name = row.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    orthonormalization = symbolic_config.orthonormalization
    #co_normalization = symbolic_config.co_normalization
    inference_method = symbolic_config.model_params['inference_method']
    kernel_name = symbolic_config.model_params['kernel_name']
    temperature = symbolic_config.model_params['temperature']
    experiment_folder = os.path.join(get_data_root(), 'experiments', symbolic_config.experiment_name, symbolic_config.experiment_id, annotator_id, f'round_{round_number}', 'inference'); os.makedirs(experiment_folder, exist_ok=True)    
    proj_path = os.path.join(experiment_folder, f'{symbolization_config_name}_prototypes_projection_{row.identifier}.npy')
    transport_map_path = os.path.join(experiment_folder, f'{symbolization_config_name}_prototypes_projection_transport_map_{row.identifier}.npy')
    distance_matrix_path = os.path.join(experiment_folder, f'{symbolization_config_name}_distance_matrix_{row.identifier}.npy')
    
    
    
    if 'test_bounds' in row.index:
        test_bounds = row['test_bounds']
    else:
        red('/!\ Warning: test_bounds not found in row index. Using default bounds [0, N]')
        raise ValueError("test_bounds not found in row index. Please ensure the DataFrame contains 'test_bounds' column.")
        test_bounds = [0, row.N]
       
       
    if inference_method in ['wasserstein_projection', 'subspace_projection']:
        labels_path = os.path.join(experiment_folder, f'{symbolization_config_name}_prototypes_projection_labels_{row.identifier}.npy')
        
    elif inference_method == 'clustering':
        
        clustering_output_folder = os.path.join(get_data_root(), 'experiments', clustering_config.experiment_name, clustering_config.experiment_id, annotator_id, f'round_{round_number}')
        labels_path = os.path.join(clustering_output_folder, f'{row.identifier}_labels.npy')
    else:
        raise ValueError(f"Invalid inference_method: {inference_method}")


    if os.path.exists(proj_path) and os.path.exists(labels_path) and not overwrite:
        labels = np.load(labels_path)
        distance_matrix = np.load(distance_matrix_path)
        cluster_dist = np.min(distance_matrix, axis=1)
        return pd.Series([labels.tolist(), cluster_dist, transport_map_path, distance_matrix_path, proj_path])    
    
    elif not do_compute:
        #print(f"Skipping computation as do_compute is False.")
        return pd.Series([[], [], np.nan, np.nan, np.nan])
    
    else:
        pass#print(f"File {proj_path} does not exist. Processing...")
        

    # if row["video_representation_path"] == '/diskA/sam_data/data-features/cuisine/G57_P44_THENic_15012019/Tobii/video_representations_VideoMAEv2_OL8dijz2fixNZgrn9AQoXQ==.npy':
    #     print('Skipping this video representation')
    #     return pd.Series([[], [], np.nan, np.nan, np.nan])
    
    if X is None:
        X = np.load(row['video_representation_path'])[test_bounds[0]:test_bounds[1]]
        X = check_train_data(normalize_data(X, normalization=normalization))
    
    #print('{}: Shape of x: {} shape of prototypes: {}'.format(row["video_representation_path"], X.shape, prototypes.shape))
    
    
    if inference_method == 'subspace_projection':
        #print(f'Computing subspace projection from {row["video_representation_path"]}')
        M, labels, cluster_dist  = project_onto_prototype_components(X, prototypes, kernel_name=kernel_name, orthonormalization=orthonormalization, normalization=normalization, sigma_rbf=sigma_rbf, temperature=temperature, verbose=False)
        #distances, labels
        
    elif inference_method == 'wasserstein_projection':
        #print(f'Computing wasserstein projection from {row["video_representation_path"]}')
        
        weights_priors = symbolic_config.model_params['weights_priors']
        
        #print(f'Orthonormalization of the embeddings and the prototypes from {row["video_representation_path"]} to compute e.g. {proj_path}')
        X_orth, P_orth = orthonormalize_embeddings(X, prototypes)

        P_orth = normalize_data(P_orth, normalization=normalization)
            
        M, X_proj, T, labels, alpha, beta = compute_wasserstein_projection(X_orth, P_orth, weights_priors=weights_priors, temperature=1)
        cluster_dist = np.min(M, axis=1)
        
        #print(labels) # Comes from per-participant Z-normalization
        #print(row['embedding_labels']) # Comes from the global-Z-normalization (adjusting for mean and std for each D dimensions using all samples)    
    
    elif inference_method == 'clustering':
        
        M = pairwise_distances(X, prototypes, metric=kernel_name)
        
        # Find the index of the nearest prototype for each data point
        labels = np.argmin(M, axis=1)
        cluster_dist = np.min(M, axis=1)
        print(f' Coverage of the label space: {len(Counter(labels))}')
        
    if verbose:
        
        row[f'{inference_method}_labels'] = labels
        
        plot_chronogames(row, labels_col=f'{inference_method}_labels')
        plot_labels_2D_encoding(row, xlim=None, labels_col=f'{inference_method}_labels',  do_order_labels=True, figsize=(25, 12), perceptually_consistent=False)
        #plot_trajectory(M, title=f'{inference_method} | M=np.exp(-gammma*d(x_i, mu_j))', figsize=(25, 12), T_high=30, add_lowest=True, xlim=None)

    if  len(labels) != row.N :
        #print(f'Label range: [{np.unique(labels).min()}:{np.unique(labels).max()}] ({len(np.unique(labels))} unique labels)')
        green(f'Participant: {row.participant_id} - {row.identifier} - {inference_method}\n' + \
                f'\tLabels range: [{np.unique(labels).min()}:{np.unique(labels).max()}]\n' + \
                f'\t({len(np.unique(labels))} unique labels)\n' + \
                f'\tlen(labels)={len(labels)}, shape M={M.shape}\nB={row.test_bounds}, N={row.N} N_raw={row.N_raw}\n')
    # Save outputs
    np.save(labels_path, labels)
    #green(f"Saved labels to {labels_path}")

    np.save(distance_matrix_path, M)
    #green(f"Saved matrix distance to {distance_matrix_path}")
    
    if inference_method == 'wasserstein_projection':
        
        np.save(transport_map_path, T)
        green(f"Saved optimal transport map/coordinates to {transport_map_path}")
        
        np.save(proj_path, X_proj)
        green(f"Saved projected embeddings to {proj_path}")
    
        
    return pd.Series([labels, cluster_dist, transport_map_path, distance_matrix_path, proj_path])

def batched_cosine_median(X, batch_size=10000):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    dists = []
    for _ in range(n // batch_size):
        i = np.random.randint(0, n, batch_size)
        j = np.random.randint(0, n, batch_size)
        dot = np.sum(X[i] * X[j], axis=1)
        dists.append(1 - dot)
    return np.median(np.concatenate(dists))


def main(config_name='SymbolicInferenceConfig', annotator_id='samperochon', round_number=0, overwrite=True, verbose=True):
    """
    Infer labels and cluster-samples distances per participants (similar to clustering exeriments but wfor each participants) 
    Temporal segmetation is performed during post-processing when analyzing the distance matrix and deciding for retained labels. 
    Performs prototypes and samples assignement (create label predictions, assigned distances, and provide saved distance matrix path) using config_name: {}'.format(args.config_name)
    """
    
    print('#############################')
    green('Infer labels and cluster-samples distances per participants (similar to clustering exeriments but wfor each participants) ')
    green('Temporal segmetation is perfromed during post-processing when analyzing the distance matrix and deciding for retained labels.')    
    print('#############################')

    # 1) Get experiment configs
    config =  import_config(config_name)
    qualification_mapping = fetch_qualification_mapping(verbose=False)

    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    normalization = clustering_config.model_params['normalization']
    cluster_types = clustering_config.model_params['cluster_types'] 
    
    
    inference_method = config.model_params['inference_method']
    kernel_name = config.model_params['kernel_name']
    
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(experiments_folder, exist_ok=True)    
    
    # 1) Prepare dataset and shrink it for memory efficiency
    #df = get_experiments_dataframe(experiment_config_name=config_name,  return_symbolization=False)
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
    dset.metadata  = load_embedding_dimensions(dset.metadata)
    df = dset.metadata.copy()
    
    df['clustering_config_name'] = clustering_config_name
    df['X_all_sbj_slice'] = add_cum_sum_col(df, col='N')    
    info = f'{config_name}\nN_s = {df.participant_id.nunique()},  N={int(df.N.sum())}'
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"DataFrame memory size: {size_mb:.3f} MB")
    necessary_indices = ['clustering_config_name', 'video_representation_path', 'identifier', 'X_all_sbj_slice']
    #df = df[necessary_indices]; size_mb =df[necessary_indices].memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"DataFrame reduced size: {size_mb:.3f} MB")

    # 2) Get and normalize data samples and prototypes, according to the clustering and dataset configs
    P = get_prototypes(clustering_config_name, cluster_types=cluster_types, annotator_id=annotator_id, round_number=round_number, verbose=True)
    
    X_train = np.vstack([dset[i][0] for i in range(len(dset)) if dset.metadata.iloc[i].split == 'train'])    
    X_all =  np.vstack([dset[i][0] for i in range(len(dset))])
    train_scaler = StandardScaler().fit(X_train)#np.vstack([X_train]))
    print(f"/!\ Inference training scaler: mean={train_scaler.mean_}, std={train_scaler.scale_}")# Only fit

    X_train =  train_scaler.transform(X_train)  # Normalize training data for rbg computation 
    X_all = train_scaler.transform(X_all)  # Normalize all data
    P_norm = train_scaler.transform(P)  # Normalize prototypes using the training data scaler

    # l2 of the samples
    X_all = check_train_data(normalize_data(X_all, normalization=normalization))
    P_norm = check_train_data(normalize_data(P, normalization=normalization))
    
    
    # 2) (Optional) Inititlaization of the kernel bandwidth sigma_rbf as the median euclidean distance within the dataset
    if kernel_name == 'gaussian_rbf':    
        print('Computing Sigma rbf)')     
        sigma_rbf = batched_cosine_median(X_train, batch_size=10000)  # Use batched cosine median for large datasets
        #M_eucl = pairwise_distances(X_train, metric='cosine')
        #sigma_rbf = np.median(M_eucl[np.triu_indices_from(M_eucl, k=1)])
        gamma = 1 /  (2 * sigma_rbf ** 2); green(f'Estimated Gaussian RBF sigma for inference: Sigma={sigma_rbf} Gamma: {gamma}')
    else:
        sigma_rbf = None
        
        
    # 3) Perform inference of the labels (yiels labels, cluster_dist and distance path (store distance matrix) 
    blue(f'Running subspace projection using ditance kernel name: {kernel_name}')

    # For susbspace projection only the embedding_labels, cluster_dist, and distance path are used
    df[['embedding_labels', 'cluster_dist',  'T_path', 'M_path',  'X_proj_path']] = df.apply(lambda row: inference_projection(row, P_norm, X=X_all[row.X_all_sbj_slice[0]:row.X_all_sbj_slice[1], :], symbolization_config_name=config_name, annotator_id=annotator_id, round_number=round_number, sigma_rbf=sigma_rbf, overwrite=overwrite, do_compute=True,  normalization=None, verbose=verbose), axis=1)

    # 4) Get the global distance matrix and labels and visualize them (and save figures) 
    M_global = np.concatenate([np.load(path) for path in df.M_path.to_list()], axis=0)
    cluster_dist_global = np.hstack(df.cluster_dist)
    labels_global = np.hstack(df.embedding_labels)
    
    config.final_label_coverage = len(np.unique(labels_global))
    filename = os.path.join(experiments_folder, "config.json"); config.to_json(filename)
    filename = os.path.join(output_folder, "config.json"); config.to_json(filename)
    pd.DataFrame(qualification_mapping).to_csv(os.path.join(output_folder, 'qualification_mapping.csv'), index=False)

    print(f'sigma_rbf: {sigma_rbf} - Coverage={config.final_label_coverage}')
    
    fi(20, 4);_ = plt.hist(M_global.flatten(), bins='fd');plt.title(f'Distance matrix with kernel={kernel_name}\n{info}')
    plt.axvline(np.mean(M_global.flatten()), color='red', linestyle='dashed', linewidth=1, label='mean')
    plt.axvline(np.median(M_global.flatten()), color='blue', linestyle='dashed', linewidth=1, label='median');plt.legend()
    figpath = os.path.join(output_folder, 'global_distance_matrix_hist.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
    
    fi(20, 5);_ = plt.hist(cluster_dist_global, bins='fd');plt.title(f'Assignement cost/distance\n{info}');
    figpath = os.path.join(output_folder, f'global_cluster_distances_hist_{kernel_name}.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
    
    fi(20, 5);_ = plt.hist(labels_global, bins='fd');plt.title(f'Labels histogram across participants\n{info}');
    figpath = os.path.join(output_folder, f'global_labels_{kernel_name}.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
    
    fi(20, 5);_ = plt.hist(np.median(M_global, axis=0), bins='fd');plt.title(f'Median distance across samples (NxG={M_global.shape}\n{info}')
    figpath = os.path.join(output_folder, f'median_distances_across_participants_{kernel_name}.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

    fi(20, 15);plt.imshow(M_global[:100000, :].T, aspect='auto', cmap='coolwarm');plt.title(f'Global distance matrix (NxG={M_global[:100000, :].shape})\n{info}')
    plt.colorbar();figpath = os.path.join(output_folder, 'global_distance_matrix_hist.png')    
    plt.savefig(figpath, dpi=100, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
    
    green(f'Saved Iinference figures in folder {output_folder}.\nFinished inference of pariwise distance matrix')
    print('#############################')
    
    
    #5) If needed (we could restrain only to the experiments assuming G-space), apply the prototypes space reduction to the distance matrix (and save it as well)
    return 
    
def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    print('Performing prototypes and samples assignement (create label predictions, assigned distances, and provide saved distance matrix path) using config_name: {}'.format(args.config_name))
    main(config_name=args.config_name)
    t1 = time.time()
    print(f'Time elapsed: {t1 - t0} seconds')
    
    sys.exit(0)
    