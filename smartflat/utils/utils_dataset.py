import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import shuffle
from tqdm import tqdm

# def train_test_val_split_by(
#     dataset: pd.DataFrame,
#     train_size: float = 0.8,
#     test_to_val_size: float = 1.0,
#     split_by="participant_id",
# ) -> pd.DataFrame:
#     """Build train/val/test splits by participant_id.
#     This makes sure all the samples from the same participant_id are in the same split
#     """

#     if dataset[split_by].isnull().any():
#         raise ValueError(f"Split column '{split_by}' contains null values.")

#     subjects = dataset[split_by].unique().tolist()
#     train_subjects, test_subjects = train_test_split(
#         subjects, train_size=train_size, shuffle=True, random_state=42
#     )
#     # get the proportion of val to test + val (default: 0.5)
#     val_to_test_and_val_size = 1 / (1 + test_to_val_size)
#     test_subjects, val_subjects = train_test_split(
#         test_subjects, test_size=val_to_test_and_val_size, shuffle=True, random_state=42
#     )

#     splits = {}
#     splits.update({sid: "train" for sid in train_subjects})
#     splits.update({sid: "val" for sid in val_subjects})
#     splits.update({sid: "test" for sid in test_subjects})

#     dataset["split"] = dataset[split_by].map(splits)
#     dataset = shuffle(dataset)  # type: ignore
#     return dataset

def train_test_val_split_by(
    dataset: pd.DataFrame,
    train_subjects: list,
    train_size: float = 0.8,
    test_to_val_size: float = 1.0,
    random_state: int = 47,
    split_by="participant_id",
) -> pd.DataFrame:
    """Grow the provided list of train_subjects until train_size is reached by instance count.
    Then split remaining subjects into val/test by subject (not instance).
    """
    if dataset[split_by].isnull().any():
        raise ValueError(f"Split column '{split_by}' contains null values.")

    all_subjects = dataset[split_by].unique().tolist()
    remaining_subjects = [s for s in all_subjects if s not in train_subjects]
    rng = np.random.RandomState(random_state)
    rng.shuffle(remaining_subjects)

    # Grow train_subjects until enough instances are collected
    target_train_instances = int(train_size * len(dataset))
    current_train_subjects = list(train_subjects)
    current_train_count = dataset[dataset[split_by].isin(current_train_subjects)].shape[0]

    for sid in remaining_subjects:
        if current_train_count >= target_train_instances:
            break
        current_train_subjects.append(sid)
        current_train_count += dataset[dataset[split_by] == sid].shape[0]

    final_train_subjects = set(current_train_subjects)
    left_subjects = [s for s in all_subjects if s not in final_train_subjects]

    # Split remaining subjects into val and test
    val_ratio = 1 / (1 + test_to_val_size)
    test_subjects, val_subjects = train_test_split(
        left_subjects, test_size=val_ratio, random_state=42
    )

    # Assign splits
    splits = {}
    splits.update({sid: "train" for sid in final_train_subjects})
    splits.update({sid: "val" for sid in val_subjects})
    splits.update({sid: "test" for sid in test_subjects})

    dataset["split"] = dataset[split_by].map(splits)
    return shuffle(dataset, random_state=42)

def normalize_data(X_train, config=None, normalization=None):
    """Normalize data using Z-score or L2 normalization.
    Args:
        X_train: Input data to be normalized of shape (N, D).
        config: Configuration object containing normalization parameters.
        normalization: Type of normalization to apply ('Z-score', 'l2', or 'identity').
    Returns:
        Normalized data across the first (samples) axis.
    """
    
    eps = 1.0e-8
    assert normalization is not None or config is not None
    
    if config is not None:
        normalization = config.model_params['normalization']
        
    if normalization == 'Z-score':
        #print(f'Z-score normalization of the {X_train.shape[0]} samples')
        X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + eps)
    

    elif normalization == 'l2':
        #print(f'L2-normalization of the {X_train.shape[0]} samples')
        
        X_train = X_train / (
            np.linalg.norm(X_train, ord=2, axis=1, keepdims=True) + eps
        )
    
    elif normalization == 'identity':
        #print(f'Identity normalization of the {X_train.shape[0]} samples')
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    return X_train

def co_z_transform(X_all, P, normalization='Z-score'):
    # Compute Z-normalized versions jointly
    X_P_concat = np.vstack([X_all, P])
    X_P_concat_norm = check_train_data(normalize_data(X_P_concat, normalization=normalization))
    X_all_norm = X_P_concat_norm[:X_all.shape[0]]
    P_norm = X_P_concat_norm[-P.shape[0]:]
    return X_all_norm, P_norm

def check_train_data(X_train, verbose=False):
    
    print(f'[INFO] Shape of X: {X_train.shape}')
    # Remove nans before clustering
    n_x = X_train.shape[0]; nan_rows = np.any(np.isnan(X_train), axis=1); X_train = X_train[~nan_rows]
    if verbose:
        print('N_out={} ({} nan vectors removed)'.format(X_train.shape, n_x - X_train.shape[0] ))

    n_x = X_train.shape[0]; nan_rows = np.any(np.isinf(X_train), axis=1); X_train = X_train[~nan_rows]
    if verbose:
       print('N_out={} ({} Inf vectors removed)'.format(X_train.shape, n_x - X_train.shape[0] ))

    # Check if X_train is empty after filtering
    if X_train.shape[0] == 0:
        raise ValueError("Error: No valid samples left after removing NaN rows.")


    return X_train.astype(np.float32)

def z_normalize(X_train, model):
    """Jointly Z-normalizes training (NxD) and model (PxD) matrices.
    X_train_norm, model, mean, std = z_normalize(X_train, model)
    train_restored, model_restored = restore(X_train_norm, model, mean, std)
    """
    mean = np.mean(X_train, axis=0)  # Compute mean over training set
    std = np.std(X_train, axis=0)    # Compute std over training set
    std[std == 0] = 1  # Prevent division by zero

    train_norm = (X_train - mean) / std
    model_norm = (model - mean) / std

    return train_norm, model_norm, mean, std

def restore(train_norm, model_norm, mean, std):
    """Restores training and model matrices to their original scales.
    
    X_train_norm, model, mean, std = z_normalize(X_train, model)
    train_restored, model_restored = restore(X_train_norm, model, mean, std)

    """
    train_orig = train_norm * std + mean
    model_orig = model_norm * std + mean
    return train_orig, model_orig

def quantize_signal(arr, n_bins=10):
    """Apply quantization row-wise

        Function to quantize acceleration values into 10 bins (0-9) and assign -1 to NaNs

        df['acceleration_quantized'] = df['acceleration_norm_mean'].apply(lambda x: quantize_signal(x, n_bins=3) if isinstance(x, np.ndarray) else x)

    """
    nan_mask = np.isnan(arr)  # Track NaN values
    
    # Ignore NaNs for bin computation
    valid_values = arr[~nan_mask]
    
    if len(valid_values) == 0:  # If all values are NaN, return all -1
        return np.full_like(arr, -1)
    
    # Compute bin edges using quantiles
    bin_edges = np.nanpercentile(valid_values, np.linspace(0, 100, n_bins + 1))
    
    # Digitize values into bins (clip ensures values exactly at max are included)
    binned_values = np.digitize(valid_values, bin_edges, right=True) - 1
    
    # Ensure values stay within [0, 9]
    binned_values = np.clip(binned_values, 0, n_bins - 1)
    
    # Assign back to original array
    quantized_arr = np.full_like(arr, -1)  # Initialize with -1 (NaN encoding)
    quantized_arr[~nan_mask] = binned_values  # Assign bin indices to valid values
    
    return quantized_arr.astype(int)

def add_cum_sum_col(df, col='N'):
    n_cum = 0
    slices = []
    for n in df[col]:
        slices.append((int(n_cum), int(n_cum + n)))
        n_cum += n
    return slices


def sample_uniform_rows_by_col(df, column, N=5):
    # Compute min and max
    min_val = df[column].dropna().min()
    max_val = df[column].dropna().max()
    max_val = df[column][np.isfinite(df[column])].max()
    print(f"Sample {N} values from {column}: (U; Min: {min_val}, Max: {max_val})")
    # Generate N evenly spaced values between min and max
    target_values = np.linspace(min_val, max_val, N)
    
    # Find the closest rows for each target value
    sampled_rows = df.loc[[np.abs(df[column] - target).idxmin() for target in target_values]]
    
    return sampled_rows

def collapse_cluster_stats(df, labels_col='segments_labels', per_segment_feature='segments_length'):
    return df.apply(lambda row: pd.Series(row[per_segment_feature]).groupby(row[labels_col]).apply(list).to_dict() if ( (isinstance(row[per_segment_feature], list)) | (isinstance(row[per_segment_feature], np.ndarray)) )  else np.nan, axis=1)

def get_long_embedding(dset, add_cols=['participant_id', 'modality']):
    """Add for each participant as many rows to the dataframe as the number of block representation."""
    df = []
    for i in range(len(dset)):
    
        x, _, id = dset[i]; row = dset.metadata.iloc[i]
        n_nan = np.isnan(x).mean()
        X = pd.DataFrame()
        X['embeddings'] = x.tolist()
        X['n_nan'] = n_nan
        for add_col in add_cols:
            X[add_col] = row[add_col]
            
        df.append(X)

    # Concatenate all the dataframes into a long dataframe
    df = pd.concat(df, ignore_index=True)
        
    return df

def calculate_dispersion_ratio(df_all, which_col="embedding", groupby=['canonical_subject_id']):
    #df_subject_std = df_all.groupby(groupby, as_index=False).agg({which_col: lambda x: np.std(np.array(x.tolist()), ddof=0, axis=0)}).rename(columns={which_col: "embedding_std"})
    df_subject_std = df_all.groupby(groupby, as_index=False).agg({which_col: lambda x: np.std(np.array(x.tolist()), ddof=0, axis=0)}).rename(columns={which_col: "embedding_std"})

    df_subject_mean = df_all.groupby(groupby, as_index=False).agg({which_col: "mean"}).rename(columns={which_col: "embedding_mean"})
        
    embed_std_mean = np.mean(df_subject_std.embedding_std.values)
    embed_mean_std = np.std(df_subject_mean.embedding_mean.values, ddof=0, axis=0)
    emned_ratio = embed_std_mean / embed_mean_std
    return pd.DataFrame(emned_ratio, columns=["ratio"]) 


def compute_matrix_stats(P, title=''):
    """Computes various statistics of a prototype matrix."""
    
    def compute_rankme(P):
        """Computes the effective rank of a prototype matrix using the RankMe metric."""
        eps = 1e-7  # Stability
        s = torch.linalg.svdvals(P.clone().detach())
        s_norm = torch.linalg.norm(s, ord=1)
        p = s / s_norm + eps
        return torch.exp(-(p * torch.log(p)).sum()).item()

    def compute_spectral_gap(embeddings, k=7):
        """Spectral Analysis: Compute spectral gap of Laplacian matrix"""
        
        eps = 1e-7  # Stability
        #print("Constructing k-NN graph...")
        W = kneighbors_graph(embeddings, k, mode='distance', metric='euclidean', include_self=False)
        W = 0.5 * (W + W.T)  # Ensure symmetry

        #print("Computing Laplacian matrix...")
        degrees = np.array(W.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - W  # Unnormalized Laplacian

        #print("Computing smallest nonzero eigenvalue of Laplacian (spectral gap)...")
        smallest_eigenvalues = spla.eigsh(L, k=2, which='SM', return_eigenvectors=False)
        low_spectral_gap = smallest_eigenvalues[0] - smallest_eigenvalues[1]  # λn-1 - λn with λn > 0 smallest eigenvalue

        largest_eigenvalues = sorted(spla.eigsh(L, k=2, which='LM', return_eigenvectors=False))
        large_spectral_gap = largest_eigenvalues[1] - largest_eigenvalues[0]  # λ1 - λ2

        return low_spectral_gap, large_spectral_gap

    def triangle_inequality_test(embeddings, num_samples=50000):
        """Triangle Inequality & Geodesic Distortion """
        indices = np.random.randint(0, embeddings.shape[0], (num_samples, 3))
        
        distortions = []
        for i, (a, b, c) in tqdm(enumerate(indices), total=num_samples):
            
            d_ab = np.linalg.norm(embeddings[a] - embeddings[b])
            d_bc = np.linalg.norm(embeddings[b] - embeddings[c])
            d_ac = np.linalg.norm(embeddings[a] - embeddings[c])
            
            distortion = d_ab + d_bc - d_ac  # Should be ≥ 0 in Euclidean, < 0 in hyperbolic
            distortions.append(distortion)
        
        distortions = np.array(distortions)
        thinness_fraction = np.mean(distortions < 0)

        return thinness_fraction, np.percentile(distortions, [5, 50, 95])

    rank_standard = np.linalg.matrix_rank(P)
    rankme_value = compute_rankme(torch.tensor(P, dtype=torch.float32))
    
    low_spectral_gap, large_spectral_gap = compute_spectral_gap(P)
    thinness_fraction, distortion_stats = triangle_inequality_test(P)
    print(f'###################### {title} Matrix report #############################')
    print(f"\tShape={P.shape}, Rank: Std={rank_standard}, Eff={rankme_value:.2f} ({100*rankme_value/P.shape[0]:.1f}%)")
    print(f'\tGap: Low={low_spectral_gap:.2f}, High={large_spectral_gap:.2f}')
    print(f"\tTriangle distortion (5%, 50%, 95%): {distortion_stats.round(2)}")
