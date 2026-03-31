"""Smartflat dataset builder."""

import os
import socket
import sys

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler



from smartflat.configs.loader import import_config
from smartflat.constants import available_dataset_names
from smartflat.datasets.base_dataset import BaseDataset, SmartflatDataset
from smartflat.datasets.dataset_hands import HandsDataset, HandsProcessingDataset
from smartflat.datasets.dataset_multimodal import MultimodalDataset
from smartflat.datasets.dataset_skeleton import SkeletonDataset
from smartflat.datasets.dataset_speech import SpeechDataset
from smartflat.datasets.dataset_video_representations import (
    PrototypesTrajectoryDataset,
    VideoBlockDataset,
    VideoSegmentDataset,
    VideoSegmentRepresentationsDataset,
    VideoSymbolsDataset,
)

# from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import fetch_qualification_mapping, get_gold_data_path


def get_dataset(
    dataset_name: str = "base",
    used_cached: bool = False,
    **kwargs,
) -> BaseDataset:
    """Build `dataset_name` using provided keyword arguments.
    Args:
        dataset_name (str): name of the dataset to be built.
        kwargs (dict): keyword arguments passed to the Dataset class constructor.

    Raises:
        ValueError: If `dataset_name` is not registered.

    Returns:
        BaseDataset: Initialized dataset.
    """

    if dataset_name == "base":

        dset = SmartflatDataset(dataset_name=dataset_name, **kwargs)


    elif dataset_name == 'feature_extraction':
        
        dset = SmartflatDataset(dataset_name=dataset_name, do_apply_fixation=False, **kwargs)
        
        
    elif dataset_name == "harddrive":

        dset = SmartflatDataset(
            dataset_name=dataset_name, root_dir="/Volumes/harddrive/data", **kwargs
        )

    elif dataset_name == "smartflat":

        dset = SmartflatDataset(
            dataset_name=dataset_name, root_dir="/Volumes/Smartflat/data", **kwargs
        )

    elif dataset_name == "smartflat-refill":

        dset = SmartflatDataset(
            dataset_name=dataset_name,
            root_dir="/Volumes/Smartflat/data-refill",
            **kwargs,
        )

    elif dataset_name == "gold":

        gold_data_dir = get_gold_data_path()
        dset = SmartflatDataset(
            dataset_name=dataset_name, root_dir=gold_data_dir, **kwargs
        )

    elif dataset_name == "multimodal_dataset":

        dset = MultimodalDataset(dataset_name=dataset_name, **kwargs)

    elif dataset_name == "video_block_representation":

        dset = VideoBlockDataset(dataset_name=dataset_name, **kwargs)
    
    elif dataset_name == "prototypes_trajectory_representation":
        
        dset = PrototypesTrajectoryDataset(dataset_name=dataset_name, **kwargs)
        
    elif dataset_name == "video_symbolization":

        dset = VideoSymbolsDataset(dataset_name=dataset_name, **kwargs)

        
    elif dataset_name == "video_segment_representation":

        dset = VideoSegmentDataset(dataset_name=dataset_name, **kwargs)

    elif dataset_name == "video_segment_representation_unexpended":

        dset = VideoSegmentRepresentationsDataset(dataset_name=dataset_name, **kwargs)

    elif dataset_name == "skeleton_landmarks":

        dset = SkeletonDataset(dataset_name=dataset_name, **kwargs)

    elif dataset_name == "hand_landmarks":

        dset = HandsDataset(dataset_name=dataset_name, **kwargs)
        
    elif dataset_name == "tracking_hand_landmarks":

        dset = HandsProcessingDataset(dataset_name=dataset_name, **kwargs)

    elif dataset_name == "speech_recognition_representation":

        dset = SpeechDataset(dataset_name=dataset_name, **kwargs)

    elif os.path.isdir(dataset_name):

        dset = SmartflatDataset(
            dataset_name=dataset_name, root_dir=dataset_name, **kwargs
        )

    else:

        raise ValueError(
            f"Dataset {dataset_name} is not a valid directory or the dataset name is not registered. Use one of {available_dataset_names}"
        )

    return dset


def build_clustering_dataset_splits(config_nam='ClusteringDeploymentKmeansRefinementTaskConfig', cluster_meta_type='all', centroid_min_dist=None, do_whiten=False):
    
    
    config = import_config(config_nam)

    # Get dataset
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
    assert (dset.metadata.task_name.nunique() == 1) & (dset.metadata.modality.nunique() == 1)
    task_name = dset.metadata.task_name.iloc[0]; modality = dset.metadata.modality.iloc[0]

    # Build datasets
    
    if config.input_clustering_config_name is not None:
        source_config = import_config(config.input_clustering_config_name)
        qualification_mapping = fetch_qualification_mapping()

    else:
        X = np.vstack([dset[i][0] for i in range(len(dset))])
    
    if do_whiten: 
        yellow('Whitening samples.')
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
    
    if config.do_add_timestamps:
        T_i =  np.hstack([np.arange(ni)/ ni for ni in N_i]).flatten()
        X = np.concatenate([X, T_i[:, None]], axis=1)


    print(f'[INFO] Shape of X: {X.shape}')
    # Remove nans before clustering
    n_x = X.shape[0]; nan_rows = np.any(np.isnan(X), axis=1); X = X[~nan_rows]
    print('Training data shape for {} clustering: {} ({} NAN vectors removed)'.format(config.model_name, X.shape, n_x - X.shape[0] ))


def prepare_training_data(df, qualification_mapping=None, cluster_meta_type='all', labels_col='embedding_labels', splits=['train'], n_max = 1e9, thresholding_method='1st', threshold_value=None, threshold_mapping=None, sample_inner=False, add_ambiguous=True, add_definitive=True, return_labels=False, verbose=True):
    """
    Note: labels_col should be the one used to compute the clust_distance 
    """
    
    total_samples = 0
    total_vectors = df.N.sum()
    
    def sample_cluster_vectors(row, cluster_values,  thresholding_method='1st', threshold_value=None, threshold_mapping=None):
        nonlocal total_samples
        X = np.load(row['video_representation_path'])[row.test_bounds[0]:row.test_bounds[1]]
        labels = row[labels_col]
        
        
        if thresholding_method == 'threshold_mapping':
                        
            cluster_dist = np.array(row['cluster_dist'])
            thresholds = np.vectorize(threshold_mapping.get)(labels)
            
            if sample_inner:
                idx_kept = np.argwhere(np.isin(labels, cluster_values) & (cluster_dist <= thresholds)).flatten()
            else:
                idx_kept = np.argwhere(np.isin(labels, cluster_values) & (cluster_dist > thresholds)).flatten()
            #print(f"[DEBUG] Selected {len(idx_kept)} samples with indices: {idx_kept}")


        elif thresholding_method in ['min_dist', 'std-1', 'std-2', 'median', '1st']:
            
            
            cluster_dist = np.array(row['cluster_dist'])
            
            if thresholding_method == 'median':
                
                centroid_min_dist = np.median(cluster_dist)
                
            elif thresholding_method == '1st':
                
                centroid_min_dist = np.quantile(cluster_dist, 0.25)
                
            
            elif thresholding_method == 'std-1':
                
                # Use the first standard deviation as the threshold
                centroid_min_dist = np.std(cluster_dist) + np.mean(cluster_dist)
                
            elif thresholding_method == 'std-2':
                
                # Use the second standard deviation as the threshold
                centroid_min_dist = 2 * np.std(cluster_dist) + np.mean(cluster_dist)
            
            elif thresholding_method == 'min_dist':
                
                # Use the minimum distance as the threshold
                centroid_min_dist = threshold_value
            
            else:
                raise ValueError(f"Unknown thresholding method: {thresholding_method}")
            
            
        
            if sample_inner:
                idx_kept = np.argwhere(np.isin(labels, cluster_values) & (cluster_dist <= centroid_min_dist)).flatten()
            else:
                idx_kept = np.argwhere(np.isin(labels, cluster_values) & (cluster_dist > centroid_min_dist)).flatten()
        
        else:
                
            # Fetch indexes of the clusters associated with e.g. task-related/exogeneous clusters
            idx_kept = np.argwhere(np.isin(labels, cluster_values)).flatten()
            proportion = total_samples / total_vectors if total_vectors > 0 else 0
        
        #print(f'[INFO] Sampling {len(idx_kept)}/{X.shape[0]} vectors from {row.participant_id}')
            
        # Update counters
        total_samples += len(idx_kept)
        
        if return_labels:
            return X[idx_kept], labels[idx_kept]
        
        return X[idx_kept]
    
    #display(df['split'].value_counts())


    X_training = []; L_training = []
    
    if cluster_meta_type == 'all':
        
        cluster_values = sorted(np.unique(np.hstack(df[df['split'].isin(splits)][labels_col].to_list())))
        for _, row in df[df['split'].isin(splits)].iterrows():
            
            if return_labels:
                X_i, l_i = sample_cluster_vectors(row, cluster_values=cluster_values, thresholding_method=thresholding_method, threshold_value=threshold_value, threshold_mapping=threshold_mapping)
                L_training.extend(l_i)
                
            else:
                X_i = sample_cluster_vectors(row, cluster_values=cluster_values, thresholding_method=thresholding_method, threshold_value=threshold_value, threshold_mapping=threshold_mapping)
            
            X_training.extend(X_i)
        
    else:
                   
        if add_definitive:
            
            # Incorporate loosely clustered samples from the "definitive" clusters
            for cluster_type in ['task-definitive', 'exo-definitive']:
                cluster_values = qualification_mapping[cluster_type]
                #rint(f'[INFO] Sampling vectors from {cluster_type} clusters: {cluster_values}')
                for _, row in [df['split'].isin(splits)].iterrows():
                    
                    if return_labels:
                        X_i, l_i = sample_cluster_vectors(row, cluster_values=cluster_values, centroid_min_dist=centroid_min_dist, threshold_mapping=threshold_mapping)
                        L_training.extend(l_i)
                        
                    else:
                        X_i = sample_cluster_vectors(row, cluster_values=cluster_values, centroid_min_dist=centroid_min_dist, threshold_mapping=threshold_mapping)
                    X_training.extend(X_i)


        if add_ambiguous: 
            
            # Incorporate ambiguous samples from the "ambiguous" clusters
            for cluster_type in ['task-definitive', 'exo-definitive']:
                if cluster_type not in qualification_mapping.keys():
                    print(f'[INFO] No ambiguous clusters found for {cluster_meta_type}')
                    
                else:
                    
                    cluster_values = qualification_mapping[cluster_type]
                    #print(f'[INFO] Sampling vectors from {cluster_type} clusters: {cluster_values}')
                    for _, row in [df['split'].isin(splits)].iterrows():
                        
                        
                        if return_labels:
                            X_i, l_i = sample_cluster_vectors(row, cluster_values=cluster_values, centroid_min_dist=None, threshold_mapping=None)
                            L_training.extend(l_i)
                            
                        else:
                            X_i = sample_cluster_vectors(row, cluster_values=cluster_values, centroid_min_dist=None, threshold_mapping=None)
                        X_training.extend(X_i)

    # Print final proportion of sampled vectors
    proportion = total_samples / total_vectors if total_vectors > 0 else 0
    if verbose:
        print(f'{thresholding_method} thresholding method with threshold_value={threshold_value}\nFinal proportion of sampled vectors: {total_samples}/{total_vectors} ({proportion:.2%})')
    
    # Downsample to max_samples
    if len(X_training) > n_max:
        print(f'[INFO] Downsampling {len(X_training)} to {n_max} vectors')
        indices = np.random.choice(len(X_training), n_max, replace=False)
        X_training = np.array(X_training)[indices]
        if return_labels:
            L_training = np.array(L_training)[indices]

    if return_labels:
        return np.vstack(X_training), np.array(L_training)
    else:
        if len(X_training) == 0:
            return []
        return np.vstack(X_training)
    
def sample_vectors(row, n_samples=5, probs_col='p_embedding_labels'):
    X = np.load(row['video_representation_path'])[row.test_bounds[0]:row.test_bounds[1]]
    assert X.shape[0] == row.p_embedding_labels.shape[0]
    indices = np.random.choice(X.shape[0], size=n_samples, p=row[probs_col])
    return X[indices]
def build_loaders(
    dataset_name: str = "",
    batch_size: int = 64,
    collate_fn=None,
    num_workers: int = 6,
    num_workers_eval: int = 2,
    drop_last: bool = True,
    **dataset_params,
):
    """Build training and validation dataloaders."""

    #
    # Train loader
    #

    train_dataset = get_dataset(dataset_name, **dataset_params)
    train_dataset.set_split("train")
    train_sampler = DistributedSampler(train_dataset, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context="spawn",
        prefetch_factor=num_workers,
        persistent_workers=True,
    )

    #
    # Validation loader
    #
    val_dataset = hcd.get_dataset(
        dataset_name, load_metadata_only=True, **dataset_params
    )
    val_dataset.set_split("val")

    val_sampler = DistributedValidationSampler(
        val_dataset,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers_eval,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context="spawn",
        prefetch_factor=num_workers_eval,
        persistent_workers=True,
    )

    #
    # Test loader
    #
    test_dataset = get_dataset(dataset_name, **dataset_params)
    test_dataset.set_split("test")

    test_sampler = DistributedValidationSampler(test_dataset)

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=num_workers_eval,
        collate_fn=collate_fn,
        pin_memory=True,
        multiprocessing_context="spawn",
        prefetch_factor=num_workers_eval,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
