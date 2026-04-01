"""Cosine k-means clustering via faiss for recursive prototyping.

Implements the core clustering step of the recursive prototyping pipeline
(Ch. 5, Section 5.2): C=100 candidate centroids per round, cosine k-means
on residual sets, with optional cross-validation splits.

Prerequisites:
    - Dataset prepared via ``smartflat.datasets.loader.get_dataset``.
    - Model and metric builders from ``smartflat.engine.builders``.

Main entry points:
    - ``main()``: Full clustering experiment with parameter sweeps.
    - ``main_deployment()``: Production clustering for a specific config/round.
    - ``main_personalized()``: DEPRECATED — raises DeprecatedWarning.
    - ``predict_with_centroids()``: Assign samples to nearest centroids.

Outputs:
    - Centroids saved as .npy files per experiment.
    - Clustering scores (silhouette, etc.) saved per experiment.
"""

import ast
import multiprocessing
import os
import sys
import time
from collections import Counter
from glob import glob
from pprint import pprint
from typing import Any, Callable, Dict, Literal, Optional

try:
    import faiss
except:
    pass
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.utils import resample
from torch.utils.data import DataLoader


import argparse

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from smartflat.configs.loader import get_complete_configs, import_config, load_config
from smartflat.datasets.loader import get_dataset, prepare_training_data
from smartflat.datasets.utils import load_embedding_dimensions
from smartflat.engine.builders import build_metrics, build_model
from smartflat.engine.clustering_evaluation import get_experiments_clustering
from smartflat.utils.utils import get_expids, pairwise
from smartflat.utils.utils_coding import fi, green, red, yellow
from smartflat.utils.utils_dataset import check_train_data, normalize_data
from smartflat.utils.utils_io import (
    fetch_qualification_mapping,
    get_data_root,
    load_df,
    parse_video_metadata,
)


def main_deployment(config_name, annotator_id, round_number, overwrite=False):
    
    
    config = import_config(config_name)
    #n_clusters = config.n_clusters
    
    if config.task_type == 'clustering':
        personalized = config.model_params['personalized']
    elif config.task_type == 'symbolization':
        personalized = config.personalized
    
    t_init = time.time(); t1 = time.time()

    for modality in ['Tobii']:#, 'GoPro1', 'GoPro2', 'GoPro3']:
        for task_name in ['cuisine']:#, 'lego']:
            
            if (task_name == 'cuisine') and (modality != 'Tobii'):
                continue
            elif (task_name == 'lego') and (modality == 'GoPro1'):
                continue
            #n_clusters = config.n_clusters_dict[task_name][modality]
            
            if config_name == 'ClusteringDeploymentKmeansCombinationCompleteConfig':
                
                input_clustering_config_names, round_numbers, qualification_cluster_types = get_complete_configs(round_number=round_number)
                model_params={'annotator_id': annotator_id,
                              'round_number': round_number,
                              'input_clustering_config_names': input_clustering_config_names,
                              'round_numbers': round_numbers,
                              'qualification_cluster_types': qualification_cluster_types,
                              }
                green(f'Creating clustering Configs: {input_clustering_config_names} outputs - with Rounds: {round_numbers}')
                
            elif config_name == 'ClusteringDeploymentKmeansInferenceConfig':
                    
                input_clustering_config_names, round_numbers, qualification_cluster_types = get_complete_configs(round_number=round_number, inference_mode=True)

                model_params={'annotator_id': annotator_id,
                                'round_number': round_number,
                                'input_clustering_config_names': input_clustering_config_names,
                                'round_numbers': round_numbers,
                                'qualification_cluster_types': qualification_cluster_types,
                                }

                green(f'Creating clustering Configs: {input_clustering_config_names} outputs - with Rounds: {round_numbers}')

            else:
                model_params = {}
            
            if personalized:
                print('Running personalized clustering.')
                main_personalized(config_name=config.config_name, 
                experiment_id=config.experiment_id,
                model_params=model_params, 
                dataset_params={'task_names': [task_name],
                                'modality': [modality]},
                do_whiten=config.do_batch_whiten,
                overwrite=overwrite)
                
            else:
                print('Running global-pool clustering type.')
                main(config_name=config.config_name, 
                        annotator_id=annotator_id,
                        round_number=round_number,
                        experiment_id=config.experiment_id,
                        model_params=model_params, 
                        dataset_params={'task_names': [task_name],
                                        'modality': [modality]},
                        overwrite=overwrite)
                        
        
            print(f'Finished clustering for {task_name}-{modality} with clusters in {time.time()-t_init:.2f} min.')
            t1 = time.time()
        
    print(f'Finished clustering for all tasks and modalities in {(time.time() - t_init)/60:.2f} min.')

def main(config_name='ClusteringAllConfig', annotator_id: str = 'samperochon', round_number: int = 0, experiment_id: str=None, model_params: dict = None, dataset_params: dict = None, continue_process=False, check_results=False, overwrite=False):

    # Get experiment config
    config = import_config(config_name)
    config.model_params['annotator_id'] = annotator_id
    config.model_params['round_number'] = round_number
    
    
    if model_params is not None:
        config.model_params.update(model_params)
        
    if dataset_params is not None:
        config.dataset_params.update(dataset_params)
        
    if experiment_id is not None:
        config.experiment_id = experiment_id
    
    elif config.experiment_id is None:
            
        config.experiment_id = get_expids()
        
    thresholding_method = config.thresholding_method
    threshold_value = config.threshold_value
    
    training_mode = config.training_mode
    normalization = config.model_params.get('normalization', None)
    kernel_name = config.model_params.get('kernel_name', None)
    co_normalization = config.model_params.get('co_normalization', None)
    print(f'Using normalization: {normalization}, co_normalization: {co_normalization}, kernel_name: {kernel_name}')

    # Check already done
    if check_results:
            
        results = get_experiments_clustering(config, use_stored=False)
        
        # Remove already done'
        if len(results) >0:
            print(f'Found {len(results)} experiments.')
            results_exp = results[(results['annotator_id'] == annotator_id) &
                                    (results['round_number'] == round_number) &
                                    (results['task_name'] == config.dataset_params['task_names'][0] ) &
                                    (results['modality'] == config.dataset_params['modality'][0]) &
                                    (results['n_clusters'] == config.model_params['n_clusters']) ]


            identifier_processed = results_exp.identifier.unique()
            experiment_id = results_exp.experiment_id.unique()
            print(f'Found {len(experiment_id)} experiment ids.')

            # New experiment
            if len(experiment_id) == 0:
                
                print('-> Running {}-{} with n_pca_components={} and n_clusters={}.'.format(config.dataset_params['task_names'][0], 
                                                                                        config.dataset_params['modality'][0], 
                                                                                        config.dataset_params['n_pca_components'], 
                                                                                        config.model_params['n_clusters']
                                                                                        ))

            elif len(experiment_id) == 1:
                print('-> Experiment {}-{} with n_pca_components={} and n_clusters={}. aleady exists with {} ids.'.format(config.dataset_params['task_names'][0], 
                                                                                                                    config.dataset_params['modality'][0], 
                                                                                                                    config.dataset_params['n_pca_components'], 
                                                                                                                    config.model_params['n_clusters'],
                                                                                                                    results_exp.identifier.nunique()
                                                                                                                    ))
                if continue_process:
                    print(f"Continuing experiment with ID: {experiment_id[0]}")
                else:
                    return 
                    
                config.experiment_id = experiment_id[0]
                
            else:
                print(f"Multiple experiments found: {experiment_id}. Exiting")
                return#raise ValueError(f"Multiple experiments found: {experiment_id}")

    output_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}')
    output_path = os.path.join(output_folder, f'cuisine_Tobii_cluster_centers.npy')
    print(f'Output folder: {output_folder}')
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f'Output file {output_path} already exists.')
    os.makedirs(output_folder, exist_ok=True)

    print(f'Saving outputs to {output_folder}.')
    filename = os.path.join(output_folder, "config.json")
            
    # Get dataset
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)

    assert (dset.metadata.task_name.nunique() == 1) & (dset.metadata.modality.nunique() == 1)
    task_name = dset.metadata.task_name.iloc[0]
    modality = dset.metadata.modality.iloc[0]
    
    # incorporate the co-Z-normalization mechanism if needed
    # if normalization == dset.normalization == 'Z-score' and not config_name == 'ClusteringDeploymentKmeansConfig':
    #     co_normalization=True
    #     dset.normalization = 'identity'
    #     config.model_params['normalization'] = 'identity'
    # else:
    #     co_normalization = False

    # 1) Get training data sampler 
    
    display(dset.metadata.split.value_counts().to_frame())
    X_train = np.vstack([dset[i][0] for i in range(len(dset)) if dset.metadata.iloc[i].split == 'train'])
    
    train_scaler = StandardScaler().fit(X_train)
    train_scaler_path = os.path.join(output_folder, f'training_scaler_N_{X_train.shape[0]}.pkl')
    joblib.dump(train_scaler, train_scaler_path)  # save
    green(f'Training scaler saved to {train_scaler_path}')

    print(f"/!\ Training scaler: mean={train_scaler.mean_}, std={train_scaler.scale_}")# Only fit
    
    
    
    # 1) Defining the training data, between weak-labels dependant (for recursive characterization) or all embeddings
    if isinstance(config.input_clustering_config_name, str):
        # Here we use the labels associated with the projection to the annnotated prorotypes space
        # The whole set of data has to be clustered using the complete partiionning of 
        # the centroids
        # Note that the rounds start again from the refinement wave of configs, needing to decrease from 1 the round to recover ssymbolizations
        print(f'[INFO] Using clustering labels from {config.input_clustering_config_name} for training.')
        qualification_mapping = fetch_qualification_mapping(verbose=False)
        
        if thresholding_method == 'threshold_mapping':
            complete_inf_config = import_config(config.input_clustering_config_name)
            n_sigma = complete_inf_config.n_sigma
            
            
            
            cl_output_folder = os.path.join(get_data_root(), 'outputs', complete_inf_config.experiment_name, complete_inf_config.experiment_id, annotator_id, f'round_{round_number-1}')
            thresholds_path = os.path.join(cl_output_folder, f'{config.input_clustering_config_name}_threshold_mapping_{n_sigma}_sigma.pkl')
            if not os.path.isfile(thresholds_path):
                raise FileNotFoundError(f'Thresholds file not found: {thresholds_path}. Please run the inference first.')
            threshold_mapping = load_df(thresholds_path)
        else:
            threshold_mapping = None
        
        dset.load_clustering_labels(config_name=config.input_clustering_config_name, add_distances=True, annotator_id=annotator_id, round_number=round_number-1)
        dset.metadata = load_embedding_dimensions(dset.metadata)
        
        # Identity transformation (from dataframe TODO make this more consistent (easy))
        X_train = prepare_training_data(dset.metadata, 
                                        splits=['train'], 
                                        qualification_mapping=qualification_mapping[config.input_clustering_config_name][annotator_id][f'round_{round_number-1}'], 
                                        thresholding_method=thresholding_method, 
                                        threshold_value=threshold_value,
                                        threshold_mapping=threshold_mapping,
                                        sample_inner=False, 
                                        add_ambiguous=True, 
                                        add_definitive=True)
        

        
    X_train = X_train[np.random.permutation(len(X_train))]
    
    train_scaler_path = os.path.join(output_folder, f'check_scaler_N_{X_train.shape[0]}.pkl')
    joblib.dump(train_scaler, train_scaler_path)  # save
    green(f'Training scaler saved to {train_scaler_path}')

    if training_mode:
        X_train = train_scaler.transform(X_train)
    else: 
        # Use instead all data for inference
        X_train = np.vstack([dset[i][0] for i in range(len(dset))])
        X_train = train_scaler.transform(X_train)
        
    X_train = normalize_data(X_train, normalization=normalization)
    
    # 2) Get test data (and delineations)
    if config.dataset_name == 'video_block_representation':
        N_i = [dset[i][0].shape[0] for i in range(len(dset))]
        
    elif config.dataset_name == 'video_segment_representation':
        #sdf = dset.metadata[dset.metadata['split'] == 'train']
        N_i = dset.metadata.drop_duplicates(subset=['participant_id', 'modality']).n_cpts.apply(lambda x: x-1).to_list()
        assert np.sum(N_i) == len(dset)
    
    if config.do_add_timestamps:
        assert config.input_clustering_config_name is None, 'Timestamps are not supported for personalized clustering.'
        T_i =  np.hstack([np.arange(ni)/ ni for ni in N_i]).flatten()
        X_train = np.concatenate([X_train, T_i[:, None]], axis=1)
        
    
    # 3) Build model 
    print(f'Building model {config.model_name} with params: {config.model_params}')
    model = build_model(config.model_name, config.model_params, verbose=True)
    
    if config.model_name == 'prototypes':
        pass
    
    # 2) Scaling
    print(f'Normalizing data with {normalization} with co_normalize={co_normalization}'); n_x = len(X_train)
    

    # 2) Labels computation  store labels
    X_all = np.vstack([dset[i][0] for i in range(len(dset))]).astype(np.float32)
    if config.do_add_timestamps:
        T_i =  np.hstack([np.arange(ni)/ ni for ni in N_i]).flatten()
        X_all = np.concatenate([X_all, T_i[:, None]], axis=1).astype(np.float32)
    X_all = train_scaler.transform(X_all)
    X_all = normalize_data(X_all, normalization=normalization)
    
    
    # if co_normalization:
    #     # Joint Z-transformation
    #     X_P_concat = np.vstack([X_train, model])
    #     X_P_concat_norm = scaler.transform(X_P_concat)
    #     X_all_scaled = scaler.transform(X_all)
    #     # X_P_concat_norm = check_train_data(normalize_data(X_P_concat, normalization=normalization))
    #     X_train = X_P_concat_norm[:X_train.shape[0]]
    #     model = X_P_concat_norm[-model.shape[0]:]
            
    # else:
    #     scaler = StandardScaler().fit(X_train)     
    #     #X_train = check_train_data(normalize_data(X_train, normalization=normalization))
    #     X_train = check_train_data(scaler.transform(X_train))
        

    # Final shape logging
    print('Training data shape for {} clustering K={}: {} ({} NAN vectors removed)'.format(
        config.model_name, config.model_params['n_clusters'], X_train.shape, n_x - X_train.shape[0]))

    # Perform clustering
    t0 = time.time(); assert X_train.flags.c_contiguous
    
    config.n_training = X_train.shape[0]

    if config.model_name in ['batch-kmeans', 'kmeans']:
        
        # 1) K-Means fitting
        X_train = X_train.astype(np.float32)
        model.fit(X_train)
        print('Fitted model with N={} vectors in {:.2f} min'.format(X_train.shape[0], (time.time() - t0)/60))
        
        for i in range(len(dset)):
            try:
                dset[i][0]
            except:
                red('Failed!')
                print(dset.metadata.iloc[i])
                
        labels = model.predict(X_all)
        
        green('Shape of centroid')
        D = X_train[0].shape[0]
        print(model.cluster_centers_[:, :D].shape)
        np.save(output_path, model.cluster_centers_[:, :D])
        
        if config.do_add_timestamps:
            output_path = os.path.join(output_folder, f'{task_name}_{modality}_cluster_centers_time.npy')
            np.save(output_path, model.cluster_centers_[:, -1])
            
    elif config.model_name == 'prototypes':
        
        # Need to adapt this from the inference code and more various distances computation
        print(f'Computing prototypes with {kernel_name} kernel.')
        labels = predict_with_centroids(X_all, model, distance=kernel_name)
        
        print(f'Number of labels used: {len(Counter(labels))} Coverage of the label space: {len(Counter(labels))}')

        config.embeddings_labels_support_size = len(Counter(labels))
        config.K_training =  model.shape[0]
        config.D =  model.shape[1]
        
        fi(20, 4)
        _ = plt.hist(labels, bins=200)
        plt.title('Distribution of N={} labels'.format(len(labels)))
        
        green(f'Shape of centroid {model.shape} saved in:\n{output_path}')
        np.save(output_path, model)
        
        
    elif config.model_name == 'faiss-kmeans-euclidean':
        
        # 1) K-Means fitting
        X_train = X_train.astype(np.float32)
        model.train(X_train)
        config.inertia = model.obj[-1]
        print('Fitted model with N={} vectors in {:.2f} min'.format(X_train.shape[0], (time.time() - t0)/60))
        
        # 2) Assign cluster labels
        _, labels = model.index.search(X_all, 1)  # Search assigns nearest centroid
        labels = labels.flatten()
    
        green(f'Shape of centroid {model.centroids.shape} asved in {output_path}')
        np.save(output_path, model.centroids)
        
    elif config.model_name == 'faiss-kmeans-cosine':
        
        # 1) Normalize data for cosine similarity
        X_train = X_train.astype(np.float32)
        faiss.normalize_L2(X_train)  # Ensures dot product = cosine similarity
        model.train(X_train)
        config.inertia = model.obj[-1]

        print('Fitted model with N={} vectors in {:.2f} min'.format(X_train.shape[0], (time.time() - t0)/60))
        
        # 2) Assign cluster labels
        faiss.normalize_L2(X_all)  # Normalize test data before searching
        _, labels = model.index.search(X_all, 1)
        labels = labels.flatten()
        
        green(f'Shape of centroid {model.centroids.shape} asved in {output_path}')
        np.save(output_path, model.centroids)
        

    elif config.model_name == 'tw-finch':
        
        centroids, num_clusters, labels = model(X)
        print('Fitted model with N={} vectors in {:.2f} min'.format(X.shape[0], (time.time() - t0)/60))
        
        green(f'Shape of centroid: {centroids.shape}')
        np.save(output_path, centroids)
        
    elif config.model_name == 'gaussian_mixture':
        

        start_time = time.time()
        model = fit_gmm_incremental(model, X, batch_size=10000)
        end_time = time.time()
        #model.fit(X)
        print(f"[INFO] Total time for fitting: {end_time - start_time:.2f} seconds")

        print('Fitted model with N={} vectors in {:.2f} min'.format(X.shape[0], (time.time() - t0)/60))
        assert model.converged_
        
        centroids = model.means_
        green(f'Shape of centroid: {centroids.shape}')
        np.save(output_path, centroids)
        
        
        output_path = os.path.join(output_folder, f'{task_name}_{modality}_clusters_weights.npy')
        weights = model.weights_
        np.save(output_path, weights)
        
        output_path = os.path.join(output_folder, f'{task_name}_{modality}_clusters_covariances.npy')
        covariances = model.covariances_
        np.save(output_path, covariances)
    
        config.n_iter_ = model.n_iter_
        config.converged_ = model.converged_
        config.lower_bound_ = model.lower_bound_
        config.aic = model.aic(X)
        config.bic = model.bic(X)
        
        
        # Compute metrics and store labels
        # Recreate X to make sure labels corresponds to the correct data (shuffling is performed inplace within the gmm fitting batch-version )
        X = np.vstack([dset[i][0] for i in range(len(dset))])
        labels = model.predict(X)
        
        np.save(output_path, model.means_)
        
    elif config.model_name == 'bayesian_gaussian_mixture':
        
        start_time = time.time()
        model = fit_gmm_incremental(model, X, batch_size=10000)
        end_time = time.time()
        #model.fit(X)
        print(f"[INFO] Total time for fitting: {end_time - start_time:.2f} seconds")

        print('Fitted model with N={} vectors in {:.2f} min'.format(X.shape[0], (time.time() - t0)/60))
        assert model.converged_
        
        centroids = model.means_
        green(f'Shape of centroid: {centroids.shape}')
        np.save(output_path, centroids)
        
        
        output_path = os.path.join(output_folder, f'{task_name}_{modality}_clusters_weights.npy')
        weights = model.weights_
        np.save(output_path, weights)
        
        output_path = os.path.join(output_folder, f'{task_name}_{modality}_clusters_covariances.npy')
        covariances = model.covariances_
        np.save(output_path, covariances)
    
        config.n_iter_ = model.n_iter_
        config.converged_ = model.converged_
        config.lower_bound_ = model.lower_bound_
        
        # Compute metrics and store labels
        # Recreate X to make sure labels corresponds to the correct data (shuffling is performed inplace within the gmm fitting batch-version )
        X = np.vstack([dset[i][0] for i in range(len(dset))])
        labels = model.predict(X)
        np.save(output_path, model.means_)

    # # Perform repeated clustering scores computation on downsampled data
    # if config.dataset_name == 'video_block_representation' or X.shape[0] >= 50000:
    #     config.silhouette_score, config.calinski_harabasz_score, config.davies_bouldin_score = stochastic_clustering_scores_computation(X, labels, n_samples=50000, n_repet=5)
    
    # elif config.dataset_name == 'video_segment_representation':
    #     config.silhouette_score = silhouette_score(X, labels)
    #     config.calinski_harabasz_score = calinski_harabasz_score(X, labels)
    #     config.davies_bouldin_score = davies_bouldin_score(X, labels)
    
    # print('\tSilhouette score: {:.2f}'.format(config.silhouette_score))
    # print('\tCalinski Harabasz score: {:.2f}'.format(config.calinski_harabasz_score))
    # print('\tDavies Bouldin score: {:.2f}'.format(config.davies_bouldin_score))
    
    config.to_json(filename)

    if config.dataset_name == 'video_block_representation':
        df_save = dset.metadata
    elif config.dataset_name == 'video_segment_representation':
        df_save = dset.metadata.drop_duplicates(subset=['participant_id', 'modality'])

    i_mem = 0
    for i in range(len(df_save)):
        row = df_save.iloc[i] 
        output_path = os.path.join(output_folder, f'{row.identifier}_labels.npy')
        l_i = labels[i_mem: i_mem + int(N_i[i])]; i_mem = i_mem + int(N_i[i])
        #print('len saved labels: ', len(l_i))
        np.save(output_path, l_i.astype(int)) # Float to avoir overflow when K > 255
    print(f'/!\ Label coverage: {len(np.unique(labels))}')
    print(f'Saved labels for e.g. {row.identifier} to {output_path}.')

def main_personalized(config_name='ClusteringAllConfig', experiment_id: str=None, model_params: dict = None, dataset_params: dict = None, do_whiten=False, continue_process=False, check_results=False):

    raise DeprecatedWarning('This function is deprecated, please use main_deployment instead (add annotator etc)')
    # Get experiment config
    config = import_config(config_name)
    
    if model_params is not None:
        config.model_params.update(model_params)
        
    if dataset_params is not None:
        config.dataset_params.update(dataset_params)
        
    if experiment_id is not None:
        config.experiment_id = experiment_id
    
    elif config.experiment_id is None:
        config.experiment_id = get_expids()

    output_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id)
    os.makedirs(output_folder, exist_ok=True)
    print(f'[INFO] Saving outputs to {output_folder}.')
    filename = os.path.join(output_folder, "config.json")
    
    # Get dataset
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)

    # Process each video individually
    for i in range(len(dset)):
        identifier = dset.metadata.iloc[i].identifier
        
        output_path = os.path.join(output_folder, f'{identifier}_labels.npy')
        cluster_centers_path = os.path.join(output_folder, f'{identifier}_cluster_centers.npy')
        if os.path.isfile(output_path) and os.path.isfile(cluster_centers_path):
            print(f'[INFO] Found existing labels for {identifier}, skipping.')
            continue
        
        video_embeddings = dset[i][0]  # Shape: N_i x D
        print(f'[INFO] Processing video {i}, embeddings shape: {video_embeddings.shape}')

        if do_whiten: 
            yellow('Whitening samples.')
            video_embeddings = (video_embeddings - video_embeddings.mean(axis=0)) / video_embeddings.std(axis=0)

        # Remove NaNs
        n = video_embeddings.shape[0]
        nan_rows = np.any(np.isnan(video_embeddings), axis=1)
        video_embeddings = video_embeddings[~nan_rows]
        #print(f'Clustering video {i}, shape after NaN removal: {video_embeddings.shape} ({n - video_embeddings.shape[0]} NAN vectors removed)')

        # Build model
        model = build_model(config.model_name, config.model_params)

        # Perform clustering
        t0 = time.time()
        assert video_embeddings.flags.c_contiguous
        
        if config.model_name == 'kmeans':

            model.fit(video_embeddings)
            print(f'[INFO] Fitted model for video {i} with N={video_embeddings.shape[0]} vectors in {(time.time() - t0) / 60:.2f} min')

            # Save labels
            labels = model.labels_
            np.save(output_path, labels)
            print(f'[INFO] Saved labels for video {i} to {output_path}')

            # Optionally save cluster centers if needed
            np.save(cluster_centers_path, model.cluster_centers_)
            print(f'[INFO] Saved cluster centers for video {i} to {cluster_centers_path}')
        
        elif config.model_name == 'tw-finch':

                
            centroids, num_clusters, labels = model(video_embeddings)
            print(f'[INFO] Fitted model for video {i} with N={video_embeddings.shape[0]} vectors in {(time.time() - t0) / 60:.2f} min')
            
            np.save(output_path, labels)
            print(f'[INFO] Saved labels for video {i} to {output_path}')

            np.save(cluster_centers_path, centroids)
            print(f'[INFO] Saved cluster centers for video {i} to {cluster_centers_path}')


    # Save config
    config.to_json(filename)
    print(f'[INFO] Saved configuration to {filename}')



def stochastic_clustering_scores_computation(X, labels, n_samples=20000, n_repet=5):
    """Perform repeated clustering scores computation on downsampled data.
    
    Notes:
    https://scikit-learn.org/stable/modules/clustering.html
    """
    
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    t0 = time.time()
    
    for _ in range(n_repet):
        # Downsample training data to compute stats (keeping cluster distribution constant)
        downsampled_X, downsampled_labels = downsample_data(X, labels, n_samples=n_samples)
        
        t0 = time.time()
        silhouette_scores.append(silhouette_score(downsampled_X, downsampled_labels))

        ch_scores.append(calinski_harabasz_score(downsampled_X, downsampled_labels))

        db_scores.append(davies_bouldin_score(downsampled_X, downsampled_labels))
    
    print('Done computing clustering scores {} times in {:.2f} min'.format(n_repet,  (time.time() - t0)/60))
    
    print('List of slihouette score: ', silhouette_scores)
    print('List of ch_scores score: ', ch_scores)
    print('List of db_scores score: ', db_scores)
    
    # Compute average scores
    avg_silhouette_score = np.mean(silhouette_scores)
    avg_ch_score = np.mean(ch_scores)
    avg_db_score = np.mean(db_scores)

    return avg_silhouette_score, avg_ch_score, avg_db_score

def downsample_data(X, labels, n_samples=20000):
    # Combine the features and labels into one dataset
    data = np.c_[X, labels]

    # Split the data by label
    data_by_label = [data[data[:,-1] == label] for label in np.unique(labels)]

    # Downsample each group separately
    downsampled_data_by_label = [resample(group, replace=False, n_samples=int(n_samples * len(group) / len(data))) for group in data_by_label]

    # Concatenate the downsampled data
    downsampled_data = np.concatenate(downsampled_data_by_label)

    # Split the features and labels
    downsampled_X = downsampled_data[:,:-1]
    downsampled_labels = downsampled_data[:,-1]

    return downsampled_X, downsampled_labels

def predict_with_centroids(X, P, distance='euclidean'):

    # # return assignments
    # if distance == 'euclidean':
    #     # Use sklearn's pairwise_distances for efficient Euclidean distance computation
    #     distances = pairwise_distances(X, P, metric='euclidean')

    # elif distance == 'cosine':
    #     # Use sklearn's pairwise_distances for efficient cosine distance computation
    #     distances = pairwise_distances(X, P, metric='cosine')

    # else:
    #     raise ValueError(f'Invalid distance metric: {distance}')

    # # Find the index of the nearest prototype for each data point
    # assignments = np.argmin(distances, axis=1)
    
    
    # Rebuild index with same dimension and spherical setting
    index = faiss.IndexFlatIP(P.shape[1])  # IP = inner product = cosine sim (since vectors are L2-normalized)
    faiss.normalize_L2(P)  # Redundant if saved from faiss.Kmeans with spherical=True, but safe
    index.add(P)

    # Normalize query samples
    faiss.normalize_L2(X)  # X_query can be X_train or any test set

    # Assign labels
    _, labels = index.search(X, 1)
    labels = labels.flatten()



    return labels

# Example usage:
# X = np.random.rand(100, 5)  # Example dataset with 100 points in 5 dimensions
# P = np.random.rand(10, 5)    # Example set of 10 prototypes in 5 dimensions
# assignments = assign_prototypes_cosine(X, P)






# Incremental GMM fitting function
def fit_gmm_incremental(gmm, data, batch_size=10000):
    
    """
    [INFO] Fitting batch 96/97 (size: 10000)
        Initialization converged: True
        106.87s - n_iter: 3, converged: True, lower_bound: -602.1828835462559, aic: 81577805.67092511, bic: 332260242.9486086, time: 21.57 seconds
        [INFO] Fitting batch 97/97 (size: 7614)
        Initialization converged: True
        83.97s - n_iter: 3, converged: True, lower_bound: -584.3358456720971, aic: 78432414.2578947, bic: 319637471.10662395, time: 16.66 seconds
        [INFO] GMM fitting completed.
        [INFO] Total time for fitting: 6331.62 seconds
        Fitted model with N=967614 vectors in 105.53 min
        
        # Cuisine Tobii K=35:
        [INFO] Fitting batch 96/97 (size: 10000)
        Initialization converged: True
        59.26s - n_iter: 3, converged: True, lower_bound: -609.0648621986547, aic: 81715445.24397309, bic: 332397882.5216565, time: 12.85 seconds
        [INFO] Fitting batch 97/97 (size: 7614)
        Initialization converged: True
        47.26s - n_iter: 3, converged: True, lower_bound: -583.3002101817037, aic: 78416643.60064699, bic: 319621700.4493762, time: 10.08 seconds
        [INFO] GMM fitting completed.
        [INFO] Total time for fitting: 5816.10 seconds
                    
         """

    # shuffle data
    np.random.shuffle(data)
    
    #data, _ = downsample_data(data, labels=np.zeros(data.shape[0]), n_samples=20000)
    num_batches = int(np.ceil(data.shape[0] / batch_size))
    print(f"[INFO] Splitting data NxD={data.shape[0]}x{data.shape[1]} into {num_batches} batches (batch size: {batch_size})")
    
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data.shape[0])
        batch_data = data[start_idx:end_idx]
        sstart_time = time.time()

        print(f"[INFO] Fitting batch {i+1}/{num_batches} (size: {batch_data.shape[0]})")
        gmm.fit(batch_data)
        n_iter_ = gmm.n_iter_
        converged_ = gmm.converged_
        lower_bound_ = gmm.lower_bound_
        end_time = time.time()
        try:
            start_time = time.time()
            aic = gmm.aic(batch_data)
            bic = gmm.bic(batch_data)
            end_time = time.time()
            print(f'{end_time - sstart_time:.2f}s - n_iter: {n_iter_}, converged: {converged_}, lower_bound: {lower_bound_}, aic: {aic}, bic: {bic}, time: {end_time - start_time:.2f} seconds')
        except:
            print(f'{end_time - sstart_time:.2f}s - n_iter: {n_iter_}, converged: {converged_}, lower_bound: {lower_bound_}')
    print("[INFO] GMM fitting completed.")
    return gmm

# Parallel loading of batches (optional for large-scale datasets)
def load_data_in_batches(data, batch_size):
    num_batches = int(np.ceil(data.shape[0] / batch_size))
    return [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='ClusteringDeploymentKmeansConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    main_deployment(config_name=args.config_name)
    sys.exit(0)
    
    
    
