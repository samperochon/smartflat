import argparse
import ast
import multiprocessing
import os
import sys
import time
from glob import glob
from pprint import pprint

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

#from typing import Any, Callable, Dict, Literal, Optional



from smartflat.configs.loader import import_config, load_config
from smartflat.datasets.loader import get_dataset
from smartflat.engine.builders import build_metrics, build_model
from smartflat.utils.utils import get_expids, pairwise
from smartflat.utils.utils_io import get_data_root, load_df, parse_video_metadata, save_df


def main(args):
    
    config_name= args.config_name
    
    # Get experiment config
    config = import_config(config_name)
    
    results = get_results_clustering(config, use_stored=False, overwrite=True, do_reversed=args.reversed) #TODO change the overwrite potentially since at the experiment level 
    
def get_results_clustering(config, experiment_folder=None, use_stored=True,  do_reversed=False, overwrite=False, do_compute=False):
    
    if experiment_folder is None:
        
        experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name)
        
        #print('Warning hardcoding results')
        #experiment_folder = os.path.join(get_data_root(), 'experiments',  'clustering-whiten-PCA')
       
        
    results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name); os.makedirs(results_folder, exist_ok=True)
    
    #print('Warning hardcoding results')
    #xresults_folder = os.path.join(get_data_root(), 'outputs', 'clustering-whiten-PCA'); os.makedirs(results_folder, exist_ok=True)
    
    results_df_path = os.path.join(results_folder, 'results.csv')
    
    if not use_stored:
        
        t_init = time.time()
        
        dset_original_space = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
        persistent_metadata = dset_original_space.metadata.copy()
        exps = glob(os.path.join(experiment_folder, '*'))
        if do_reversed:
            print('Taking reversed order of experiments')
            exps = list(reversed(exps))
        
        
        print(f'Found a total of {len(exps)} experiments folders.')
        results = []
        for i, exp in enumerate(exps):
            train_config = load_config(os.path.join(exp, 'config.json'))
            
            if (train_config.dataset_params['normalization'] == 'PCA') and (train_config.dataset_params['n_pca_components'] is None):
                print(f'Skipping experiment {os.path.basename(exp)} because of missing PCA components.')
                continue
            
            
            dset_transformed_space = get_dataset(dataset_name=train_config.dataset_name, **train_config.dataset_params)

        
            if not hasattr(train_config, 'clustering_results') and not overwrite and do_compute:
                
                print('Compute metrics...')
                t_init_score = time.time()
                silhouette_scores, davies_bouldin_scores = compute_clustering_scores(dset_transformed_space, exp)

                train_config.clustering_results = {}
                train_config.clustering_results['silhouette_score'] = {identifier: score for identifier, score in zip(dset_transformed_space.metadata.identifier.to_list(), silhouette_scores) }
                train_config.clustering_results['davies_bouldin_score'] = {identifier: score for identifier, score in zip(dset_transformed_space.metadata.identifier.to_list(), davies_bouldin_scores) }
                #config.results['calinski_harabasz_score'] = {identifier: score for identifier, score in zip(dset.metadata.identifier.to_list(), calinski_harabasz_scores) }
                print('Done in {} minutes'.format((time.time() - t_init_score)/60))
                filename = os.path.join(exp, "config.json"); train_config.to_json(filename)
                print('Saving config to {}'.format(filename))
                                
            elif hasattr(train_config, 'clustering_results'):
                
                print('Do not compute clusering scores')
                res_df = pd.DataFrame({'identifier': train_config.clustering_results['silhouette_score'].keys(), 
                            'silhouette_score': train_config.clustering_results['silhouette_score'].values(),
                            'davies_bouldin_score': train_config.clustering_results['davies_bouldin_score'].values(),
                           # 'calinski_harabasz_score': config.clustering_results['calinski_harabasz_score'].values()
                            })

                dset_transformed_space.metadata = dset_transformed_space.metadata.merge(res_df, on='identifier', how='left')
                silhouette_scores = dset_transformed_space.metadata['silhouette_score'].to_list()
                davies_bouldin_scores = dset_transformed_space.metadata['davies_bouldin_score'].to_list()
                #calinski_harabasz_scores = dset.metadata['calinski_harabasz_score'].to_list()

            else:
                print('Do not compute clusering scores')
                silhouette_scores = len(dset_transformed_space) * [np.nan]
                davies_bouldin_scores = len(dset_transformed_space) * [np.nan]
                #calinski_harabasz_scores = len(dset) * [np.nan]
                
            results.append(pd.DataFrame({'experiment_id': train_config.experiment_id,
                                        'identifier': dset_transformed_space.metadata.identifier.to_list(), 
                                         'n_frames':  dset_transformed_space.metadata.n_frames.to_list(), 
                                         'fps':  dset_transformed_space.metadata.fps.to_list(),
                                         'task_name':  dset_transformed_space.metadata.task_name.to_list(),
                                         'modality':  dset_transformed_space.metadata.modality.to_list(),
    
                                         'model_name': train_config.model_name,
                                         'n_clusters': train_config.model_params['n_clusters'],
                                         'whiten': train_config.dataset_params['whiten'],
                                         'n_pca_components': train_config.dataset_params['n_pca_components'],
                                         'n_training': train_config.n_training,
                                         'normalization': train_config.dataset_params['normalization'],
                                         
                                         'pred_labels': dset_transformed_space.metadata.identifier.apply(lambda x: np.load(os.path.join(exp, f'{x}_labels.npy')) if os.path.isfile(os.path.join(exp, f'{x}_labels.npy')) else np.nan).to_list(),
                                         
                                         'silhouette_score': silhouette_scores,
                                         'davies_bouldin_score': davies_bouldin_scores,
                                         #'calinski_harabasz_score': calinski_harabasz_scores,
                                         
                                         'glob_silhouette_score': train_config.silhouette_score,
                                         #'glob_calinski_harabasz_score': config.calinski_harabasz_score,
                                         'glob_davies_bouldin_score': train_config.davies_bouldin_score
                                        }))
            #dset.metadata = persistent_metadata.copy()
            
            print(f'Done parsing experiment {i}/{len(exps)}')
    
        results = pd.concat(results)
        results['n_pca_components'].replace({None: -1}, inplace=True)
        results['duration'] = results['n_frames'] / results['fps'] / 60

        # Remove outliers
        n = results.identifier.nunique()
        results.dropna(subset=['duration'], inplace=True); print(f'/!\ {n - results.identifier.nunique()}/{n} identifers discarded because of missing duration.')
        #results = results[(results['fps'] > 10) & (results['duration'] > 20) & (results['duration'] < 90)]
        #print(f'/!\ {n - results.identifier.nunique()}/{n} identifers discarded because of fps and duration criteria.')

        # Add bins label
        results['n_frames_binned'], bins_label = pd.cut(results['duration'], bins=4, retbins=True, labels=False)
        results['duration_labels'] = results.n_frames_binned.map({i: j for i, j in enumerate(['{}-{}'.format(int(s), int(e)) for s,e in pairwise(bins_label)]) })
        Ns=results.drop_duplicates(subset=['identifier']).groupby(['task_name', 'modality'])['duration_labels'].value_counts()
        results['duration_labels_counts'] = results.apply(lambda x: f'{x.duration_labels} (N={Ns.loc[x.task_name].loc[x.modality].loc[x.duration_labels]})', axis=1)
        
        # Convert arrays to string and save results
        
        #results['pred_labels'] = results['pred_labels'].apply(lambda x: np.array2string(x, separator=',') if type(x) == np.ndarray else np.nan)
        save_df(results, results_df_path)
        print(f'Saved dataframe to {results_df_path} - {(time.time() - t_init)/60} minutes')
        #results['pred_labels'] = results['pred_labels'].apply(lambda x: ast.literal_eval(x) if type(x) == str else np.nan)
    
    elif not os.path.isfile(results_df_path):
        raise ValueError(f'No results found at {results_df_path}.')
    
    else:
        results = load_df(results_df_path)
        print(f'Loading results from {results_df_path}: {results.shape}.')
        
        
        #results['pred_labels'] = results.apply(lambda x: np.load(os.path.join(get_data_root(), 'experiments', config.experiment_name, x.experiment_id, f'{x.identifier}_labels.npy')), axis=1)
        #results['pred_labels'] = results['pred_labels'].apply(lambda x: ast.literal_eval(x) if type(x) == str else np.nan)


    #TODO: done earlier
    results['n_pca_components'].replace({None: -1}, inplace=True)
    return results

def get_experiments_clustering(config, experiment_folder=None, use_stored=True):
    
    if config:
        raise DeprecationWarning('need to handle dset_transformed/raw')
    
    if experiment_folder is None:
        
        experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name)
        
    results_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name); os.makedirs(results_folder, exist_ok=True)
    results_df_path = os.path.join(results_folder, 'experiments.csv')
    
    if not use_stored:
        
        t_init = time.time()
        
        dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
        exps = glob(os.path.join(experiment_folder, '*'))
        print(f'Found a total of {len(exps)} experiments folders.')
        results = []
        for exp in exps:
            print(f'Parsing {exp}...')
            if not os.path.isfile(os.path.join(exp, 'config.json')):
                yellowf(f'Deleting experiment folder because of missing config.json: {exp}')
                subprocess.run(['rm', '-r', exp])
                continue
            config = load_config(os.path.join(exp, 'config.json'))
            
            if (config.dataset_params['normalization'] == 'PCA') and (config.dataset_params['n_pca_components'] is None):
                print(f'Skipping experiment {os.path.basename(exp)} because of missing PCA components.')
                continue
            
            results.append(pd.DataFrame({'experiment_id': config.experiment_id,
                                        'identifier': dset.metadata.identifier.to_list(), 
                                         'task_name':  dset.metadata.task_name.to_list(),
                                         'modality':  dset.metadata.modality.to_list(),
    
                                         'model_name': config.model_name,
                                         'n_clusters': config.model_params['n_clusters'],
                                         
                                         'n_pca_components': config.dataset_params['n_pca_components'],
                                         #'n_training': config.n_training,
                                         
                                         'pred_labels': dset.metadata.identifier.apply(lambda x: np.load(os.path.join(exp, f'{x}_labels.npy')) if os.path.isfile(os.path.join(exp, f'{x}_labels.npy')) else np.nan).to_list(),
                                        
                                         'glob_silhouette_score': config.silhouette_score if hasattr(config, 'silhouette_score') else np.nan,
                                         'glob_calinski_harabasz_score': config.calinski_harabasz_score if hasattr(config, 'calinski_harabasz_score') else np.nan,
                                         'glob_davies_bouldin_score': config.davies_bouldin_score if hasattr(config, 'davies_bouldin_score') else np.nan,
                                        
                                        }))
            
        if len(results) == 0:
            print('No experiemnts found.')
            return pd.DataFrame()
            
        results = pd.concat(results)
        
        results.to_csv(results_df_path, index=False)
        print(f'Saved dataframe to {results_df_path} - {(time.time() - t_init)/60} minutes')
    
    elif not os.path.isfile(results_df_path):
        pass#raise ValueError(f'No results found at {results_df_path}.')
    
        return pd.DataFrame()
    else:
        if not os.path.isfile(results_df_path):
            return pd.DataFrame()
        print(f'Loading results from {results_df_path}.')
        results = pd.read_csv(results_df_path, low_memory=False)
        results['pred_labels'] = results.apply(lambda x: np.load(os.path.join(get_data_root(), 'experiments',  config.experiment_name, x.experiment_id, f'{x.identifier}_labels.npy')), axis=1)
        #results['pred_labels'] = results['pred_labels'].apply(lambda x: ast.literal_eval(x) if type(x) == str else np.nan)

    return results

def compute_scores_for_data_point(i, dset, experiment_path):
    labels_file = os.path.join(experiment_path, f'{dset.metadata.identifier.iloc[i]}_labels.npy')
    if os.path.isfile(labels_file):
        labels = np.load(labels_file)
        embeddings = dset[i][0]
        
        # Add a small random noise
        noise = np.random.normal(0, 1e-9, embeddings.shape)
        embeddings = embeddings + noise
        #print('Computing scores for {} with {} embeddings'.format(dset.metadata.identifier.iloc[i], embeddings.shape[0]))
        try:
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            #calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        except ValueError:
            silhouette = np.nan
            davies_bouldin = np.nan
            #calinski_harabasz = np.nan
            
    else:
        silhouette = np.nan
        davies_bouldin = np.nan
        #calinski_harabasz = np.nan
    return silhouette, davies_bouldin#, calinski_harabasz

def compute_clustering_scores(dset, experiment_path):
    num_data_points = len(dset)
    scores = Parallel(n_jobs=-1)(
        delayed(compute_scores_for_data_point)(i, dset, experiment_path) for i in range(num_data_points)
    )
    silhouette_scores, davies_bouldin_scores = zip(*scores)
    return silhouette_scores, davies_bouldin_scores


def parse_args():
    parser = argparse.ArgumentParser(
        'Create clustering results.', add_help=False)

    parser.add_argument(
        '--config_name',
        default='ClusteringAllConfig',
        type=str,
        help='Config name to use for clustering evaluation')
    parser.add_argument('--reversed', action='store_true', help='Whether exploring experiment paths in reversed order (for speed with 2 VM)')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    main(args)    
    main(args)