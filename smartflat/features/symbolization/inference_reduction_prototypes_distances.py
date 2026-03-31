
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import time
from collections import defaultdict

import matplotlib.pyplot as plt

from smartflat.configs.loader import import_config
from smartflat.features.symbolization.co_clustering import get_prototypes_mapping
from smartflat.features.symbolization.utils_dataset import get_experiments_dataframe
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import fetch_qualification_mapping, get_data_root


def source_distance_shape(x):
    return np.array(list(x)).shape

def meta_cluster_average_dist(x):
    x = np.vstack(x.values)  # some x[i] might be 1D arrays
    return pd.Series(np.mean(x, axis=0))

def meta_cluster_std_dist(x):
    x = np.vstack(x.values)  # some x[i] might be 1D arrays
    return pd.Series(np.std(x, axis=0))

def meta_cluster_min_dist(x):
    x = np.vstack(x.values) # some x[i] might be 1D arrays
    return pd.Series(np.min(x, axis=0))

def meta_cluster_max_dist(x):
    x = np.vstack(x.values) # some x[i] might be 1D arrays
    return pd.Series(np.max(x, axis=0))

def meta_cluster_maxmin_dist(x):
    x = np.vstack(x.values) # some x[i] might be 1D arrays
    return pd.Series((np.max(x, axis=0) - np.min(x, axis=0)))

def agglomerate_distance_matrix(M, mapping_prototypes_reduction, distance_aggregate_col='meta_cluster_min_dist', return_clusterdf=False, verbose=False):
    
    # 1) Get the distance matrix with proper indexes mapping
    cdf = pd.DataFrame(mapping_prototypes_reduction, index=['meta_cluster_index']).T
    cdf.reset_index(names=['source_cluster_index'], inplace=True)
    cdf.sort_values(['source_cluster_index'], inplace=True)
    
    excluded_clusters = [i for i in range(M.shape[1]) if i not in sorted(mapping_prototypes_reduction.keys())]
    exdf = pd.DataFrame({'source_cluster_index': excluded_clusters, 'meta_cluster_index': [-3] * len(excluded_clusters)})
    cdf = pd.concat([cdf, exdf], axis=0)

    
    # 2) "Trick" to aggregate distances based on the mapping: use pandas grouping abality
    cdf['distance_samples'] = list(M.T)
    cdf = cdf.groupby(['meta_cluster_index']).distance_samples.agg([source_distance_shape, meta_cluster_average_dist, meta_cluster_std_dist, meta_cluster_min_dist, meta_cluster_max_dist, meta_cluster_maxmin_dist])    
    cdf.reset_index(inplace=True)
    
    # 3) Refactor the distance matrix
    M_transformed = np.vstack(cdf[distance_aggregate_col]).T
    
    if verbose:
        # list of distance types to plot
        distance_types = [
            'meta_cluster_average_dist',
            'meta_cluster_std_dist',
            'meta_cluster_min_dist',
            'meta_cluster_max_dist',
            'meta_cluster_maxmin_dist'
        ]

        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.flatten()

        for i, dist_type in enumerate(distance_types):
            ax = axes[i]

            # transform matrix
            M_transformed = np.vstack(cdf[dist_type]).T
            M_show = M_transformed @ M_transformed.T

            # plot
            im = ax.imshow(M_show, aspect='auto', cmap='coolwarm', origin='lower', interpolation='nearest')
            ax.set_title(dist_type.replace("meta_cluster_", ""), fontsize=14)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # remove the unused 6th subplot
        fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()

    
    
    if return_clusterdf:
        return cdf, M_transformed
    
    return M_transformed

def reduce_similarity_matrix(D_x, mapping_prototypes_reduction, agg_fn=np.mean):
    
    K = D_x.shape[0]
    G = np.max(list(mapping_prototypes_reduction.values()))
    D_xr = np.zeros((G, G))

    # Bucket indices by group
    groups = defaultdict(list)
    for idx, g in mapping_prototypes_reduction.items():
        groups[g].append(idx)

    # Aggregate between groups
    for i in range(G):
        for j in range(G):
            rows = groups[i]
            cols = groups[j]
            values = D_x[np.ix_(rows, cols)]
            D_xr[i, j] = agg_fn(values)

    return D_xr

def plot_distance_aggregations(df, mapping_prototypes_reduction, distance_aggregate_cols = ['meta_cluster_average_dist', 'meta_cluster_std_dist', 'meta_cluster_min_dist', 'meta_cluster_max_dist', 'meta_cluster_maxmin_dist'], figsize=None, n_subj_rows=1, crop_x=None, crop_y=None):
    
    n_plots = 1 + len(distance_aggregate_cols)
    
    if figsize is None:
        figsize=(6 * n_plots, 4 * n_subj_rows)

    fig, axes = plt.subplots(n_subj_rows, n_plots, figsize=figsize, sharex=False, sharey=False)
    fig.suptitle(f"Distance matrix and aggregations for {n_subj_rows} subjects", y=1.03, fontsize=16)
    
    if n_subj_rows == 1:
        axes = np.expand_dims(axes, 0)  # Ensure 2D array of axes

    for i in range(n_subj_rows):
        row = df.sample(1).iloc[0]
        M = np.load(row.M_path)
        
        if crop_x is None:
            crop_x = [0, M.shape[0]]
        if crop_y is None:
            crop_y = [0, M.shape[1]]
        M = M[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
        
        # --- Original
        ax = axes[i, 0]
        im = ax.imshow(M.T, cmap='coolwarm', aspect='auto', interpolation='nearest')
        ax.set_title("Original distance")
        fig.colorbar(im, ax=ax)

        # --- Aggregated
        for j, distance_aggregate_col in enumerate(distance_aggregate_cols, start=1):
            cdf, M_transformed = agglomerate_distance_matrix(
                M, mapping_prototypes_reduction, distance_aggregate_col=distance_aggregate_col, return_clusterdf=True
            )
            ax = axes[i, j]
            im = ax.imshow(M_transformed.T, cmap='coolwarm', aspect='auto', interpolation='nearest')
            ax.set_title(f"Aggregation: {distance_aggregate_col.replace('meta_cluster_', '').replace('_dist', '')}")
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_distances_matrix_tuples(distance_matrices, n_rows=4, n_cols=3, suptitle='', figsize=(25, 15), figpath=None):

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16, weight='bold', y=1.02)
    axes = axes.flatten()
    for ax, (title, matrix) in zip(axes, distance_matrices.items()):
        if 'placeholder' in title:
            ax.axis('off')
            continue
        
        im = ax.imshow(matrix, cmap='coolwarm')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set off remaining axes
    for ax in axes[len(distance_matrices):]:
        ax.axis('off')
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    if figpath is not None:
        plt.savefig(figpath, dpi=100, bbox_inches='tight')
    plt.show()

def plot_distances_and_hists(distance_matrices, suptitle='', figsize=(18, 3), figpath=None):
    items = [(k, v) for k, v in distance_matrices.items() if isinstance(v, np.ndarray)]
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(figsize[0], figsize[1]*n))
    fig.suptitle(suptitle, fontsize=16, weight='bold', y=1.01)
    
    for i, (title, matrix) in enumerate(items):
        axes[i, 0].imshow(matrix, cmap='coolwarm')
        axes[i, 0].set_title(title)
        axes[i, 0].axis('off')
        axes[i, 1].hist(matrix.flatten(), bins=50, color='gray')
        axes[i, 1].set_title('Histogram')
    
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=100, bbox_inches='tight')
    plt.show()

def main_reduction_row(row, symbolization_config_name, mapping_prototypes_reduction, annotator_id, round_number, input_space='K_space', overwrite=False, verbose=False):

    # 1) get configs
    symbolic_config = import_config(symbolization_config_name)
    clustering_config_name = symbolic_config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    distance_aggregate_col = symbolic_config.distance_aggregate_col

    # 2) Define saving folders
    experiment_folder = os.path.join(get_data_root(), 'experiments', symbolic_config.experiment_name, symbolic_config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrices_reductions'); os.makedirs(experiment_folder, exist_ok=True)    
    if input_space == 'K_space':
        reduced_distance_matrix_path = os.path.join(experiment_folder, f'{symbolization_config_name}_reduced_distance_matrix_{row.identifier}.npy')
    elif input_space == 'G_space':
        reduced_distance_matrix_path = os.path.join(experiment_folder, f'{symbolization_config_name}_reduced_G_space_distance_matrix_{row.identifier}.npy')
    
    if os.path.exists(reduced_distance_matrix_path) and not overwrite:
        return reduced_distance_matrix_path
    
    
    # 3)  Load the participant distance matrix (rbf) 
    M = np.load(row.M_path)
    
    # 4) Reduce the distance matrix according to the mapping and an aggregation function (min in practice)
    M_reduced = agglomerate_distance_matrix(M, mapping_prototypes_reduction, distance_aggregate_col=distance_aggregate_col,  return_clusterdf=False)
    
    # 5) Save the reduced distance matrix
    np.save(reduced_distance_matrix_path, M_reduced)
    print(f"Saved reduced distance matrix for {row.participant_id} at {reduced_distance_matrix_path}")
        
    # 6) Update the dataframe with the reduced distance matrix path    
    return reduced_distance_matrix_path

def main(config_name='SymbolicInferenceConfig', annotator_id='samperochon', round_number=0, input_space='K_space', df=None, mapping_prototypes_reduction=None, overwrite=False, verbose=True):
    """Compute the distance matrix for the dataset based on the prototypes mapping from source to meta-prototypes.
    
    Based on the state dataframe produced by the `utils_dataset.py`.
    """
    
    print('#############################')
    green('Transform distance matrix according ot the prototypes spaces mapping')
    green(config_name)
    print('#############################')

    # 0) Get the experiment configuration
    config =  import_config(config_name)
    qualification_mapping = fetch_qualification_mapping(verbose=False)

    experiment_id = config.experiment_id
    task_names = config.dataset_params['task_names']
    modality = config.dataset_params['modality']
    cpts_config_name = config.cpts_config_name
    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    normalization = clustering_config.model_params['normalization']
    cluster_types = clustering_config.model_params['cluster_types'] 
    kernel_name = config.model_params['kernel_name']
    inference_method = config.model_params['inference_method']
    
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'figures-distance-matrix-reduction'); os.makedirs(output_folder, exist_ok=True)    
    experiments_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    

    # 1) Get data
    if df is None:
        print(f'Loading dataframe')
        df = get_experiments_dataframe(experiment_config_name=config_name, annotator_id=annotator_id, round_number=round_number, return_symbolization=False)
    
    info = f'{config_name}\nN_s = {df.participant_id.nunique()},  N={int(df.N.sum())}, Coverage={len(np.unique(np.hstack(df.embedding_labels)))}'
    
    
    # 2) Get mapping across source and meta prototypes % TODO/ check if symbolic or sometime clustering config ? 
    if mapping_prototypes_reduction is None:
        mapping_prototypes_reduction = get_prototypes_mapping(config_name, annotator_id=annotator_id, round_number=round_number, input_space=input_space, is_foreground=True, return_cluster_types=False, verbose=False)
    # 3) Compute reduced distance matrix, save the result and prvide the path 
    df[f'M_reduced_{input_space}_path'] = df.apply(lambda row: main_reduction_row(row, symbolization_config_name=config_name, mapping_prototypes_reduction=mapping_prototypes_reduction, annotator_id=annotator_id, round_number=round_number, input_space=input_space, overwrite=overwrite, verbose=verbose), axis=1)

    # 4) Visulization
    if verbose:
        distance_aggregate_cols = ['meta_cluster_min_dist','meta_cluster_maxmin_dist'] #  'meta_cluster_std_dist', 'meta_cluster_min_dist', 'meta_cluster_max_dist'
        plot_distance_aggregations(df, mapping_prototypes_reduction, distance_aggregate_cols=distance_aggregate_cols, figsize=(35, 25), n_subj_rows=3, crop_x=None)
        
    M_global = np.concatenate([np.load(path) for path in df[f'M_reduced_{input_space}_path'].to_list()], axis=0)
    filename = os.path.join(experiments_folder, "config.json"); config.to_json(filename)
    pd.DataFrame(qualification_mapping).to_csv(os.path.join(output_folder, 'qualification_mapping.csv'), index=False)
            
    fi(20, 4);_ = plt.hist(M_global.flatten(), bins='fd');plt.title(f'Reduced distance matrix with kernel={kernel_name}\n{info}')
    plt.axvline(np.mean(M_global.flatten()), color='red', linestyle='dashed', linewidth=1, label='mean')
    plt.axvline(np.median(M_global.flatten()), color='blue', linestyle='dashed', linewidth=1, label='median');plt.legend()
    figpath = os.path.join(output_folder, 'global_reduced_distance_matrix_hist.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
    
    fi(20, 5);_ = plt.hist(np.median(M_global, axis=0), bins='fd');plt.title(f'Median distance across samples (NxG={M_global.shape}\n{info}')
    figpath = os.path.join(output_folder, f'median_reduced_distances_across_participants_{kernel_name}.png')    
    plt.savefig(figpath, dpi=80, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()
        
    fi(20, 15);plt.imshow(M_global[:100000, :].T, aspect='auto', cmap='coolwarm', interpolation='nearest');plt.title(f'Global distance matrix (NxG={M_global[:500000, :].shape})\n{info}')
    plt.colorbar();figpath = os.path.join(output_folder, 'global_distance_matrix_hist.png')    
    plt.savefig(figpath, dpi=100, bbox_inches='tight')
    
    if verbose:
        plt.show()
    else:
        plt.close()    
    green(f'Saved figures in folder {output_folder}. Finished inference of pariwise distance matrix reduction')
    return df 

def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    print('Distances matrices computation for the dataset based on the prototypes mapping from source to meta: {}'.format(args.config_name))
    main(config_name=args.config_name)
    t1 = time.time()
    print(f'Time elapsed: {t1 - t0} seconds')
    
    sys.exit(0)
    