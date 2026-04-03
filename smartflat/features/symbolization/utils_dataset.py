"""Data preparation for the symbolization pipeline.

Builds the symbolization DataFrame (``build_symbdf``) and loads experiment
DataFrames (``get_experiments_dataframe``) consumed by main, inference,
and co_clustering modules.

Prerequisites:
    - Clustering experiments completed via ``smartflat.engine.clustering``.
    - Change points computed via ``smartflat.engine.change_point_detection``.
    - Annotation data loaded via ``smartflat.annotation_smartflat``.

Key functions:
    - ``build_symbdf()``: Construct the full symbolization DataFrame with
      embeddings, labels, change points, and optionally gaze features.
    - ``get_experiments_dataframe()``: Load or create experiment DataFrame
      for a given config, annotator, and round.
"""

import argparse
import datetime
import hashlib
import os
import sys
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display



from smartflat.annotation_smartflat import add_ground_truth_labels
from smartflat.configs.loader import import_config
from smartflat.constants import available_modality, gaze_features
from smartflat.datasets.dataset_gaze import compute_segments_gaze_features, populate_gaze_data
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
from smartflat.features.symbolization.utils import (
    compute_cluster_probabilities,
    compute_sample_entropy,
    predict_segments_from_embed_labels,
    propagate_embeddings_labels_to_frames,
    propagate_segment_labels_to_embeddings,
    qc_sanity_check_filters,
    reduce_segments_labels,
    retrieve_segmentation_costs,
    update_segmentation_from_embedding_labels,
)
from smartflat.utils.utils import pairwise, upsample_sequence
from smartflat.utils.utils_coding import blue, green, red
from smartflat.utils.utils_dataset import quantize_signal, sample_uniform_rows_by_col
from smartflat.utils.utils_io import fetch_has_gaze, get_data_root, load_df, save_df
from smartflat.utils.utils_visualization import plot_chronogames


def build_symbdf(experiment_config_name, overwrite_inference=False, overwrite=False, do_filter=True, has_video=False, parse_annotations=False, add_gaze=False, add_low_dimension_projection=False, max_identifier=None, annotator_id='samperochon', round_number=0, verbose=False):
    """
    Load and process symbolization dataframes for a list of cluster folders.
    
    Parameters:
    n_cluster_folder_list (list): List of cluster folder names.
    cpts_config_name (str): Configuration name for change point detection.
    experiment_config_name (str): Configuration name for clustering.
    overwrite (bool, optional): If True, overwrite existing dataframes. Default is False.
    do_filter (bool, optional): If True, apply filtering. Default is True.
    has_video (bool, optional): If True, indicates presence of video data. Default is False.
    add_low_dimension_projection (bool, optional): If True, add low dimension projection. Default is False.
    max_identifier (int, optional): Maximum identifier value for filtering. Default is max_identifier.
    Returns:
    pd.DataFrame: Processed symbolization dataframe.
    """

    config =  import_config(experiment_config_name)
    config.dataset_params['annotator_id'] = annotator_id
    config.dataset_params['round_number'] = round_number
    
    experiment_id = config.experiment_id
    cpts_config_name = config.cpts_config_name
    extended_experiment = False
    
    print(f'Using change point detection config: {cpts_config_name}')
    
    if config.task_type == 'clustering':
        
        raise ValueError("Deprecated.")
        # outputs of build_symbdf
        clustering_config_name = experiment_config_name
        working_df_path = os.path.join(get_data_root(), 'dataframes', 'states',  '{}_{}_{}_{}.pkl'.format(experiment_id, cpts_config_name, experiment_config_name, 'light'))

    elif config.task_type == 'symbolization':

        #print('Loading symbolization dataframe')
        # Outputs of define_symbolization
        clustering_config_name = config.clustering_config_name
        output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)
        working_df_path = os.path.join(output_folder, f'{experiment_config_name}_states_dataframe.pkl')
        clusterdf_path = os.path.join(output_folder, f'{experiment_config_name}_clusterdf.pkl')

    if not overwrite and os.path.exists(working_df_path):
        print(f"Overwrite={overwrite} Loading existing dataframe from old path names {working_df_path}")
        df = load_df(working_df_path)
        return df
    
    green(f'Generating symbolization dataframe for {experiment_config_name}')
    dset = get_dataset(dataset_name=config.dataset_name, **config.dataset_params)
    dset.metadata  = load_embedding_dimensions(dset.metadata)
    dset.load_change_point_detection(config_name=cpts_config_name, annotator_id=annotator_id, round_number=round_number)
    #dset.load_optimal_lambdas(config_name=cpts_config_name, annotator_id=annotator_id, round_number=round_number)
    dset.load_gamma(config_name=cpts_config_name,  annotator_id=annotator_id, round_number=round_number)
    if overwrite_inference:
        print('Inference computation...')
        from smartflat.features.symbolization.inference import main as main_inference
        main_inference(config_name=experiment_config_name, annotator_id=annotator_id, round_number=round_number, verbose=False)
    
    dset.load_clustering_labels(config_name=experiment_config_name, annotator_id=annotator_id, round_number=round_number, add_distances=True, load_clustering=False)
            
    if parse_annotations:
        print('Start parsing the annotations...')
        dset.parse_annotations(verbose=False)            
        dset.metadata = add_ground_truth_labels(dset.metadata)
        display(dset.metadata.groupby(['task_name', 'modality', 'has_annotation', 'annotation_software']).size().to_frame().rename(columns={0: 'N'}).T)
    
        print('Size of filled category A labels: {}'.format(dset.metadata.dropna(subset=['embedding_labels_A'])[f'embedding_labels_A'].size))
    
    
    # Add ground truth labels
    # Post-processing  and filtering 
    df = dset.metadata.copy()
    df['cpts'] = df.cpts.apply(lambda x: x if x[-1] != x[-2] else x[:-1])
    # Clip last element of cpts to N  and display those for which it was different:
    #display(df[df.apply(lambda x: True if x.cpts[-1] != x.N else False, axis=1)])
    df['cpts'] = df.apply(lambda x: np.clip(x.cpts, 0, x.N), axis=1)
    

    n=len(df)
    #display( df[~df.apply(lambda x: x.embedding_labels.shape[0] - x.N, axis=1) == 0])
    #df = df[df.apply(lambda x: x.embedding_labels.shape[0] - x.N, axis=1) == 0]
    print(f"Removed {n-len(df)} rows with N != |embedding_labels| ")

    #assert df.apply(lambda x: (x.cpts[-1] - x.N) != 0, axis=1).sum() == 0
    #n=len(df)
    #display(df[df.apply(lambda x: x.cpts[-1] - x.N, axis=1) != 0])
    #df = df[df.apply(lambda x: x.cpts[-1] - x.N, axis=1) == 0]
    #print(f"Removed {n-len(df)} rows with cpts[-1] != N ")
    #red(f"/!\ Keep {n-len(df)} rows with cpts[-1] != N ")

    df = qc_sanity_check_filters(df, do_filter=do_filter, has_video=has_video, max_identifier=max_identifier, verbose=verbose)
    #df = filter_progress_cols(df, progress_cols)

    
    # 2) Reduce segments labels and cpts by removing redundant contiguous segments labels, and propagate labels (top-down approach) 
    if extended_experiment:
        # 1) Propagate labels  to segment per majority voting (bottom-up approach)
        df["segments_labels"]= df.apply(predict_segments_from_embed_labels, axis=1, temporal_segmentation_col='cpts', combine_func_name="majority_voting")
        
        df["upsampled_labels"] = df.embedding_labels.apply(upsample_sequence, args=(df.n_embedding_labels.max(),))
        df[["segments_start", "segments_end"]] = df.apply(lambda x: pd.Series([x["cpts"][:-1], x["cpts"][1:]], index=["segment_start", "segment_end"]),axis=1)

        df[['reduced_cpts', 'reduced_segments_labels', 'segments_has_separations', 'n_segmented']]  = df.apply(reduce_segments_labels, temporal_segmentation_col='cpts', segments_labels_col='segments_labels', axis=1)
        df['n_reduced_cpts'] = df['reduced_cpts'].apply(lambda x: len(x) if type(x) == np.ndarray else np.nan).to_list()
        df['reduced_segments_length'] = df['reduced_cpts'].apply(np.ediff1d)
        
        
        df['refined_reduced_embedding_labels'] = df.apply(propagate_segment_labels_to_embeddings, temporal_segmentation_col = 'reduced_cpts', segment_labels_col='reduced_segments_labels', axis=1)

        # This represents the percentage of per-sample labels that have changed after accounting for the segment labels (chnge-point detection)
        df['percent_embed_changes'] = df.apply(lambda x: np.mean(x.refined_reduced_embedding_labels != x.embedding_labels), axis=1)
        # Show the number of withdrawn segments per cluster    
        
            
        # Plots 
        
        # 1) Number of withdrawn segments per cluster
        if  df.modality.nunique() == 1:
            matrix = pd.DataFrame(df['n_segmented'].tolist()).fillna(0)

            # Calculate 5th and 95th percentiles
            vmin, vmax = np.percentile(matrix.values, [5, 95])

            # Plot with improved colormap and color bar
            plt.figure(figsize=(20, 5))
            sns.heatmap(
                matrix, 
                cmap='coolwarm', 
                vmin=vmin, 
                vmax=vmax, 
                cbar_kws={'label': 'Nombre de speratations comptée par clusters'}
            )

            plt.title("Observation du nombre de séparations observées par clusters (filtré 5th & 95th percentiles)")
            plt.xlabel("Cluster idx")
            plt.ylabel("Clusters")
            plt.show()
        
        # 2) Percent embed chamge distribution with uniformly-sampled examples.
        
        plt.figure(figsize=(10, 6))
        df['percent_embed_changes'].hist(bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlim([0, 1])
        # Ajouter un titre, une légende et des labels en français
        plt.title("Distribution of the percentage label switch after temporal regularisation using change-point detections ", fontsize=14)
        plt.xlabel("%", fontsize=12)
        plt.ylabel("F", fontsize=12)
        plt.legend(["Percentage embedding label switch"], loc="upper right")
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.show()
                
        for _, row in sample_uniform_rows_by_col(df, column='percent_embed_changes', N=3).iterrows():
            green(f'Percentage of embedding changed after refinements: {row.percent_embed_changes:.2f}')
            plot_chronogames(row, labels_col="embedding_labels", time_calibration='embedding', n_labels=20, figsize=(25, 1))
            plot_chronogames(row, labels_col="refined_reduced_embedding_labels", time_calibration='embedding', n_labels=20, figsize=(25, 1))

    else:

        # 0) Save results for comparisons and statistics computation
        col_names_update = ['segm_embedding_labels', 'segments_labels',
                        'cpts', 'n_cpts', 'segments_has_separations', 'n_segmented', 'segments_length',
                        'n_embed_changes', 'percent_embed_changes', 
                        'sum_cpts_withdrawn', 'percent_cpts_withdrawn']
        df['cpts'] = df.cpts.apply(lambda x: x if x[-1] != x[-2] else x[:-1])
        
                
        df[col_names_update] = df.apply(update_segmentation_from_embedding_labels, axis=1, 
                                        temporal_segmentation_col='cpts', 
                                        embedding_labels_col='raw_embedding_labels', 
                                        combine_func_name='majority_voting', 
                                        filter_noise_labels=False, verbose=verbose)

    # 4) Propagate labels to frames and compute agreement for overlapping portions
    if extended_experiment:
        
        df[['frame_labels_vote', 'frame_labels', 'agreement', 'n_frame_labels']]  = propagate_embeddings_labels_to_frames(df, embed_labels_col='embedding_labels', return_detailed_outputs=True)
        df[['refined_frame_labels_vote', 'refined_frame_labels', 'refined_agreement', 'refined_n_frame_labels']]  = propagate_embeddings_labels_to_frames(df, embed_labels_col='refined_reduced_embedding_labels', return_detailed_outputs=True)
        df['frame_labels']  = propagate_embeddings_labels_to_frames(df, embed_labels_col='embedding_labels', return_detailed_outputs=False)
        
    if add_low_dimension_projection:
        
        # Assuming your DataFrame is `df` with embeddings in column 'X':
        #df = add_umap_grouped(df, embed_col='X', task_col='task_name', modality_col='modality')
        df = add_pca_3d_grouped(df, embed_col='X', task_col='task_name', modality_col='modality')
        #df['umap_embed'] = add_umap(df, embed_col='X')
        #df['pca_embed'] = add_pca_3d(df, embed_col='X')

        # for covar_col in ['participant_id', 'modality']:
        #         df[f'{covar_col}_label'] = add_covar_label(df, covar_col=covar_col)
        
    if add_gaze:
        
        df['has_gaze'] = df.apply(fetch_has_gaze, axis=1, verbose=verbose)
        display(df['has_gaze'].value_counts())
        #df = df[df['has_gaze']==True]
        #df = df.iloc[:1]
        df = compute_segments_gaze_features(df, verbose=False)
        display(df.has_gaze.value_counts())
        
        # TODO: Continue here adding all energy nmovement
        df['acceleration_norm_mean'] = df['acceleration_norm_mean'].apply(lambda x: np.array(x, dtype=np.float32))
        df['acceleration_quantized'] = df['acceleration_norm_mean'].apply(lambda x: quantize_signal(x, n_bins=6) if isinstance(x, np.ndarray) else x)
        #plot_chronogames(df[df['has_gaze']==True].sort_values('duration'), labels_col="acceleration_quantized", time_calibration='embeddings',  figsize=(25, 8), upsampling='interpolation')
        
    if extended_experiment:
        
        # Load segmentation costs
        df = retrieve_segmentation_costs(df, 
                                        config_name='ChangePointDetectionDeploymentConfig', 
                                        temporal_segmentation_col='reduced_cpts', 
                                        penalty_column='lambda_0', 
                                        suffix='')

        df = retrieve_segmentation_costs(df, 
                                        config_name='ChangePointDetectionDeploymentConfig', 
                                        temporal_segmentation_col='cpts', 
                                        penalty_column='lambda_0', 
                                        suffix='_raw')

        df[["segments_start", "segments_end"]] = df.apply(lambda x: pd.Series([x['reduced_cpts'][:-1], x['reduced_cpts'][1:]], index=["0", "1"]),axis=1)
        df[["segments_start_raw", "segments_end_raw"]] = df.apply(lambda x: pd.Series([x['cpts'][:-1], x['cpts'][1:]], index=["0", "1"]),axis=1)

        # Comptute embedding-level sampling probablities
        df['p_embedding_labels'] = compute_cluster_probabilities(df, labels_col='refined_reduced_embedding_labels')

    print('---------------------------- Dataset summary statistics -----------------------------')
    green('N.  unique subjects (trigram): {}, participant_id: {}, modalities: {}, length: {}'.format(df.trigram.nunique(), df.participant_id.nunique(), df.identifier.nunique(), len(df)))
    display(df.drop_duplicates(['trigram']).groupby(['task_name'])['group'].value_counts().to_frame())
    print("Number of unique clusters for the dataset (support of the clustering): {}".format(len(np.unique(np.hstack(df['embedding_labels'])))))
    save_df(df, working_df_path)

    print(f"Saved working dataframe to {working_df_path}")
    
    return df  
    
def build_clindf(n_cluster, cpts_config_name, clustering_config_name, do_load_df=False, filter_gold=True, add_cpts_costs=False):
    
    n_cluster_folder = f'K_{n_cluster}'
    
    # 1) Load dataset with demographics and clinical variables
    input_columns = ['task_name', 'participant_id', 'modality', 'identifier',  'duration', 'fps', 'n_frames', 
                    'age', 'sex', 'group', 'pathologie']

    dset = get_dataset(dataset_name='base', verbose=False)
    df = dset.metadata[input_columns].copy()

    green(f"Summary of the df: {df.shape} - Number of unique participant_id: {df.participant_id.nunique()} - Number of unique identifier: {df.identifier.nunique()}")

    # 2) Pull regularisation estimated parameters (A, k_0, x_0, etc)
    input_columns = ['participant_id', 'identifier', 'model_name', 
                    
                    # SHOULD NOT BE LOADED: (we drop duplicates experiments to create the result file per administration to store the information. Deployment cpts and lambdas are added in datasets or second-level dataset building)
                    # 'penalty', 'cpts', 'n_cpts', 'cpts_frequency', 'penalty_n', 'log_penalty', 
                    
                    # 'L_hat^p', 'k_hat^p','x0_hat^p', 'L_var^p', 'k_var^p', 'x0_var^p', 'lambda_0_log^p','lambda_1_log^p', 'lambda_0^p', 'lambda_1^p', 
                    # 'L_hat^c', 'k_hat^c','x0_hat^c', 'L_var^c', 'k_var^c', 'x0_var^c', 'lambda_0_log^c','lambda_1_log^c', 'lambda_0^c', 'lambda_1^c', 
                    'n_frames_binned','duration_labels', 'duration_labels_counts']

    # config = import_config(cpts_config_name); results_output_path = os.path.join(get_data_root(), "outputs", config.experiment_name, "lambda_optimal.csv")
    # fitdf = pd.read_csv(results_output_path, usecols=input_columns)
    # green(f'Loaded fitdf from {results_output_path}')
    # green(f"Summary of the fitdf: {fitdf.shape} - Number of unique participant_id: {fitdf.participant_id.nunique()} - Number of unique identifier: {fitdf.identifier.nunique()}")

        
        
    # 3) Pull Temporal segmentation and clustering deployment # TODO: Add n_reduced_cpts
    gaze_features_list = [ i for k, v in gaze_features.items() for i in v]

    input_columns = ['participant_id', 'identifier', 'N', 'n_cpts', 'n_cpts_raw', 'percent_embed_changes',# 'segments_fit_cost', 'fit_cost', 'reg_cost', 'total_cost',
                    'cpts', 'embedding_labels', 'segments_labels', 'n_segments', #'p_embedding_labels', 
                    
                    'segments_length', 'n_segmented'] #+ gaze_features_list

    working_df_path = os.path.join(get_data_root(), 'dataframes', 'states',  '{}_{}_{}_{}.pkl'.format(n_cluster_folder, cpts_config_name, clustering_config_name, 'light'))
    blue(f'Loading symbolization dataset from {working_df_path}')

    tsdf = load_df(working_df_path)
    tsdf['video_representation_path'] = tsdf['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    tsdf['video_path'] = tsdf.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)


    n=len(tsdf)
    tsdf = tsdf[tsdf.segments_labels.apply(len) > 5]
    print(f'Filtered {n - len(tsdf)}/{n} rows with less than 5 segments')
    tsdf['n_segments'] = tsdf.segments_labels.apply(len)


    tsdf['video_representation_path'] = tsdf['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    tsdf['video_path'] = tsdf['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    if add_cpts_costs:
        tsdf = retrieve_segmentation_costs(tsdf, 
                                            config_name=cpts_config_name,
                                            temporal_segmentation_col='cpts', 
                                            penalty_column='lambda_0', 
                                            suffix='')
    else:
        tsdf['segments_fit_cost'] = np.nan
        tsdf['fit_cost'] = np.nan
        tsdf['reg_cost'] = np.nan
        tsdf['total_cost'] = np.nan

    tsdf = tsdf[input_columns]
    # Analysis modification
    #columns_name_mapping = {'n_cpts': 'n_cpts_raw'}
    #tsdf.rename(columns=columns_name_mapping, inplace=True)
    tsdf['n_cpts'] = tsdf['cpts'].apply(lambda x: len(x) if type(x) == np.ndarray else np.nan).to_list()
    assert False # Check below
    tsdf['cpts_percentiles'] = tsdf['cpts'] / tsdf['N']
    tsdf['cpts_percentiles'] = tsdf['cpts_percentiles'].apply(lambda x: [min(e, 1) for e in x])


    green(f"Summary of the symbolization_df: {tsdf.shape} - Number of unique participant_id: {tsdf.participant_id.nunique()} - Number of unique identifier: {tsdf.identifier.nunique()}")


    df_int = df.merge(fitdf, on=['participant_id', 'identifier'], how='right')
    df_merged = df_int.merge(tsdf, on=['participant_id', 'identifier'], how='left')


    if filter_gold:
        n = len(df_merged)
        df_merged.dropna(subset=['embedding_labels'], inplace=True)
        print(f'Filtered {n - len(df_merged)}/{n} rows without embedding_labels')
        
        n = len(df_merged)
        df_merged.dropna(subset=['n_frames'], inplace=True)
        print(f'Filtered {n - len(df_merged)}/{n} rows without n_frames')

    df_merged['fps'] = df_merged['fps'].round(1)
    df_merged['pathologie'].fillna('N.A', inplace=True)
    df_merged['task_group'] = df_merged['task_name'] + "_" + df_merged['pathologie'].astype(str)
    df_merged['sum_cpts_withdrawn'] = df_merged.n_segmented.apply(lambda x: np.sum(x))
    df_merged['has_gaze'] = df_merged.apply(fetch_has_gaze, axis=1, verbose=False)
    display(df_merged['has_gaze'].value_counts())



    # Add the reduced segments labels
    df_merged['cpts_frames'] = df_merged.apply(lambda row: [int(cpt * row.n_frames / row.N) for cpt in row.cpts], axis=1)
    df_merged['cpts_temporal'] = df_merged.apply(lambda row: [cpt / row.fps / 60 for cpt in row.cpts_frames], axis=1)
    df_merged['segments_bounds'] = df_merged.cpts_temporal.apply(lambda x: [(s, e) for s, e in pairwise(x)])

    df_merged['percent_cpts_withdrawn'] = df_merged.apply(lambda x: x.sum_cpts_withdrawn / x.n_cpts_raw, axis=1)
    df_merged['symbols_freq'] = df_merged['n_segments'] / df_merged['duration']

    # Optional custom bin edges (set to None if you want automatic binning)
    custom_bins = [0, 15, 30, 40, 120]  # Example of custom bins; replace with your own
    custom_bins = [15, 30, 40, 100]
    df_merged = df_merged[(df_merged['duration'] < 80) & (df_merged['duration'] > 15)]

    if custom_bins:
        bins_label = custom_bins
        df_merged['n_frames_binned'] = pd.cut(df_merged['duration'], bins=bins_label, labels=False)
    else:
        df_merged['n_frames_binned'], bins_label = pd.cut(df_merged['duration'], bins=5, retbins=True, labels=False)


    df_merged['duration_labels'] = df_merged.n_frames_binned.map({i: j for i, j in enumerate(['{}-{}'.format(int(s), int(e)) for s,e in pairwise(bins_label)]) })
    Ns=df_merged.drop_duplicates(subset=['identifier']).groupby(['task_name', 'modality'])['duration_labels'].value_counts()
    df_merged['duration_labels_counts'] = df_merged.apply(lambda x: f'{x.duration_labels} (N={Ns.loc[x.task_name].loc[x.modality].loc[x.duration_labels]})', axis=1)
    df_merged['duration_labels'] = pd.Categorical(df_merged['duration_labels'], categories=['15-30', '30-40', '40-100'], ordered=True)

    df_merged['cpts_frequency'] = df_merged['n_cpts'] / df_merged['duration']


    df_merged.sort_values('duration_labels', inplace=True)

    green(f"Summary of the merged df: {df_merged.shape} - Number of unique participant_id: {df_merged.participant_id.nunique()} - Number of unique identifier: {df_merged.identifier.nunique()}")

        
    return df_merged

def get_experiments_dataframe(experiment_config_name='ClusteringDeploymentKmeansConfig', annotator_id='samperochon', round_number=0, return_symbolization=False, return_data_only=True, verbose=False):
    
    config = import_config(experiment_config_name)
    experiment_id = config.experiment_id
    cpts_config_name = config.cpts_config_name
    
    if config.task_type == 'clustering':
        # outputs of build_symbdf
        raise ValueError("Look first whether these states should be annotator dependent and re-run if necessary.")
        working_df_path = os.path.join(get_data_root(), 'dataframes', 'states',  '{}_{}_{}_{}.pkl'.format(experiment_id, cpts_config_name, experiment_config_name,'light'))
        if not os.path.exists(working_df_path):
            red(f"Working dataframe not found at {working_df_path}")
            return None
        
        df = load_df(working_df_path)
        green(f"Loading: {working_df_path}")
    
    elif config.task_type == 'symbolization':
        
        #print('Loading symbolization dataframe')
        # Outputs of define_symbolization
        output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)
        clusterdf_path = os.path.join(output_folder, f'{experiment_config_name}_clusterdf.pkl')
        
        if return_symbolization:
            working_df_path = os.path.join(output_folder, f'{experiment_config_name}_symbolization_registration_dataframe.pkl')
            green(f"Loading symbolization (symb_labels) dataset: {working_df_path}")

        else:
            working_df_path = os.path.join(output_folder, f'{experiment_config_name}_states_dataframe.pkl')
            if verbose:
                green(f"Loading raw subspace projection assignement +cpts (embedding_labels) dataset: {working_df_path}")

        if not os.path.exists(working_df_path):
            red(f"Symbolization not found at {working_df_path}")
            return None
        
        df = load_df(working_df_path).sort_values('task_number_int')
        
        df.drop(columns=['symb_p_edges', 'symb_p_noise'], inplace=True, errors='ignore')
        
        # Get file metadata
        stat = os.stat(working_df_path)
        created = datetime.datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        # Compute file hash
        with open(working_df_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:8]

        if verbose:
            green(f"Loaded symbolization dataframe from {working_df_path} "
                f"(mean dist={df['cluster_dist'].apply(np.sum).sum():.2e})")
            green(f"  ↪ created: {created} | file ID: {file_hash}")
        #green(f"Loaded symbolization dataframe from {working_df_path} (mean dist={df['cluster_dist'].apply(np.mean).mean():.4f})")
        df['annotator_id'] = annotator_id
        df['round_number'] = round_number
        if verbose:
            display(df[['annotator_id', 'round_number', 'participant_id', 'task_number_int', 'split', 'raw_embedding_labels', 'cluster_dist', 'N_raw', 'test_bounds', 'N']].head(3))
            
    # Linux:
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    #df['video_path'] = df['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    # Cheetah
    df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}', 'inference')    
    df['M_path'] = df.apply(lambda x: os.path.join(experiment_folder, f'{experiment_config_name}_distance_matrix_{x.identifier}.npy'), axis=1)

    #df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    
    
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root('light')))
    df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    df['has_video'] = df.video_path.apply(lambda x: True if ((type(x) == str) and os.path.isfile(x)) else False)
    df = use_light_dataset(df, light_by_default=True)

    # for col in df.columns:
    #     if isinstance(df[col].iloc[0], np.ndarray):
    #         print(f"Column '{col}' is of dtype: {df[col].apply(lambda arr: arr.dtype).iloc[0]}")
        

    if 'symb_labels' in df.columns:
        df_new = pd.DataFrame({
            'symb_p_noise': df['symb_labels'].apply(lambda x: np.sum([i == -1 for i in x]) / len(x)),
            'symb_p_edges': df['symb_labels'].apply(lambda x: np.sum([i == -2 for i in x]) / len(x)),
        })
        df = pd.concat([df, df_new], axis=1)

    if return_data_only:
        return df
    
    # Only for symbolic tasks for now 
    if not not os.path.exists(clusterdf_path):
        red(f"Working dataframe not found at {clusterdf_path}")
        return None
    
    clusterdf = load_df(clusterdf_path)
    if verbose:
                    
        green(f"Loaded clusterdf dataframe from {clusterdf_path}")
        
    labels = (Counter(np.hstack(df['symb_labels']))); K = df['K'].mean()
    info = f'{experiment_config_name}\nN_s = {df.participant_id.nunique()},  N={int(df.N.sum())}, K={K:.2f}, Coverage={len(labels)} N_segments = {df.n_cpts_symb.sum()} p_residu={df.symb_p_residuals.mean():.2%}'
    
    if verbose:
        green('--------------------------------------------------------------')
        green('2) Symbolization with (i) cluster prevalence+noise labelization')
        green(f'{info}')
        green('--------------------------------------------------------------')
        
    

    return df, clusterdf

def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='ClusteringDeploymentKmeansConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    clustering_config_name = args.config_name
    start_time = time.time()
    df = build_symbdf(clustering_config_name, 
                      overwrite_inference=False, 
                      do_load_df=False,
                      overwrite=True, do_filter=True, has_video=False, 
                      parse_annotations=False, 
                      add_gaze=False, add_low_dimension_projection=False, max_identifier=None, return_symbolization=False)

    print(f"Number of unique participant_id: {df.participant_id.nunique()}")
    end_time = time.time()
    print(f"Function build_symbdf took {end_time - start_time:.2f} seconds")
    

    exit(0)  