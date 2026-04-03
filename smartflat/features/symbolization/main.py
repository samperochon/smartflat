"""Orchestrates the full recursive prototyping and symbolization pipeline.

Implements the P-round recursive prototyping loop (Ch. 5, Section 5.2):
C=100 candidate centroids per round via cosine k-means, followed by
human annotation (discriminative step), residual set construction
(d_min=0.3 threshold), and prototype accumulation across P rounds.

Also performs symbolization (Ch. 5, Section 5.5): assigning each temporal
segment to its nearest prototype to produce symbolic sequences, with
optional morphological filtering for smoothing.

Prerequisites:
    - Embedding dataset loaded via ``smartflat.datasets.loader``.
    - Change points computed via ``smartflat.engine.change_point_detection``.

Main entry points:
    - ``define_symbolization()``: Full symbolization for a given config.
    - ``build_clusterdf()``: Construct cluster-level summary DataFrame.
    - ``filter_prototypes_per_prevalence()``: Remove low-prevalence prototypes.
    - ``perform_edges_detection()``: Morphological filtering of segment labels.

Notebooks:
    - ``demo_recursive_procedure.ipynb``: Shows the multi-round loop.
    - ``demo_symbolization_gold.ipynb``: Full gold pipeline.
"""

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

import seaborn as sns
from IPython.display import display
from matplotlib.colors import ListedColormap

from smartflat.configs.loader import get_complete_configs, import_config
from smartflat.features.symbolization.co_clustering import clustering_prototypes_space
from smartflat.features.symbolization.inference_reduction_prototypes_distances import (
    main as main_prototypes_reduction,
)
from smartflat.features.symbolization.main_prototypes_annotation import main
from smartflat.features.symbolization.utils import (
    compute_threshold_per_cluster_index,
    get_prototypes,
    update_segmentation_from_embedding_labels,
)
from smartflat.features.symbolization.utils_dataset import (
    build_symbdf,
    get_experiments_dataframe,
)
from smartflat.features.symbolization.visualization import (
    plot_distances_distribution_subplots,
    plot_labels_distributions_subplots,
)
from smartflat.utils.utils_coding import blue, green, red
from smartflat.utils.utils_dataset import collapse_cluster_stats, compute_matrix_stats
from smartflat.utils.utils_io import (
    fetch_qualification_mapping,
    get_data_root,
    get_video_loader,
    load_df,
    save_df,
)
from smartflat.utils.utils_visualization import get_base_colors, plot_chronogames


def build_clusterdf(df, clustering_config_name=None, embeddings_label_col='opt_embedding_labels', segments_labels_col='opt_segments_labels', segments_length_col='segments_length', qualification_mapping=None, annotator_id='samperochon', round_number=0, filter_cluster_index_outside_qm=False, verbose=False):
    
    if clustering_config_name is None:
        assert df.clustering_config_name.nunique() == 1
        clustering_config_name = df.clustering_config_name.unique()[0]
        print(clustering_config_name)
    
    if qualification_mapping is None:
        
        qualification_mapping = fetch_qualification_mapping(verbose=False)
    
    if clustering_config_name in qualification_mapping.keys() and annotator_id in qualification_mapping[clustering_config_name].keys() and f'round_{round_number}' in qualification_mapping[clustering_config_name][annotator_id].keys():
        qm_reverse = {cluster_index: cluster_type for cluster_type, cluster_index_list in qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}'].items() for cluster_index in cluster_index_list }

            #raise ValueError(f"Qualification mapping for {clustering_config_name} and annotator {annotator_id} not found for round {round_number}. Available keys: {qualification_mapping.keys()}")
    elif f'{clustering_config_name}_reduced' in qualification_mapping.keys() and annotator_id in qualification_mapping[f'{clustering_config_name}_reduced'].keys() and f'round_{round_number}' in qualification_mapping[f'{clustering_config_name}_reduced'][annotator_id].keys():
        
        print('Using reduced qualification mapping')
        qm_reverse = {cluster_index: cluster_type for cluster_type, cluster_index_list in qualification_mapping[f'{clustering_config_name}_reduced'][annotator_id][f'round_{round_number}'].items() for cluster_index in cluster_index_list }

    else:
        qm_reverse = {cluster_index: 'Unknown' for cluster_index in df[embeddings_label_col].explode().unique()}
        
    segments_label_df = pd.DataFrame([
        {"cluster_index": k, "n_appearance": v, "participant_id": row["participant_id"], "group": row["group"]}
        for _, row in df.iterrows()
        for k, v in Counter(row[segments_labels_col]).items()
    ])
    if verbose:
        green('segments_label_df:')
        display(segments_label_df.head(2))
    
    embeddings_label_df = pd.DataFrame({
        'cluster_index': np.hstack(df[embeddings_label_col]),
        'cluster_dist': np.hstack(df['cluster_dist']),
        'participant_id': np.hstack([[row['participant_id']] * len(row[embeddings_label_col]) for _, row in df.iterrows()])
    })
    if verbose:
        green('embeddings_label_df:')
        display(embeddings_label_df.head(2))

    prevalence_df = (
        segments_label_df.groupby('cluster_index')['participant_id'].nunique().div(segments_label_df['participant_id'].nunique())
        .reset_index(name='sbj_prevalence')
        .merge(
            embeddings_label_df.groupby('cluster_index')['participant_id'].count()
            .div(embeddings_label_df.shape[0])
            .reset_index(name='embedding_prevalence'),
            on='cluster_index'
        )
    )
    if verbose:
        green('prevalence_df:')
        display(prevalence_df.head(2))
    
    
    clusterdf = prevalence_df.merge(
        segments_label_df.groupby('cluster_index')['n_appearance'].mean().reset_index().rename(columns={'n_appearance': 'n_appearance_mean'}),
        on='cluster_index'
    ).merge(
        segments_label_df.groupby('cluster_index')['n_appearance'].std().reset_index().rename(columns={'n_appearance': 'n_appearance_std'}),
        on='cluster_index'
    ).merge(
        embeddings_label_df.groupby('cluster_index')['cluster_dist'].median().reset_index().rename(columns={'cluster_dist': 'cluster_dist_median'}),
        on='cluster_index'
    ).merge(
        segments_label_df.groupby('cluster_index')['participant_id'].nunique().reset_index().rename(columns={'participant_id': 'n_subjects'}),
        on='cluster_index'
    ).merge(
        segments_label_df.groupby('cluster_index')['n_appearance'].apply('sum').to_frame().reset_index(),
        on='cluster_index'
    ).merge(
        embeddings_label_df.groupby('cluster_index')['cluster_dist'].mean().reset_index().rename(columns={'cluster_dist': 'cluster_dist_mean'}),
        on='cluster_index'
    ).merge(
        embeddings_label_df.groupby('cluster_index')['cluster_dist'].std().reset_index().rename(columns={'cluster_dist': 'cluster_dist_std'}),
        on='cluster_index'
    ).merge(
        embeddings_label_df.groupby('cluster_index')['cluster_dist'].sum().reset_index().rename(columns={'cluster_dist': 'cluster_dist_sum'}),
        on='cluster_index'
    )
    #df['cluster_lengths'] = collapse_cluster_stats(df, labels_col=segments_labels_col, per_segment_feature='segments_length')
    
    df[f'{segments_length_col}_s'] = df.apply(lambda x: 60 * x[segments_length_col] / x['N']* x['duration'], axis=1)
    df['cluster_lengths'] = collapse_cluster_stats(df, labels_col=segments_labels_col, per_segment_feature=f'{segments_length_col}_s')
    
    long_df = pd.DataFrame([
        {"cluster_index": cluster_index, "cluster_lengths": np.mean(value_list), "participant_id": row["participant_id"], "group": row["group"], "pathologie": row["pathologie"]}
        for _, row in df.iterrows() if isinstance(row['cluster_lengths'], dict)
        for cluster_index, value_list in row['cluster_lengths'].items()
    ])

    
    clusterdf = clusterdf.merge(
        long_df.groupby('cluster_index')['cluster_lengths'].mean().reset_index().rename(columns={'cluster_lengths': 'cluster_length_mean'}),
        on='cluster_index'
    )
    clusterdf = clusterdf.merge(
        long_df.groupby('cluster_index')['cluster_lengths'].std().reset_index().rename(columns={'cluster_lengths': 'cluster_length_std'}),
        on='cluster_index'
    )
    
    clusterdf = clusterdf.merge(
        embeddings_label_df.groupby('cluster_index')['participant_id'].count().reset_index().rename(columns={'participant_id': 'embedding_count'}),
        on='cluster_index'
    )
    if verbose:
        green('clusterdf:')
        display(clusterdf.head(2))

    clusterdf['normalized_embedding_prevalence'] = clusterdf['embedding_count'] / clusterdf['embedding_count'].sum()

    ssd_tot = clusterdf.cluster_dist_sum.sum()
    clusterdf['cluster_dist_sum_pn'] = clusterdf.apply(lambda x: x.cluster_dist_sum / ssd_tot / x.normalized_embedding_prevalence, axis=1)
    
    
    if clustering_config_name in qualification_mapping.keys():
        clusterdf['cluster_type'] = clusterdf.cluster_index.map(qm_reverse)

        if filter_cluster_index_outside_qm:
            n = len(clusterdf)
            clusterdf.dropna(subset=['cluster_type'], inplace=True)
            #print(f"Removed {n - len(clusterdf)} clusters index as outside the scope of {clustering_config_name} (CT={clusterdf.cluster_type.nunique()})."
    else:
        clusterdf['cluster_type'] = 'Unknown' 
        
    clusterdf.loc[clusterdf['cluster_index'] == -1, 'cluster_type'] = 'Noise'
    clusterdf.loc[clusterdf['cluster_index'] == -2, 'cluster_type'] = 'Edge'
    
    if verbose:
        pass#display(clusterdf.groupby(['cluster_type'])[['n_subjects', 'n_appearance', 'normalized_embedding_prevalence', 'cluster_dist_mean', 'cluster_dist_median','n_source_prototypes', 'cluster_length_mean']].agg(['count', 'mean', 'std', 'min', 'max']))

    return clusterdf


def format_info_cluster(row):
    return (
        f"Cluster {row.cluster_index}  (d_avg={row.cluster_dist_mean:.2f} SSDn={row.cluster_dist_sum_pn:.2f}):\n"
        f"N={row.n_subjects} sbjs ({row.sbj_prevalence*100:.0f}%)   N={int(row.embedding_count)} ({row.normalized_embedding_prevalence*100:.1f}%)\n"
        f"N_segm/session = {row.n_appearance_mean:.1f} +/- {row.n_appearance_std:.1f}   T = {row.cluster_length_mean:.1f}s +/- {row.cluster_length_std:.1f} s"
    )
    
def filter_prototypes_per_prevalence(clusterdf, min_subjects=10, min_occurences=10,title='',  verbose=True):
    
    if verbose:
        print(f'Gathered cluster df with K={clusterdf.shape[0]}')
        display(clusterdf.head(3))
        
    exluded_clusters = clusterdf[(clusterdf['n_subjects'] < min_subjects) | (clusterdf['n_appearance'] < min_occurences)]['cluster_index'].values

    print(f'Excluded clusters: N={len(exluded_clusters)}')
    if verbose:
        plt.figure(figsize=(20, 4))
        sns.scatterplot(
            data=clusterdf, 
            x='n_subjects', 
            y='n_appearance', 
            hue='cluster_index', 
            palette='tab20', 
            legend=False
        )
        plt.axvline(min_subjects, linestyle='--', color='r', label='Min Subjects threshold')
        plt.axhline(min_occurences, linestyle='--', color='b', label='Min appearance threshold')
        plt.title(f"Cluster prevalence across participants and number of appearance (Kout={len(exluded_clusters)})\n{title}")
        plt.xlabel("Prevalence across participants")
        plt.ylabel("Number of segments occurences")
        plt.legend()
        plt.show()
        plt.figure(figsize=(20, 4))
        sns.scatterplot(
            data=clusterdf, 
            x='n_subjects', 
            y='n_appearance', 
            hue='cluster_index', 
            palette='tab20', 
            legend=False
        )
        plt.ylim([0, 100]); plt.xlim([0, 40])
        plt.axvline(min_subjects, linestyle='--', color='r', label='Min Subjects threshold')
        plt.axhline(min_occurences, linestyle='--', color='b', label='Min appearance threshold')
        plt.title(f"Cluster prevalence across participants and number of appearance (Kout={len(exluded_clusters)})\n{title}")
        plt.xlabel("Prevalence across participants")
        plt.ylabel("Number of segments occurences")
        plt.legend()
        plt.show()

        # Pairplot for relationships
        # sns.pairplot(clusterdf[['n_subjects', 'n_appearance', 'cluster_length_mean', 'cluster_dist_median']], diag_kind='kde')
        # plt.show()

    return list(exluded_clusters)

def add_qualification_mapping_for_reduced_space(annotator_id, round_number, reduced_clustering_config_name=['ClusteringDeploymentKmeansCombinationInferenceI5Config'], qualification_mapping=None):
    raise DeprecationWarning("This function is deprecated and will be removed in future versions. Use the new clustering_prototypes_space function instead.")
    for clustering_config_name in reduced_clustering_config_name:
        _, _, _, _, qualification_mapping = clustering_prototypes_space(symbolization_config_name=clustering_config_name, 
                                                                                    annotator_id=annotator_id,
                                                                                    round_number=round_number, 
                                                                                    model_names=['hierarchical-clustering'], 
                                                                                    verbose=False, 
                                                                                    do_save=False, 
                                                                                    overwrite=False,
                                                                                    qualification_mapping=qualification_mapping)     

    return qualification_mapping


def process_symb_labels(row, embedding_labels_col='symb_labels', min_valid_duration=60, opening_size=60, closing_size=2, verbose=False, figsize=(35, 8)):
    """
    Applies morphological opening and closing to clean temporal label sequences,
    finds start/end of valid regions, and optionally visualizes results including final chronograms.
    """
    start_time = time.time()

    N = row['N']
    duration = row['duration']
    fps = row['fps']
    stride = duration / N
    
    labels = np.array(row[embedding_labels_col])
    valid_mask = labels != -2

    cleaned = binary_opening(valid_mask, structure=np.ones(opening_size))
    cleaned = binary_closing(cleaned, structure=np.ones(closing_size))

    labeled_arr, n = label(cleaned)
    slices = [np.where(labeled_arr == i)[0] for i in range(1, n+1)]
    long_segments = [s for s in slices if  int(len(s) * row['fps'] / row['delta_t_eff'])  >= min_valid_duration]
                
    if not long_segments:
        print("All segments are shorter than the minimum valid length.")
        return [], (0, 0), (0, len(labels))  # all cropped

    start = long_segments[0][0]
    end = long_segments[-1][-1] + 1  # make it exclusive
    
    # Compute seconds for final segment bounds
    start_min =  start * row['duration'] / row['N']
    end_min =  end * row['duration'] / row['N']

    crop_start = start
    crop_end = N - end
    pad_secs = 5  # seconds
    pad_embeddings = int(pad_secs * row['fps'] / row['delta_t_eff'])

    # Left edge
    left_idx = crop_start
    left_time = round(left_idx * stride, 2)
    left_x0 = max(0, left_idx - pad_embeddings)
    left_t0 = max(0.0, left_time - pad_secs)
    left_x1 = min(N, left_idx + pad_embeddings)
    left_t1 = min(duration, left_time + pad_secs)

    # Right edge
    right_idx = N - crop_end
    right_time = round(right_idx * stride, 2)
    right_t0 = max(0.0, right_time - pad_secs)
    right_x0 = max(0.0, right_idx - pad_embeddings)
    right_x1 = min(N, right_idx + pad_embeddings)
    right_t1 = min(duration, right_time + pad_secs)
    
    # Output signal
    info_title = f'{row.participant_id}\nDuration={duration:.2f} minutes, N={N}, support K={len(np.unique(labels))} prototypes'
    info_results = f"Test minutes bounds: {start, end} --> {start_min:.2f} to {end_min:.2f} minutes"
    processed = labels[start:end]

    if verbose:
        fig, ax = plt.subplots(5, 1, figsize=figsize, sharex=False)
        fig.suptitle(f"{info_title}\n{info_results}", fontsize=16, weight='bold')
        # --- Compute consistent color map ---
        unique_labels = np.sort(np.unique(labels[labels != -1]))
        n_labels = len(unique_labels)
        color_space = np.sort(np.unique(np.append(unique_labels, -1)))

        colors = get_base_colors(n_colors=n_labels + 1, verbose=False)
        color_mapping = {label: colors[i] for i, label in enumerate(color_space)}
        color_mapping[-1] = (0, 0, 0, 1)  # noise = black

        cmap = ListedColormap([color_mapping[label] for label in color_space])
        norm = BoundaryNorm(np.append(color_space, color_space[-1] + 1), cmap.N, clip=False)

        # 1. Original Labels
        ax[0].imshow([labels], cmap=cmap, norm=norm, aspect="auto")
        ax[0].set_title("Original Labels")

        # 2. Initial Valid Mask
        ax[1].imshow([valid_mask], cmap="gray", aspect="auto")
        ax[1].set_title("Initial Valid Mask")

        # 3. Cleaned Valid Mask with Start/End Bounds
        ax[2].imshow([cleaned], cmap="gray", aspect="auto")
        ax[2].axvline(start, color='r', linestyle='--')
        ax[2].axvline(end, color='g', linestyle='--')
        ax[2].set_title(f"Cleaned Valid Mask with Start/End Bounds")
        
        # 4. Final Cleaned Sequence (with Edges in Black)
        full_cleaned = np.full_like(labels, fill_value=-1)
        full_cleaned[start:end] = processed
        ax[3].imshow([full_cleaned], cmap=cmap, norm=norm, aspect='auto')
        ax[3].axvline(start, color='r', linestyle='--')
        ax[3].axvline(end, color='g', linestyle='--')
        ax[3].set_title(f"Final Cleaned Sequence (Edges in Black)\n{info_results}")

        # 5. Final Cropped Signal Only
        ax[4].imshow([processed], cmap=cmap, norm=norm, aspect='auto')
        ax[4].set_title(f"Final Cropped Signal Only\n{info_title}")

        plt.tight_layout()
        plt.show()

    # Print everything
    elapsed = time.time() - start_time

    return processed.tolist(), (start, end), (start, len(labels) - end), (left_time, right_time), (left_x0, left_x1), (right_x0, right_x1)
      
def perform_edges_detection(df=None, symbolization_config_name='SymbolicInferenceConfig', opening_size=None, closing_size=None, verbose=False, n_sbj=5):
    """
    Applies morphological opening and closing to clean temporal label sequences,
    finds start/end of valid regions, and optionally visualizes results including final chronograms.
    """
    
    def create_mask(label_array, start, end, value=-2):
        """
        Create a mask for the label array to indicate valid regions.
        """
        mask = [value] * np.ones_like(label_array, dtype=int)
        mask[start:end] = 1
        return mask

    # Load the experiment configuration
    config = import_config(symbolization_config_name)
    if opening_size is None:
        opening_size = config.edges_opening_size
    
    if closing_size is None:
        closing_size = config.edges_closing_size
    
    if df is None:
            # Get the dataframe with the symbolization
        df = get_experiments_dataframe(experiment_config_name=symbolization_config_name,  return_symbolization=True) 

    df['opening_size'] = df.apply(lambda row: int(opening_size * row['fps'] / row['delta_t_eff'] ), axis=1)
    df['closing_size'] = df.apply(lambda row: int(closing_size * row['fps']/ row['delta_t_eff']), axis=1)
    print(f' Average opening size: {df["opening_size"].mean()} embedding and closing size: {df["closing_size"].mean()}')

    # Process
    df['processed_segm_embedding_symb_labels'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels', opening_size=row['opening_size'],closing_size=row['closing_size'])[0],axis=1)
    df['test_bounds'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels',  opening_size=row['opening_size'],closing_size=row['closing_size'])[1],axis=1)
    df['edges_durations'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels', opening_size=row['opening_size'], closing_size=row['closing_size'])[2],axis=1)
    df['test_bounds_timestamps'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels', opening_size=row['opening_size'], closing_size=row['closing_size'])[3],axis=1)
    df['left_bounds_embeddings'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels', opening_size=row['opening_size'], closing_size=row['closing_size'])[4],axis=1)
    df['right_bounds_embeddings'] = df.apply(lambda row: process_symb_labels(row, embedding_labels_col='segm_embedding_symb_labels', opening_size=row['opening_size'], closing_size=row['closing_size'])[5],axis=1)
    df['mask_test'] = df.apply(lambda x: create_mask(x['segm_embedding_symb_labels'], x['test_bounds'][0], x['test_bounds'][1]), axis=1)
    df['processed_N'] = df.processed_segm_embedding_symb_labels.apply(len) 

    # Propagate results to embedding-level variables
    
    df['processed_cluster_dist'] = df.apply(lambda row: row.cluster_dist[row.test_bounds[0]: row.test_bounds[1]], axis=1)
    df['processed_raw_embedding_labels'] = df.apply(lambda row: row.raw_embedding_labels[row.test_bounds[0]: row.test_bounds[1]], axis=1)
    df['processed_embedding_labels'] = df.apply(lambda row: row.embedding_labels[row.test_bounds[0]: row.test_bounds[1]], axis=1)
    df['processed_opt_embedding_labels'] = df.apply(lambda row: row.opt_embedding_labels[row.test_bounds[0]: row.test_bounds[1]], axis=1)
    
    # Propagate results to segments-level variables
    
    # temporal segmentation
    df['processed_cpts_0'] = df.apply(lambda x: [0] + [cpt - x.test_bounds[0] for cpt in x.cpts_0 if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] + [x.processed_N], axis=1)
    df['processed_cpts_1'] = df.apply(lambda x: [0] + [cpt - x.test_bounds[0] for cpt in x.cpts_1 if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] + [x.processed_N], axis=1)
    df['processed_opt_cpts'] = df.apply(lambda x: [0] + [cpt - x.test_bounds[0] for cpt in x.opt_cpts if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] + [x.processed_N], axis=1)
    df['processed_symb_cpts'] = df.apply(lambda x: [0] + [cpt - x.test_bounds[0] for cpt in x.symb_cpts if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] + [x.processed_N], axis=1)
    df['processed_raw_cpts'] = df.apply(lambda x: [0] + [cpt - x.test_bounds[0] for cpt in x.raw_cpts if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] + [x.processed_N], axis=1)

    df['n_processed_cpts_0'] = df['processed_cpts_0'].apply(len)
    df['n_processed_cpts_1'] = df['processed_cpts_1'].apply(len)
    df['n_processed_cpts_raw'] = df['processed_raw_cpts'].apply(len)
    df['n_processed_opt_cpts'] = df['processed_opt_cpts'].apply(len)
    df['n_processed_symb_cpts'] = df['processed_symb_cpts'].apply(len)
    
    # Segments attributes
    df['processed_segments_length'] = df['processed_cpts_0'].apply(np.ediff1d)
    df['processed_opt_segments_length'] = df['processed_opt_cpts'].apply(np.ediff1d)
    df['processed_symb_segments_length'] = df['processed_symb_cpts'].apply(np.ediff1d)
    
    
    # Segments labels 
    df['processed_segments_labels'] = df.apply(lambda x:  [x['segments_labels'][i] for i, cpt in enumerate(x.cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)
    df['processed_opt_segments_labels'] = df.apply(lambda x:  [x['opt_segments_labels'][i] for i, cpt in enumerate(x.opt_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)
    df['processed_symb_segments_labels'] = df.apply(lambda x:  [x['symb_segments_labels'][i] for i, cpt in enumerate(x.symb_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)

    # Propagate results intermediate experimental variables at different steps

    df['processed_n_embed_changes'] = df.apply(lambda x: np.sum(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.embedding_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_percent_embed_changes'] = df.apply(lambda x: np.mean(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.embedding_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_sum_cpts_withdrawn'] = df.apply(lambda x: np.sum([x['n_segmented'][i] for i, cpt in enumerate(x.cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ]) , axis=1)
    df['processed_percent_cpts_withdrawn'] = df.apply(lambda x: x.processed_sum_cpts_withdrawn / x.n_processed_cpts_raw , axis=1)
    df['processed_n_segmented'] =  df.apply(lambda x:  [x['n_segmented'][i] for i, cpt in enumerate(x.cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)

    df['processed_opt_n_embed_changes'] = df.apply(lambda x: np.sum(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.opt_embedding_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_opt_percent_embed_changes'] = df.apply(lambda x: np.mean(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.opt_embedding_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_opt_sum_cpts_withdrawn'] = df.apply(lambda x: np.sum([x['opt_n_segmented'][i] for i, cpt in enumerate(x.opt_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ]) , axis=1)
    df['processed_opt_percent_cpts_withdrawn'] = df.apply(lambda x: x.processed_opt_sum_cpts_withdrawn / x.n_processed_cpts_raw , axis=1)
    df['processed_opt_n_segmented'] =  df.apply(lambda x:  [x['opt_n_segmented'][i] for i, cpt in enumerate(x.opt_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)


    df['processed_symb_n_embed_changes'] = df.apply(lambda x: np.sum(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.symb_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_symb_percent_embed_changes'] = df.apply(lambda x: np.mean(x.raw_embedding_labels[x.test_bounds[0]:x.test_bounds[1]] != x.symb_labels[x.test_bounds[0]:x.test_bounds[1]]), axis=1)
    df['processed_symb_sum_cpts_withdrawn'] = df.apply(lambda x: np.sum([x['symb_n_segmented'][i] for i, cpt in enumerate(x.symb_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ]) , axis=1)
    df['processed_symb_percent_cpts_withdrawn'] = df.apply(lambda x: x.processed_symb_sum_cpts_withdrawn / x.n_processed_cpts_raw , axis=1)
    df['processed_symb_n_segmented'] =  df.apply(lambda x:  [x['symb_n_segmented'][i] for i, cpt in enumerate(x.symb_cpts[1:]) if ((cpt > x.test_bounds[0]) and (cpt < x.test_bounds[1]) ) ] , axis=1)
    
    # 6) Record some variables of interest and propagate changes 
    df['processed_K_raw'] = df['processed_raw_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['processed_K_segm'] = df['processed_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['processed_G_opt'] = df['processed_opt_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['processed_G_segm'] = df['processed_segm_embedding_symb_labels'].apply(lambda labels: len(np.unique(labels)))
    df['processed_G_symb'] = df['processed_symb_labels'].apply(lambda labels: len(np.unique(labels)))

    
    # 7) Visualization of the results
    start_durations = df.apply(lambda x: x['edges_durations'][0] / x['N'] * x['duration'] , axis=1)
    end_durations = df.apply(lambda x: x['edges_durations'][1] / x['N'] * x['duration'] , axis=1)
    total_duration = df['duration'].sum()
    if verbose: 

        for i in range(n_sbj):  
            row = df.sample(1).iloc[0]
            title_info = f"Participant ID: {row['participant_id']}, Duration: {row['duration']:.2f} minutes, N: {row['N']}, Support K: {len(np.unique(row['symb_labels']))} prototypes"
            blue(f'Process {title_info}')
            process_symb_labels(row, embedding_labels_col='symb_labels', opening_size=row['opening_size'], closing_size=row['closing_size'], verbose=True)
        
        display(df[['participant_id', 'mask_test', 'symb_labels', 'test_bounds', 'edges_durations', 'test_bounds_timestamps', 'left_bounds_embeddings', 'right_bounds_embeddings']].head(5))
        
        plt.figure(figsize=(10, 2))
        _, bins, _ = plt.hist(start_durations, bins=50, alpha=0.6, label='Start Crop Duration', color='tomato')
        plt.hist(end_durations, bins=bins, alpha=0.6, label='End Crop Duration', color='steelblue')
        plt.xlabel('Cropped edges durations [minutes]')
        plt.ylabel('Number of administrations')
        plt.title('Distribution of cropped edges [minutes] (starting and ending edges)')
        plt.legend();plt.tight_layout(); plt.show()

    total_crop_start = start_durations.sum(); total_crop_end = end_durations.sum(); total_duration_hr = total_duration / 60
    dataset_summary_report = (
        f"📊 Total edges cropped — START: {total_crop_start / 60:.2f}h "
        f"({100 * total_crop_start / total_duration:.2f}%), "
        f"END: {total_crop_end / 60:.2f}h "
        f"({100 * total_crop_end / total_duration:.2f}%) "
        f"of total {total_duration_hr:.2f}h"
    )
    return df, dataset_summary_report

def create_symbolic_representation(row, embedding_labels_col='embedding_labels', cluster_type_col='opt_cluster_type', exluded_clusters=[], thresholds_mapping=None, radius=None, verbose=True):
    
    labels = row[embedding_labels_col]
    # source_labels = row['raw_embedding_labels']
    cluster_distance = row['cluster_dist']
    cluster_types = row[cluster_type_col]
    
    if not len(cluster_distance) == len(cluster_types):
        red(f'/!\ lengths are unequal:, {len(labels)}, {len(cluster_distance)}, {len(cluster_types)}\n{row.identifier}')
        return np.array(labels)

    total_labels = len(labels)
    excluded_count = 0
    distance_excluded_count = 0

    symbolic_representation = []
    # for l_x, l_p, d_x in zip(source_labels, labels, cluster_distance):
    for l_p, d_x in zip(labels, cluster_distance):

        if radius is not None:
            bound = radius
            
        elif thresholds_mapping is not None:
            bound = thresholds_mapping[l_p]
                
        if l_p in exluded_clusters:
            symbolic_representation.append(-1) # EDGE
            excluded_count += 1
            
        elif d_x >= bound:
            symbolic_representation.append(-1) # NOISE
            distance_excluded_count += 1
        else:
            symbolic_representation.append(l_p)
    
    # Compute proportions
    excluded_ratio = excluded_count / total_labels if total_labels > 0 else 0
    distance_excluded_ratio = distance_excluded_count / total_labels if total_labels > 0 else 0
    total_excluded_ratio = (excluded_count + distance_excluded_count) / total_labels if total_labels > 0 else 0

    # Print details for debugging
    # print(f"Row index {row.name}: N={total_labels}, "
    #       f"Prototypes excluded K_out={excluded_count} ({excluded_ratio:.2%}), "
    #       f"Further distance={distance_excluded_count} ({distance_excluded_ratio:.2%}), "
    #       f"Total proportion of exluded sample={total_excluded_ratio:.2%}")
    if verbose:
        pass
        # print(f"Row index {row.name:5d}: "
        # f"N={total_labels:<6d} "
        # f"Exluded prototypes (K_out={len(exluded_clusters)}): {excluded_ratio:6.2%} "
        # f"N residus=:{distance_excluded_count:<6d}: {distance_excluded_ratio:6.2%} " # "Latent distance residual set:
        # #f"Temporal distance residual set: Not implemented TODO"
        # f"Total proportion of excluded sample={total_excluded_ratio:6.2%}")
    return np.array(symbolic_representation)

def get_threshold_mapping(df, config_name, label_col='G_opt_embedding_labels', annotator_id='samperochon', round_number=0, n_sigma=None, overwrite_distance_thresholds=False, verbose=True, alpha=1.0):

    config = import_config(config_name)
    if config.task_type == 'symbolization':
        clustering_config_name = config.clustering_config_name    
    else:
        clustering_config_name = config_name
        
    clustering_config = import_config(clustering_config_name)
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    cl_output_folder = os.path.join(get_data_root(), 'outputs', clustering_config.experiment_name, clustering_config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(experiment_folder, exist_ok=True)    
    if n_sigma is None:
        n_sigma = config.n_sigma
    radius_symbolization = config.radius_symbolization
        
    if n_sigma is not None:
        if alpha == 1: 
            thresholds_path1 = os.path.join(output_folder, f'{config_name}_threshold_mapping_{label_col}_{n_sigma}_sigma.pkl')
            thresholds_path2 = os.path.join(cl_output_folder, f'{clustering_config_name}_threshold_mapping_{label_col}_{n_sigma}_sigma.pkl')
        else:
            thresholds_path1 = os.path.join(output_folder, f'{config_name}_threshold_mapping_{label_col}_{n_sigma}_sigma_alpha_{alpha}.pkl')
            thresholds_path2 = os.path.join(cl_output_folder, f'{clustering_config_name}_threshold_mapping_{label_col}_{n_sigma}_sigma_{alpha}.pkl')

        if os.path.isfile(thresholds_path1) and not overwrite_distance_thresholds:
            thresholds_mapping = load_df(thresholds_path1)
        else:
            thresholds_mapping = compute_threshold_per_cluster_index(df, n_sigma=n_sigma, source_embedding_labels_col=label_col, thresholds_path=[thresholds_path1, thresholds_path2], verbose=verbose)

        print(f'Final thresholds mapping for {label_col}: (K={len(thresholds_mapping)}): {str(thresholds_mapping)[:35]}')
        
        if verbose:
            plt.figure(figsize=(10, 2))
            plt.title(f'Threshold Mapping for {label_col}')
            plt.xlabel('Cluster Index')
            plt.ylabel('Distance Threshold')
            plt.hist(list(thresholds_mapping.values()), bins=30, alpha=0.7)
            plt.grid()
            plt.show()
    else:
        print(f'No thresholds mapping provided, using default value of {radius_symbolization} for all clusters')
        thresholds_mapping = None
        
    return thresholds_mapping


def smooth_segments(segments, cpts, min_length=5):
    """
    Generalized smoothing: merges consecutive identical segments and removes any
    short segments flanked by identical neighbors.

    Rules applied in a single pass:
      - Any value that is flanked by identical values (before/after) can be "removed"
        if its length >= min_length
      - Consecutive identical segments are merged
      - Returns:
          smoothed_segments, smoothed_cpts, interruptions, smoothed_frames
    """
    cleaned = []
    cleaned_cpts = []
    interruptions = []

    i = 0
    while i < len(segments):
        label = segments[i]
        cpt = cpts[i]
        # length of this segment
        seg_len = cpts[i+1] - cpts[i] if i < len(cpts)-1 else 0

        # check if flanked by same label
        if i > 0 and i < len(segments) - 1 and segments[i-1] == segments[i+1]:
            if seg_len <= min_length:
                # increment interruption counter of previous segment
                interruptions[-1] += 1
                i += 1
                continue

        # merge consecutive identical labels
        if cleaned and cleaned[-1] == label:
            i += 1
            continue

        cleaned.append(label)
        cleaned_cpts.append(cpt)
        interruptions.append(0)
        i += 1

    # make sure last cpt is included
    if cpts[-1] not in cleaned_cpts:
        cleaned_cpts.append(cpts[-1])

    # reconstruct frame-level embeddings
    smoothed_frames = []
    for start, end, label in zip(cleaned_cpts[:-1], cleaned_cpts[1:], cleaned):
        if end > start:  # avoid 0 or negative intervals
            smoothed_frames.extend([label] * (end - start))
        

    return np.array(cleaned), np.array(cleaned_cpts), np.array(interruptions), np.array(smoothed_frames)


def define_symbolization(symbolization_config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', do_save=True, alpha=None, overwrite_inference=False, reset_init_build=True, do_compute_clustering=False, overwrite_distance_thresholds=False, parse_annotations=False, clustering_input_spaces=[], overwrite=False,  annotator_id='samperochon', round_number=0, verbose=True):
    """
    Perform symbolization (registration) of the data using clustering and prototype-based methods.
    
    1) Get 'utils_dataset.py' state dataframe with embedding_labels and temporal segmentation
    2) Get the final meta-prototypes prototype vectors (V_k Voronoi regions associated with prototypes $c_k \subset [1, K]])
        (i) Define the radius thresholds for each cluster as mean(dist_cluser_i) + n_sigma * std(dist_cluster_i)
        (ii) Perform clustering on prototypes space to get the labels mapping and create optimal symbolization
            - This step may involve using different clustering models (e.g., hierarchical, spectral, GMM)
            - The clustering results are saved in a DataFrame
            - The best clustering results are selected based on silhouette score and other metrics
            - The aggregation of clusters into meaningful reliable syllabes is performed using co-clustering between the prototypes embedding space (mu_i) and the temporal kernel desity estimation (KDE) of the prototypes temporal space (tau_i)
            
    3) Create symbolic representation of the data using the optimal labels and thresholds
    4) Save the results to a specified output folder.
    
    TODO: just assign the input_space to the clusterdf to make it evolving throughout the code 
    """
    config = import_config(symbolization_config_name)

    if symbolization_config_name == 'SymbolicSourceInferenceCompleteGoldConfig':
        input_clustering_config_names, round_numbers, qualification_cluster_types = get_complete_configs(round_number=round_number)
        config.model_params['input_clustering_config_names'] = input_clustering_config_names
        config.model_params['round_numbers'] = round_numbers
        config.model_params['qualification_cluster_types'] = qualification_cluster_types
        green(f'Creating Symbolic Configs: {symbolization_config_name} outputs - with Rounds: {round_numbers}')
        
    elif symbolization_config_name in ['SymbolicSourceInferenceGoldConfig', 'SymbolicSourcePrototypesTSInferenceGoldConfig']:
        input_clustering_config_names, round_numbers, qualification_cluster_types = get_complete_configs(round_number=round_number, inference_mode=True)

        config.model_params['input_clustering_config_names'] = input_clustering_config_names
        config.model_params['round_numbers'] = round_numbers
        config.model_params['qualification_cluster_types'] = qualification_cluster_types
        
    config.model_params['annotator_id'] = annotator_id
    config.model_params['round_number'] = round_number
            
    
    task_name = config.task_name
    modality = config.modality
    task_type = config.task_type
    
    if config.task_type == 'symbolization':
        clustering_config_name = config.clustering_config_name    
    else:
        clustering_config_name = symbolization_config_name
        
    clustering_config = import_config(clustering_config_name)
    cluster_types = clustering_config.model_params['cluster_types'] 
    clustering_model_names = config.clustering_model_names
    radius_symbolization = config.radius_symbolization
    min_subjects = config.min_subjects
    min_occurences = config.min_occurences
    n_sigma = config.n_sigma
    perform_prototypes_reduction = config.perform_prototypes_reduction

    
    output_folder = os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    cl_output_folder = os.path.join(get_data_root(), 'outputs', clustering_config.experiment_name, clustering_config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(output_folder, exist_ok=True)    
    experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}'); os.makedirs(experiment_folder, exist_ok=True)    
    qualification_mapping = fetch_qualification_mapping(verbose=False)
    #qualification_mapping[symbolization_config_name][annotator_id][f'round_{round_number}']['foreground'] = qualification_mapping[symbolization_config_name][annotator_id][f'round_{round_number}']['task-definitive'] + qualification_mapping[symbolization_config_name][annotator_id][f'round_{round_number}']['exo-definitive']
    #qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['foreground'] = qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['task-definitive'] + qualification_mapping[clustering_config_name][annotator_id][f'round_{round_number}']['exo-definitive']

    if len(qualification_mapping) > 0:
        try:
            display(pd.DataFrame(qualification_mapping)[clustering_config_name].apply(lambda x: len(x) if isinstance(x, list) else np.nan).to_frame().fillna(0).T)
        except:
            print(f'Qualification mapping for {clustering_config_name} is not available')
    # Part 1: Dataset and model preparation
    

    # 1) Get dataset with original labels and temporal segmentation and distances computed
    if reset_init_build:
        df = build_symbdf(symbolization_config_name, 
                          annotator_id=annotator_id,
                            round_number=round_number,
                           overwrite_inference=overwrite_inference,
                            overwrite=overwrite, 
                            do_filter=True, 
                            has_video=False, 
                            parse_annotations=False, 
                            add_gaze=False, 
                            add_low_dimension_projection=False, 
                            max_identifier=None)
        
    else:
        
        df = get_experiments_dataframe(experiment_config_name=symbolization_config_name, annotator_id=annotator_id, round_number=round_number, return_symbolization=False) 
   
    if df is None:
        
        return None, None, None, None, None, None, None, None, None
    
    green('###############################################################')
    green(f'experiment_folder={experiment_folder}\noutput_folder={output_folder}\ncl_output_folder={cl_output_folder}\n with clustering {clustering_config_name} and task type {task_type}')
    green('###############################################################')
    
    
    df['normalized_timestamps'] = df.apply(lambda row: np.arange(row.N) / row['N'], axis=1)
    df['clustering_config_name'] = clustering_config_name
    df['symbolization_config_name'] = symbolization_config_name
    df['task_type'] = task_type
    
    clusterdf = build_clusterdf(df, clustering_config_name=symbolization_config_name, embeddings_label_col='raw_embedding_labels', segments_labels_col='segments_labels', segments_length_col='segments_length', annotator_id=annotator_id, round_number=round_number)
    
    if 'cluster_type' not in df.columns:
        cluster_type_mapping = clusterdf.set_index('cluster_index')['cluster_type'].to_dict()
        df['cluster_type'] = df['raw_embedding_labels'].apply(lambda labels: [cluster_type_mapping[l] if l in cluster_type_mapping.keys() else 'Noise' for l in labels])

    threshold_mapping = get_threshold_mapping(df, symbolization_config_name, label_col='raw_embedding_labels', annotator_id=annotator_id, round_number=round_number, overwrite_distance_thresholds=overwrite_distance_thresholds, verbose=True)
    # 3) Create the symbolization by excluding (i) noise and prevalence clusters, (ii) mis-fitted participant's sample compared to the prototypes (d>mean+2*sigma) and propagate to segments levels
    df['filtered_raw_embedding_labels'] = df.apply(lambda row: create_symbolic_representation(row, 
                                                                            embedding_labels_col='raw_embedding_labels', 
                                                                            cluster_type_col='cluster_type',
                                                                            exluded_clusters=[], 
                                                                            radius=radius_symbolization if radius_symbolization is not None else None,
                                                                            thresholds_mapping=threshold_mapping,
                                                                            verbose=verbose), axis=1)

    
    if verbose:
        plot_labels_distributions_subplots(df, [symbolization_config_name], labels_col='embedding_labels', clusterdf=clusterdf)

    # 2) Get prototypes vectors
    prototypes = {}
    for cluster_type in cluster_types:
        prototypes[cluster_type] = get_prototypes(clustering_config_name, cluster_types=[cluster_type], annotator_id=annotator_id, round_number=round_number, verbose=False)
    
    prototypes['all'] = get_prototypes(clustering_config_name, cluster_types=cluster_types, annotator_id=annotator_id, round_number=round_number, verbose=False)
    if verbose:
        compute_matrix_stats(prototypes['all'], title='All prototypes')
        #compute_matrix_stats(prototypes['task-definitive'], title='Task prototypes')
        #compute_matrix_stats(prototypes['exo-definitive'], title='Exo prototypes')
        #compute_matrix_stats(prototypes['Noise'], title='Noise prototypes')
    
    K = prototypes['all'].shape[0]
    labels = np.hstack(df['embedding_labels'])
    
    # 3) Perform clustering on prototypes space to get the labels mapping and create optimal symbolization
    if 'K_space' in clustering_input_spaces:
        
        results, best_df, mapping_prototypes_reduction_K_space, mapping_index_cluster_type = clustering_prototypes_space(symbolization_config_name=symbolization_config_name, 
                                                                                                                                        annotator_id=annotator_id,
                                                                                                                                        input_space='K_space',     
                                                                                                                                        round_number=round_number,
                                                                                                                                        model_names=clustering_model_names,
                                                                                                                                        overwrite=overwrite,
                                                                                                                                        verbose=verbose)

        green(f'Using clustering input space K_space with {len(mapping_prototypes_reduction_K_space)} prototypes\n{str(mapping_prototypes_reduction_K_space)[:100]}...')
        df['opt_embedding_labels'] = df['raw_embedding_labels'].apply(lambda labels: [mapping_prototypes_reduction_K_space[l] if l in mapping_prototypes_reduction_K_space.keys() else -1 for l in labels ])        
        df['opt_cluster_type'] = df['opt_embedding_labels'].apply(lambda labels: [mapping_index_cluster_type[l] if l in mapping_prototypes_reduction_K_space.keys() else 'Noise' for l in labels])
        
        #return df, mapping_prototypes_reduction_K_space
        if perform_prototypes_reduction:
            print('Performing prototypes space distances reduction for K_space ')
            df = main_prototypes_reduction(config_name=symbolization_config_name, df=df, annotator_id=annotator_id, round_number=round_number, input_space='K_space', mapping_prototypes_reduction=mapping_prototypes_reduction_K_space, verbose=verbose)
            print('Done!\n')
            
    else:
        df['opt_embedding_labels'] = df['raw_embedding_labels'].copy()
        #qualification_mapping = {symbolization_config_name: {'Unknown': np.unique(np.hstack(df['opt_embedding_labels']))}}
        qualification_mapping = fetch_qualification_mapping(verbose=False)
        mapping_index_cluster_type = {l: 'Unknown' for l in np.unique(np.hstack(df['opt_embedding_labels']))}
        mapping_index_cluster_type[-1] = 'Noise'
    
        mapping_prototypes_reduction_K_space = {l: l for l in np.unique(np.hstack(df['opt_embedding_labels']))}
        df['opt_cluster_type'] = df['opt_embedding_labels'].apply(lambda labels: [mapping_index_cluster_type[l] for l in labels])

        best_df = None
        results = None
        
    col_names_update = ['segm_opt_embedding_labels', 'opt_segments_labels',
                    'opt_cpts', 'n_cpts_opt', 'opt_segments_has_separations', 'opt_n_segmented', 'opt_segments_length',
                    'opt_n_embed_changes', 'opt_percent_embed_changes',
                    'opt_sum_cpts_withdrawn', 'opt_percent_cpts_withdrawn']
    

    #print('Labels:', sorted(np.unique(np.hstack(df['opt_embedding_labels']))))
    #print(df['opt_embedding_labels'].apply(lambda x: len(np.unique(x))))
    df[col_names_update] = df.apply(update_segmentation_from_embedding_labels, axis=1, 
                                    temporal_segmentation_col='raw_cpts', 
                                    embedding_labels_col='opt_embedding_labels', 
                                    combine_func_name='majority_voting_inner', 
                                    filter_noise_labels=False, verbose=verbose)

    
    
    df['opt_p_residuals'] = df['opt_embedding_labels'].apply(lambda x: np.sum([1 for i in x if i == -1]) / len(x) )


    # 4) Perform clustering on [G] space to get the labels mapping and create optimal symbolization
    if 'G_space' in clustering_input_spaces:
        
        results, best_df_G, mapping_prototypes_reduction_G_space, mapping_index_cluster_type = clustering_prototypes_space(symbolization_config_name=symbolization_config_name, 
                                                                                                                                        annotator_id=annotator_id,
                                                                                                                                        round_number=round_number,
                                                                                                                                        input_space='G_space',
                                                                                                                                        model_names=clustering_model_names,
                                                                                                                                        alpha=alpha, 
                                                                                                                                        overwrite=False,
                                                                                                                                        verbose=False)
        best_df = pd.concat([best_df, best_df_G], axis=0, ignore_index=True)
        mapping_prototypes_reduction_G_space = {i_K: mapping_prototypes_reduction_G_space[i_G] for i_K, i_G in mapping_prototypes_reduction_K_space.items() if i_G in mapping_prototypes_reduction_G_space.keys()}

        mapping_index_cluster_type[-1] = 'Noise'
        df['G_opt_embedding_labels'] = df['raw_embedding_labels'].apply(lambda labels: [mapping_prototypes_reduction_G_space[l] if l in mapping_prototypes_reduction_G_space.keys() else -1 for l in labels ])        
        df['G_opt_cluster_type'] = df['G_opt_embedding_labels'].apply(lambda labels: [mapping_index_cluster_type[l] if l in mapping_prototypes_reduction_G_space.keys() else 'Noise' for l in labels])

        if perform_prototypes_reduction:
            print('Performing prototypes space distances reduction for K_space ')
            df = main_prototypes_reduction(config_name=symbolization_config_name, df=df, annotator_id=annotator_id, round_number=round_number, input_space='G_space', mapping_prototypes_reduction=mapping_prototypes_reduction_G_space, verbose=verbose)
            print('Done!\n')

        
    else:
        df['G_opt_embedding_labels'] = df['opt_embedding_labels'].copy()
        #qualification_mapping = {symbolization_config_name: {'Unknown': np.unique(np.hstack(df['opt_embedding_labels']))}}
        #qualification_mapping = fetch_qualification_mapping(verbose=False)
        mapping_index_cluster_type = {l: 'Unknown' for l in np.unique(np.hstack(df['G_opt_embedding_labels']))}
        mapping_index_cluster_type[-1] = 'Noise'
    
        #mapping_prototypes_reduction_G_space = {l: l for l in np.unique(np.hstack(df['G_opt_embedding_labels']))}
        df['G_opt_cluster_type'] = df['G_opt_embedding_labels'].apply(lambda labels: [mapping_index_cluster_type[l] for l in labels])

        #best_df = None
        #results = None
        
    col_names_update = ['segm_G_opt_embedding_labels', 'G_opt_segments_labels',
                    'G_opt_cpts', 'n_cpts_G_opt', 'G_opt_segments_has_separations', 'G_opt_n_segmented', 'G_opt_segments_length',
                    'G_opt_n_embed_changes', 'G_opt_percent_embed_changes',
                    'G_opt_sum_cpts_withdrawn', 'G_opt_percent_cpts_withdrawn']
    

    #print('Labels:', sorted(np.unique(np.hstack(df['opt_embedding_labels']))))
    #print(df['opt_embedding_labels'].apply(lambda x: len(np.unique(x))))
    df[col_names_update] = df.apply(update_segmentation_from_embedding_labels, axis=1, 
                                    temporal_segmentation_col='raw_cpts', 
                                    embedding_labels_col='G_opt_embedding_labels', 
                                    combine_func_name='majority_voting_inner', 
                                    filter_noise_labels=False, verbose=verbose)

    
    
    df['G_opt_p_residuals'] = df['G_opt_embedding_labels'].apply(lambda x: np.sum([1 for i in x if i == -1]) / len(x) )
    #return df, best_df, results, mapping_prototypes_reduction_K_space, mapping_index_cluster_type, qualification_mapping

    labels = (Counter(np.hstack(df['G_opt_embedding_labels'])))
    info = f'{symbolization_config_name}\nN_s = {df.participant_id.nunique()},  N={int(df.N.sum())}, K={K}, Coverage={len(labels)} N_segments = {df.n_cpts_opt.sum()} p_residu={df.opt_p_residuals.mean():.2%}'
    
    green('--------------------------------------------------------------')
    green('1) Projection to the optimal clustering subspaces for each source prototypes')
    green(f'{info}')
    green('--------------------------------------------------------------')
    green('All prototypes statistics (task+exo+noise definitive):'); compute_matrix_stats(prototypes['all'], title='Source prototypes')
    # Part 2: Symbolization with (i) cluster prevalence+noise labelization 
    
    # 1) Filter prototypes per prevalence and noise clusters
    clusterdf = build_clusterdf(df, 
                                annotator_id=annotator_id, 
                                round_number=round_number,
                                embeddings_label_col='G_opt_embedding_labels', 
                                segments_labels_col='G_opt_segments_labels', 
                                segments_length_col='G_opt_segments_length', 
                                qualification_mapping=qualification_mapping)    

    opt_mapping_prototypes_df =  pd.DataFrame(mapping_prototypes_reduction_G_space, index=['projected_cluster_index']).transpose().reset_index(names=['original_cluster_index'])
    mapping_prototypes_reduction_G_space_list = opt_mapping_prototypes_df.groupby(['projected_cluster_index']).original_cluster_index.agg(list).to_dict()#.apply(len)#.hist(bins=10)
    mapping_prototypes_reduction_G_space_list[-1] = []
    clusterdf['source_index'] = clusterdf.cluster_index.map(mapping_prototypes_reduction_G_space_list); clusterdf['n_source_prototypes'] = clusterdf['source_index'].apply(len)


    print('Start exluding clusters n_clusterdf: K_init=', len(clusterdf))
    exluded_clusters = filter_prototypes_per_prevalence(clusterdf, min_subjects=min_subjects, min_occurences=min_occurences, verbose=verbose, title=info)
    
    
    # if perform_prototypes_reduction:
    #     exluded_clusters.extend(qualification_mapping[f'{clustering_config_name}_reduced']['Noise'])
    #     n = len(exluded_clusters)
    #     print(f'Number of prevalence-excluded prototypes: {n} Additional Noise (meta-)clusters: {len(exluded_clusters) - n}')
    # else:
    #     exluded_clusters = []

    if verbose:
        plot_labels_distributions_subplots(df, [symbolization_config_name], labels_col='opt_embedding_labels', clusterdf=clusterdf)

    # 2) Define the radius thresholds for each cluster as mean(dist_cluser_i) + n_sigma * std(dist_cluster_i)
    if 'G_space' in clustering_input_spaces:
        threshold_mapping = get_threshold_mapping(df, symbolization_config_name, label_col='G_opt_embedding_labels', alpha=alpha, annotator_id=annotator_id, round_number=round_number, overwrite_distance_thresholds=overwrite_distance_thresholds, verbose=True)
    elif 'K_space' in clustering_input_spaces:
        threshold_mapping = get_threshold_mapping(df, symbolization_config_name, label_col='opt_embedding_labels', alpha=alpha, annotator_id=annotator_id, round_number=round_number, overwrite_distance_thresholds=overwrite_distance_thresholds, verbose=True)
    else:
        threshold_mapping = get_threshold_mapping(df, symbolization_config_name, label_col='raw_embedding_labels', alpha=alpha, annotator_id=annotator_id, round_number=round_number, overwrite_distance_thresholds=overwrite_distance_thresholds, verbose=True)
    # 3) Create the symbolization by excluding (i) noise and prevalence clusters, (ii) mis-fitted participant's sample compared to the prototypes (d>mean+2*sigma) and propagate to segments levels
    df['symb_labels'] = df.apply(lambda row: create_symbolic_representation(row, 
                                                                            embedding_labels_col='G_opt_embedding_labels', 
                                                                            exluded_clusters=exluded_clusters, 
                                                                            radius=radius_symbolization if radius_symbolization is not None else None,
                                                                            thresholds_mapping=threshold_mapping,
                                                                            verbose=verbose), axis=1)
   
    df['symb_cluster_type'] = df['symb_labels'].apply(lambda labels: [mapping_index_cluster_type[l] for l in labels])

    col_names_update = ['segm_embedding_symb_labels', 'symb_segments_labels',
                      'symb_cpts', 'n_cpts_symb', 'symb_segments_has_separations', 'symb_n_segmented', 'symb_segments_length',
                      'symb_n_embed_changes', 'symb_percent_embed_changes',
                      'symb_sum_cpts_withdrawn', 'symb_percent_cpts_withdrawn']
    
    df[col_names_update] = df.apply(update_segmentation_from_embedding_labels, axis=1, 
                                    temporal_segmentation_col='raw_cpts', 
                                    embedding_labels_col='symb_labels', 
                                    combine_func_name='majority_voting_inner', 
                                    filter_noise_labels=False, 
                                    verbose=verbose)
    
    df[['symb_segments_labels', 'symb_cpts','symb_interruptions','segm_embedding_symb_labels']] = df.apply(lambda x: pd.Series(smooth_segments(x['symb_segments_labels'], x['symb_cpts'], min_length=4)), axis=1)
    df['symb_segments_length'] = df['symb_cpts'].apply(np.ediff1d)
    #print('Labels:', sorted(np.unique(np.hstack(df['symb_segments_labels']))))
    # 4) Update meta-prototypes information using updated segments definition and -1 labels  
    clusterdf = build_clusterdf(df, annotator_id=annotator_id, round_number=round_number, embeddings_label_col='segm_embedding_symb_labels', segments_labels_col='symb_segments_labels', segments_length_col='symb_segments_length', qualification_mapping=qualification_mapping)
    
    opt_mapping_prototypes_df =  pd.DataFrame(mapping_prototypes_reduction_G_space, index=['projected_cluster_index']).transpose().reset_index(names=['original_cluster_index'])
    mapping_prototypes_reduction_G_space_list = opt_mapping_prototypes_df.groupby(['projected_cluster_index']).original_cluster_index.agg(list).to_dict()#.apply(len)#.hist(bins=10)
    mapping_prototypes_reduction_G_space_list[-1] = exluded_clusters
    # mapping_prototypes_reduction_G_space_list[-1] = []
    clusterdf['source_index'] = clusterdf.cluster_index.map(mapping_prototypes_reduction_G_space_list)
    clusterdf['n_source_prototypes'] = clusterdf['source_index'].apply(len)
    
    
    # # 5) Apply edges masking and morphological operations (openings and closing) to perform the estimation of the task bounds
    # if 'test_bounds' not in df.columns:
    #     print('Performing edges detection')
    #     df, dataset_summary_report = perform_edges_detection(df, symbolization_config_name, verbose=verbose)
    # else:
    #     print('Edges detection already performed, skipping this step')
    #     dataset_summary_report = ''

    # 6) TODO: Post-processing variables computation: should be in analysis otebooks only ? 
    df['K'] = prototypes['all'].shape[0]
    df['K_raw'] = df['raw_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['K_segm'] = df['embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['G_opt'] = df['opt_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['G_G_opt'] = df['G_opt_embedding_labels'].apply(lambda labels: len(np.unique(labels)))
    df['G_symb'] = df['symb_labels'].apply(lambda labels: len(np.unique(labels)))
    df['G_segm'] = df['segm_embedding_symb_labels'].apply(lambda labels: len(np.unique(labels)))
    df['symb_p_residuals'] = df['symb_labels'].apply(lambda x: np.sum([1 for i in x if i in [-1, -2]]) / len(x) )
    df['symb_p_noise'] = df['symb_labels'].apply(lambda x: np.sum([1 for i in x if i == -1]) / len(x) )
    #df['symb_p_edges'] = df['symb_labels'].apply(lambda x: np.sum([1 for i in x if i == -2]) / len(x) )
    
    # 6) Save results 
    if do_save:
        results_path = os.path.join(output_folder, f'{symbolization_config_name}_symbolization_registration_dataframe.pkl')
        
        if not os.path.exists(results_path) or overwrite:
            
            save_df(df, results_path)
            green(f'Saved df to {results_path}')
            results_path = os.path.join(experiment_folder, f'{symbolization_config_name}_symbolization_registration_dataframe.pkl')
            save_df(df, results_path)
            green(f'Saved df to {results_path}')

            results_path = os.path.join(output_folder, f'{symbolization_config_name}_clusterdf.pkl')
            save_df(clusterdf, results_path)
            green(f'Saved clcusterdf to {results_path}')
            results_path = os.path.join(experiment_folder, f'{symbolization_config_name}_clusterdf.pkl')
            save_df(clusterdf, results_path)
            green(f'Saved clusterdf to {results_path}')
            
    # 7) Visualization
    labels = (Counter(np.hstack(df['symb_labels'])))
    info = f'{symbolization_config_name}\nN_s = {df.participant_id.nunique()},  N={int(df.N.sum())}, K={K}, Coverage={len(labels)} N_segments = {df.n_cpts_symb.sum()} p_residu={df.symb_p_residuals.mean():.2%}'
    green('--------------------------------------------------------------')
    green('2) Symbolization with (i) cluster prevalence+noise labelization')
    green(f'{info}')
    green('--------------------------------------------------------------')
    print('verbose:', verbose)
    if verbose:
        
        plot_labels_distributions_subplots(df, [symbolization_config_name], labels_col='symb_labels', clusterdf=clusterdf)
        plot_distances_distribution_subplots(df, clustering_config_names=[symbolization_config_name], distance_col='cluster_dist', radius_symbolization=radius_symbolization)
        
        plot_chronogames(df, labels_col="raw_embedding_labels", n_t_max=99999, time_calibration='embeddings',  figsize=(25, 12))
        plot_chronogames(df, labels_col="filtered_raw_embedding_labels", n_t_max=99999, time_calibration='embeddings',  figsize=(25, 12))
        
        
        #plot_chronogames(df, labels_col="embedding_labels", n_t_max=99999, time_calibration='embeddings',  figsize=(25, 20))
        plot_chronogames(df, labels_col="segm_embedding_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 12))
        plot_chronogames(df, labels_col="opt_embedding_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 20))
        plot_chronogames(df, labels_col="segm_opt_embedding_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 20))
        plot_chronogames(df, labels_col="G_opt_embedding_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 20))
        plot_chronogames(df, labels_col="segm_G_opt_embedding_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 20))
        
        plot_chronogames(df, labels_col="segm_embedding_symb_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='interpolation', figsize=(25, 20))
        plot_chronogames(df, labels_col="segm_embedding_symb_labels", n_t_max=99999, time_calibration='embeddings',  upsampling='padding', figsize=(25, 20))

        
        plot_chronogames(df, labels_col="symb_labels", n_t_max=99999, time_calibration='embeddings',  figsize=(25, 12))
        plot_chronogames(df, labels_col="cluster_dist", n_t_max=99999, time_calibration='embeddings',  figsize=(25, 12))

    # Compute memory usage in MB
    size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    green(f"DataFrame df memory size: {size_mb:.3f} MB")
    
    unique_clusters = sorted(np.unique(np.hstack(df['embedding_labels'])))
    print(f'embedding_labels: Unique clusters: {len(unique_clusters)}')

    unique_clusters = sorted(np.unique(np.hstack(df['opt_embedding_labels'])))
    print(f'opt_embedding_labels: Unique clusters: {len(unique_clusters)}')

    unique_clusters = sorted(np.unique(np.hstack(df['symb_labels'])))
    print(f'symb_labels: Unique clusters: {len(unique_clusters)}')
    
    
    display(clusterdf.groupby(['cluster_type'])[['n_subjects', 'n_appearance', 'normalized_embedding_prevalence', 'cluster_dist_mean', 'cluster_dist_median','n_source_prototypes', 'cluster_length_mean']].agg(['count', 'mean', 'std', 'min', 'max']))

    return df, clusterdf, prototypes, qualification_mapping, results, best_df, mapping_prototypes_reduction_K_space, mapping_prototypes_reduction_G_space, exluded_clusters, info

def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='SymbolicInferenceConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = parse_args()
    t0 = time.time()
    print('Process registration by loading vector labels and timestamps, meta-prototypes basis, radius per clusters, noise encoding,   using config_name: {}'.format(args.config_name))
    outputs = define_symbolization(symbolization_config_name=args.config_name, 
                                   overwrite_inference=False, do_save=True, reset_init_build=True, 
                                   do_compute_projection=False, do_compute_clustering=False, 
                                   overwrite_distance_thresholds=False, overwrite=True,  verbose=True)
    print(outputs[-1])
    t1 = time.time()
    print(f'Time elapsed: {t1 - t0} seconds')
    
    sys.exit(0)
    