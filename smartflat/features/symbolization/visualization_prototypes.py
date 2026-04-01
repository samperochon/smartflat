"""Prototype display functions: nearest/furthest frames for each cluster.

Renders grid visualizations of video frames closest to and furthest from
each prototype centroid, supporting quality assessment of the vocabulary
via the Visual Probing Mechanism (Ch. 5, Section 5.2).

Used primarily by:
    - ``demo_prototypes.ipynb``
    - ``prototypes_annotator.ipynb``

External dependencies:
    - decord (video frame extraction)
    - cv2 (image processing)
"""

import argparse
import json
import logging
import os
import random
import socket
import sys
import time
from collections import Counter, defaultdict
from tqdm import tqdm
import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde


from decord import bridge
from IPython.display import display
from sklearn.decomposition import PCA

from smartflat.configs.loader import import_config
from smartflat.constants import progress_cols
from smartflat.datasets.filter import filter_progress_cols
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import use_light_dataset
from smartflat.utils.utils import pairwise, upsample_sequence
from smartflat.utils.utils_coding import green
from smartflat.utils.utils_dataset import collapse_cluster_stats
from smartflat.utils.utils_io import (
    fetch_output_path,
    fetch_root_dir,
    get_api_root,
    get_data_root,
    get_host_name,
    get_video_loader,
    load_df,
    parse_flag,
    parse_identifier,
    save_df,
)
from smartflat.utils.utils_visualization import get_colors, plot_gram

bridge.set_bridge('native')

# Use .png for quicly visualize figures in the navigator/GUI or .svg for Keynote/papers
FIGURES_EXTENSION = '.png'
ANONYMIZE = True
import matplotlib.patheffects as pe

def plot_clusters_subjects_rows(df, config_name="ClusteringDeploymentConfig", cpts_cols='reduced_cpts_frames', labels_col='reduced_segments_labels', cluster_values=None, n_sbj = 5, n_frame_per_sbj = 5, suffix='', sampling_mode='random_within_different_segments', smallest=True, show=False, figure_folder=None, overwrite=False, verbose=False):
    """
        Here we plot a figure for each cluster represented across the `frame_labels_col` columns (arrays of different length), 
        with `n_sbj` rows and `n_frame_per_sbj` columns representing for each subejcts `n_frame_per_sbj` randomly sampled frames across the cluster index.
        
        1) Find the total number of clusters 
        
        2) For each cluster value (between 0 and P):
            i) Sample `n_subj` random participant_id and iterate over subjects
            ii) Find `n_frame_per_sbj` indexes (without replacement) of segments of the query label
            iii) Draw an index between the `cpts` of that segment to select a frame number 
    """
    
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()
    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1)
    
    # 1) Find total number of clusters and get colors
    if cluster_values is None:
        cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)

    # Combine colors from tab20, tab20b, and tab20c
    colors = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors) *20
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])

    
    if sampling_mode=='global_centroid_proximity':
            
        results = find_extreme_embeddings(df, labels_col=labels_col, n_closest=n_frame_per_sbj, n_frames_per_subject=n_frame_per_sbj, smallest=smallest)
    
    
    for cluster_value in cluster_values:
                
        df['n_cv'] = df.apply(lambda x: np.sum(x[labels_col] == cluster_value), axis=1)
        sdf = df.sort_values('n_cv', ascending=False).iloc[:n_sbj].copy()
        
        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
            if os.path.exists(figure_path) and not overwrite:
                print(f'Figure {figure_path} already exists. Skipping')
                continue


        _n_frames_per_subjects = min(df['n_cv'].max(), n_frame_per_sbj)
        fig, ax = plt.subplots(figsize=(20, n_sbj * 3))
        plt.title('Cluster value: {}'.format(cluster_value), weight='bold', fontsize=20)
        print(f'Plotting cluster {cluster_value} with {_n_frames_per_subjects} frames per subject for {n_sbj} subjects')
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        tiled_frames = np.zeros((n_sbj * L, _n_frames_per_subjects * l, C), dtype=np.uint8)
        #print(f'Final tile of shape {tiled_frames.shape}')
        n_plotted=-1
        n_tent = 0
        while (n_plotted < n_sbj - 1) and (n_tent < 10):
            if len(sdf) == 0:
                print('No more subjects to sample. Stopping')
                break
            row = sdf.iloc[0]
            sdf.drop(row.name, inplace=True)
            try:
                vr = video_loader(row.video_path)
            except:
                print(f'Error loading {row.video_path}. Continue')
                import subprocess
                subprocess.run(['open', os.path.dirname(row.video_path)])
                subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
                continue
            
        
            if sampling_mode=='random_within_different_segments':
                
                labels = row[labels_col]
                cluster_indexes = np.argwhere(labels == cluster_value).flatten()   
                
                if len(cluster_indexes) == 0:
                    #print('No cluster found. Continue')
                    n_tent+=1
                    continue

                else:
                    #print(f'Found {len(cluster_indexes)} cluster segments for {cluster_value}')         
                    n_plotted+=1                        
                        
                sampled_segments_idx = sorted(random.sample(list(cluster_indexes), min(len(cluster_indexes), _n_frames_per_subjects)))
                cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in  sampled_segments_idx]
                #sampled_frames = [int(start + (end-start) * np.random.sample()) for start, end in  cpts_tuples]
                #TODO: here try to sample the closest sample to the centroid (filter distance and label ?)
                sampled_frames = [int(start + (end-start) / 2) for start, end in  cpts_tuples]
                #sampled_frames = [min(max(0, frame_idx), len(vr) - 1) for frame_idx in sampled_frames]
                
                #sampled_frames_info = [f'min={frame_idx/row.n_frames*row.duration:.1f}\n[{cpt_t[0]/row.n_frames*row.duration:.1f}-{cpt_t[1]/row.n_frames*row.duration:.1f}]' for frame_idx, cpt_t in zip(sampled_frames, cpts_tuples)]
                sampled_frames_info = [f'{ divmod(round( frame_idx / row.n_frames * row.duration * 60), 60)[0]:d}m { divmod(round( frame_idx / row.n_frames * row.duration * 60), 60)[1]:02d}s' for frame_idx, cpt_t in zip(sampled_frames, cpts_tuples)]
                if verbose:
                    print(f'Exploring label {cluster_value}')
                    print(f'Segments labels: {labels}')
                    print(f'Cpts start-end: {cpts_tuples}')
                    print(f'Final sampled frames: {sampled_frames}')

                # Compute cluster prevalence
                prevalence_sum = np.sum([end - start for (start, end) in cpts_tuples])
                #norm = row[cpts_cols][-1]
                prevalence = prevalence_sum / 25 / 60
                plt.text(-50, n_plotted * L + L // 2, f'{row.participant_id}\n{len(cpts_tuples)} segments , total={prevalence:.2f} min', 
                        fontsize=20, ha='right', va='center', color='black')
            
            elif sampling_mode=='centroid_proximity':
                
                labels = row[labels_col]
                cluster_indexes = np.argwhere(labels == cluster_value).flatten()   
                
                if len(cluster_indexes) == 0:
                    #print('No cluster found. Continue')
                    n_tent+=1
                    continue

                else:
                    #print(f'Found {len(cluster_indexes)} cluster segments for {cluster_value}')         
                    n_plotted+=1
                        
                
                assert 'embedding_labels' in labels_col
                
                
                labels = row[labels_col]

                # Get 'cluster_dist' values and their corresponding indices
                cluster_distances = row['cluster_dist']  # Assuming it's an array-like structure
                
                if smallest:
                    # Get indices of n smallest cluster_dist values
                    sorted_indices = np.argsort(cluster_distances)
                    filtered_cluster_indexes = np.intersect1d(cluster_indexes, sorted_indices)[:n_frame_per_sbj]
                    
                else:
                    # Get indices of n largest cluster_dist values
                    sorted_indices = np.argsort(cluster_distances)
                    filtered_cluster_indexes = np.intersect1d(cluster_indexes, sorted_indices)[-n_frame_per_sbj:]

                # Select the corresponding cluster indices
                #print(f'Cluster indexes: {cluster_indexes}')
                #print(f'Sorted indices: {sorted_indices}')
                #print(f'Filtered cluster indexes: {filtered_cluster_indexes}')
                
                # Find frames for each cluster index
                sampled_frames = [int( idx / row.N * len(vr)) for idx in filtered_cluster_indexes]
                sampled_frames_info = [f't_n={(idx / row.N)*row.duration:.2f}, d={cluster_distances[idx]:.2f}' for idx in filtered_cluster_indexes]
                #print('Sampled_frames: ', sampled_frames)
            
            
            # Place each sampled frame in the array
            for i, frame_idx in enumerate(sampled_frames):
                
                try:
                    frame = np.array(vr[frame_idx].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                    
                tiled_frames[n_plotted * L:(n_plotted + 1) * L, i * l:(i + 1) * l, :] = frame
                
                # Add sampled frame number at the top-right corner
                text_x = i * l + 0.44*l     # small left margin
                text_y = n_plotted * L + 2  # Slight offset from the top
                ax.text(
                    text_x, text_y, str(sampled_frames_info[i]),
                    color="#030508", fontsize=22, ha='left', va='top',
                    fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")]
                )
                #if i == 0:
                #    ax.text(text_x-100, text_y, f'{row.participant_id}', color='black', fontsize=10, ha='right', va='top')
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    (i * l, n_plotted * L), l, L,
                    linewidth=3,
                    edgecolor=cluster_colors(cluster_value),
                    facecolor='none'
                )
                ax.add_patch(rect)
                
        # After the while loop finishes, slice to the filled part
        if n_plotted >= 0:  # at least one subject was plotted
            filled_rows = (n_plotted + 1) * L
            tiled_frames = tiled_frames[:filled_rows, :, :]
        else:
            # Handle the case where no subjects were found
            tiled_frames = np.zeros((0, _n_frames_per_subjects * l, C), dtype=np.uint8)

        print(f'Plotted {n_plotted+1} subjects for cluster {cluster_value}')
        
        
        # Plot the result
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path, bbox_inches='tight', dpi=80)
            #print(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
            

    return 


def find_extreme_embeddings(df, labels_col, n_closest=5, n_frames_per_subject=2, smallest=True, cluster_values=None, indexing='use_embedding_indexing'):
    """
    Find the n_closest or n_farthest embeddings for each cluster across all subjects, 
    with a constraint of n_frames_per_subject indices per subject.

    Args:
    - df (pd.DataFrame): DataFrame where each row represents a video/subject and stores labels, distances, etc.
    - labels_col (str): Column name containing cluster labels (array-like).
    - n_closest (int): Number of closest/farthest embeddings to find per cluster.
    - n_frames_per_subject (int): Maximum number of indices to pick per subject.
    - closest (bool): If True, find closest; if False, find farthest.

    Returns:
    - dict: Dictionary where keys are cluster indices and values are lists of tuples 
            (subject_index, embedding_index) for the selected embeddings.
    """
    df.reset_index(drop=True, inplace=True)
    if cluster_values is None:
        cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    results = {cluster: [] for cluster in cluster_values}
    
    df['N_raw'] = df['N_raw'].astype(int)
    for cluster in tqdm(cluster_values):
        cluster_candidates = []
        
        for subj_idx, row in df.iterrows():
            distances = np.array(row['cluster_dist'])
            labels = np.array(row[labels_col])
            
            if 'test_bounds' in df.columns:
                offset_embedding = row['test_bounds'][0] 
                #print(f'Subject {subj_idx} has offset_embedding {offset_embedding}')
            else:
                raise ValueError('Column "test_bounds" not found in DataFrame. Please ensure it exists.')
        
            # Validate lengths of cluster_dist and labels
            if len(distances) != len(labels):
                print(f"Warning: Mismatch in lengths for subject {subj_idx}. Skipping.")
                continue
            
            # Select embeddings belonging to the current cluster
            cluster_indices = np.where(labels == cluster)[0]
            cluster_distances = distances[cluster_indices]
            if smallest:
                # Get indices of n_frames_per_subject closest distances
                sorted_indices = cluster_indices[np.argsort(cluster_distances)[:n_frames_per_subject]]
            else:
                # Get indices of n_frames_per_subject farthest distances
                sorted_indices = cluster_indices[np.argsort(cluster_distances)[-n_frames_per_subject:]]
            
            # Collect (subject_index, embedding_index) for the current subject
            #print(f"Subject {subj_idx} has {len(sorted_indices)} embeddings in cluster {cluster}")
            cluster_candidates.extend([(subj_idx, row.identifier, offset_embedding, offset_embedding+idx, idx / row['N'], distances[idx]) for idx in sorted_indices])
            #print(f'Adding {len(sorted_indices)} candidates for subject {subj_idx} in cluster {cluster} with offset {offset_embedding} N={row.N} N_raw={row.N_raw}')
    
        # Sort all candidates by distance globally and select top n_closest
        global_distances = [df.iloc[subj_idx]['cluster_dist'][embedding_idx-offset_embedding] for subj_idx, _, offset_embedding, embedding_idx, _, _ in cluster_candidates]
        global_sorted_indices = np.argsort(global_distances) if smallest else np.argsort(global_distances)[::-1]
        
        # Track frames selected per subject
        selected_candidates = []
        subject_frame_count = {}

        for i in global_sorted_indices:
            subj_idx, identifier, offset_embedding, embedding_idx, tn, d = cluster_candidates[i]
            if subject_frame_count.get(subj_idx, 0) < n_frames_per_subject:
                
                
                if indexing == 'use_frame_indexing':
                    print('TODO: not sure of  still work (deprecated)')
                    selected_candidates.append((df.iloc[subj_idx].identifier, None, int(embedding_idx / df.iloc[subj_idx].N_raw * df.iloc[subj_idx].n_frames), tn,  d))
                elif indexing == 'use_embedding_indexing':
                    selected_candidates.append((df.iloc[subj_idx].identifier, offset_embedding, embedding_idx, tn, d))
                    
                subject_frame_count[subj_idx] = subject_frame_count.get(subj_idx, 0) + 1
            if len(selected_candidates) >= n_closest:
                break
        
        results[cluster] = selected_candidates
        #print(f'Cluster {cluster} has {len(selected_candidates)} embedding indexes.')
    
    return results


def plot_nearest_centroid_frames(df, labels_col='embedding_labels', cluster_values=None, sampled_index=0, show=True,  figure_folder=None, suffix='', n_cols=5,title='',  overwrite=False):
    
    assert df.task_name.nunique() == 1
    assert df.modality.nunique() == 1
    
    task_name, modality = df.task_name.iloc[0], df.modality.iloc[0]
    # No seize for this function 
    L, l, C = 224 // 4, 398 // 4, 3
    n_tent_max = 50
    video_loader = get_video_loader()
    if not (df.modality.nunique() == 1) and (df.task_name.nunique() == 1):
        print(df.modality.unique())
        print(df.task_name.unique())
        raise ValueError('Multiple modalities or tasks found. Limiting to the first one')
    # 1) Find total number of clusters
    if cluster_values is None:
        cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
        
    # Get unique cluster values and their count
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)

    # Combine colors from tab20, tab20b, and tab20c
    colors = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors) * 20

    # Ensure the combined colormap has enough colors for the number of unique labels
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")

    # Create a ListedColormap with the required colors
    cluster_colors = ListedColormap(colors[:n_labels])

    if df['has_video'].mean() != 1:
        print('Warning: not all subjects have videos. Limiting to subjects with videos')
        n = len(df)
        display(df[df['has_video'] == False])
        print(df[df['has_video'] == False].video_representation_path.to_list())
        df = df[df['has_video'] == True]
        print(f'-> Kept {len(df)}/{n} identifiers')

    results = find_extreme_embeddings(df, labels_col=labels_col, n_closest=500, n_frames_per_subject=1, smallest=True)

    figure_path=None

    if figure_folder is not None:
        figure_path = os.path.join(figure_folder, 'nearest_frame_'+task_name + '_' + modality  + suffix + FIGURES_EXTENSION)
        if os.path.exists(figure_path) and not overwrite:
            print(f'Figure {figure_path} already exists. Skipping')

    fig, axes = plt.subplots(n_labels // n_cols + (n_labels % n_cols > 0), n_cols, figsize=(20, n_labels // n_cols * 2 + 5))
    axes = axes.flatten()
    fig.suptitle('Closest frame to cluster centroid\n{}- {}\nK={}'.format(task_name, modality, n_labels), weight='bold', y=1.03, fontsize=20)

    for (cluster, ax) in zip(cluster_values, axes[:n_labels]):
        
        try:
            sampled_info = results[cluster]
        except:
            print(f'No sampled info for cluster {cluster}. Continue')
            ax.axis('off')
            continue
        try:
            identifier = sampled_info[sampled_index][0]
        except:
            print(f'No identifier found for cluster {cluster}. Continue')
            print('Sampled info:', sampled_info)
            ax.axis('off')
            continue
        try:
            row = df[df['identifier'] == identifier].iloc[0]
            vr = video_loader(row.video_path)
        except:
            print(f'Error loading {row.video_path}. Continue')
            continue


        embedding_idx = sampled_info[sampled_index][1]
        sampled_frames = int( embedding_idx / row.N * len(vr))
        sampled_frames_info = f'{row.participant_id}\nt_n={embedding_idx / row.N:.1f} d={row.cluster_dist[embedding_idx]:.1f} '

        # Place each sampled frame in the array
        #try:
        frame = np.array(vr[sampled_frames].asnumpy())
        #frame = cv2.resize(frame, (l, L))  # resize to the target size
        ax.imshow(frame)
        #except:
        #    ax.axis('off')
        #    continue
        
        ax.axis('off')
        ax.set_title(f'Cluster {cluster}\n{sampled_frames_info}' + title, fontsize=10)
                
    for ax in axes[n_labels:]:
        ax.axis('off')
    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path, dpi=90, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return 

import multiprocessing
from multiprocessing import Pool
from functools import partial

def process_cluster(cluster_value, df, results, n_sbj, cluster_colors, n_tent_max,
                    anonymize, figure_folder, clusterdf, prefix, suffix, show,
                    overwrite, FIGURES_EXTENSION):
    if cluster_value not in clusterdf['cluster_index'].to_list():
        print(f'Cluster {cluster_value} not found in clusterdf. Continue')
        return

    if figure_folder is not None:
        figure_path = os.path.join(
            figure_folder,
            prefix + df.task_name.iloc[0] + '_' + df.modality.iloc[0] +
            '_K_' + str(cluster_value) + suffix + FIGURES_EXTENSION
        )
        if os.path.exists(figure_path) and not overwrite:
            print(f'Figure {figure_path} already exists. Skipping')
            return

    plot_cluster_figure(
        df, results, cluster_value, n_sbj, cluster_colors, n_tent_max,
        anonymize=anonymize, figure_folder=figure_folder, clusterdf=clusterdf,
        prefix=prefix, suffix=suffix, show=show
    )





def plot_clusters_global(df, config_name="ClusteringDeploymentConfig", labels_col='reduced_segments_labels', n_sbj = 5,  prefix='', suffix='', smallest=True, show=False, figure_folder=None, anonymize=True, overwrite=False, cluster_values=None, clusterdf=None, n_cores=20, verbose=False):
    """
        Here we plot a figure for each cluster represented across the `frame_labels_col` columns (arrays of different length), 
        with `n_sbj` rows and `n_frame_per_sbj` columns representing for each subejcts `n_frame_per_sbj` randomly sampled frames across the cluster index.
        
        1) Find the total number of clusters 
        
        2) For each cluster value (between 0 and P):
            i) Sample `n_subj` random participant_id and iterate over subjects
            ii) Find `n_frame_per_sbj` indexes (without replacement) of segments of the query label
            iii) Draw an index between the `cpts` of that segment to select a frame number 
    """
    

    n_tent_max = 500
    
    if not (df.modality.nunique() == 1) and (df.task_name.nunique() == 1):
        print(df.modality.unique())
        print(df.task_name.unique())
        raise ValueError('Multiple modalities or tasks found. Limiting to the first one')
    # 1) Find total number of clusters
    
    if cluster_values is None:
        cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    # Get unique cluster values and their count
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)

    colors = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors) * 20

    # Ensure the combined colormap has enough colors for the number of unique labels
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")

    # Create a ListedColormap with the required colors
    cluster_colors = ListedColormap(colors[:n_labels])
    labels = np.hstack(df[labels_col].to_list()); count_dict = Counter(labels)
    
    if df['has_video'].mean() != 1:
        print('Warning: not all subjects have videos. Limiting to subjects with videos')
        display(df[df['has_video'] == False])
        n = len(df)
        df = df[df['has_video'] == True]
        print(f'Kept {len(df)}/{n} identifiers')
    
    results = find_extreme_embeddings(df, labels_col=labels_col, n_closest=n_sbj+5 + n_tent_max, n_frames_per_subject=3, cluster_values=cluster_values, smallest=smallest)

    if n_cores == 'all':
        n_cores =multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores")
    
    if n_cores == 1:
                    
        for cluster_value in cluster_values:

            if not cluster_value in clusterdf['cluster_index'].to_list():
                print(f'Cluster {cluster_value} not found in clusterdf. Continue')
                continue
            
            
            if figure_folder is not None:
                figure_path = os.path.join(figure_folder, prefix+df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
                if os.path.exists(figure_path) and not overwrite:
                    print(f'Figure {figure_path} already exists. Skipping')
                    continue
                
            plot_cluster_figure(df, results, cluster_value, n_sbj, cluster_colors, n_tent_max, anonymize=anonymize, figure_folder=figure_folder, clusterdf=clusterdf, prefix=prefix, suffix=suffix, show=show)
                

    else:
        worker = partial(
            process_cluster, df=df, results=results, n_sbj=n_sbj,
            cluster_colors=cluster_colors, n_tent_max=n_tent_max,
            anonymize=anonymize, figure_folder=figure_folder, clusterdf=clusterdf,
            prefix=prefix, suffix=suffix, show=show,
            overwrite=overwrite, FIGURES_EXTENSION=FIGURES_EXTENSION
        )

        with Pool(processes=n_cores) as pool:
            pool.map(worker, cluster_values)




def plot_cluster_figure(df, results, cluster_value, n_sbj, cluster_colors,  n_tent_max=500, n_col = 5, anonymize=True, figure_folder=None, clusterdf=None, prefix='', suffix='', show=False):
    
    
    def format_info_cluster(row):
        return (
            f"Cluster {row.cluster_index}  (d_avg={row.cluster_dist_mean:.2f} SSDn={row.cluster_dist_sum_pn:.2f}):\n"
            f"N={row.n_subjects} sbjs ({row.sbj_prevalence*100:.0f}%)   N={int(row.embedding_count)} ({row.normalized_embedding_prevalence*100:.1f}%)\n"
            f"N_segm/session = {row.n_appearance_mean:.1f} +/- {row.n_appearance_std:.1f}   T = {row.cluster_length_mean:.1f}s +/- {row.cluster_length_std:.1f} s"
        )
        
    L, l, C = 224 // 4, 398 // 4, 3
   
    video_loader = get_video_loader()
    fig, ax = plt.subplots(figsize=(30, 14))
    
    try:
        sampled_info = results[cluster_value]
    except:
        print(f'No info found for cluster {cluster_value}. Continue')
        print(results.keys())
        raise ValueError
    
    acc_timestamps_list = []
    list_identifier_plotted = []
    # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
    tiled_frames = np.zeros([n_sbj // n_col * L, n_col * l, C], dtype=np.uint8)

    #print(f'Final tile of shape {tiled_frames.shape}')
    #print(f'Final tile of shape {tiled_frames.shape}')
    
    n_plotted = 0
    list_identifier_plotted = []
        
    participant_to_samples = defaultdict(list)
    for _sample_info in sampled_info:
        participant_to_samples[_sample_info[0]].append(_sample_info)
        
    # Now iterate over participants instead of raw list
    for identifier, samples in participant_to_samples.items():
        if n_plotted >= n_sbj:
            break
        if identifier in list_identifier_plotted:
            print(f'Identifier {identifier} already plotted. Continue')
            continue

        # Try each candidate until one works
        success = False
        for _sample_info in samples:
            
            if n_plotted >= n_sbj:
                break
            if identifier in list_identifier_plotted:
                continue

            #try:
            row = df[df['identifier'] == identifier].iloc[0]
            vr = video_loader(row.video_path)
            #except:
            # print(f'Error loading {row.video_path}. Continue')
            # import subprocess

            # #subprocess.run(['open', os.path.dirname(row.video_path)])
            # red(f'{os.path.dirname(row.video_path)}')
            # n_tent+=1
            # #subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
            # continue

            embedding_idx = _sample_info[2]
            offset_embedding = _sample_info[1]
            normalized_timestamp = _sample_info[3]
            acc_timestamps_list.append(normalized_timestamp)
            sampled_frames = int( embedding_idx / row.N_raw * len(vr))
            minute = embedding_idx / row.N_raw * row['duration']
            
            sampled_frames_info = f'{row.participant_id}\nt_n={100*embedding_idx / row.N_raw:.1f} d={row.cluster_dist[embedding_idx-offset_embedding]:.2f} min={minute:.2f}'
        
            # Place each sampled frame in the array
            #try:
            frame = np.array(vr[sampled_frames].asnumpy())
            frame = cv2.resize(frame, (l, L))  # resize to the target size
            #except:
            #    print(f'Error loading frame {sampled_frames} of  {row.video_path}. Use black frame instead')

            #    frame = np.zeros((L, l, C), dtype=np.uint8)
                    
            
            tiled_frames[(n_plotted // n_col) * L:(n_plotted // n_col + 1) * L, (n_plotted % n_col) * l:(n_plotted % n_col + 1) * l, :] = frame
            # Add sampled frame number at the top-right corner
            text_x = n_plotted % n_col * l + l - 0  # Slight offset from the right
            text_y = (n_plotted // n_col) * L + 5  # Slight offset from the top
            if not anonymize:
                ax.text(text_x, text_y, sampled_frames_info, color='red', fontsize=12, ha='right', va='top')
            # if (n_plotted % 5) == 0:
            #     ax.text(text_x-50, text_y, f'{row.participant_id}', color='black', fontsize=20, ha='right', va='top')
            
            # Add quadrillage (cluster-specific border)
            rect = Rectangle(
                ((n_plotted % n_col) * l, (n_plotted // n_col) * L), l, L,
                linewidth=3,
                edgecolor='black',
                facecolor='none'
            )
            ax.add_patch(rect)
            success = True #TODO: handle failure case if they occur
            
        if success:
            list_identifier_plotted.append(identifier)
            n_plotted += 1

        
    print(f'Plotted {n_plotted} with plotted subjects for cluster {cluster_value}')
    t_hist, bins =  np.histogram(acc_timestamps_list, bins=np.linspace(0, 1, 11))
    cl_row = clusterdf[clusterdf['cluster_index'] == cluster_value].iloc[0]
    cl_title = format_info_cluster(cl_row)
    figure_title = f'{cl_title}\nT_bins={t_hist}'
    plt.title(figure_title, weight='bold', fontsize=20, )

    # Plot the result
    plt.imshow(tiled_frames);plt.axis('off')

    if figure_folder is not None:
        
        figure_path = os.path.join(figure_folder, prefix + df.task_name.iloc[0] + '_' + df.modality.iloc[0] + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path, bbox_inches='tight', dpi=80)
        green(f'Saved figure to {figure_path}')
        
    if show:
        plt.show()
    else:
        plt.close()
 


 
def plot_participant_clusters(df, labels_col='segments_labels', cpts_cols='cpts', n_rows=10, n_max_segments=10, max_length=500, prefix='', suffix='', n_sbj=10, verbose=False, show=True, figure_folder=None):
    

    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1) and (df.n_cluster.nunique() == 1)
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()

    n_cluster = df.n_cluster.iloc[0]
    df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                            .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))



    # 1) Find total number of clusters and get colors
    cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)
    
    colors = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors) * 20
    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])
    
    print('n_max_segments:', n_max_segments, 'max_length:', max_length, 'n_cluster:', n_cluster)
    if n_max_segments >  max_length /  n_cluster:
        n_max_segments = int(min(n_max_segments, max_length/n_cluster))
    elif n_max_segments < max_length/n_cluster:
        n_max_segments = int(max(n_max_segments, max_length/n_cluster))
    print('n_max_segments:', n_max_segments)
    df['cluster_dist_cluster'] = collapse_cluster_stats(df, labels_col='embedding_labels', per_segment_feature='cluster_dist')

    for _, row in df.sample(n_sbj).iterrows():
        
        labels = row[labels_col]
        #sbj_cluster_values = sorted(np.unique(labels)); 
        sbj_cluster_values = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False).index
        sbj_prevalence = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False)[0].astype(str).to_list()
        n_clusters = len(sbj_cluster_values)
        print(f'Subjects shows {n_clusters} clusters')

        try:
            vr = video_loader(row.video_path)
        except:
            print(f'Error loading {row.video_path}. Continue')
            #import subprocess
            #subprocess.run(['open', os.path.dirname(row.video_path)])
            #subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
            continue
        
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(f'Action segmentation prediction (per increasing prototypes occurence prevalence)\nId: {row.task_name} - {row.modality} - {row.participant_id} - K={row.n_cluster} (|S|={len(labels)})\n' + '-'.join(sbj_prevalence), weight='bold', y=1.02,fontsize=20)
        
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        
        n_plots =  min(max_length, len(labels), n_max_segments * n_cluster )
        tiled_frames = np.zeros([n_rows * L, (n_plots // n_rows+1) * l, C], dtype=np.uint8)
        #print(f'Initial tile of shape {tiled_frames.shape}')
        n_plotted = 0
        for _, cluster_value in enumerate(sbj_cluster_values):
            
            cluster_indexes = np.argwhere(labels == cluster_value).flatten(); n_s = len(cluster_indexes) 
            sampled_segments_idx = sorted(random.sample(list(cluster_indexes), min(len(cluster_indexes), n_max_segments)))
            cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in sampled_segments_idx]
            sampled_embeddings = [(start, end, int(start + (end-start) * np.random.sample())) for  start, end in  cpts_tuples]
            
            sampled_embeddings = sampled_embeddings[:n_max_segments]
            
            try:
                std_d_c = np.std(row['cluster_dist_cluster'][cluster_value])
                range_c = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) 
                range_c_n = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) / np.max(row['cluster_dist_cluster'][cluster_value])
                dn = ((row.cluster_dist[embedding_idx] - np.min(row.cluster_dist_cluster[cluster_value])) / range_c)
            except:
                std_d_c = np.nan
                range_c = np.nan
                range_c_n = np.nan
                dn = np.nan

            if verbose:
                print(f'Exploring label {cluster_value}')
                print(f'Segments labels: {labels}')
                print(f'Sampled segments idx: {sampled_segments_idx}')
                print(f'Cpts start-end: {cpts_tuples}')
                print(f'Final sampled frames: {sampled_embeddings}')

        
            #print(f'Final tile of shape {tiled_frames.shape}')
            for i_seg , (start, end, embedding_idx) in enumerate(sampled_embeddings):
                
                sampled_frame = int( embedding_idx / row.N * len(vr))
                if i_seg == 0:
                     #sampled_frame_info = f'C={cluster_value} (x{n_s}) ({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m)t={100*embedding_idx / row.N:.1f}% d={10*row.cluster_dist[embedding_idx]:.1f}'#\nt_n={100*embedding_idx / row.N:.1f}  '
                     sampled_frame_info = f'C={cluster_value} (x{n_s}) sig_c={100*std_d_c:.1f} 10*r_c={10*range_c:.1f}'
                else:
                    sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) - d={100*dn:.1f}'

                #sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) d={100*(row.cluster_dist[embedding_idx] / range_c):.1f}%'
                    
                    
                    
                # Place each sampled frame in the array
                try:
                    frame = np.array(vr[sampled_frame].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    print(n_plotted, 'Error loading frame', sampled_frame)
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                        
                
                #print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                tiled_frames[(n_plotted % n_rows) * L:(n_plotted % n_rows + 1) * L, (n_plotted // n_rows) * l:(n_plotted // n_rows +1) * l, :] = frame
                
                
                # Add sampled frame number at the top-right corner
                text_x = n_plotted // n_rows * l  +l - 3  # Slight offset from the right
                text_y = (n_plotted % n_rows) * L + 5  # Slight offset from the top
                #print('text_y=', text_y,'text_x=', text_x,  sampled_frame_info)
                #ax.text(text_x, text_y, sampled_frame_info, color='k', fontsize=5, ha='right', va='top')
                ax.text(text_x, text_y, sampled_frame_info, 
                        color='r' if i_seg==0 else 'k', fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                # if (n_plotted % 5) == 0:
                #     ax.text(text_x-50, text_y, f'{row.participant_id}', color='black', fontsize=20, ha='right', va='top')
                
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    ((n_plotted // n_rows) * l, (n_plotted % n_rows) * L), l, L,
                    linewidth=4,
                    edgecolor=cluster_colors(unique_clusters.tolist().index(cluster_value)),
                    facecolor='none'
                )
                
                ax.add_patch(rect)
                n_plotted += 1
                
        
        tiled_frames = tiled_frames[0:n_rows * L, 0:((n_plotted-1) // n_rows+1) * l, :]
        #print(f'Final tile of shape {tiled_frames.shape}')
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, prefix+row.identifier + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path, bbox_inches='tight', dpi=80)
            green(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
            
    return


def plot_participant_clusters_chronological(df, labels_col='segments_labels', cpts_cols='cpts', n_rows=10,  max_length=500, prefix='', suffix='', n_sbj=10, verbose=False, show=True, figure_folder=None):

    
    assert (df.modality.nunique() == 1) and (df.task_name.nunique() == 1) and (df.n_cluster.nunique() == 1)
    L, l, C = 224 // 4, 398 // 4, 3
    video_loader = get_video_loader()

    n_cluster = df.n_cluster.iloc[0]
    df['cluster_counts'] = (df[labels_col].apply(lambda row: dict(Counter(row)))
                            .apply(lambda d: {int(k): d.get(k, 0) for k in range(n_cluster)}))


    print('cpts_cols:', cpts_cols, 'labels_col', labels_col,  'n_rows', n_rows, 'n_sbj', n_sbj, 'max_length:', max_length, 'n_cluster:', n_cluster)

    # 1) Find total number of clusters and get colors
    cluster_values = sorted(np.unique(np.hstack(df[labels_col].to_list())))
    unique_clusters = np.unique(cluster_values)
    n_labels = len(unique_clusters)
    colors = list(plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors) * 20

    if len(colors) < n_labels:
        raise ValueError(f"Not enough colors for {n_labels} labels. Increase the color palette size.")
    cluster_colors = ListedColormap(colors[:n_labels])

    df['cluster_dist_cluster'] = collapse_cluster_stats(df, labels_col='embedding_labels', per_segment_feature='cluster_dist')

    for _, row in df.sample(n_sbj).iterrows():
        
        labels = row[labels_col]
        #sbj_cluster_values = sorted(np.unique(labels)); 
        sbj_cluster_values = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False).index
        sbj_prevalence = pd.DataFrame(row.cluster_counts, index=[0]).T.sort_values(0, ascending=False)[0].astype(str).to_list()
        n_clusters = len(sbj_cluster_values)        
        
        if figure_folder is not None:
            figure_path = os.path.join(figure_folder, prefix+row.identifier + suffix+ FIGURES_EXTENSION)
            if os.path.exists(figure_path):
                print(f'Figure {figure_path} already exists. Skipping')
                continue


        try:
            vr = video_loader(row.video_path)
        except:
            print(f'Error loading {row.video_path}. Continue')
            #import subprocess
            #subprocess.run(['open', os.path.dirname(row.video_path)])
            #subprocess.run(['open', os.path.dirname(row.video_path).replace('light', 'final')])
            continue
        
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.title(f'Chronological action segmentation\n{row.task_name} - {row.modality} - {row.participant_id} - K={row.n_cluster} (|S|={len(labels)})\n' + '-'.join(sbj_prevalence), weight='bold', y=1.02,fontsize=20)
        
        # Initialize a large array to store frames (shape: n_clusters * L, n_frames_to_plot * l, C)
        
        n_plots =  min(len(labels), max_length)
        tiled_frames = np.zeros([n_rows * L, (n_plots // n_rows+1) * l, C], dtype=np.uint8)
        #print(f'Initial tile of shape {tiled_frames.shape}')
        n_plotted = 0
        for cluster_indexes, cluster_value  in enumerate(labels):
            
            cpts_tuples = [(row[cpts_cols][idx], min(row[cpts_cols][idx+1], len(vr))) for idx in [cluster_indexes]]
            sampled_embeddings = [(start, end, int(start + (end-start) * np.random.sample())) for  start, end in  cpts_tuples]
            
            
            try:
                std_d_c = np.std(row['cluster_dist_cluster'][cluster_value])
                range_c = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) 
                range_c_n = (np.max(row['cluster_dist_cluster'][cluster_value]) - np.min(row['cluster_dist_cluster'][cluster_value])) / np.max(row['cluster_dist_cluster'][cluster_value])
            except:
                std_d_c = np.nan
                range_c = np.nan
                range_c_n = np.nan

            if verbose:
                print(f'Exploring label {cluster_value}')
                print(f'Segments labels: {labels}')
                print(f'Cpts start-end: {cpts_tuples}')
                print(f'Final sampled frames: {sampled_embeddings}')

        
            #print(f'Final tile of shape {tiled_frames.shape}')
            for i_seg , (start, end, embedding_idx) in enumerate(sampled_embeddings):
                
                sampled_frame = int( embedding_idx / row.N * len(vr))
                sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) {cluster_indexes}/{len(labels)}'

                #sampled_frame_info = f'({(start / row.N) * row.duration:.1f}-{(end / row.N) * row.duration:.1f} m) d={100*(row.cluster_dist[embedding_idx] / range_c):.1f}%'
                
                    
                # Place each sampled frame in the array
                try:
                    frame = np.array(vr[sampled_frame].asnumpy())
                    frame = cv2.resize(frame, (l, L))  # resize to the target size
                except:
                    print(n_plotted, 'Error loading frame', sampled_frame)
                    frame = np.zeros((L, l, C), dtype=np.uint8)
                        
                
                try:
                    #print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                    tiled_frames[(n_plotted % n_rows) * L:(n_plotted % n_rows + 1) * L, (n_plotted // n_rows) * l:(n_plotted // n_rows +1) * l, :] = frame
                except:
                    print('Error placing frame', frame.shape, tiled_frames.shape)
                    print(f'Tile of shape {tiled_frames.shape}')
                    print('y=', (n_plotted % n_rows) * L, (n_plotted % n_rows + 1) * L,'-----x=',  (n_plotted // n_rows) * l, (n_plotted // n_rows + 1) * l)
                    
                # Add sampled frame number at the top-right corner
                text_x = n_plotted // n_rows * l  +l - 3  # Slight offset from the right
                text_y = (n_plotted % n_rows) * L + 5  # Slight offset from the top
                #print('text_y=', text_y,'text_x=', text_x,  sampled_frame_info)
                #ax.text(text_x, text_y, sampplot_cluster_prevalenceled_frame_info, color='k', fontsize=5, ha='right', va='top')
                ax.text(text_x, text_y, sampled_frame_info, 
                        color=cluster_colors(unique_clusters.tolist().index(cluster_value)), #'r' if i_seg==0 else 'k', 
                        fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                    )
                # if (n_plotted % 5) == 0:
                #     ax.text(text_x-50, text_y, f'{row.participant_id}', color='black', fontsize=20, ha='right', va='top')
                
                # Add quadrillage (cluster-specific border)
                rect = Rectangle(
                    ((n_plotted // n_rows) * l, (n_plotted % n_rows) * L), l, L,
                    linewidth=4,
                    edgecolor=cluster_colors(unique_clusters.tolist().index(cluster_value)),
                    facecolor='none'
                )
                
                ax.add_patch(rect)
                n_plotted += 1
                
        
        tiled_frames = tiled_frames[0:n_rows * L, 0:((n_plotted-1) // n_rows+1) * l, :]
        #print(f'Final tile of shape {tiled_frames.shape}')
        plt.imshow(tiled_frames);plt.axis('off')

        if figure_folder is not None:
            
            figure_path = os.path.join(figure_folder, prefix+row.identifier + '_K_' + str(cluster_value) + suffix+ FIGURES_EXTENSION)
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path, bbox_inches='tight', dpi=80)
            green(f'Saved figure to {figure_path}')
            
        if show:
            plt.show()
        else:
            plt.close()
    return


def plot_clusters_global_helper(cpts_config_name='ChangePointDetectionDeploymentConfig', 
                                    clustering_config_name='ClusteringDeploymentKmeansConfig', 
                                    n_clusters=[50]):
        
    n_cluster_folder_list = [f'K_{f}' for f in n_clusters]

    df = get_all_clusters_dataframes(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_cluster_folder_list=n_cluster_folder_list, extended_version=False)
    
    # Linux:
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    #df['video_path'] = df['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    # Cheetah
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    #df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)

    df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root('light')))
    df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    df['has_video'] = df.video_path.apply(lambda x: True if ((type(x) == str) and os.path.isfile(x)) else False)
    df = use_light_dataset(df, light_by_default=True)

    col_used = ['identifier', 'task_name', 'participant_id', 'modality', 'duration', 'N', 
    'cpts', 'has_video', 'video_path', 'embedding_labels', 'cluster_dist']
    df  = df[col_used]

    clustering_config = import_config(clustering_config_name)

    t_init = time.time()


    for task_name in ['cuisine',]:# 'lego']:
        
        for modality in ['Tobii']:#, 'GoPro1', 'GoPro2', 'GoPro3']:
            
            
            if (task_name == 'cuisine') and (modality != 'Tobii'):
                continue
            elif (task_name == 'lego') and (modality == 'GoPro1'):
                continue
            
            mdf = df[(df['modality'] == modality ) & (df['task_name'] == task_name)].sort_values(['duration', 'identifier'])

            #try:                
            figure_folder = os.path.join(get_data_root(), 
                                        'outputs',
                                        clustering_config.experiment_name, 
                                        'figures-clusters', 
                                        task_name,
                                        modality,
                                        clustering_config.experiment_id,
                                        'closest')
            print(f"Processing {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder} ")

            plot_clusters_global(mdf,
                            config_name=clustering_config_name, 
                            labels_col='embedding_labels', 
                            n_sbj = 30 , 
                            show=False,
                            smallest=True, 
                            prefix = '', 
                            suffix='_closest',
                            figure_folder=figure_folder)#figure_folder)
            
            figure_folder = os.path.join(get_data_root(), 
                                        'outputs',
                                        clustering_config.experiment_name, 
                                        'figures-clusters', 
                                        task_name,
                                        modality,
                                        clustering_config.experiment_id,
                                        'furthest')
            print(f"Processing {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder} ")
            # plot_clusters_global(mdf,
            #                 config_name=clustering_config_name, 
            #                 labels_col='embedding_labels', 
            #                 n_sbj = 30, 
            #                 show=False,
            #                 smallest=False, 
            #                 prefix = '', 
            #                 suffix = '_furthest',
            #                 figure_folder=figure_folder)
            
            # print(f"Processed {task_name} {modality} with {len(mdf)} ids - saved figures to {figure_folder} ")
            # # except Exception as e:
            #     print(f'Error processing {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder} ')
            #     print(e)
            #     continue
                
    print('Done in ', time.time() - t_init)  
        
        
def plot_sbj_clusters_global_helper(cpts_config_name='ChangePointDetectionDeploymentConfig', 
                                clustering_config_name='ClusteringDeploymentKmeansConfig', 
                                n_clusters=['K_50'], 
                                chronological=False):
        
    n_cluster_folder_list = [f'K_{f}' for f in n_clusters]

    df = get_all_clusters_dataframes(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_cluster_folder_list=n_cluster_folder_list, extended_version=False)

    # Linux:
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    #df['video_path'] = df['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    # Cheetah
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    #df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)

    df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root('light')))
    df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    df['has_video'] = df.video_path.apply(lambda x: True if ((type(x) == str) and os.path.isfile(x)) else False)
    df = use_light_dataset(df, light_by_default=True)

    col_used = ['identifier', 'task_name', 'participant_id', 'modality', 'duration', 'N', 
    'cpts', 'has_video', 'video_path', 'n_cluster', 'embedding_labels', 'segments_labels', 'cluster_dist']
    df  = df[col_used]

    clustering_config = import_config(clustering_config_name)

    t_init = time.time()
    for n_cluster in n_clusters:

        for task_name in ['cuisine',]:# 'lego']:
            
            for modality in ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3']:
                
                
                if (task_name == 'cuisine') and (modality != 'Tobii'):
                    continue
                elif (task_name == 'lego') and (modality == 'GoPro1'):
                    continue
                
                
                
                figure_folder = os.path.join(get_data_root(), 
                                            'outputs',
                                            clustering_config.experiment_name, 
                                            'figures-subjects', 
                                            task_name,
                                            modality,
                                            f'K_{n_cluster}',
                                            'chronological' if chronological else 'prevalence_ordered')
                
                mdf = df[(df['modality'] == modality ) & (df['task_name'] == task_name) & (df['n_cluster'] == n_cluster)].sort_values(['duration', 'identifier', 'n_cluster'])
                print(f"Processing {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder} ")
                if chronological:
                    plot_participant_clusters_chronological(mdf, labels_col='segments_labels', cpts_cols='cpts', n_rows=20,  max_length=500, prefix='', suffix='_chonological', n_sbj=3, verbose=False, show=False, figure_folder=figure_folder)
                else:
                    
                    plot_participant_clusters(mdf, labels_col='segments_labels', cpts_cols='cpts', n_rows=15, n_max_segments=5, max_length=200, prefix='', suffix='_prevalence_ordered', n_sbj=3, verbose=False, show=False, figure_folder=figure_folder)

    print('Done in ', time.time() - t_init)  


def plot_nearest_centroid_frames_helper(cpts_config_name='ChangePointDetectionDeploymentConfig', 
                                    clustering_config_name='ClusteringDeploymentKmeansConfig', 
                                    n_clusters=[50]):
    
    n_cluster_folder_list = [f'K_{f}' for f in n_clusters]

    df = get_all_clusters_dataframes(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_cluster_folder_list=n_cluster_folder_list, extended_version=False)

    # Linux:
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    #df['video_path'] = df['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    # Cheetah
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    #df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)

    df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root('light')))
    df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    df['has_video'] = df.video_path.apply(lambda x: True if ((type(x) == str) and os.path.isfile(x)) else False)
    df = use_light_dataset(df, light_by_default=True)

    col_used = ['identifier', 'task_name', 'participant_id', 'modality', 'duration', 'N', 
    'cpts', 'has_video', 'video_path', 'n_cluster', 'embedding_labels', 'cluster_dist']
    df  = df[col_used]

    clustering_config = import_config(clustering_config_name)

    t_init = time.time()
    for n_cluster in n_clusters:

        for task_name in ['cuisine',]:# 'lego']:
            
            for modality in ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3']:
                
                
                if (task_name == 'cuisine') and (modality != 'Tobii'):
                    continue
                elif (task_name == 'lego') and (modality == 'GoPro1'):
                    continue

                mdf = df[(df['modality'] == modality ) & (df['task_name'] == task_name) & (df['n_cluster'] == n_cluster)].sort_values(['duration', 'identifier', 'n_cluster'])


                figure_folder = os.path.join(get_data_root(), 
                            'outputs',
                            clustering_config.experiment_name, 
                            task_name,
                            modality, 
                            'artifacts'); os.makedirs(figure_folder, exist_ok=True)

                print(f"Processing {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder} ")
                
                plot_nearest_centroid_frames(mdf, config_name=clustering_config_name, labels_col='embedding_labels', sampled_index=0, show=False,  figure_folder=figure_folder, suffix='_0')
                plot_nearest_centroid_frames(mdf, config_name=clustering_config_name, labels_col='embedding_labels', sampled_index=1, show=False,  figure_folder=figure_folder, suffix='_1')
                plot_nearest_centroid_frames(mdf, config_name=clustering_config_name, labels_col='embedding_labels', sampled_index=2, show=False,  figure_folder=figure_folder, suffix='_2')
                

                green(f'Done {task_name} {modality} with {len(mdf)} ids - saving figures to {figure_folder}')
       
def plot_clusters_subjects_rows_helper(cpts_config_name='ChangePointDetectionDeploymentConfig', 
                                    clustering_config_name='ClusteringDeploymentKmeansConfig', 
                                    n_clusters=[50]):
    
    n_cluster_folder_list = [f'K_{f}' for f in n_clusters]

    df = get_all_clusters_dataframes(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_cluster_folder_list=n_cluster_folder_list, extended_version=False)

    # Linux:
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/'))
    #df['video_path'] = df['video_path'].apply(lambda x: x.replace('/media/sam/', '/Volumes/') if pd.notnull(x) else x)

    # Cheetah
    #df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root()))
    #df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)

    df['video_representation_path'] = df['video_representation_path'].apply(lambda x: x.replace('/diskA/sam_data/data-features', get_data_root('light')))
    df['video_path'] = df.apply(lambda x: os.path.join(get_data_root(), x.task_name, x.participant_id, x.modality, x.video_name + '.mp4'), axis=1)
    df['has_video'] = df.video_path.apply(lambda x: True if ((type(x) == str) and os.path.isfile(x)) else False)
    df = use_light_dataset(df, light_by_default=True)

    
    col_used = ['identifier', 'task_name', 'participant_id', 'modality', 'duration',# 'N', 'n_frames', 
            'segments_labels','cpts_frames',
            'cpts', 'has_video', 'video_path', 'n_cluster', 'embedding_labels', 'cluster_dist']
    df['cpts_frames'] = df.apply(lambda row: [int(cpt * row.n_frames / row.N) for cpt in row.cpts], axis=1)
    df  = df[col_used]

    clustering_config = import_config(clustering_config_name)

    t_init = time.time()
    for n_cluster in n_clusters:

        for task_name in ['cuisine',]:# 'lego']:
            
            for modality in ['Tobii', 'GoPro1', 'GoPro2', 'GoPro3']:
                
                
                if (task_name == 'cuisine') and (modality != 'Tobii'):
                    continue
                elif (task_name == 'lego') and (modality == 'GoPro1'):
                    continue

                figure_folder = os.path.join(get_data_root(), 
                                                'outputs',
                                                clustering_config.experiment_name, 
                                                'figures-clusters', 
                                                task_name,
                                                modality,
                                                f'K_{n_cluster}', 
                                                'subjects_rows'); os.makedirs(figure_folder, exist_ok=True)
                
                
                sub_sd = df[(df['task_name'] == task_name) & 
                            (df['modality'] == modality) & 
                            (df['has_video']==1) &
                            (df['n_cluster'] == n_cluster) ]
                
                if sub_sd.empty:
                    print(f"Skipping {task_name} {modality} with {len(sub_sd)} ids - no video")
                    continue
                

                print(f"Processing {task_name} {modality} with {len(sub_sd)} ids - saving figures to {figure_folder} ")
                
                plot_clusters_subjects_rows(sub_sd,
                                            config_name=clustering_config_name, 
                                        cpts_cols='cpts_frames', 
                                            labels_col='segments_labels', 
                                            suffix = '',
                                            n_sbj = 5, 
                                            n_frame_per_sbj = 5, 
                                            show=False,
                                            figure_folder=figure_folder)

                green(f'Done {task_name} {modality} with {len(sub_sd)} ids - saving figures to {figure_folder}')
              
def parse_args():
    parser = argparse.ArgumentParser(description='Clustering Deployment Script')
    parser.add_argument('--config_name', type=str, default='ClusteringDeploymentKmeansConfig', help='Configuration name for clustering deployment')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    n_clusters = [250]
    cpts_config_name = 'ChangePointDetectionDeploymentConfig'

    plot_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=args.config_name, n_clusters=n_clusters)

    #clustering_config_name = 'ClusteringDeploymentKmeansTaskCombinationConfig' #args.clustering_config_name
    #plot_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)



    # plot_nearest_centroid_frames_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    # blue('[INFO] Done plot_nearest_centroid_frames_helper')
    #plot_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    #blue('[INFO] Done plot_clusters_global_helper')
    # plot_clusters_subjects_rows_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    # blue('[INFO] Done plot_clusters_subjects_rows_helper')
    # plot_sbj_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters, chronological=True)
    # blue('[INFO] Done plot_sbj_clusters_global_helper chronological')
    # plot_sbj_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters, chronological=False)
    # blue('[INFO] Done plot_sbj_clusters_global_helper prevalence_ordered')

    #blue('[INFO] Done plot_clusters_global_helper')

    # clustering_config_name = 'ClusteringDeploymentPersonalizedKmeansConfig'
    # plot_nearest_centroid_frames_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    # plot_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    # plot_clusters_subjects_rows_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters)
    # plot_sbj_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters, chronological=True)
    # plot_sbj_clusters_global_helper(cpts_config_name=cpts_config_name, clustering_config_name=clustering_config_name, n_clusters=n_clusters, chronological=False)

    # rsync -ahuvzL --progress pomme:/home/perochon/data-gold-final/outputs/<experiment_name>  ~/github-repositories/smartflat/data/data-gold-final/outputs

    # conda activate temporal_segmentation
    # python3 /home/perochon/smartflat/api/features/symbolization/visualization_prototypes.py --config ClusteringDeploymentKmeansExogeneousCombinationConfig
