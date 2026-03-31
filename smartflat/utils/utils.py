import ast
import base64
import io
import os
import socket
import subprocess
import sys
import uuid
from collections import Counter
from copy import deepcopy
from glob import glob
from itertools import tee
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

#from dash import Dash, dcc, html, Input, Output, no_update
#from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import seaborn as sns

#from spectralcluster import SpectralClusterer, RefinementOptions, ThresholdType, ICASSP2018_REFINEMENT_SEQUENCE
#from spectralcluster.utils import compute_affinity_matrix
#import skvideo.io
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm import tqdm

#from ruptures.base import BaseCost
#from ruptures.exceptions import NotEnoughPoints


# add tools path and import our own tools

#from smartflat.const import *


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# Function to rename columns with a specified suffix
def add_cols_suffixes(df, cols, suffix):
    df = df.rename(columns={col: f"{col}{suffix}" for col in cols})
    return df

def get_expids():
    experiment_id = uuid.uuid4()
    return str(experiment_id)


def smartflat_range(num_frames, segment_length, delta_t):
    """For the smarflat project we want precise modeling of the video, 
    with baseline representations having overlap between each (so delta_t < segment_length)"""
    return range(0, num_frames - segment_length-1, delta_t)


# def collect_embeddings(df):
# Deprecated ?
#     """Copy folder structure and computed representations in the data/dump folder"""
    
#     dump_location = os.path.join(get_data_root(), 'dump'); os.makedirs(dump_location, exist_ok=True)
#     for task, participant_id, modality, embedding_path in  df[df['video_representation_computed']][['task', 'participant_id', 'modality', 'embedding_path']].to_numpy():
    
#         embed_folder = os.path.join(dump_location, task, participant_id, modality); os.makedirs(embed_folder, exist_ok=True)
#         subprocess.run(['cp', embedding_path, embed_folder])
#         print("Copied embedding to {}/ ...".format(embed_folder))

#     # Copy the compute log file
#     metrics_path = os.path.join(get_data_root(), 'dataframes', 'compute_time_video_representation_learning.csv')
#     output_metric_path = os.path.join(dump_location, '{}_compute_time_video_representation_learning.csv'.format(socket.gethostname()))
#     subprocess.run(['mv', metrics_path, output_metric_path])

#     print("Done.")

#     return
    

def check_and_convert(x):
    try:
        # First attempt to evaluate the string (e.g., if it is a string representation of a number)
        x = ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # If evaluation fails, it remains as is
        pass
    # Use pd.to_numeric to convert to numeric and coerce errors to NaN
    return pd.to_numeric(x, errors='coerce')


def retrieve_video_path(dataset_name):
    
    if dataset_name=='50Salads':
        root = '/home/perochon/temporal_segmentation/data/50Salads/rgb/'
        video_paths = glob(os.path.join(root, '*')); video_paths = [p for p in video_paths if p[-4:] == '.avi']
    
    elif dataset_name=='Breakfast':
        
        root = '/home/perochon/temporal_segmentation/data/Breakfast/BreakfastII_15fps_qvga_sync'
        video_paths = glob(os.path.join(root, '*', '*', '*'));video_paths = [p for p in video_paths if p[-4:] == '.avi']

        
    elif dataset_name=='GTEA':
        
        root = '/home/perochon/temporal_segmentation/data/GTEA/videos/'
        video_paths = glob(os.path.join(root, '*')); video_paths = [p for p in video_paths if p[-4:] == '.mp4']

    return video_paths



# def upsample_sequence(sequence, target_length):

#     if not (isinstance(sequence, list) or isinstance(sequence, np.ndarray)):
#         return np.nan
    
#     # Calculate the length of the original sequence
#     original_length = len(sequence)
    
#     # Calculate the ratio by which to upscale the sequence
#     upscale_ratio = target_length / original_length
    
#     # Generate indices for the new sequence
#     new_indices = np.arange(target_length) / upscale_ratio
    
#     # Interpolate values using linear interpolation
#     new_sequence = np.interp(new_indices, np.arange(original_length), sequence).astype(int)
    
#     return new_sequence

def predict_segments_from_embed_labels(row, temporal_segmentation_col='cpts', embedding_labels_col='embedding_labels', output_col='segments_label', combine_func_name="majority_voting", filter_noise_labels=True, verbose=False):
    """Row-function that predicts segment label based on the point-wise labels within the segment, as defined by the temporal segmentation

    Use `cpts` and `embedding_labels`
    """
    
    def majority_voting(labels, filter_noise_labels=False, verbose=False):

        if len(labels) == 0:
            return -1  # Return noise label if empty
        
        # Remove -1 values encoding (i) empty annotation for the Boris label arrays, and (ii) noise and exogeneous for symbolic sequences
        valid_labels = labels[labels >= 0] if filter_noise_labels else labels

        
        # Compute majority label and stats
        counter = Counter(valid_labels)
        total_votes = len(valid_labels)
        
        majority_label = counter.most_common(1)[0][0]
        top5 = counter.most_common(5)

        # Print results in a concise format
        if verbose and len(top5) > 1:
            pass#print(f"Majority Label: {majority_label} (L={len(labels)} L_f={len(valid_labels)}) Top 5 Clusters (percentage): ", end="")
            #print(", ".join(f"{label}: {round((count / total_votes) * 100)}%" for label, count in top5))
        # print(valid_labels)
        #return np.bincount(valid_labels).argmax() if valid_labels.size > 0 else -1
        #if majority_label == -1:
        #    raise ValueError(f"Majority label is -1, which is unexpected. Check the input labels: {labels}")
        return majority_label
    
    def median_label(labels):
        return np.median(labels)
    
    # def closest_voting(labels, distances):
    #     print(distances, np.std(distances))
        
    #     valid_labels = labels[labels >= 0]  # Remove -1 values encoding (i) empty annotation for the Boris label arrays, and (ii) noise and exogeneous for symbolic sequences
    #     valid_distances = distances[labels >= 0]  # Remove -1 values encoding (i) empty annotation for the Boris label arrays, and (ii) noise and exogeneous for symbolic sequences

    #     label_majority_voting = np.bincount(valid_labels).argmax() if valid_labels.size > 0 else -1
    #     print(f'Majority voting - clostes voting  ? {label_majority_voting}=={valid_labels[np.argmin(valid_distances)]}')
    #     return valid_labels[np.argmin(valid_distances)]


    def closest_voting(labels, distances):
        labels = np.array(labels)
        distances = np.array(distances)

        valid_labels = labels[labels >= 0]  # Remove invalid labels
        valid_distances = distances[labels >= 0]  # Remove corresponding distances

        if valid_labels.size == 0 or valid_distances.size == 0:
            #print("Warning: No valid labels or distances found. Returning -1.")
            return -1  # Return a default value if no valid data exists

        label_majority_voting = np.bincount(valid_labels).argmax()
        label_closest_voting = valid_labels[np.argmin(valid_distances)]  # Now safe
        if label_majority_voting != label_closest_voting:
            print(f'Majority voting: {label_majority_voting}, Closest voting: {label_closest_voting}')
        return label_majority_voting

    row[output_col] = [0] * (len(row[temporal_segmentation_col]) - 1)

    for segment_index, (i, j) in enumerate(pairwise(row[temporal_segmentation_col])):
        #try:
        #print(label_selection_fn, row)
        if combine_func_name == "majority_voting":
            row[output_col][segment_index] = majority_voting(row[embedding_labels_col][int(i):int(j)], filter_noise_labels=filter_noise_labels, verbose=verbose)
        elif combine_func_name == "median_label":
            row[output_col][segment_index] = median_label( row[embedding_labels_col][int(i):int(j)])
        elif combine_func_name == "closest_voting":
            row[output_col][segment_index] = closest_voting( row[embedding_labels_col][int(i):int(j)], row['cluster_dist'][int(i):int(j)])
        elif combine_func_name == "majority_voting_inner":
            seg_labels = row[embedding_labels_col][int(i):int(j)]
            seg_len = len(seg_labels)
            if seg_len >= 4:
                q1 = seg_len // 4
                q3 = 3 * seg_len // 4
                inner_labels = seg_labels[q1:q3]
            else:
                # fallback if too short → use full segment
                inner_labels = seg_labels
            row[output_col][segment_index] = majority_voting(inner_labels, verbose=verbose)
                
            
        #except:
            #print(f"Error for {row.participant_id} - {row.modality} in segment {segment_index} with i={i} and j={j}")
            #print('row["embedding_labels"].shape:', row["embedding_labels"].shape)
            #print('row["cpts"][-1]:', temporal_segmentation_col, row[temporal_segmentation_col][-1], len(row[temporal_segmentation_col]))
        #    break
    return np.asarray(row[output_col])

# def upsample_sequence(sequence, target_length):
#     if not isinstance(sequence, (list, np.ndarray)):
#         return np.nan
    
#     #print('Upsampling sequence from {} to {}'.format(len(sequence), target_length))

#     original_length = len(sequence)
#     new_indices = np.linspace(0, original_length - 1, target_length)

#     # Nearest-neighbor interpolation to preserve original values
#     new_sequence = np.array(sequence)[np.round(new_indices).astype(int)]
    
#     return new_sequence


def upsample_sequence(sequence, target_length):
    if not isinstance(sequence, (list, np.ndarray)) or len(sequence) == 0:
        return np.full(target_length, np.nan)  # Return NaNs if sequence is invalid or empty

    original_length = len(sequence)
    
    if target_length <= 1 or original_length <= 1:
        # Handle small cases safely
        return np.full(target_length, sequence[0]) if original_length > 0 else np.full(target_length, np.nan)
    
    new_indices = np.linspace(0, original_length - 1, target_length)
    new_sequence = np.array(sequence)[np.clip(np.round(new_indices).astype(int), 0, original_length - 1)]
    
    return new_sequence

def pad_sequence_with_zeros(sequence, target_length, value=0):
    # Ensure sequence is a list or ndarray
    if not (isinstance(sequence, list) or isinstance(sequence, np.ndarray)):
        return np.nan
    
    # Pad the sequence with zeros up to the target length
    padded_sequence = np.pad(sequence, (0, target_length - len(sequence)), mode='constant', constant_values=value)
    
    return padded_sequence


def get_upsampled_labels(df, labels_col='symb_labels', use_shortest=False, n_samples=1000, upsampling='padding'):
    if use_shortest:
        subset = df.nsmallest(n_samples, 'N')[labels_col].dropna().iloc[:n_samples]
    else:
        subset = df[labels_col].dropna().iloc[:n_samples]
        
    max_duration = subset.apply(len).max()
    if upsampling == 'interpolation':
        upsampled = subset.apply(upsample_sequence, args=(max_duration,))
    else:  # assumes padding if not interpolation
        upsampled = subset.apply(pad_sequence_with_zeros, args=(max_duration,))
    return np.vstack(upsampled)


def detect_blur_fft(image):
    size=60
    # grab the dimensions of the image and use the dimensions to
    # # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
            
    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
        
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean









def cluster_frames(
    embedding,
    cpts,
    method="kmeans",
    n_clusters=None,
    mask_outliers=None,
    threshold_outliers=0.5,
    centroid_dict=None,
    verbose=False,
    *args,
    **kwargs,
):
    """
    Takes n_clusters as input (optionnal), and use the self.segments_embedding to cluster each of the segment.

    output:

    self.segments_label

    """

    # Define number of cluster
    if n_clusters is None:
        pass  # self.n_clusters = self._find_optimal_nb_cluster(method=method, verbose=verbose)
    else:
        if n_clusters == 1:
            n_clusters += 1

    D, N = embedding.shape

    # Init. the vectors used
    vectors = embedding.T

    if mask_outliers is not None:
        vectors = embedding[self.mask_outliers == 0]

    if method == "supervised":

        if centroid_dict is None:
            self.centroid_dict = self._init_clusters_centroid()

        self.mapping_label_name = {
            i: name
            for i, name in enumerate(["Outliers"] + list(self.centroid_dict.keys()))
        }
        print(
            "Using custom initialization of the centroids, there are: {} clusters".format(
                n_clust
            )
        )
        kmeans = KMeans(
            n_clusters=len(self.centroid_dict.keys()),
            algorithm="auto",
            init=np.array(list(self.centroid_dict.values())),
            random_state=0,
        ).fit(vectors)
        labels = kmeans.labels_ + 1
        self.n_clusters = len(self.centroid_dict.keys())

    # Cluster the representations using K-Means
    elif method == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters, algorithm="auto", init="k-means++", random_state=0
        ).fit(vectors)
        labels = kmeans.labels_ + 1

    elif method == "kernel":
        kmeans = KernelKMeans(
            n_clusters=self.n_clusters, kernel="cosine", random_state=0
        ).fit(vectors)
        labels = kmeans.labels_ + 1

    elif method == "spectral_clustering_embedding":

        if n_clusters is not None:
            min_clusters, max_clusters = n_clusters - int(
                np.ceil(0.05 * n_clusters)
            ), n_clusters + int(np.ceil(0.05 * n_clusters))

        labels = (
            spectral_clustering_embedding(
                vectors,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                *args,
                **kwargs,
            )
            + 1
        )
        self.n_clusters = len(np.unique(self.segments_label))

    elif method == "spectral_clustering":
        segments_gram = vectors @ vectors.T
        # self.n_clusters, self.segments_label  = spectral_clustering(self._embedding.gram, custom_dist='cosine', max_iter=300, *args, **kwargs)
        self.n_clusters, labels = spectral_clustering(
            segments_gram, custom_dist="cosine", max_iter=300, *args, **kwargs
        )
        labels += 1

    elif method == "TW-FINCH":

        # Load the Descriptor
        _, _, labels = FINCH(
            vectors, req_clust=n_clusters, verbose=False, tw_finch=True
        )
        labels = labels + 1

    # Assign the labels of each inlier vectors
    embedding_label = deepcopy(labels)

    if mask_outliers is not None:
        embedding_label[np.argwhere(mask_outliers == 1).flatten()] = 0

    embedding_label = embedding_label.astype(int)

    segments_label = np.array([np.nan] * (len(cpts) - 1))

    for segment_index, (i, j) in enumerate(pairwise(cpts)):
        segments_label[segment_index] = np.bincount(embedding_label[i:j]).argmax()

    # Unlabelled background segments if isolated between two same activities (if length is less than 2s)
    min_size = np.ceil(25 * 0.25 / DEFAULT_TAU_SAMPLE).astype(int)
    segments_length = [j - i for i, j in pairwise(cpts)]
    previous_label = segments_label[0]
    for i, segment_label in enumerate(segments_label[1:-1]):
        next_label = segments_label[i + 2]

        # if segment_label==0 and next_label==previous_label!=0 and segments_length[i+1]<=min_size:

        if next_label == previous_label != 0 and segments_length[i + 1] <= min_size:
            segments_label[i + 1] = next_label

        previous_label = segment_label

    # self._find_outlier_segments(verbose=True)

    # if 'silhouette_score' in self.outlier_methods:
    #    self._find_outliers_frame_silhouette(verbose=verbose)

    # self._segments_outliers_removal(threshold_outliers=threshold_outliers, verbose=verbose)

    return embedding_label, segments_label

def init_df_ruptures(path):#=os.path.join(DATA_DIR,'results_ruptures.csv')):
    

    if not os.path.isfile(path):
        results_dict = {'dataset':[],
                    'video_name': [],
                    'embedding_method': [],
                    'model_name': [],
                    'reduce_embedding': [],
                    'dim_embedding': [],
                    'gram_post_process':[],
                    'outlier_methods':[],
                    #'embedding_score_kernel': [], 
                    #'embedding_score_adjacent': [], 
                    #'embedding_score_mmd': [], 
                    #'embedding_score_kernel_optical_flow': [], 
                    #'embedding_score_adjacent_optical_flow': [], 
                    #'embedding_score_mmd_optical_flow': [], 
                    'ruptures_on': [],
                    'ruptures_on_gram': [],
                    'penalty': [],
                    'f1-mean': [],
                    'f1_segmentation': [],
                    'mof': [], 
                    'f1_macro' : [],
                    'iou': [],
                    'mof_twfinch': [], 
                    'f1_macro_twfinch' : [],
                    'iou_twfinch': []
                    }    
        df = pd.DataFrame(results_dict)

    else:

        df = pd.read_csv(path)
        results_dict = df.to_dict(orient='list')
        
    return df

def save_experiment_ruptures(df, pipeline, path, type=None):
    
    df = df.append(pd.DataFrame({'dataset': pipeline.dataset.task,
                                'video_name': pipeline.dataset.video_name, 
                                'embedding_method': pipeline._embedding.embedding_method, 
                                'reduce_embedding': pipeline._embedding.reduce_embedding, 
                                'dim_embedding': pipeline._embedding.dim_embedding, 
                                'model_name': pipeline._embedding._extractor.model_name,
                                'gram_post_process': str(pipeline._embedding.gram_post_process),
                                'outlier_methods': str(pipeline.outlier_methods), 

                                #'embedding_score_kernel': pipeline.embedding_score_kernel['embedding'], 
                                #'embedding_score_adjacent': pipeline.embedding_score_adjacent['embedding'], 
                                #'embedding_score_mmd': pipeline.embedding_score_mmd['embedding'],



                                #'embedding_score_kernel_optical_flow': pipeline.embedding_score_kernel['optical_flow'] if 'optical_flow' in pipeline.embedding_score_kernel.keys() else np.nan, 

                                #'embedding_score_adjacent_optical_flow': pipeline.embedding_score_adjacent['optical_flow'] if 'optical_flow' in pipeline.embedding_score_adjacent.keys() else np.nan, 

                                #'embedding_score_mmd_optical_flow': pipeline.embedding_score_mmd['optical_flow'] if 'optical_flow' in pipeline.embedding_score_mmd.keys() else np.nan, 

                                'ruptures_on_gram': 'gram' in pipeline.ruptures_on,
                                'ruptures_on': pipeline.ruptures_on,
                                
                                'penalty': pipeline.penalty, 
                                 'f1_segmentation': pipeline.annotation.f1_segmentation,

                                'f1-mean': np.mean(pipeline.annotation.f1_overlap['f1']), 
                                 'mof': pipeline.annotation.mof,
                                 'f1_macro': pipeline.annotation.f1_macro,
                                 'iou': pipeline.annotation.iou,
                                 'mof_twfinch': pipeline.annotation.mof_twfinch,
                                 'f1_macro_twfinch': pipeline.annotation.f1_macro_twfinch,
                                 'iou_twfinch': pipeline.annotation.iou_twfinch                               
                                }, index=[0]), ignore_index=True)

    df.to_csv(path, index=False)
    #display(df.tail(1))
    return df

def init_df():#path=os.path.join(DATA_DIR,'results_embedding_orb.csv'), embedding_method='D+FV'):
    

    if not os.path.isfile(path):
        if embedding_method=='D+FV':
            results_dict = {'dataset':[],
                        'video_name': [],
                        'embedding_method': [],
                        'reduce_embedding': [],
                        'dim_embedding': [],
                        'N_fit': [],
                        'K': [],
                        'color_descriptor': [],
                        'descriptor_name': [],
                        'extraction_mode': [],
                        'reduce_descriptor': [],
                        'D': [], 
                        'gram_post_process':[],
                        'outlier_methods':[],
                        'embedding_score_kernel': [], 
                        'embedding_score_adjacent': [], 
                        'embedding_score_mmd': []
                        }
        elif embedding_method=='deep_features':
            results_dict = {'dataset':[],
                        'video_name': [],
                        'embedding_method': [],
                        'reduce_embedding': [],
                        'dim_embedding': [],
                        'model_name': [],
                        'gram_post_process':[],
                        'outlier_methods':[],
                        'embedding_score_kernel': [], 
                        'embedding_score_adjacent': [], 
                        'embedding_score_mmd': []
                        }        
        df = pd.DataFrame(results_dict)

    else:

        df = pd.read_csv(path)
        results_dict = df.to_dict(orient='list')
        
    return df

def save_experiment(df, pipeline, path):
    if pipeline.config['embedding_method']=='D+FV':
        df = df.append(pd.DataFrame({'dataset': pipeline.dataset.task,
                            'video_name': pipeline.dataset.video_name, 
                            'embedding_method': pipeline._embedding.embedding_method, 
                            'reduce_embedding': pipeline._embedding.reduce_embedding, 
                            'dim_embedding': pipeline._embedding.dim_embedding, 
                            'N_fit': pipeline._embedding._extractor.N_fit, 
                            'K': pipeline._embedding._extractor.K, 
                            'color_descriptor': pipeline._embedding._extractor.color_descriptor, 
                            'descriptor_name': pipeline._embedding._extractor._descriptors.descriptor_name, 
                            'extraction_mode': pipeline._embedding._extractor._descriptors.extraction_mode, 
                            'reduce_descriptor': pipeline._embedding._extractor._descriptors.reduce_descriptor, 
                            'D': pipeline._embedding._extractor._descriptors.D, 
                            'gram_post_process': str(pipeline._embedding.gram_post_process),
                            'outlier_methods': str(pipeline.outlier_methods), 
                            'embedding_score_kernel': pipeline.embedding_score_kernel, 
                            'embedding_score_adjacent': pipeline.embedding_score_adjacent, 
                            'embedding_score_mmd': pipeline.embedding_score_mmd}, index=[0]), ignore_index=True)

    elif pipeline.config['embedding_method']=='deep_features':
        df = df.append(pd.DataFrame({'dataset': pipeline.dataset.task,
                                    'video_name': pipeline.dataset.video_name, 
                                    'embedding_method': pipeline._embedding.embedding_method, 
                                    'reduce_embedding': pipeline._embedding.reduce_embedding, 
                                    'dim_embedding': pipeline._embedding.dim_embedding, 
                                    'model_name': pipeline._embedding._extractor.model_name,
                                    'gram_post_process': str(pipeline._embedding.gram_post_process),
                                    'outlier_methods': str(pipeline.outlier_methods), 
                                    'embedding_score_kernel': pipeline.embedding_score_kernel, 
                                    'embedding_score_adjacent': pipeline.embedding_score_adjacent, 
                                    'embedding_score_mmd': pipeline.embedding_score_mmd}, index=[0]), ignore_index=True)

    df.to_csv(path, index=False)
    #display(df.tail(1))
    return df

def check_experiment_already_done(df, verbose=False, return_df=False, **kwargs):
    return False #TODO
    
    narrowed_df=deepcopy(df)
    if verbose:
        print(len(narrowed_df)) 
        
        
    if 'model_name' in narrowed_df.columns:
        del kwargs['descriptor_name'], kwargs['reduce_descriptor'], 
    elif 'descriptor_name' in narrowed_df.columns:
        del kwargs['model_name']
    
    for key, value in kwargs.items():
        
        if key in ['reduce_embedding', 'reduce_descriptors']:
            if value is None:
                narrowed_df = narrowed_df[narrowed_df[key].isnull()]
            else:
                narrowed_df = narrowed_df[narrowed_df[key]==value]
        else:
        
            narrowed_df = narrowed_df[narrowed_df[key]==value]
            
        print(len(narrowed_df), key, value) if verbose else None
        
    if not return_df:
        
        return len(narrowed_df) > 0
    
    else:
        
        return narrowed_df

def check_experiment_already_done_ol(df, video_name, config):

    if config['embedding_method']=='D+FV':
            
        
        if len(df[(df['video_name']==video_name) &
                (df['embedding_method']==config['embedding_method']) &
                (df['reduce_embedding']==config['reduce_embedding']) &
                (df['dim_embedding']==config['dim_embedding']) &
                (df['N_fit']==config['N_fit']) &
                (df['K']==config['K']) &
                (df['descriptor_name']==config['descriptor_name']) &
                (df['color_descriptor']==config['color_descriptor']) &
                (df['extraction_mode']==config['extraction_mode']) &
                (df['reduce_descriptor']==config['reduce_descriptor']) &
                (df['gram_post_process']==str(config['gram_post_process'])) &
                (df['outlier_methods']==str(config['outlier_methods'])) &
                (df['D']==config['D'])]) > 0 : 
            return True
        else:
            return False

    elif config['embedding_method']=='deep_features':

        if len(df[(df['video_name']==video_name)]) > 0 : 
            return True
        else:
            return False     

def add_default_column(results_paths, column_name, default_value):
    for results_path in results_paths:   
        results_dict = {'dataset':[],
                    'video_name': [],
                    'embedding_method': [],
                    'reduce_embedding': [],
                    'dim_embedding': [],
                    'N_fit': [],
                    'K': [],
                    'color_descriptor': [],
                    'descriptor_name': [],
                    'extraction_mode': [],
                    'reduce_descriptor': [],
                    'outlier_methods': [],
                    'D': [], 
                    'embedding_score_kernel': [], 
                    'embedding_score_adjacent': [], 
                    'embedding_score_mmd': [], 
                    }



        performances_df = pd.DataFrame(results_dict)
        performances_df = performances_df.append(pd.read_csv(results_path))
        performances_df[column_name] = default_value
        performances_df.to_csv(results_path, index=False)
    return

def compute_cost_adjacent(gram, start, end, start_b=None, end_b=None, start_a=None, end_a=None):
    """
        segments = np.floor(np.array(tuple(pairwise(annotation.gt_cpt)))/pipeline.tau_sample).astype(int)
        n = segments.shape[0]

        gram = pipeline.gram
        start=segments[i, 0]
        end=segments[i, 1]
        start_b=segments[i-1, 0] 
        end_b=segments[i-1, 1]
        start_a=segments[i+1, 0]
        end_a=segments[i+1, 1]

        c_sub_gram = gram[start:end, start:end]
        diag = np.diagonal(c_sub_gram).sum()
        NUM = diag - sub_gram.sum() / (end - start)
        DEN = 0

        if start_b is not None:
            b_sub_gram = gram[start_b:end_b, start_b:end_b]
            bi_sub_gram = gram[start_b:end_b, start:end]
            DEN +=  diag - bi_sub_gram.sum() * (2/(end_b-start_b)) + b_sub_gram.sum() * (end-start)/(end_b-start_b)**2
        if start_a is not None:
            a_sub_gram = gram[start_a:end_a, start_a:end_a]
            ai_sub_gram = gram[start:end, start_a:end_a]
            DEN += diag - ai_sub_gram.sum() * (2/(end_a-start_a)) + a_sub_gram.sum() * (end-start)/(end_a-start_a)**2


    """
    
    c_sub_gram = gram[start:end, start:end]
    diag = np.diagonal(c_sub_gram).sum()
    NUM = diag - c_sub_gram.sum() / (end - start)
    DEN = np.finfo(float).eps
    
    if start_b is not None:
        b_sub_gram = gram[start_b:end_b, start_b:end_b]
        bi_sub_gram = gram[start_b:end_b, start:end]
        DEN +=  diag - bi_sub_gram.sum() * (2/(end_b-start_b)) + b_sub_gram.sum() * (end-start)/(end_b-start_b)**2
        
    if start_a is not None:
        a_sub_gram = gram[start_a:end_a, start_a:end_a]
        ai_sub_gram = gram[start:end, start_a:end_a]
        DEN += diag - ai_sub_gram.sum() * (2/(end_a-start_a)) + a_sub_gram.sum() * (end-start)/(end_a-start_a)**2
    return NUM/DEN

def compute_mmd(gram, start, end, start_b=None, end_b=None, start_a=None, end_a=None, verbose=False):
    """
        segments = np.floor(np.array(tuple(pairwise(annotation.gt_cpt)))/pipeline.tau_sample).astype(int)
        n = segments.shape[0]

        gram = pipeline.gram
        start=segments[i, 0]
        end=segments[i, 1]
        start_b=segments[i-1, 0] 
        end_b=segments[i-1, 1]
        start_a=segments[i+1, 0]
        end_a=segments[i+1, 1]

        c_sub_gram = gram[start:end, start:end]
        diag = np.diagonal(c_sub_gram).sum()
        NUM = diag - sub_gram.sum() / (end - start)
        DEN = 0

        if start_b is not None:
            b_sub_gram = gram[start_b:end_b, start_b:end_b]
            bi_sub_gram = gram[start_b:end_b, start:end]
            DEN +=  diag - bi_sub_gram.sum() * (2/(end_b-start_b)) + b_sub_gram.sum() * (end-start)/(end_b-start_b)**2
        if start_a is not None:
            a_sub_gram = gram[start_a:end_a, start_a:end_a]
            ai_sub_gram = gram[start:end, start_a:end_a]
            DEN += diag - ai_sub_gram.sum() * (2/(end_a-start_a)) + a_sub_gram.sum() * (end-start)/(end_a-start_a)**2


    """
    
    c_sub_gram = gram[start:end, start:end]
    score = 0 
    cpt_to_plot = [start, end]
    array_to_plot = [c_sub_gram]
    label_to_plot = ['K22']
    if start_b is not None:
        b_sub_gram = gram[start_b:end_b, start_b:end_b]
        bi_sub_gram = gram[start_b:end_b, start:end]
        score +=  c_sub_gram.sum() / ( (end-start)*(end-start))
        score += b_sub_gram.sum() / ( (end_b-start_b)*(end_b-start_b))
        score -= 2*bi_sub_gram.sum()/( (end_b-start_b)*(end-start))
        if verbose:
            cpt_to_plot.extend([start_b, end_b])
            array_to_plot.extend([b_sub_gram, bi_sub_gram])
            label_to_plot.extend(['K11', 'K12'])
    if start_a is not None:
        a_sub_gram = gram[start_a:end_a, start_a:end_a]
        ai_sub_gram = gram[start:end, start_a:end_a]
        score +=  c_sub_gram.sum() / ( (end-start)*(end-start))
        score += a_sub_gram.sum() / ( (end_a-start_a)*(end_a-start_a))
        score -= 2*ai_sub_gram.sum()/( (end_a-start_a)*(end-start))
        if verbose:
            cpt_to_plot.extend([start_a, end_a])
            array_to_plot.extend([a_sub_gram, ai_sub_gram])
            label_to_plot.extend(['K33', 'K23'])
    if verbose:
        fig, axes = plt.subplots(1, 6, figsize=(25, 8)); axes = axes.flatten()
        
        axes[0].imshow(gram, vmin=0, vmax=1);axes[0].set_title("K Distance={:.2f}".format(score))
        for cpt in cpt_to_plot:
            axes[0].axvline(cpt, color='tab:red', linewidth=4)
            axes[0].axhline(cpt, color='tab:red', linewidth=4)
        
        for i, (arr, lab) in enumerate(zip(array_to_plot, label_to_plot)):

            axes[i+1].imshow(arr, vmin=0, vmax=1);axes[i+1].set_title(lab)

        plt.tight_layout()
        plt.show()
    return score

def error(gram, start, end):
    """Return the approximation cost on the segment [start:end].
    Args:
        start (int): start of the segment
        end (int): end of the segment
    Returns:
        segment cost
    Raises:
        NotEnoughPoints: when the segment is too short (less than `min_size` samples).
    """

    sub_gram = gram[start:end, start:end]
    val = np.diagonal(sub_gram).sum()
    val -= sub_gram.sum() / (end - start)
    return val

def decompose(cpt):

    from itertools import tee
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


    p_start  = [a for a, _ in pairwise(cpt)]
    p_end  = [b for _, b in pairwise(cpt)]
    return p_start, p_end
    
def create_descriptors_video(self, mode = 'vanilla', color_type='GRAY', *args, **kwargs):
    
    raise NotImplementedError
                
    height, width, channels = self[0].shape
    out_video =  np.empty([self.n_frames, height, width, channels], dtype = np.uint8)
    
    for i, idx_frame in tqdm(enumerate(range(self.n_frames))):
        video_frame_raw = self[idx_frame]
        if color_type=='GRAY':
            video_frame = video_frame_raw
        if color_type == 'H':
            video_frame = cv2.cvtColor(video_frame_raw, cv2.COLOR_BGR2HSV)[:,:,0]
        elif color_type == 'S':
            video_frame = cv2.cvtColor(video_frame_raw, cv2.COLOR_BGR2HSV)[:,:,1]
        elif color_type == 'V':
            video_frame = cv2.cvtColor(video_frame_raw, cv2.COLOR_BGR2HSV)[:,:,2]

        key_points, _ = self.extract(video_frame, idx_frame)
        #print("{}/{} Size Descriptors: {}".format(i, self.n_frames, descriptors.shape[0]))
        frame_with_kp=cv2.drawKeypoints(video_frame_raw, key_points, video_frame_raw, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_video[i] = frame_with_kp
    
    # Writes the the output image sequences in a video file
    dir_name = os.path.join(OUTPUT_DIR, 'video_descriptors', self.video_name)
    os.makedirs(dir_name, exist_ok=True)
    filename = os.path.join(dir_name,"color{}_{}.avi".format(color_type, self.descriptor_name))

    skvideo.io.vwrite(filename, out_video)
    return

def plot_segmentations(gt_start_frame, gt_stop_frame, pred_start_frame, pred_stop_frame, names=['True Segmentation', 'Predicted Segmentation']):
    
    gt_tuple = [(start_frame, stop_frame-start_frame) for start_frame, stop_frame in zip(gt_start_frame, gt_stop_frame)]
    pred_tuple = [(start_frame, stop_frame-start_frame) for start_frame, stop_frame in zip(pred_start_frame, pred_stop_frame)]
    
    if False:
        fig, ax = plt.subplots(figsize=(30,8))
        ax.broken_barh([gt_tuple[0]], (30, 9), facecolors='tab:green', alpha=.5, label=names[0])
        ax.broken_barh([pred_tuple[0]], (30, 9), facecolors='tab:red', alpha=.5, label=names[1])
        for i, segment in enumerate(gt_tuple[1:]):
            ax.broken_barh([segment], (20-i*10, 9), facecolors='tab:green', alpha=.5)
        for i, segment in enumerate(pred_tuple[1:]):
            ax.broken_barh([segment], (20-i*10, 9), facecolors='tab:red', alpha=.5)
        ax.set_yticks([])
        ax.set_title("{} and {}".format(names[0], names[1]), weight='bold', fontsize=18);ax.set_xlabel('Time [Frame]', weight='bold', fontsize=18);ax.legend()
        plt.show()
    fig, ax = plt.subplots(figsize=(30,8))
    ax.broken_barh(gt_tuple, (20, 9), facecolors='tab:green', alpha=.5, label=names[0])
    ax.broken_barh(pred_tuple, (10, 9),facecolors=('tab:red'), alpha=.5, label=names[1])
    for start_frame, stop_frame in zip(gt_start_frame, gt_stop_frame):
        ax.axvline(x=start_frame, ymin=0.55, ymax=.95, color='k',linestyle='-.')
        ax.axvline(x=stop_frame, ymin=0.55, ymax=.95, color='k',linestyle='-.')

    for start_frame, stop_frame in zip(pred_start_frame, pred_stop_frame):
        ax.axvline(x=start_frame, ymin=0.05, ymax=.47, color='k',linestyle='-.')
        ax.axvline(x=stop_frame, ymin=0.05, ymax=.47, color='k',linestyle='-.')
    ax.set_xlabel('Time [Frame]');ax.set_yticks([])
    ax.set_title("{} and {}".format(names[0], names[1]), weight='bold', fontsize=18);ax.set_xlabel('Time [Frame]', weight='bold', fontsize=18);ax.legend()
    plt.show()

def create_epic_datasets(data_root='/Users/samperochon/EPIC-KITCHENS/', max_number=None, video_name=None):
    from glob import glob
    
    subjects = glob(os.path.join(data_root, '*'))

    for subject in subjects:
        path_videos = glob(os.path.join(subject, 'videos','*'))
        
    if max_number is None:
        max_number = len(path_videos)

    datasets = {}
    for path_video in path_videos[:max_number]:
        if video_name is not None:
            if os.path.basename(path_video)==video_name:
                datasets[os.path.basename(path_video)] = VideoFrameDataset(path_video, device='Tobii', task='EPIC-KITCHENS', D=64, descriptor_name="surf")
        else:
            datasets[os.path.basename(path_video)] = VideoFrameDataset(path_video, device='Tobii', task='EPIC-KITCHENS', D=64, descriptor_name="surf")

    return datasets

def fit_gmm_with_multiple_vides(datasets, K=128, N_frames=5000, mode='vanilla', pyramid_level=0, color_descriptor=False):

    videos_name = list(datasets.keys())
    pipeline = Pipeline(datasets[videos_name[0]], K=K, tau_sample=1, mode=mode, pyramid_level=pyramid_level, color_descriptor=color_descriptor)
    N_per_video = int(N_frames/len(videos_name))

    # Init. of the descriptor array 
    video_frame = datasets[videos_name[0]][0]
    _, descriptors = datasets[videos_name[0]].extract(video_frame, idx=0, mode=mode)
    data_to_fit = descriptors

    for videos_name, dataset in datasets.items():

        random_index = np.random.randint(0, dataset.n_frames, N_per_video)
        for idx in tqdm(random_index):
            video_frame = dataset[idx]
            _, descriptors = dataset.extract(video_frame, idx=idx, mode=mode)
            data_to_fit = np.vstack((data_to_fit, descriptors))


    print("Number of extracted patches: {}".format(data_to_fit.shape[0]))

    # Fitting the GMM 
    pipeline.fv['GRAY'].gmm.fit(data_to_fit)

    # Saving the GMM 
    output_directory = os.path.join(OUTPUT_DIR, 'models', pipeline.dataset.task)
    os.makedirs(output_directory, exist_ok=True)
    output_directory = os.path.join(OUTPUT_DIR, 'models', pipeline.dataset.task, pipeline.dataset.device)
    os.makedirs(output_directory, exist_ok=True)

    filename = os.path.join(output_directory, '{}_{}_D{}_K{}_{}_{}.joblib'.format(pipeline.dataset.device, "all", pipeline.dataset.D, pipeline.fv['GRAY'].K, pipeline.dataset.descriptor_name, 'GRAY'))
    dump(pipeline.fv['GRAY'].gmm, filename)
    print("Saved GMM model in file {}\nGMM is fitted.".format(filename))
    return

    
def join_and_discard(frame, join_len, discard_len, binary_mask):
    """
    This function applies smoothing to frame such that:
    1) Gaps smaller or equal to "join_len" are filled,
    2) Isolated continuous frames with length smaller than "discard_len" are removed
    
    Arguments:
        frame {[int]} -- The indices
        join {int} -- threshold for joining discrete groups
        discard {int} -- threshold for discarding groups
    """
    
    if binary_mask:
        original_size=len(frame)
        frame = np.argwhere(frame==1).squeeze()
        
        
    # print(frame)
    try:
        if len(frame) == 0:
            return np.zeros(original_size)
    except:
        return np.zeros(original_size)
    
    # First join
    frame = sorted(frame)
    joined_frame = []
    prev = frame[0]
    for f in frame[1:]:
        if f > prev + 1 and f < prev + join_len:
            joined_frame.extend(list(range(prev+1, f)))
        prev = f

    frame = sorted(frame + joined_frame)
    # print(joined_frame)
    
    # Then discard
    discard_frame = []
    prev = frame[0]
    island = [prev]
    for f in frame[1:]:
        if f == prev + 1:
            island.append(f)
        else:
            # check island length
            if len(island) < discard_len:
                discard_frame.extend(island)
            island = [f]
        prev = f

    if len(island) < discard_len:
        discard_frame.extend(island)

    # print(discard_frame)
    new_frame = [f for f in frame if f not in discard_frame]
    
    
    if binary_mask:
        new_mask = np.zeros(original_size)
        new_mask[new_frame] = 1
        return new_mask
    
    return new_frame

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def _lauch_umap_fv(df, dataset, mode='jupyterlab'):
    
    plt.figure(figsize=[30,40])
    fig = px.scatter(df, x='emb_x1', y='emb_x2',  hover_name='video_name', color='label', hover_data=['index','time', 'label'],
                        template="simple_white", width=1300, height=700)
    fig.update_traces(hoverinfo="none",hovertemplate=None)

    app = JupyterDash(__name__)
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )
    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]

        nframe_num = hover_data['customdata'][0]
        time = hover_data['customdata'][1]
        label = hover_data['customdata'][2]

        im_matrix = dataset[nframe_num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "500px", 'display': 'block', 'margin': '0 auto'},
                ),

                html.P("Frame number {} Label {}".format(nframe_num, label), style={'font-weight': 'bold'}),
                html.P("Time: {}".format(time), style={'font-weight': 'bold'}),
            ])
        ]

        return True, bbox, children
    app.run_server(mode=mode, port = 8091, height=800, dev_tools_ui=True,  inline_exceptions=False,#debug=True,
                  dev_tools_hot_reload =True, threaded=True)
    return 

def _lauch_umap_segments(df, dataset, mode='jupyterlab'):
    plt.figure(figsize=[30,30])
    fig = px.scatter(df, x='emb_x1', y='emb_x2',  hover_name='video_name', color='label', hover_data=[ 'index','start', 'stop', 'label'], #symbol='marker', 
                        template="simple_white", width=1300, height=800)
    fig.update_traces(hoverinfo="none",hovertemplate=None)


    app = JupyterDash(__name__)
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )    
    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update
        print(hoverData)

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]

        nframe_num = hover_data['customdata'][0]
        start = hover_data['customdata'][1]
        stop = hover_data['customdata'][2]
        label = hover_data['customdata'][3]

        im_matrix = dataset[nframe_num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "500px", 'display': 'block', 'margin': '0 auto'},
                ),

                html.P("Frame {} Label {}".format(nframe_num, label), style={'font-weight': 'bold'}),
                html.P("Start: {}".format(start), style={'font-weight': 'bold'}),
                html.P("Stop: {}".format(stop), style={'font-weight': 'bold'}),
            ])
        ]

        return True, bbox, children
    app.run_server(mode=mode, port = 8091, height=800, dev_tools_ui=True,  inline_exceptions=False,#debug=True,
                  dev_tools_hot_reload =True, threaded=True)
    return 

def extend_labels(labels, cpts, n_fv):
    
    # Create the labels associated to each Fiisher vectors and plot the results 
    fv_labels = np.zeros(n_fv)
    idx_segment = 0
    for idx_fv in range(n_fv):    
        if idx_fv < cpts[idx_segment]:
            fv_labels[idx_fv] = labels[idx_segment]
        else:
            idx_segment+=1
            fv_labels[idx_fv] = labels[idx_segment]
    return fv_labels


def create_frames_label(index_fv_subsampled, index_outliers, n_frames, tau_sample, labels):
    
    idx_full = np.zeros(n_frames)
    counter_outliers = 0
    for i, ii in enumerate(range(0, n_frames, tau_sample)):
        if ii in index_fv_subsampled:
            idx_full[ii:ii+tau_sample] = labels[i-counter_outliers]
        elif ii in index_outliers*tau_sample:
            idx_full[ii:ii+tau_sample] = 0
            counter_outliers+=1
        else:
            print("Something went wrong")
    return idx_full


def spectral_clustering_embedding(embeddings, n_cluster=None, min_clusters=None, max_clusters=None, *args, **kwargs):
    
    from spectralcluster import (
        ICASSP2018_REFINEMENT_SEQUENCE,
        EigenGapType,
        LaplacianType,
        RefinementOptions,
        SpectralClusterer,
        ThresholdType,
    )
    from spectralcluster.utils import (
        compute_affinity_matrix,
        compute_number_of_clusters,
        compute_sorted_eigenvectors,
    )

    refinement_options = RefinementOptions(gaussian_blur_sigma=.4,
                                            p_percentile=0.9,
                                            thresholding_soft_multiplier=0.01,
                                            thresholding_type=ThresholdType.Percentile,
                                            refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE)

    clusterer = SpectralClusterer(  
        min_clusters=min_clusters if min_clusters is not None else MIN_CLUSTERS,
        max_clusters=max_clusters if max_clusters is not None else MAX_CLUSTERS,
        refinement_options=refinement_options,
        laplacian_type=LaplacianType.RandomWalk,
        row_wise_renorm=False,
        max_iter=1000,
        custom_dist="cosine",
        eigengap_type= EigenGapType.Ratio, 
        affinity_function=compute_affinity_matrix)
    
    return clusterer.predict(embeddings)


def spectral_clustering(gram, n_clusters=None, custom_dist='cosine', max_iter=300, *args, **kwargs):
    
    from spectralcluster import EigenGapType, LaplacianType, laplacian
    from spectralcluster.custom_distance_kmeans import run_kmeans
    from spectralcluster.utils import (
        compute_number_of_clusters,
        compute_sorted_eigenvectors,
    )

    # Compute affinity matrix.
    affinity = gram

    # Compute Laplacian matrix
    laplacian_norm = laplacian.compute_laplacian(affinity, laplacian_type=LaplacianType.RandomWalk)

    # Perform eigen decomposion. Eigen values are sorted in an ascending order
    eigenvalues, eigenvectors = compute_sorted_eigenvectors(laplacian_norm, descend=False)


    if n_clusters is None:
        # Get number of clusters. Eigen values are sorted in an ascending order
        n_clusters, max_delta_norm = compute_number_of_clusters(eigenvalues,
                                                            max_clusters=MAX_CLUSTERS,
                                                            eigengap_type=EigenGapType.Ratio,#Ratio,#NormalizedDiff,
                                                            descend=False)
    # Get spectral embeddings.
    spectral_embeddings = eigenvectors[:, :n_clusters]

    if kwargs['row_wise_renorm']:
        # Perform row wise re-normalization.
        rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
        spectral_embeddings = spectral_embeddings / np.reshape(rows_norm, (spectral_embeddings.shape[0], 1))

    # Run clustering algorithm on spectral embeddings. This defaults
    # to customized K-means.
    labels = run_kmeans(spectral_embeddings=spectral_embeddings,
                        n_clusters=n_clusters,
                        custom_dist=custom_dist,
                        max_iter=max_iter)

    return n_clusters, labels

def create_annotated_video(pipeline, add_ruptures = False, add_gt=True, on_flow=False, suffix='segmented'):
    
    # Setting paraneters
    colors = np.concatenate([plt.get_cmap('Set2')(np.arange(0,10))[-1:,:], plt.get_cmap('tab10')(np.arange(0,10))], axis=0)
    height, width, layers = pipeline.dataset[0].shape
    bandwidth=50

    if add_ruptures:
        n_rupt = len(pipeline.cpt)
        length=5
        red_frame = np.concatenate([255*np.ones((height, width))[:,:,np.newaxis], np.zeros((height, width))[:,:,np.newaxis], np.zeros((height, width))[:,:,np.newaxis]], axis=2)
        out_video =  np.empty([pipeline.dataset.n_frames+length*n_rupt, height, width, layers], dtype = np.uint8)
    else:
        out_video =  np.empty([pipeline.dataset.n_frames, height, width, layers], dtype = np.uint8)
        
    # Create the video by iterating on the frames
    cont_rupt, cont_video = 0, 0
    for idx_frame in tqdm(range(pipeline.dataset.n_frames)):

        img = pipeline.dataset[idx_frame]    
        img_out = cv2.rectangle(img, (0, 0), (width, bandwidth), colors[int(pipeline.full_labels[idx_frame])][:-1]*256, -1)
        img_out = cv2.rectangle(img_out, (0, 0), (bandwidth, height), colors[int(pipeline.full_labels[idx_frame])][:-1]*256, -1)
        img_out = cv2.rectangle(img_out, (width-bandwidth, 0), (width, height), colors[int(pipeline.full_labels[idx_frame])][:-1]*256, -1)
        img_out = cv2.rectangle(img_out, (0, height-bandwidth), (width, height), colors[int(pipeline.full_labels[idx_frame])][:-1]*256, -1)
        out_video[cont_video] = img_out
        cont_video+=1
        if add_ruptures and idx_frame == pipeline.cpt[cont_rupt]:
            for _ in range(length):
                out_video[cont_video] = red_frame
                cont_video+=1
            if cont_rupt<n_rupt-1:
                cont_rupt+=1      

    # Create the directory if it does not exists and the associated video
    os.makedirs(os.path.join(OUTPUT_DIR, 'videos_ruptures', pipeline.dataset.video_name), exist_ok=True)
    filename = os.path.join(OUTPUT_DIR,'videos_ruptures', pipeline.dataset.video_name, '{}_{}.avi'.format(pipeline.dataset.video_name, suffix))
    skvideo.io.vwrite(filename, out_video)
    print("Done. You can find the video in : {}".format(filename))
    return

def plot_frames(dataset, frame_list=None, rows=1, cols=3, plot_width=30, plot_height=12, add_descriptors=True):

    if frame_list is None:
        frame_list = np.random.randint(0,dataset.n_frames, rows*cols)
            
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for index, (ax, idx) in enumerate(zip(grid, sorted(frame_list))):
        frame = dataset[idx]
        if add_descriptors:
            key_points, descriptors = dataset.extract(frame, idx)
            frame_with_kp=cv2.drawKeypoints(frame, key_points, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            ax.imshow(frame_with_kp)
            ax.set_title('{} - {}'.format(idx, dataset.idx2time(idx)))
        else: 
            # Iterating over the grid returns the Axes.
            ax.imshow(frame)
            ax.set_title('{} - {}'.format(idx, dataset.idx2time(idx)))
    plt.show()
    return frame_list

def plot(dataset, idx1, idx2=None, title = ''):

    f1 = dataset[idx1]
    if idx2 is not None:
        f2 = dataset[idx2]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle(title, fontsize = 18, weight = 'bold')

    ax1.imshow(f1)
    ax1.set_title('{}%'.format(np.round(100*(idx1/dataset.n_frames))))

    if idx2 is not None:
        ax2.imshow(f2)

        ax2.set_title('{}%'.format(np.round(100*(idx2/dataset.n_frames))))
        
def create_xticklabels(pipeline, array, idxs):
    #idx should be in the reference of the entire video, ie between 0 and self.dataset.n_frames
    xticklabels = [pipeline.idx2time(idxs[0])]
    xticklocation = [0]
    
    for i in range(1, len(array)-1):
        if array[i]!=array[i-1]:
            xticklabels.append(pipeline.idx2time((idxs[0]+i*pipeline.tau_sample)))
            xticklocation.append(i)

    xticklabels.append(pipeline.idx2time(idxs[1]))
    xticklocation.append(len(array)-1)
    
    if len(xticklabels) >20:
        xticklabels = np.array(xticklabels)[range(0, len(xticklabels), len(xticklabels)//20)]
        xticklocation = np.array(xticklocation)[range(0, len(xticklocation), len(xticklocation)//20)]
            
    return xticklabels, xticklocation

def frame_before_after_ruptures(pipeline, array, n=10):
    rupt_mask = np.zeros(pipeline.fisher_vectors.shape[1])
    for i in range(1,len(array)):
        rupt_mask[i] = (array[i]-array[i-1])!=0 
    pos_rupt = np.argwhere(rupt_mask==1)
    pos_rupt_doubled = []
    for i in pos_rupt:
        pos_rupt_doubled.append(i[0]-n)
        pos_rupt_doubled.append(i[0]+n)
    return pos_rupt_doubled

def fi(x=25, y=4):
    return plt.figure(figsize=(x,y))
    
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def compute_segment_cost(pipeline):

    segments = [0] + pipeline.cpt[:-1] + [pipeline.cpt[-1]-1]

    fig, axes = plt.subplots(len(segments)//5+1, 5, figsize=(30, 3*len(segments)//5))
    axes = axes.flatten()
    for i, (idx_start, idx_end) in enumerate(pairwise(segments)):   
        # Compute the cost associated to this segment
        sub_gram = pipeline.gram[idx_start:idx_end, idx_start:idx_end]
        cost = np.diagonal(sub_gram).sum() - sub_gram.sum() / (idx_end - idx_start)

        # Compute the variance of the Fisher Vectors
        var = pipeline.fisher_vectors[:,idx_start:idx_end].var(axis=1).sum()
        
        #Visaulize the heatmap
        sns.heatmap(pd.DataFrame(sub_gram, columns=[pipeline.idx2time(i) for i in range(idx_start, idx_end)], index=[pipeline.idx2time(i) for i in range(idx_start, idx_end)]), ax = axes[i])
        axes[i].set_title("Error cost : {:.2f}|Var: {:.2f}\n{} - {}".format(cost, var, pipeline.idx2time(idx_start), pipeline.idx2time(idx_end)), weight='bold', fontsize=18)
    [ax.axis('off') for ax in axes], plt.tight_layout()
    return

def compute_gathered_segment_cost(pipeline):

    # Gather the indexes of the ruptures that separate gathered segments
    cpts = [0] + pipeline.cpt[:-1]+[pipeline.cpt[-1]-1]
    rupt_scene_idx = [0]
    for i in range(1, len(pipeline.kmeans_label)):
        if pipeline.kmeans_label[i]!= pipeline.kmeans_label[i-1]:
            rupt_scene_idx.append(i)
    rupt_scene_idx.append(len(cpts)-1)

    fig, axes = plt.subplots(len(rupt_scene_idx)//5+1, 5, figsize=(30, 5*len(rupt_scene_idx)//5))
    axes = axes.flatten()
    for i, (start, end) in enumerate(pairwise(rupt_scene_idx)):
            
        idx_start = cpts[start]
        idx_end = cpts[end]

        # Compute the cost associated to this segment
        sub_gram = pipeline.gram[idx_start:idx_end, idx_start:idx_end]
        cost = np.diagonal(sub_gram).sum() - sub_gram.sum() / (idx_end - idx_start)

        # Compute the variance of the Fisher Vectors
        var = pipeline.fisher_vectors[:,idx_start:idx_end].var(axis=1).sum()

        sns.heatmap(pd.DataFrame(sub_gram, columns=[pipeline.idx2time(i) for i in range(idx_start, idx_end)], index=[pipeline.idx2time(i) for i in range(idx_start, idx_end)]), ax = axes[i])
        axes[i].set_title("Error cost : {:.2f}|Var: {:.2f}\n{} - {}".format(cost, var, pipeline.idx2time(idx_start), pipeline.idx2time(idx_end)), weight='bold', fontsize=18)
    [ax.axis('off') for ax in axes], plt.tight_layout()
    return



def repr(object_, indent=0):
    
    import numpy as np
    import seaborn as sns
    
    if indent==0:
        
        print("{0:10}{1:30}\t {2:40}\t {3:150}".format("","Attribute Name", "type", "Value or first element"))
        print("{0:10}-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n".format(""))
    
    for _ in range(indent):
        print("\t")
    
    if not isinstance(object_, dict):
    
        dict_ = object_.__dict__
    else:
        dict_ = object_
    
    for k, o in dict_.items():
        if type(o) == dict:
            print("{0:10}{1:30}\t {2:40}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o)))
            repr(o, indent=indent+1)
        else:
            print("{0:10}{1:30}\t {2:40}\t {3:150}".format(indent*"\t" if indent > 0 else "", k, _print_correct_type(o), _print_correct_sample(o, indent=indent)))
    
    print("\n")
    return 

def _print_correct_sample(o, indent=0):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the element o, taking the type into account.

    """


    
    if o is None:
        return "None"    
    
    elif isinstance(o, (int, float, np.float32, np.float64)):
        return str(o)
    
    elif isinstance(o, str):
        return o.replace('\n', '-') if len(o) < 80 else o.replace('\n', '-')[:80]+'...'
    
    elif isinstance(o, (list, tuple)) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} len: {}".format(str(o[0]), len(o))
    
    elif isinstance(o, np.ndarray) :#and not type(o) == sns.palettes._ColorPalette:
        return "{} shape: {}".format(str(o[0]), str(o.shape))
    
    elif type(o) == dict : 
        return repr(o, indent=indent+1)

    elif type(o) == pd.core.frame.DataFrame:
        return 'dataframe'
    
    else:
        
        return str(o)
    
def _print_correct_type(o):
    """
        This helper function is associated with the show method, used to print properly classes object.
        This one output a string of the type of the element o.

    """
    if o is None:
        return "None"    
    
    elif isinstance(o, int):
        return "int"
    
    elif isinstance(o, float):
        return "float"
    
    elif isinstance(o, np.float32):
        return "np.float32"
    
    elif isinstance(o, np.float64):
        return "np.float64"    
    
    elif isinstance(o, list):
        return "list"
    
    elif isinstance(o, str):
        return "str"
    
    elif isinstance(o, tuple):
        return "tuple"
    
    elif isinstance(o, np.ndarray):
        return "np.ndarray"
    
    elif type(o) == dict : 
        return "dict"
    
    else:
        
        return str(type(o))
        
