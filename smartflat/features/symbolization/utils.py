"""Shared utility functions for the symbolization pipeline.

Provides segment label propagation, prototype extraction, entropy
computation, QC sanity checks, distance normalization, and threshold
estimation used by the recursive prototyping (Ch. 5, Section 5.2)
and symbolization (Section 5.5) stages.

Called by:
    - ``smartflat.features.symbolization.main``
    - ``smartflat.features.symbolization.inference``
    - ``smartflat.features.symbolization.co_clustering``
    - ``smartflat.features.symbolization.run_code``

External dependencies:
    - cv2, decord (video frame access)
    - pyentrp (sample entropy computation)
    - ruptures (change point detection models)
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

import cv2
import numpy as np
import pandas as pd
import ruptures as rpt
from pyentrp import entropy as ent


import matplotlib.lines as mlines

#import umap.umap_ as umap
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from decord import bridge
from IPython.display import display
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize

from smartflat.configs.loader import get_complete_configs, import_config
from smartflat.constants import progress_cols
from smartflat.datasets.filter import filter_progress_cols
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import load_embedding_dimensions
from smartflat.engine.builders import build_model, compute_metrics
from smartflat.utils.utils import pairwise, upsample_sequence
from smartflat.utils.utils_coding import blue, green
from smartflat.utils.utils_dataset import (
    add_cum_sum_col,
    check_train_data,
    co_z_transform,
    collapse_cluster_stats,
    normalize_data,
)
from smartflat.utils.utils import predict_segments_from_embed_labels
from smartflat.utils.utils_io import (
    fetch_output_path,
    fetch_qualification_mapping,
    get_api_root,
    get_data_root,
    get_host_name,
    get_video_loader,
    parse_flag,
    parse_identifier,
    save_df,
)
from smartflat.utils.utils_transform import zscore
from smartflat.utils.utils_visualization import plot_gram


def fix_clinical_diagnosis(df):
    
    
    df['group_folder'] = df['diag_number'].apply(lambda x: 1 if 'P' in x else 0)

    
    df.loc[df['group_folder'] == 0, 'pathologie'] = 'HEALTHY'
    df.loc[df['participant_id'] == 'SC_G130_P111_RICDam_29092023', 'pathologie'] = 'HEALTHY'
    df.loc[df['participant_id'] == 'SC_G130_P111_RICDam_29092023', 'group'] = 0
    df.loc[df['participant_id'] == 'SC_G130_P111_RICDam_29092023', 'group_folder'] = 0
    df.loc[df['participant_id'] == 'G173_MORFab_SDS2_P_M12_V2_19072024', 'pathologie'] = 'TBI'

    df.loc[df['participant_id'] == 'G173_MORFab_SDS2_P_M12_V2_19072024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'SC_G129_P110_DELMAr_27092023', 'pathologie'] = 'HEALTHY'
    df.loc[df['participant_id'] == 'SC_G129_P110_DELMAr_27092023', 'group'] = 0
    df.loc[df['participant_id'] == 'SC_G129_P110_DELMAr_27092023', 'group_folder'] = 0


    df.loc[df['participant_id'] == 'SC_G126_P107_GOUJul_06092023', 'group_folder'] = 0
    df.loc[df['participant_id'] == 'SC_G126_P107_GOUJul_06092023', 'pathologie'] = 'HEALTHY'
    df.loc[df['participant_id'] == 'SC_G126_P107_GOUJul_06092023', 'group'] = 0

    df.loc[df['participant_id'] == 'SC_G133_P114_LEBEmm_05102023', 'group_folder'] = 0
    df.loc[df['participant_id'] == 'SC_G133_P114_LEBEmm_05102023', 'pathologie'] = 'HEALTHY'
    df.loc[df['participant_id'] == 'SC_G133_P114_LEBEmm_05102023', 'group'] = 0

    df.loc[df['participant_id'] == 'G155_VANBru_SDS2_P_M24_V3_24042024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G155_VANBru_SDS2_P_M24_V3_24042024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G155_VANBru_SDS2_P_M24_V3_24042024', 'group'] = 1

    df.loc[df['participant_id'] == 'G156_CONTit_SDS2_P_M12_V2_16052024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G156_CONTit_SDS2_P_M12_V2_16052024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G156_CONTit_SDS2_P_M12_V2_16052024', 'group'] = 1

    df.loc[df['participant_id'] == 'G157_ILIMil_SDS2_P_M12_V2_22052024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G157_ILIMil_SDS2_P_M12_V2_22052024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G157_ILIMil_SDS2_P_M12_V2_22052024', 'group'] = 1


    df.loc[df['participant_id'] == 'G158_LONCat_SDS2_P_M24_V3_29052024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G158_LONCat_SDS2_P_M24_V3_29052024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G158_LONCat_SDS2_P_M24_V3_29052024', 'group'] = 1

    df.loc[df['participant_id'] == 'G159_RAYVia_SDS2_P_M12_V2_31052024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G159_RAYVia_SDS2_P_M12_V2_31052024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G159_RAYVia_SDS2_P_M12_V2_31052024', 'group'] = 1


    df.loc[df['participant_id'] == 'G160_BAIAnn_SDS2_P_M0_V1_12062024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G160_BAIAnn_SDS2_P_M0_V1_12062024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G160_BAIAnn_SDS2_P_M0_V1_12062024', 'group'] = 1

    df.loc[df['participant_id'] == 'G161_AMEAmo_SDS2_P_M24_V3_14062024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G161_AMEAmo_SDS2_P_M24_V3_14062024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G161_AMEAmo_SDS2_P_M24_V3_14062024', 'group'] = 1


    df.loc[df['participant_id'] == 'G162_FAUJea_SDS2_P_M24_V3_19062024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G162_FAUJea_SDS2_P_M24_V3_19062024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G162_FAUJea_SDS2_P_M24_V3_19062024', 'group'] = 1


    df.loc[df['participant_id'] == 'G165_FONWil_SDS2_P_M24_V3_28062024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G165_FONWil_SDS2_P_M24_V3_28062024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G165_FONWil_SDS2_P_M24_V3_28062024', 'group'] = 1


    df.loc[df['participant_id'] == 'G167_BOUAde_SDS2_P_M24_V3_02072024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G167_BOUAde_SDS2_P_M24_V3_02072024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G167_BOUAde_SDS2_P_M24_V3_02072024', 'group'] = 1

    df.loc[df['participant_id'] == 'G171_SABNic_SDS2_P_M0_V1_12072024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G171_SABNic_SDS2_P_M0_V1_12072024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G171_SABNic_SDS2_P_M0_V1_12072024', 'group'] = 1

    df.loc[df['participant_id'] == 'G176_ZIELud_SDS2_P_M12_V2_11092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G176_ZIELud_SDS2_P_M12_V2_11092024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G176_ZIELud_SDS2_P_M12_V2_11092024', 'group'] = 1

    df.loc[df['participant_id'] == 'G177_LUCXav_SDS2_P_M12_V2_13092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G177_LUCXav_SDS2_P_M12_V2_13092024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G177_LUCXav_SDS2_P_M12_V2_13092024', 'group'] = 1

    df.loc[df['participant_id'] == 'G178_PITAla_SDS2_P_M0_V1_18092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G178_PITAla_SDS2_P_M0_V1_18092024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G178_PITAla_SDS2_P_M0_V1_18092024', 'group'] = 1


    df.loc[df['participant_id'] == 'G180_GOGFlo_SDS2_P_M24_V3_20092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G180_GOGFlo_SDS2_P_M24_V3_20092024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G180_GOGFlo_SDS2_P_M24_V3_20092024', 'group'] = 1

    df.loc[df['participant_id'] == 'G181_DAVArc_SDS2_P_M12_V2_25092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G181_DAVArc_SDS2_P_M12_V2_25092024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G181_DAVArc_SDS2_P_M12_V2_25092024', 'group'] = 1

    df.loc[df['participant_id'] == 'G182_RICJon_SDS2_P_M12_V2_27092024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G182_RICJon_SDS2_P_M12_V2_27092024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G182_RICJon_SDS2_P_M12_V2_27092024', 'group'] = 1

    df.loc[df['participant_id'] == 'G184_GOBPao_SDS2_P_M0_V1_03102024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G184_GOBPao_SDS2_P_M0_V1_03102024', 'pathologie'] = 'RIL'
    df.loc[df['participant_id'] == 'G184_GOBPao_SDS2_P_M0_V1_03102024', 'group'] = 1

    df.loc[df['participant_id'] == 'G185_JABSyl_SDS2_P_M12_V2_09102024', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G185_JABSyl_SDS2_P_M12_V2_09102024', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G185_JABSyl_SDS2_P_M12_V2_09102024', 'group'] = 1

    df.loc[df['participant_id'] == 'G31_P20_PETDom_20032018', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G31_P20_PETDom_20032018', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G31_P20_PETDom_20032018', 'group'] = 1


    df.loc[df['participant_id'] == 'G81_P68_GAUSte_08082019', 'group_folder'] = 1
    df.loc[df['participant_id'] == 'G81_P68_GAUSte_08082019', 'pathologie'] = 'TBI'
    df.loc[df['participant_id'] == 'G81_P68_GAUSte_08082019', 'group'] = 1

    return df


def load_mapping_prototypes_reduction(config_name, annotator_id='samperochon', round_number=0):
    
    config = import_config(config_name)
    dict_load_path= os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id,f'round_{round_number}', f'mapping_prototypes_reduction_K_space.json')
    if os.path.exists(dict_load_path):
        with open(dict_load_path) as f:
            print('dict_load_path', dict_load_path)
            mapping_prototypes_reduction_K_space = json.load(f)
    else:
        print(f"File {dict_load_path} does not exist.")
        mapping_prototypes_reduction_K_space = {}
        
    mapping_prototypes_reduction_K_space = {int(k): int(v) for k, v in mapping_prototypes_reduction_K_space.items()}

    
    dict_load_path= os.path.join(get_data_root(), 'outputs', config.experiment_name, config.experiment_id, annotator_id,f'round_{round_number}', f'mapping_prototypes_reduction_G_space.json')
    if os.path.exists(dict_load_path):
        with open(dict_load_path) as f:
            mapping_prototypes_reduction_G_space = json.load(f)
    else:
        print(f"File {dict_load_path} does not exist.")
        mapping_prototypes_reduction_G_space = {}
        
    mapping_prototypes_reduction_G_space = {int(k): int(v) for k, v in mapping_prototypes_reduction_G_space.items()}
    
    
    mapping_prototypes_reduction_G_space = {i_K: mapping_prototypes_reduction_G_space[i_G] for i_K, i_G in mapping_prototypes_reduction_K_space.items() if i_G in mapping_prototypes_reduction_G_space.keys()}
     
    return mapping_prototypes_reduction_K_space, mapping_prototypes_reduction_G_space

def add_pyramid_labels(df, config_name, annotator_id, round_number):
    
    mapping_prototypes_reduction_K_space, mapping_prototypes_reduction_G_space = load_mapping_prototypes_reduction(config_name, annotator_id=annotator_id, round_number=round_number)

    mapping_prototypes_reduction_G_space_r = {}
    for k, v in mapping_prototypes_reduction_G_space.items():
        mapping_prototypes_reduction_G_space_r.setdefault(v, []).append(k)

    mapping_prototypes_reduction_K_space_r = {}
    for k, v in mapping_prototypes_reduction_K_space.items():
        mapping_prototypes_reduction_K_space_r.setdefault(v, []).append(k)
    config = import_config(config_name)
    G_opt_space_path= os.path.join(get_data_root(), 'dataframes', 'annotations', 'pigeon-annotations', config.experiment_name, config.experiment_id, annotator_id,f'round_{round_number}', 'round_8_meta-prototypes_K_100.csv')
    G_opt_space_annotdf = pd.read_csv(G_opt_space_path, sep=';')
    G_space_path = os.path.join(get_data_root(), 'dataframes', 'annotations', 'pigeon-annotations', config.experiment_name, config.experiment_id, annotator_id,f'round_{round_number}', 'round_8_G_space_meta-prototypes_K_100.csv')
    G_space_annotdf = pd.read_csv(G_space_path, sep=';')
    G_space_annotdf['n_cluster'] = G_space_annotdf.path.apply(lambda  x : int(x.split('_')[2]))
    G_opt_qm = G_opt_space_annotdf.set_index('n_cluster')['cluster_type'].to_dict()
    G_qm = G_space_annotdf.set_index('n_cluster')['cluster_type'].to_dict()

    # x = df['filtered_raw_embedding_labels'].iloc[0]

    # for i in x:
    #     print('Investigating index ', i)
    #     if (G_opt_qm[mapping_prototypes_reduction_G_space[i]] == 'valid'):
    #         print(f'Index {i} in raw space maps to {mapping_prototypes_reduction_G_space[i]} in G_opt space which is {G_opt_qm[mapping_prototypes_reduction_G_space[i]]}')
    #     elif (G_qm[mapping_prototypes_reduction_K_space[i]] == 'valid'):
    #         print(f'Index {i} in raw space maps to {mapping_prototypes_reduction_K_space[i]} in G space which is {G_qm[mapping_prototypes_reduction_K_space[i]]}')
    green('Adding column "pyr_filtered_raw_embedding_labels"')
    df['pyr_filtered_raw_embedding_labels'] = df['filtered_raw_embedding_labels'].apply(lambda x: ['G'+str(mapping_prototypes_reduction_G_space[i]) if ((i != -1) and (G_opt_qm[mapping_prototypes_reduction_G_space[i]] == 'valid')) else
                                                                                                'K'+str(mapping_prototypes_reduction_K_space[i]) if ((i != -1) and (G_qm[mapping_prototypes_reduction_K_space[i]] == 'valid')) else
                                                                                                'R'+str(i) if i != -1 else '-1' for i in x])
    
    
    
    col_names_update = ['pyr_segm_embedding_symb_labels', 'pyr_symb_segments_labels',
                    'pyr_symb_cpts', 'pyr_n_cpts_symb', 'pyr_symb_segments_has_separations', 'pyr_symb_n_segmented', 'pyr_symb_segments_length',
                    'pyr_symb_n_embed_changes', 'pyr_symb_percent_embed_changes',
                    'pyr_symb_sum_cpts_withdrawn', 'pyr_symb_percent_cpts_withdrawn']

    df[col_names_update] = df.apply(update_segmentation_from_embedding_labels, axis=1, 
                                    temporal_segmentation_col='raw_cpts', 
                                    embedding_labels_col='pyr_filtered_raw_embedding_labels', 
                                    combine_func_name='majority_voting_inner', 
                                    filter_noise_labels=False, 
                                    verbose=True)
    
    
    return df

def qc_sanity_check_filters(df, do_filter=True, has_video=True, max_identifier=None,  verbose=False):
    
    def filter_variable_around(df, column_name, tol, ideal_value=1, verbose=False):
        """
        Analyze fluctuations around an ideal value in a DataFrame column, visualize with histograms, 
        and return a filtered DataFrame based on the tolerance.

        Parameters:
        - df: DataFrame to process.
        - column_name: The column in the DataFrame to analyze.
        - tol: Tolerance level around the ideal value.
        - ideal_value: The target value around which fluctuations are analyzed (default is 1).
        - verbose: If True, prints details about filtering and plotting.

        Returns:
        - filtered_df: DataFrame containing only the rows within the tolerance range.
        

        Notes:
        - Used for filtering e.g. the modalities that have a maximum cpts percentile outside of 1+/-token length variability/edge effects.
        """

        # Adding a column to categorize within/out of tolerance
        df['in_tolerance'] = ((df[column_name] >= (ideal_value - tol)) & 
                            (df[column_name] <= (ideal_value + tol))).astype(int)

        # Splitting dataframes
        in_tolerance = df[df['in_tolerance'] == 1].copy()
        out_of_tolerance = df[df['in_tolerance'] == 0].copy()

        if verbose:
            print(f"Filtering completed:\n{len(in_tolerance)} values are within the tolerance range.")
            print(f"{len(out_of_tolerance)} values are outside the tolerance range.\n")
            if len(out_of_tolerance) > 0:
                print("Out-of-tolerance values (sample):")
                print(out_of_tolerance[[column_name]].head())

            # Plotting histograms
            plt.figure(figsize=(20, 3))
            plt.hist(df[column_name], bins=100, alpha=0.5, label='All data', color='gray', edgecolor='black')
            plt.hist(in_tolerance[column_name], bins=100, alpha=0.7, label='Within tolerance', color='green', edgecolor='black')
            plt.hist(out_of_tolerance[column_name], bins=100, alpha=0.7, label='Out of tolerance', color='red', edgecolor='black')
            plt.axvline(ideal_value - tol, color='red', linestyle='--', linewidth=1.5, label='Tolerance Lower Bound')
            plt.axvline(ideal_value + tol, color='red', linestyle='--', linewidth=1.5, label='Tolerance Upper Bound')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Column Values with Tolerance Ranges')
            plt.legend()

            plt.show()

        # Returning the filtered DataFrame with the in-tolerance data
        return in_tolerance.drop(columns='in_tolerance')

    print(f"do_filter={do_filter} (N={len(df)})")
    
    # Subsample the dataset
    if max_identifier is not None:
        df = df.iloc[0:max_identifier]
        
    # 2) Exlusion part
    n = len(df)
    if do_filter:
        df.dropna(subset=["fps", 'n_frames'], inplace=True)
        df.n_frames = df.n_frames.astype(int)
        #df.segment_length = df.segment_length.astype(int)
    print(f"Removed {n - len(df)} rows with missing fps or n_frames (N={len(df)})")

    # We exclude for now the ebedding array that don't seem to match in terms of size
    n = len(df)
    df['delta_t_eff'] = df.apply(lambda x: x.n_frames/x.N, axis=1).astype(int)
    if verbose:
        print('df[df[delta_t_eff] != 8')
        display(df[df['delta_t_eff'] != 8].head(3))
        df[df['delta_t_eff'] < 50].delta_t_eff.hist(bins=50)

    #if do_filter:
    print('Participants without effective VBR slices every 8 frames:', len(df[df['delta_t_eff'] != 8]))
    #print('\n'.join((df[df['delta_t_eff'] != 8].participant_id.to_list())))
    #df = df[df['delta_t_eff'] in  8]
    print(f"/!\ Keep {len(df[df['delta_t_eff'] != 8])} rows with delta_t_eff != 8 ie  N_f/N_x != stride (N={len(df)})" )

    # We exclude incomplete embedding computation
    # n = len(df)
    # df["M_is_close"] = df.apply(lambda x: np.isclose(x.N, x.n_idx_embedding, atol=16), axis=1)
    # if do_filter: 
    #     df = df[df["M_is_close"]==True]
    # print(f"Removed {n - len(df)} rows with M != n_idx_embedding")

    # We exlude administrations without labels of the right shape
    # n = len(df)
    # df["n_labels_N_is_close"] = df.apply(lambda x: x.N == x.n_embedding_labels, axis=1)
    # if do_filter:
    #     df = df[df["n_labels_N_is_close"] == True]
    # print(f"Removed {n - len(df)} rows with n_labels_N_is_close != True (N={len(df)})")

    # # For now, we filter out the modalities that have a maximum cpts percentile outside of 1+/-token length variability edge effects.
    # df['cpts_percentiles'] = df['cpts'] / df['N']
    # df['cpts_percentiles'] = df['cpts_percentiles'].apply(lambda x: [min(e, 1) for e in x])
    # df['max_cpts_percentile'] = df['cpts_percentiles'].apply(lambda x: x[-1])
    # if verbose:
    #     df.max_cpts_percentile.hist(bins=50)
    #     plt.xlim([0.95, 1.05])
    # if do_filter:
    #     df = filter_variable_around(df, column_name='max_cpts_percentile', tol=0.05)
        
    # df.drop(columns=['cpts_percentiles', 'max_cpts_percentile'], inplace=True)
    
    if has_video:
        
        # We exclude administrations without videos (for visualization)
        n = len(df)
        df = df[df['has_video']==True]
        print(f"Removed {n - len(df)} rows with without video data")
    print(f"Final from qc_sanity_check_filterse (N={len(df)})")

    return df

# def update_segmentation_from_embedding_labels(row, temporal_segmentation_col='cpts', embedding_labels_col='embedding_labels', combine_func_name='majority_voting', filter_noise_labels=False, verbose=False):
#     row = row.copy()

#     # Compute segment per majority voting within each segments by keeping the noise, edges, or exogeneous labels. 
#     row["pred_segments_labels"] = predict_segments_from_embed_labels(row, embedding_labels_col=embedding_labels_col, temporal_segmentation_col=temporal_segmentation_col, combine_func_name=combine_func_name, filter_noise_labels=filter_noise_labels, verbose=verbose)

#     # Compute n_cpts_raw
#     n_cpts_raw = len(row[temporal_segmentation_col]) if isinstance(row[temporal_segmentation_col], np.ndarray) else np.nan
    
#     # Reduce segment labels
#     row['cpts'], row['opt_segments_labels'], row['segments_has_separations'], row['n_segmented'] = reduce_segments_labels(
#         row, temporal_segmentation_col=temporal_segmentation_col, segments_labels_col='pred_segments_labels'
#     )
    
#     # Print number of removed cpts
#     if verbose:
#         pass#print(f'Row index {row.name:5d}: Removed {row["n_cpts_raw"] - len(row["cpts"])} cpts')
    
#     # Compute final n_cpts and propagate back to embedding space (accounting for segment-level regularization)
#     row['n_cpts'] = len(row['cpts']) if isinstance(row['cpts'], np.ndarray) else np.nan
        
#     row['segm_opt_embedding_labels'] = propagate_segment_labels_to_embeddings(row, temporal_segmentation_col = temporal_segmentation_col, segment_labels_col='pred_segments_labels')
    

#     # Compute the percentage of per-sample labels that have changed after accounting for the segment labels (chnge-point detection)
#     row['n_embed_changes'] = np.sum(row['segm_opt_embedding_labels'] != row[embedding_labels_col])
#     row['percent_embed_changes'] = np.mean(row['segm_opt_embedding_labels'] != row[embedding_labels_col])
    
#     # Compute the number and percentage of segments that have been withdrawn
#     row['sum_cpts_withdrawn'] = np.sum(row.n_segmented)
#     row['percent_cpts_withdrawn'] = row.sum_cpts_withdrawn / n_cpts_raw

#     row['segments_length'] = np.ediff1d(row['cpts'])
    
#     return row[['segm_opt_embedding_labels', 'opt_segments_labels', 'cpts', 'n_cpts', 'segments_has_separations', 'n_segmented', 'segments_length', 'n_embed_changes', 'percent_embed_changes', 'sum_cpts_withdrawn', 'percent_cpts_withdrawn']]

def update_segmentation_from_embedding_labels(row, temporal_segmentation_col='cpts', embedding_labels_col='embedding_labels', combine_func_name='majority_voting', filter_noise_labels=False, verbose=False):
    
    result = {}

    pred_segments_labels = predict_segments_from_embed_labels(
        row, embedding_labels_col=embedding_labels_col, temporal_segmentation_col=temporal_segmentation_col,
        combine_func_name=combine_func_name, filter_noise_labels=filter_noise_labels, verbose=verbose
    )
    n_cpts_raw = len(row[temporal_segmentation_col]) if isinstance(row[temporal_segmentation_col], np.ndarray) else np.nan
    
    cpts, opt_segments_labels, segments_has_separations, n_segmented = reduce_segments_labels(
        {**row, "pred_segments_labels": pred_segments_labels},
        temporal_segmentation_col=temporal_segmentation_col, segments_labels_col='pred_segments_labels', 
        verbose=False
    )
    
    segm_opt_embedding_labels = propagate_segment_labels_to_embeddings(
        {**row, "pred_segments_labels": pred_segments_labels},
        temporal_segmentation_col=temporal_segmentation_col, segment_labels_col='pred_segments_labels'
    )

    result['segm_opt_embedding_labels'] = segm_opt_embedding_labels
    result['opt_segments_labels'] = opt_segments_labels
    result['cpts'] = cpts
    result['n_cpts'] = len(cpts) if isinstance(cpts, np.ndarray) else np.nan
    result['segments_has_separations'] = segments_has_separations
    result['n_segmented'] = n_segmented
    result['segments_length'] = np.ediff1d(cpts)
    
    result['n_embed_changes'] = np.sum(segm_opt_embedding_labels != row[embedding_labels_col])
    result['percent_embed_changes'] = np.mean(segm_opt_embedding_labels != row[embedding_labels_col])
    
    result['sum_cpts_withdrawn'] = np.sum(n_segmented)
    result['percent_cpts_withdrawn'] = result['sum_cpts_withdrawn'] / n_cpts_raw

    return pd.Series(result)


def propagate_segment_labels_to_embeddings(row, temporal_segmentation_col = 'cpts', segment_labels_col='segments_labels'):
    """Row-function that propagates segment labels to the point-wise embedding labels within each segment."""

    embedding_labels = np.empty_like(row["raw_embedding_labels"], dtype=object)


    for segment_index, (i, j) in enumerate(pairwise(row[temporal_segmentation_col])):
        i, j = int(i), int(j)
        embedding_labels[i:j] = row[segment_labels_col][segment_index]
  
    return np.asarray(embedding_labels)

def propagate_embeddings_labels_to_frames(df, embed_labels_col='refined_embedding_labels', return_detailed_outputs=False):

    def calculate_frame_labels_vote(row):
        """Calculate the votes for each frame based on the embedding labels."""

        segment_length = row.segment_length
        n_frames = row.n_frames
        embedding_labels = row[embed_labels_col]
        n_embeddings = row.n_embedding_labels

        frame_votes = np.zeros((n_frames, n_clusters_global), dtype=int)

        for start_frame, embedding_label in zip(
            row.idx_embedding, row[embed_labels_col]
        ):
            end_frame = start_frame + segment_length
            if end_frame > n_frames:
                end_frame = n_frames

            for frame in range(start_frame, end_frame):
                # print(f'Frame {frame} embeding label: {embed_labels_col}')
                frame_votes[frame][embedding_label] += 1

        return frame_votes

    def assign_frame_labels_and_agreement(frame_votes):
        """Assign labels to frames based on majority voting and compute agreement."""
        n_frames, n_labels = frame_votes.shape
        frame_labels = np.zeros(n_frames, dtype=int)
        agreement = np.zeros(n_frames, dtype=bool)

        for frame in range(n_frames):

            votes = frame_votes[frame]

            most_common_label = np.argmax(votes)
            count_for_majority = votes[most_common_label]
            num_votes = votes.sum()

            # Assign the most common label to the frame
            frame_labels[frame] = most_common_label

            # Check agreement: all labels must be the same for agreement
            agreement[frame] = int(count_for_majority == num_votes)

            # print(f'Frame {frame}, most_common_label={most_common_label} count_for_majority={count_for_majority}, num_votes={num_votes}, agreement[frame]={agreement[frame]}')

        return frame_labels, agreement

    n_clusters_global = df[embed_labels_col].apply(max).max() + 1
    
    results = pd.DataFrame()
    results['frame_labels_vote'] = df.apply(calculate_frame_labels_vote, axis=1)
    results['frame_labels'], results['agreement'] = zip(*results.frame_labels_vote.apply(assign_frame_labels_and_agreement))
    results['n_frame_labels'] = results.frame_labels.apply(lambda x: len(x) if not np.isnan(x).all() else np.nan)
    if return_detailed_outputs:
        return results
    else:
        return results.frame_labels

def retrieve_segmentation_costs(df, input_space, config_name='ChangePointDetectionDeploymentConfig', temporal_segmentation_col = 'cpts', penalty_column = 'lambda_0', annotator_id='samperochon', round_number=0, include_segments=True, suffix=''):

    def retrieve_segmentation_costs_group(group, config_name='ChangePointDetectionDeploymentConfig'):
        
        config = import_config(config_name)
        

        if config.dataset_name == 'video_block_representation':
            
            signal = np.load(group.iloc[0].video_representation_path).astype(np.double)
            signal = signal[group.test_bounds.iloc[0][0]:group.test_bounds.iloc[0][1]]
            
        elif config.dataset_name == 'prototypes_trajectory_representation':
            
                
            # Add paths to the distance matrix
            inference_config_name = config.dataset_params['config_name']
            inference_config = import_config(inference_config_name)
            
            experiment_folder = os.path.join(get_data_root(), 'experiments', inference_config.experiment_name, inference_config.experiment_id,  inference_config.config_name)
            group['M_path'] = group.apply(lambda x: os.path.join(experiment_folder, f'{inference_config_name}_distance_matrix_{x.identifier}.npy'), axis=1)
            r_experiment_folder = os.path.join(get_data_root(), 'experiments', inference_config.experiment_name, inference_config.experiment_id, annotator_id, f'round_{round_number}', 'distance_matrices_reductions'); 
            if input_space == 'K_space':
                group['M_reduced_path'] = group.apply(lambda x: os.path.join(r_experiment_folder, f'{inference_config_name}_reduced_distance_matrix_{x.identifier}.npy'), axis=1)
            elif input_space == 'G_space':
                group['M_reduced_path'] = group.apply(lambda x: os.path.join(r_experiment_folder, f'{inference_config_name}_reduced_G_space_distance_matrix_{x.identifier}.npy'), axis=1)
            
            if config.dataset_params['use_reduced_distance_matrix']:
                
                signal = np.load(group.iloc[0].M_reduced_path).astype(np.double)
                
            else:
                signal = np.load(group.iloc[0].M_path).astype(np.double)
                
        signal = zscore(signal)
        
        if config.model_params['kernel'] == 'rbf':
            config.model_params['gamma'] = group.iloc[0].gamma
        model = build_model(config.model_name, config.model_params)
        c = model.cost.fit(signal)
        # Compute sum_of_costs for each row
        if include_segments:
            group[f'segments_fit_cost{suffix}'] = group[temporal_segmentation_col].apply(lambda cpts: np.array([model.cost.error(start, end) for (start, end) in pairwise(np.array(cpts).astype(int))]))
            
        group[f'fit_cost{suffix}'] = group[temporal_segmentation_col].apply(lambda cpts: c.sum_of_costs(np.array(cpts).astype(int)))
        group[f'reg_cost{suffix}'] = group.apply(lambda x: x[penalty_column] / np.log(x.N) * (len(x[temporal_segmentation_col]) -2) , axis=1)
        group[f'total_cost{suffix}'] = group.apply(lambda x: x[f'fit_cost{suffix}'] + x[f'reg_cost{suffix}'], axis=1)
        return group
    
    config = import_config(config_name)
    
    #columns_to_use = ['identifier', 'video_representation_path', temporal_segmentation_col, penalty_column, 'N', 'gamma']

    # Process only relevant columns, keeping the original columns aside
    #relevant_df = df[columns_to_use]
    df = df.groupby('identifier', group_keys=False).apply(retrieve_segmentation_costs_group, config_name=config_name)
    
    
    # resultdfs_standard = df.reset_index(drop=False)
    # computed_df = computed_df.reset_index(drop=True)

    # Merge the computed columns back into the original DataFrame
    # for col in [f'fit_cost{suffix}', f'reg_cost{suffix}', f'total_cost{suffix}']:
    #     df[col] = computed_df.get(col)
    # if include_segments:
    #     df[f'segments_fit_cost{suffix}'] = computed_df.get(f'segments_fit_cost{suffix}')

    return df#.groupby('identifier', group_keys=False).apply(retrieve_segmentation_costs_group)
    
def reduce_segments_labels(row, temporal_segmentation_col='cpts', segments_labels_col='segments_labels', verbose=False):
    reduced_cpt, reduced_segments_label, segmented_segments, _, n_segmented = prune_segmentation(row[temporal_segmentation_col], row[segments_labels_col], return_all=True, verbose=verbose)
    return pd.Series([reduced_cpt, reduced_segments_label, segmented_segments, n_segmented])

def prune_segmentation(cpts, segments_label, return_all=True, verbose=True):
    """
    This function taskes as input an array of change points and labels, and prune it (remove similar consecutive labels), while recording whether
    the segment were discontinued or not (contained change-points or not), as this could help to characterize the segments.
    The cpts and segments lavbels must correspond as of labels delineations.

    """
    new_cpt = []
    new_segments_label = []
    segmented_segments = []
    n_segmented = []
    prev_label = None
    segmented = False
    
    assert len(cpts) == len(segments_label) + 1, f"Number of cpts {len(cpts)} should be equal to number of segments {len(segments_label)} + 1"

    for (i, j), label in zip(pairwise(cpts), segments_label):
        if prev_label is None:

            prev_label = label
            new_cpt.append(i)
            new_segments_label.append(label)
            n=0
            
            # /!\ We add at the end the last n_segmented
            
            
            # t = "add {} to the list of cpts, label {} the segment is segmented: {}, ".format(i, label, segmented)

        else:

            if label == prev_label:
                segmented = True
                n += 1
                # t = "add nothing to the list of cpts, label {} was there, segment is segmented: {}, ".format(label, segmented)

                pass

            else:
                # t = "add {} to the list of cpts, label {} the segment is segmented: {}, ".format(i, label, segmented)
                new_cpt.append(i)
                new_segments_label.append(label)
                segmented_segments.append(segmented)
                n_segmented.append(n)

                segmented = False
                n=0
                
        

        prev_label = label
    new_cpt.append(j)
    segmented_segments.append(segmented)
    n_segmented.append(n)

    new_cpt = np.array(new_cpt).astype(int)
    new_segments_label = np.array(new_segments_label)#.astype(int)
    segmented_segments = np.array(segmented_segments)#.astype(int)
    n_segmented = np.array(n_segmented).astype(int)    

    df_raw_segmentation = pd.DataFrame(
        {
            "start": cpts[:-1],
            "end": cpts[1:],
            "label": segments_label,
        }
    )

    df_new_segmentation = pd.DataFrame(
        {
            "start": new_cpt[:-1],
            "end": new_cpt[1:],
            "label": new_segments_label,
            "segmented": segmented_segments,
            'n_segmented': n_segmented
        }
    )

    if verbose:
        print("Raw segmentation")
        display(df_raw_segmentation.T)
        print("Pruned segmentation")
        display(df_new_segmentation.T)
    
    if return_all:
        
        return new_cpt, new_segments_label, segmented_segments, df_new_segmentation, n_segmented
    else:
        return new_cpt, new_segments_label, segmented_segments,

def compute_cluster_probabilities(df, labels_col='segments_labels'):
    """
    df['p_embedding_labels'] = compute_cluster_probabilities(df, labels_col='embedding_labels')
    """
    def calculate_distribution(segments_labels, K):
        labels = np.array(segments_labels).astype(int)
        counts = np.bincount(labels, minlength=int(K))
        probabilities = counts / counts.sum()
        p_x = probabilities[labels]
        return p_x / p_x.sum()
    return  df.apply(lambda row: calculate_distribution(row[labels_col], K=row['n_cluster']), axis=1)

#def get_prototypes(config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', cluster_types=None, return_counting=False, normalization=None, annotator_id='samperochon', round_number=0, verbose=False):
def get_prototypes(config_name='ClusteringDeploymentKmeansCombinationInferenceI5Config', cluster_types=None, return_counting=False, normalization=None, annotator_id='samperochon', round_number=0, verbose=False):

    config = import_config(config_name)
    
    if config.task_type == 'clustering':
        
        modality = config.model_params['modality']
        task_name = config.model_params['task_name']
        experiment_folder = os.path.join(get_data_root(),"experiments", config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}')
        prototypes = np.load(os.path.join(experiment_folder, f'{task_name}_{modality}_cluster_centers.npy'))
        clusters_path = os.path.join(experiment_folder, f'{task_name}_{modality}_cluster_centers.npy')
        print(f'Loading all prototypes from {clusters_path}, shape: {prototypes.shape}')
        #print(f'Return clustering prototype, shape: {prototypes.shape}')
        return prototypes
        
        
    elif config.task_type == 'symbolization':
        
        clustering_config_name = config.clustering_config_name
        clustering_config = import_config(clustering_config_name)
        modality = clustering_config.model_params['modality']
        task_name = clustering_config.model_params['task_name']
        if normalization is None:
            normalization = clustering_config.model_params['normalization']
         
        
    if return_counting:
        count_prototypes_types = []
            
    
    # if cluster_types is None:

    #     clusters_path = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, annotator_id, f'round_{round_number}',  f'{task_name}_{modality}_cluster_centers.npy')
    #     prototypes = np.load(clusters_path)
    #     print(f'Loading all prototypes from {clusters_path}, shape: {prototypes.shape}')
    #     prototypes = normalize_data(prototypes, normalization=normalization)
        
    #     if return_counting:
    #         return prototypes, [len(prototypes)]
    #     else:
    #         return prototypes
         
    
    # if not isinstance(cluster_types, list):
    #     cluster_types = [cluster_types]
    
    
    qualification_mapping = fetch_qualification_mapping(verbose=False)
    
    if clustering_config_name == 'ClusteringDeploymentKmeansInferenceConfig':
            

        input_clustering_config_names, round_numbers_list, qualification_cluster_types_list = get_complete_configs(round_number=round_number, inference_mode=True)
        # print(f"input_clustering_config_names:\n\t{input_clustering_config_names}\n"
        #         f"round_numbers:\n\t{round_numbers_list}\n"
        #         f"qualification_cluster_types:\n\t{qualification_cluster_types_list}")
    
    else:
        
        input_clustering_config_names = [clustering_config_name]
        round_numbers_list = [round_number]
        qualification_cluster_types_list = [cluster_types]
          

    centroids_list = []
    index_hook_list = []
    round_hook = []
    
    # For each clustering (ROUND >1), assume that each config has the concatenation of 
    # previous centroids and the new ones, and that the qualification mapping contain the mutual explusive 
    # partitioning for all types (i.e the sum of entries (cardinal) is equal to the definitive centroids from the previous
    # clustering, plus the new centrroids brought by a rfinement step of the algorithm))
    
    # Here we DONt Need the associated qualification mapping per config, since it is user-overwritten (actuallly the essence of this method?)
    for input_clustering_config_name, qualification_cluster_types, inner_round_number in zip(input_clustering_config_names, qualification_cluster_types_list, round_numbers_list):
        
        input_config = import_config(input_clustering_config_name)
        cl_input_config = input_config.clustering_config_name
        cl_config = import_config(cl_input_config)
        clusters_path = os.path.join(get_data_root(), 'experiments', cl_config.experiment_name, cl_config.experiment_id, annotator_id, f'round_{inner_round_number}', f'{task_name}_{modality}_cluster_centers.npy')
        centroids = np.load(clusters_path)
        c1 = []
        
        if verbose:
            print(f'Initial centroids shape: {centroids.shape}')
        
        
        if input_clustering_config_name in qualification_mapping.keys() and f'round_{inner_round_number}' in qualification_mapping[input_clustering_config_name][annotator_id].keys():
        
            for cluster_type in qualification_cluster_types:
                
                if cluster_types is not None and (cluster_type not in cluster_types):
                    continue
                
                # # Todo: remove after v2 THIBAUTMARIE
                if (config_name == 'ClusteringDeploymentKmeansCombinationConfig') and (cl_input_config == 'ClusteringDeploymentKmeansConfig') and (cluster_type != 'Noise'):
                    continue
                
                if cluster_type == 'Noise':
                    continue

                c = centroids[np.array(qualification_mapping[input_clustering_config_name][annotator_id][f'round_{inner_round_number}'][cluster_type])]
                if return_counting:
                    index_hook_list.append({'cluster_type': cluster_type, 
                                            'round_number': inner_round_number,
                                            'n_clusters': len(c),
                                            'config_name': input_clustering_config_name})
                    
                c1.append(c)
                
                if verbose:
                    print(f'\t{cluster_type}: shape: {c1[-1].shape}')
            
            if len(c1) > 0:
                c1 = np.concatenate(c1, axis=0)
                c1 = normalize_data(c1, normalization=normalization)
                #print(f'Final sub-clustering centroids shape: {c1.shape}')
                centroids_list.append(c1)
                
            else:
                print(f'/!\ No prototypes sampled from {cl_input_config} with cluster types {qualification_cluster_types}')
        else:
            if verbose:
                print(f'Skipping {cl_input_config} with cluster types {qualification_cluster_types} and round {inner_round_number} as no qualification mapping found')
            continue
            
    prototypes = np.concatenate(centroids_list, axis=0)
    prototypes = normalize_data(prototypes, normalization=normalization)
    
    if verbose:
        print(f'Cumulated model shape: {prototypes.shape}')

    if return_counting:
        return prototypes, index_hook_list
    else:
        return prototypes


def compute_threshold_per_cluster_index(df, n_sigma=1, source_embedding_labels_col='embedding_labels', thresholds_path=[], verbose=False):
    
    source_embedding_labels = np.unique(np.hstack(df[source_embedding_labels_col]))
    # Initialize an empty dictionary to store the threshold values
    threshold_mapping = {}
    for l_x in source_embedding_labels:
        
        # Get distances for the current label
        cluster_distances = np.hstack(df.apply(lambda x: [d for i, d in enumerate(x['cluster_dist']) if x[source_embedding_labels_col][i] == l_x] , axis=1))
        # all_distances = []

        # for idx, row in df.iterrows():
        #     dists = row['cluster_dist']
        #     labels = row[source_embedding_labels_col]

        #     if len(dists) != len(labels):
        #         print(f" Row {idx}: Mismatched lengths — {len(dists)} distances vs {len(labels)} labels")
        #         continue  # or raise Exception if you prefer to stop

        #     filtered = [d for i, d in enumerate(dists) if labels[i] == l_x]
        #     all_distances.extend(filtered)
        # cluster_distances = np.array(all_distances)

        # Compute statistics
        mean_val = np.mean(cluster_distances)
        std_val = np.std(cluster_distances)
        threshold = mean_val + n_sigma * std_val  # Compute threshold
        if verbose:
            print(f"Cluster {l_x}: Threshold (mean + {n_sigma} * std): {threshold}")
        
        threshold_mapping[l_x] = threshold

    if len(thresholds_path) > 0:
        for thresholds_path in thresholds_path:
            if not os.path.exists(os.path.dirname(thresholds_path)):
                os.makedirs(os.path.dirname(thresholds_path))
            save_df(threshold_mapping, thresholds_path)
            green(f'Saved threshold_mapping with n_sigma={n_sigma} to {thresholds_path}')
        
        plt.figure(figsize=(10, 3))
        plt.hist(threshold_mapping.values(), bins=100, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Threshold Values (K={len(threshold_mapping)})', fontsize=16, weight='bold')
        plt.xlabel('Threshold Value', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


    
    return threshold_mapping
# def get_centroids_type(centroids, qualification_mapping, cluster_type):
    
#     return centroids[np.array(qualification_mapping[cluster_type])]

def compute_sample_entropy(row, symbolization_col='segments_labels', sample_length=4):
    S_M = row[symbolization_col]
    std_ts = np.std(S_M)
    return ent.sample_entropy(S_M, sample_length, 0.2 * std_ts)[0]

def compute_labeled_cosine_similarity(x, P, labels=None):
    
    N, D = np.shape(X)
    #print('Initial shapes (NxD) and KxD:', embeddings.shape, cluster_centers.shape)
    if prototypes.shape[1] == D+1:
        #print('Removing temporal information from the centroid clusters') 
        prototypes = prototypes[:, :-1]
        
    distances = pairwise_distances(X, prototypes, metric='cosine')

    est_labels = np.argmin(distances, axis=1)
    
    if labels is not None:
        print(f'Cosine Sanity check: estimated labels == raw_embedding_labels ? : {np.allclose(est_labels, labels)}')
    
    return np.min(distances, axis=1)

def compute_labeled_euclidean_distance(X, prototypes, labels):

    N, D = np.shape(X)
    #print('Initial shapes (NxD) and KxD:', embeddings.shape, cluster_centers.shape)
    if prototypes.shape[1] == D+1:
        #print('Removing temporal information from the centroid clusters') 
        prototypes = prototypes[:, :-1]

    distances = pairwise_distances(X, prototypes, metric='euclidean')
        
    # Compute 1st neireast neighborpartiitoning/Voronoi regions assignement and associated strength () and clusyer 
    est_labels = np.argmin(distances, axis=1)
    if not np.allclose(est_labels, labels):
        print(f'Euclidean Sanity check: estimated labels == raw_embedding_labels ? : {np.allclose(est_labels, labels)}')
        print(f'Estimated labels: {est_labels[:50]}')
        print(f'Raw embedding labels: {labels[:50]}')
        #raise ValueError('Estimated labels do not match raw embedding labels')
    
    return np.min(distances, axis=1)

def normalize_distance(x, cluster_stats, kernel_name):
    normalized = []
    for i, d in enumerate(x[f'cluster_dist_{kernel_name}']):
        c_min, c_max = cluster_stats[x['raw_embedding_labels'][i]]

        # Ensure non-zero denominator
        range_val = c_max - c_min + 1e-6
        norm_d = (d - c_min) / range_val

        # Clip extreme values to avoid exploding values
        #norm_d = np.clip(norm_d, -0.5, 1.5)  # Allows a bit above 1 for robustness

        normalized.append(norm_d)
    return normalized


    
def compute_centroid_distances_row(row, config_name, X, prototypes, kernel_name = 'euclidean', sigma_rbf = None, normalize_per_cluster=True, normalization='identity'):

    config = import_config(config_name)

    if config.personalized:

        X = np.load(row.video_representation_path)
        prototypes = np.load(os.path.join(experiment_folder, f'{row.identifier}_cluster_centers.npy'))

        # Normalize data
        if normalization == 'Z-score':
            X, prototypes = co_z_transform(X, prototypes)
        else:
            # Normalize data
            prototypes = normalize_data(prototypes, normalization=normalization)
            X = normalize_data(X, normalization=normalization)
            
    # 1) Get raw inference/clustering labels
    labels = row['raw_embedding_labels']

    if kernel_name == 'euclidean':
        
        distances = compute_labeled_euclidean_distance(X, prototypes, labels)
        
    elif kernel_name == 'cosine':

        distances = compute_labeled_cosine_similarity(X, prototypes, labels)
    
    elif kernel_name == 'gaussian_rbf':
        
        if config.use_reduced_distance_matrix:
            
            experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id, 'distance_matrices_reduction')
            M_path = os.path.join(experiment_folder, f'{config_name}_reduced_distance_matrix_{row.identifier}.npy')
            
        else:
                            
            experiment_folder = os.path.join(get_data_root(), 'experiments', config.experiment_name, config.experiment_id,  config_name)
            M_path = os.path.join(experiment_folder, f'{config_name}_distance_matrix_{row.identifier}.npy')

        if os.path.isfile(M_path):
            
            distances = list(np.min(np.load(M_path), axis=1))
        else:
            
            print(f"Distance matrix file not found: {M_path}")
            distances = np.nan
    
    else: 
        raise ValueError(f"Unknown distance metric {kernel_name}")
    
    return distances

def compute_centroid_distances(df, config_name, normalize_per_cluster=True, normalization=None, verbose=False):


    print(f'Computing distances with {config_name}')    
    config = import_config(config_name)
    co_normalization = config.co_normalization
    
    # Note that we use the kernel_name from the symbolization but other attributes are bounded to tthe clustering config
    if config.task_type == 'clustering':
        
        clustering_config_name = config_name
        clustering_config = import_config(clustering_config_name)
        cluster_types = clustering_config.model_params['cluster_types']
        kernel_name = clustering_config.model_params['kernel_name']
        if normalization is None:
            normalization = clustering_config.model_params['normalization']

        
    elif config.task_type == 'symbolization':  
        
        clustering_config_name = config.clustering_config_name
        clustering_config = import_config(clustering_config_name)
        cluster_types = clustering_config.model_params['cluster_types']
        kernel_name = config.model_params['kernel_name']
        
    else:
        raise ValueError(f"Unknown task type {config.task_type}")

    P = get_prototypes(clustering_config_name, cluster_types=cluster_types, verbose=verbose)
    X_train = np.vstack([np.load(p) for p in df.video_representation_path])
    
    if co_normalization:
        # Joint Z-transformation
        X_P_concat = np.vstack([X_train, P])
        X_P_concat_norm = check_train_data(normalize_data(X_P_concat, normalization='Z-score'))
        X_train = X_P_concat_norm[:X_train.shape[0]]
        P = X_P_concat_norm[-P.shape[0]:]
    
    else:
        # Z-transformation of the samples
        X_train = check_train_data(normalize_data(X_train, normalization=normalization))
        P = check_train_data(normalize_data(P, normalization=normalization))
    
    if kernel_name == 'gaussian_rbf':
        M_eucl = pairwise_distances(X_train, P, metric='euclidean')
        sigma_rbf = np.median(M_eucl[np.triu_indices_from(S, k=1)]); gamma = 1 /  (2 * sigma_rbf ** 2); blue(f'Sigma={sigma_rbf} Gamma: {gamma}')
    else:
        sigma_rbf = None
        

    # Define X slice for each participant
    if 'N' not in df.columns:
        df = load_embedding_dimensions(df)
        
    df['X_all_sbj_slice'] = add_cum_sum_col(df, col='N') 
    
    # Compute distances & labels using inference_projection (re-uses project_onto_prototype_components)
    df['cluster_dist'] =  df.apply(lambda row: compute_centroid_distances_row(row, config_name, X_train[row.X_all_sbj_slice[0]:row.X_all_sbj_slice[1]], P, sigma_rbf=sigma_rbf, normalization=normalization), axis=1)
    df['raw_cluster_dist'] =  df['cluster_dist'].copy()

    # Normalize per-cluster statistics if required
    if normalize_per_cluster:
        
        df[f'cluster_dist_{kernel_name}'] = df['cluster_dist']
        
        cluster_values = np.unique(np.hstack(df['raw_embedding_labels']))
        print('cluster_values', cluster_values[:5])
        cluster_stats = {}
        for cluster in cluster_values:
            cluster_distances = np.hstack(df.apply(lambda x: [d for i, d in enumerate(x[f'cluster_dist_{kernel_name}']) if x['raw_embedding_labels'][i] == cluster] , axis=1))
            #print(cluster, len(cluster_distances))
            if len(cluster_distances) > 0:
                cluster_stats[cluster] = np.percentile(cluster_distances, [0, 98])  # Use 1st & 98th percentile
            else:
               raise ValueError(f"Cluster {cluster} has no distances")
        
        df['cluster_dist'] = df.apply(lambda x: normalize_distance(x, cluster_stats, kernel_name), axis=1)

        #df.drop(columns=[f'cluster_dist_{kernel_name}'], inplace=True)

    return df

def gaussian_rbf_matrix(X, Y, sigma, temperature=1, metric='cosine'):
    """
    Computes the Gaussian RBF kernel matrix between two sets of points.
    In our work sigma is computd as the median euclidean distance across the dataset, around 53 (after Z-normalizaton of both Prototypes and training sampels)"""
    
    if metric == 'cosine':
        # Normalize to unit vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        # Compute cosine similarity and scale
        cos_sim = X_norm @ Y_norm.T
        scaled = cos_sim / temperature
        return np.exp(scaled)

    elif metric == 'euclidean':
        X_sq = np.sum(X**2, axis=1)[:, None]  # Shape (m, 1)
        Y_sq = np.sum(Y**2, axis=1)[None, :]  # Shape (1, n)
        distances_sq = X_sq + Y_sq - 2 * X @ Y.T  # Squared Euclidean distance
        return np.exp(-distances_sq / (2 * temperature *  sigma ** 2))


def reduce_similarity_matrix_per_cluster_type(D_K, mapping_prototypes_reduction, mapping_index_cluster_type, agg_fn=np.mean, excluded_clusters={}, cluster_type=None):
    """Compute the distance matrix for the dataset based on the prototypes mapping from source to meta-prototypes."""
    
    K = D_K.shape[0]

    # Bucket indices by group
    groups = defaultdict(list)
    for idx, g in mapping_prototypes_reduction.items():
        groups[g].append(idx)
    groups_ct = defaultdict(list)
    for idx, g in mapping_index_cluster_type.items():
        groups_ct[g].append(idx)
        
    for cluster_type, excluded_cluster in excluded_clusters.items():
        for ec in excluded_cluster:
            try:
                groups_ct[cluster_type].remove(ec)
            except:
                print(f'/!\ Cluster {ec} not found in {cluster_type}')
                pass

    try:
        groups_ct['Noise'].pop(-1)
    except:
        pass
    
    
    Gt = len(groups_ct['task-definitive'])
    Ge = len(groups_ct['exo-definitive'])
    Gn = len(groups_ct['Noise']) 
    Gf = len(groups_ct['foreground'])

    D_xtr = np.zeros((Gt, Gt))
    D_xer = np.zeros((Ge, Ge))
    D_xnr = np.zeros((Gn, Gn))
    D_xfr = np.zeros((Gf, Gf))

    for i, i_meta in enumerate(groups_ct['task-definitive']):
        for j, j_meta in enumerate(groups_ct['task-definitive']):
            #print(f'i_meta: {i_meta}, j_meta: {j_meta}')
            # if (i_meta in excluded_clusters['task-definitive']) or (j_meta in excluded_clusters['task-definitive']):
            #     continue
            rows = groups[i_meta]
            cols = groups[j_meta]
            values = D_K[np.ix_(rows, cols)]
            D_xtr[i, j] = agg_fn(values)
            
    for i, i_meta in enumerate(groups_ct['exo-definitive']):
        for j, j_meta in enumerate(groups_ct['exo-definitive']):

            # if (i_meta in excluded_clusters['exo-definitive']) or (j_meta in excluded_clusters['exo-definitive']):
            #     continue
            rows = groups[i_meta]
            cols = groups[j_meta]
            values = D_K[np.ix_(rows, cols)]
            D_xer[i, j] = agg_fn(values)


    for i, i_meta in enumerate(groups_ct['Noise']):
        for j, j_meta in enumerate(groups_ct['Noise']):

            if (i_meta == -1) or (j_meta == -1):
                continue
            rows = groups[i_meta]
            cols = groups[j_meta]
            values = D_K[np.ix_(rows, cols)]
            D_xnr[i, j] = agg_fn(values)

    for i, i_meta in enumerate(groups_ct['foreground']):
        for j, j_meta in enumerate(groups_ct['foreground']):

            if (i_meta == -1) or (j_meta == -1):
                continue
            rows = groups[i_meta]
            cols = groups[j_meta]
            values = D_K[np.ix_(rows, cols)]
            D_xfr[i, j] = agg_fn(values)



    #print(f'Shape of D_xtr: {D_xtr.shape}, D_xer: {D_xer.shape}, D_xnr: {D_xnr.shape}')
    
    return D_xtr, D_xer, D_xnr, D_xfr

def reduce_centroid_vectors_per_cluster_type(mu, mapping_prototypes_reduction, mapping_index_cluster_type, agg_fn=np.mean, excluded_clusters={}):
    
    
    K, D = mu.shape

    # Bucket indices by group
    groups = defaultdict(list)
    for idx, g in mapping_prototypes_reduction.items():
        groups[g].append(idx)
    groups_ct = defaultdict(list)
    for idx, g in mapping_index_cluster_type.items():
        groups_ct[g].append(idx)
        
    for cluster_type, excluded_cluster in excluded_clusters.items():
        for ec in excluded_cluster:
            try:
                groups_ct[cluster_type].remove(ec)
            except:
                print(f'/!\ Cluster {ec} not found in {cluster_type}')
                pass

    try:
        groups_ct['Noise'].pop(-1)
    except:
        #print('/!\ No Noise cluster found in the mapping, skipping it')
        pass
    
    # Gt = len(groups_ct['task-definitive'])
    # Ge = len(groups_ct['exo-definitive'])
    # Gn = len(groups_ct['Noise']) 
    Gf = len(groups_ct['foreground']) 

    # D_xtr = np.zeros((Gt, D))
    # D_xer = np.zeros((Ge, D))
    # D_xnr = np.zeros((Gn, D))
    D_xfr = np.zeros((Gf, D))

    # for i, i_meta in enumerate(groups_ct['task-definitive']):
    #     rows = groups[i_meta]
    #     D_xtr[i, :] = agg_fn(mu[rows, :], axis=0).reshape(1, -1)
            
    # for i, i_meta in enumerate(groups_ct['exo-definitive']):
    #     rows = groups[i_meta]
    #     D_xer[i, :] = agg_fn(mu[rows, :], axis=0).reshape(1, -1)
            
            
    # for i, i_meta in enumerate(groups_ct['Noise']):
    #     rows = groups[i_meta]
    #     D_xnr[i, :] = agg_fn(mu[rows, :], axis=0).reshape(1, -1)
            
    for i, i_meta in enumerate(groups_ct['foreground']):
        rows = groups[i_meta]
        D_xfr[i, :] = agg_fn(mu[rows, :], axis=0).reshape(1, -1)
            
    return D_xfr

def check_pairwise_distance_matrix(D_M):
    """Validate the distance matrix."""
    if not isinstance(D_M, np.ndarray):
        raise ValueError("Distance matrix must be a numpy array.")

    if D_M.shape[0] != D_M.shape[1]:
        raise ValueError("Distance matrix must be square.")

    if np.any(D_M < 0):
        raise ValueError("Distance matrix must contain non-negative values.")

    if not np.allclose(D_M, D_M.T):
        raise ValueError("Distance matrix must be symmetric.")

    if np.any(np.diag(D_M) != 0):
        raise ValueError("Diagonal elements of the distance matrix must be zero.")

def find_subset_index(subset_matrix: np.ndarray, full_matrix: np.ndarray, full_idx: int):
    """
    Given:
        - subset_matrix: np.ndarray of shape (200, 1408)
        - full_matrix: np.ndarray of shape (646, 1408)
        - full_idx: int, index in full_matrix (between 0 and 645)
    
    Returns:
        - The index in subset_matrix corresponding to full_matrix[full_idx] if it exists
        - None if not found
    """
    # Extract the row from the full matrix
    target_row = full_matrix[full_idx]

    # Find rows in subset_matrix that match target_row
    matches = np.all(subset_matrix == target_row, axis=1)

    if np.any(matches):
        if np.sum(matches) > 1:
            print( np.where(matches))
            print(f"Warning: Multiple matches found for index {full_idx}. Returning the first match.")
        return np.where(matches)#[0][0]  # Return the first match
    else:
        return None

def find_matching_prototypes(symbolization_config_name, annotator_id, round_number):
    """Finds matching prototypes in the clustering configurations based on the provided prototypes.
    Args:
        prototypes (np.ndarray): The prototypes to match.
        annotator_id (str): The ID of the annotator.
        round_number (int): The round number for the clustering configurations.
        cluster_types (list, optional): List of cluster types to consider. Defaults to None.
        normalization (str, optional): Normalization method to use. Defaults to None.
    """
    
    
    prototypes = get_prototypes(symbolization_config_name, annotator_id=annotator_id, round_number=round_number, verbose=False, return_counting=False)
    config = import_config(symbolization_config_name)
    clustering_config_name = config.clustering_config_name
    clustering_config = import_config(clustering_config_name)
    modality = clustering_config.model_params['modality']
    task_name = clustering_config.model_params['task_name']
    normalization = clustering_config.model_params['normalization']

    qualification_mapping = fetch_qualification_mapping(verbose=False)

    if clustering_config_name == 'ClusteringDeploymentKmeansInferenceConfig':
            
        input_clustering_config_names, round_numbers_list, qualification_cluster_types_list = get_complete_configs(round_number=round_number, inference_mode=True)

    else:
        
        input_clustering_config_names = [clustering_config_name]
        round_numbers_list = [round_number]
        qualification_cluster_types_list = [cluster_types]
            
            
        
    matching_results = []
    for search_idx in range(len(prototypes)):
        find = False 
        
        for input_clustering_config_name, qualification_cluster_types, inner_round_number in zip(input_clustering_config_names, qualification_cluster_types_list, round_numbers_list):
            
        
            input_config = import_config(input_clustering_config_name)
            cl_input_config = input_config.clustering_config_name
            cl_config = import_config(cl_input_config)
            clusters_path = os.path.join(get_data_root(), 'experiments', cl_config.experiment_name, cl_config.experiment_id, annotator_id, f'round_{inner_round_number}', f'{task_name}_{modality}_cluster_centers.npy')
            centroids = np.load(clusters_path)
            


            result = find_subset_index(centroids, prototypes, search_idx)
            if result is not None:
                find=True
                matching_results.append({
                    'input_clustering_config_name': input_clustering_config_name,
                    'qualification_cluster_types': qualification_cluster_types,
                    'inner_round_number': inner_round_number,
                    'accumulated_cluster_index': search_idx,
                    'original_cluster_index': result[0][0]
                })
            
            if find: 
                break
                
    return  pd.DataFrame(matching_results)