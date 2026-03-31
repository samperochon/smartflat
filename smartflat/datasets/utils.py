import ast
import datetime
import os
import socket
import sys
from copy import deepcopy
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import umap
except ImportError:
    umap = None

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        for a in args:
            print(a)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# add tools path and import our own tools

from smartflat.annotation_smartflat import AnnotationSmartflat
from smartflat.constants import (
    available_output_type,
    mapping_incorrect_modality_name,
    mapping_participant_id_fix,
    tasks_duration_lims,
)
from smartflat.utils.utils import check_and_convert
from smartflat.utils.utils_clinical import diagnosis_logic
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_io import (
    check_video,
    fetch_flag_path,
    fetch_output_path,
    get_data_root,
    get_file_size_in_gb,
    get_video_loader,
    parse_dir_name,
    parse_flag,
    parse_participant_id,
    parse_path,
    parse_task_number,
)


def append_percy_metadata(df_video, verbose=False):

    # 2) Merge with Percy dataframe
    usecols = [
        "task_name",
        "participant_id",
        "modality",
        "passation_id",
        "modality_id",
        "video_id_list",
        "n_videos",
        "video_name_list",
        "dates_list",
        "fps",
        "n_fps",
        "size_list",
        "total_size",
        "n_frames_list",
        "total_n_frames",
        "duration_list",
        "total_duration",
        "mod_identifier",
    ]

    # TODO: enforce that the -persitent dataframes are truly persistent and passed over (components of the dataset)
    metadata = pd.read_csv(
        os.path.join(
            get_data_root(), "dataframes", "persistent_metadata", "metadata.csv"
        )
    )

    metadata.rename(
        columns={
            "fps": "fps_percy",
            "total_size": "size_percy",
            "total_n_frames": "n_frames_percy",
            "total_duration": "duration_percy",
        },
        inplace=True,
    )
    metadata["duration_percy"] = metadata["duration_percy"] / 60

    df_video = df_video.merge(
        metadata,
        on=["task_name", "participant_id", "modality"],
        how="left",
        indicator=True,
    ).rename(columns={"_merge": "percy_metadata_merge"})

    # if not (df_video.passation_id.isna().mean() == 0):
    #     print("Some rows have missing  passation_id informations...")
    #     display(
    #         df_video[df_video.passation_id.isna()][
    #             [
    #                 "task_name",
    #                 "participant_id",
    #                 "modality",
    #                 "video_name",
    #                 "has_video",
    #                 "passation_id",
    #             ]
    #         ]
    #     )

    # df_video.dropna(subset=['passation_id'], inplace=True)
    # df_video['passation_id'] = df_video['passation_id'].astype(int)
    # df_video['n_frames_percy'] = df_video['n_frames_percy'].astype(int)
    # df_video['n_fps'] = df_video['n_fps'].astype(int)

    # metadata[metadata['participant_id'].apply(lambda x: True if 'L38' in x else False)]
    if verbose:
        yellow("Present rows not in the metadata files: {}".format(len(df_video[df_video["percy_metadata_merge"] == "left_only"])))
        display(df_video[df_video["percy_metadata_merge"] == "left_only"])  # .hist())

    #display(df_video.percy_metadata_merge.value_counts().to_frame())
    return df_video


def append_video_metadata(df_video, verbose=False):
    """
    Appends video metadata to a DataFrame.

    Parameters:
    df_video (pandas.DataFrame): The DataFrame containing video data.

    Returns:
    pandas.DataFrame: The DataFrame with video metadata appended.

    #TODO remove
    For removing corrupted videos:
    import subprocess
    n = 0
    commands = []; process=False

    for i, row in df_video[(~df_video['video_path'].isna()) & (df_video['fps'].isna())].iterrows():

        commands.append(['rm', row.video_path])

    if process:
        for command in commands:
            blue(' '.join(command))
            subprocess.run(command)
    else:
        for command in commands:
            yellow(' '.join(command))
    """

    def fill_nan_from_percy(df, col_name, verbose=False):
        n_before = df[col_name].isna().sum()
        df[col_name] = df[col_name].fillna(df[f"{col_name}_percy"])
        n_after = df[col_name].isna().sum()
        if verbose:
            print(
                f"Added {col_name} from percy (no videos found): {n_before - n_after} ({n_after} still missing.)"
            )



    # 1) Append video metadata form the clean Percy dataset.
    # Note: Some participant_id where rename to fit conventons, and don't have matching in Percy.
    # For now we keep them and retrieve their metadata from the videos themselves.
    videos_metadata_path = os.path.join(
        get_data_root(),
        "dataframes",
        "persistent_metadata",
        f"smartflat_video_metadata_updated.csv",
    )
    usecols = [
        "task_name",
        "participant_id",
        "folder_modality",
        #"modality",
        "video_name",
        "duration",
        "n_frames",
        "fps",
        "size",
        "date",
    ]
    vmetadata = pd.read_csv(videos_metadata_path, usecols=usecols)
    
    df_video = df_video.merge(
        vmetadata,
        left_on=["task_name", "participant_id", "modality", "video_name"], 
        right_on=["task_name", "participant_id", "folder_modality", "video_name"], 

        how="left",
        indicator=True,
    ).rename(columns={"_merge": "video_metadata_merge"})

    # 2) Fill missing video metadata using percy information (combination)

    # First convert list of percy fps (TODO: fix this inconsistency upstream)
    df_video["fps_percy"] = df_video["fps_percy"].apply(check_and_convert)
    df_video["fps_percy"] = df_video["fps_percy"].apply(lambda x: np.mean(x))

    # Fill missing video metadata
    for col_name in ["fps", "n_frames", "duration"]:
        fill_nan_from_percy(
            df_video, col_name, verbose=verbose if col_name == "fps" else False
        )

    if verbose:
        yellow("Present rows not in the video-metadata files:")
        display(df_video[df_video["video_metadata_merge"] == "left_only"])  # .hist())
        
        

    # df_video.loc[(df_video['participant_id'] == 'G107_P91_RAYVia_03052023') & (df_video['modality'] == 'Tobii'), 'fps'] = 25.02
    # df_video.loc[(df_video['participant_id'] == 'G107_P91_RAYVia_03052023') & (df_video['modality'] == 'Tobii'), 'duration'] = 21.25
    # df_video.loc[(df_video['participant_id'] == 'G107_P91_RAYVia_03052023') & (df_video['modality'] == 'Tobii'), 'n_frames'] = 31900

    return df_video


def append_clinical_data(df, verbose=False):
    """Append clinical data based on participant trigram"""
    
    
    clinical_data_path = os.path.join(get_data_root(), 'dataframes', 'clinical', 'merged-clinical-data-mupt.csv')
    
    cdf = pd.read_csv(clinical_data_path)
    assert cdf.trigram.nunique() == cdf.shape[0] 
    
    
    df = df.merge(cdf, on='trigram', how='left')
    df['group'] = df.apply(diagnosis_logic, axis=1)

    # 1) Statistics
    if verbose:
        green(f'Loaded merged clinical data to: {clinical_data_path}' )
        green(f'Number of unique trigrams: {cdf.trigram.nunique()}')
        display(cdf[['bras', 'pathologie']].value_counts())
        green('Diagnosis groups (0=control, 1=patient) :')
        display(df.drop_duplicates(["trigram"])["group"].value_counts().to_frame())        
        
    df['pathologie'].fillna('N.A', inplace=True)
    return df


def append_annotations(df_video, verbose=False):
    

    def apply_retrieve_annotation(modality_folder):
        
        task_name, participant_id, modality = parse_dir_name(modality_folder)
        participant_id_folder = os.path.dirname(modality_folder)
        
        # Retrieve task_name for using the correct mapping dictionnary task_name  'cuisine' 
        annotation_paths = list(
            glob(os.path.join(participant_id_folder, "Annotation", "*.json"))
        ) + list(glob(os.path.join(participant_id_folder, "Annotation", "*.boris")))
        if len(annotation_paths) > 1:
            #yellow("[WARNING] Multiple json annotation path here: {}".format(annotation_paths))
            annotation_path = annotation_paths[0]
            
            if verbose:
                pass#yellow("\n/!\ Multiple annotation files found, use first of:")
                #yellow("\n".join(annotation_paths))
            annotation = AnnotationSmartflat(task_name=task_name, annotation_path=annotation_path)

        elif len(annotation_paths) == 1:

            annotation_path = annotation_paths[0]
            annotation = AnnotationSmartflat(task_name=task_name, annotation_path=annotation_path)

        else:
            annotation = np.nan

        return annotation

    # Seek for annotation files in Annotation folder
    df_video["annotation"] = df_video.groupby(
        ["task_name", "participant_id", "modality"]
    ).folder_path.transform(
        lambda x: apply_retrieve_annotation(x.iloc[0])
    )

    df_video["has_annotation"] = df_video.annotation.apply(
        lambda x: True if isinstance(x, AnnotationSmartflat) else False
    )
    df_video["annotation_software"] = df_video.annotation.apply(
        lambda x: x.annotations_software if isinstance(x, AnnotationSmartflat) else np.nan
    )
    return df_video


def use_light_dataset(df_video, light_by_default=True):
    
    def explore_light_dataset(row, feature_name, use_light=True):
        if row[f'{feature_name}_computed'] and not use_light:
            return pd.Series([row[f'{feature_name}_path'], row[f'{feature_name}_computed'], False])
        
        # Construct potential light version path
        light_path = row[f'{feature_name}_path'].replace('final', 'light')
        
        if os.path.exists(light_path):
            
            return pd.Series([light_path, True, True])
        else:
            return pd.Series([row[f'{feature_name}_path'], row[f'{feature_name}_computed'], False])
        
    # 2.1) Look for light versions if present
    if os.path.isdir(get_data_root('light')):
        # Look for light version of the output features
        for feature_name in available_output_type:
            df_video[f'{feature_name}_is_light'] = False
            if f'{feature_name}_computed' in df_video.columns:
                df_video[[f'{feature_name}_path', f'{feature_name}_computed', f'{feature_name}_is_light']] = df_video.apply(explore_light_dataset, axis=1, args=(feature_name, light_by_default, ))
            else:
                #print(f'Feature {feature_name}_computed not found in the dataframe: No updates performed on the features path, computed etc. ')
                pass
    # 2.2) Update video presence status
    df_video['has_light_video'] = df_video.apply(lambda x: 1 if os.path.exists(os.path.join(get_data_root('light'), x.task_name, x.participant_id, x.folder_modality, x.video_name+'.mp4')) else 0, axis=1)
    
    # TODO: what when building the dataset ? 
    #df_video['video_path'] = df_video.apply(lambda x: os.path.join(get_data_root('light'), x.task_name, x.participant_id, x.modality, x.video_name+'.mp4') if x.has_light_video else x.video_path, axis=1)
    df_video['has_video'] = df_video.apply(lambda x: int(os.path.exists(x.video_path)) if (not pd.isna(x.video_path)) else 0, axis=1)

    return df_video

def add_test_bounds(df, verbose=False):
    """
    Add test bounds to the dataframe.
    """
    def to_sec(t):
        try:
            parts = str(t).strip().split(":")
            parts = list(map(int, parts))
            if len(parts) == 2:      # MM:SS
                return parts[0]*60 + parts[1]
            elif len(parts) == 3:    # HH:MM:SS
                return parts[0]*3600 + parts[1]*60 + parts[2]
            else:
                return float('nan')
        except:
            return float('nan')

    def get_bounds(row):
        stride = row['duration'] * 60 / row['N_raw']
        start_idx = int(np.ceil(row['start_timestamp_sec'] / stride))
        end_idx = int(np.floor(row['end_timestamp_sec'] / stride))
        return (start_idx, end_idx)

    test_bounds_data_path = os.path.join(get_data_root(),'dataframes', 'persistent_metadata','smartflat_test_limites.csv')
    df_test_bounds = pd.read_csv(test_bounds_data_path, sep=';'); df_test_bounds.drop(columns=['video_path', 'duration_timestamp'], inplace=True)

    df = df.merge(df_test_bounds, on='participant_id', how='left', indicator='merge_test_bounds', )
    
    #display(df[df.merge_test_bounds == 'left_only'])  # .hist())
    df = df[df.merge_test_bounds != 'left_only']
    assert (df.merge_test_bounds == 'both').all()

    #df = df[~df['start_timestamp_sec'].isna()]
    for col in ['pred_start_timestamp', 'pred_end_timestamp', 'start_timestamp', 'end_timestamp']:
        df[f'{col}_sec'] = df[col].apply(to_sec)
    #display(df[df.start_timestamp_sec.isna()])

    if 'N_raw' not in df.columns:
        df['N_raw'] = df['video_representation_path'].apply(lambda x: np.load(x).shape[0] if os.path.exists(x) else np.nan)
    
    df['test_bounds'] = df.apply(get_bounds, axis=1)
        
    return df

# local utility functions (for datasets)
def load_embedding_dimensions(row):
    def robust_dimension_loading(path):
        try:
            return tuple(np.load(path).shape)
        except:
            return (np.nan, np.nan)
    
    if 'test_bounds' in row.columns:
        #green('Using test bounds to load dimensions')
        row[["N_raw", "D"]] = row.video_representation_path.apply(robust_dimension_loading).apply(pd.Series).astype(float)
        row['N'] = row.test_bounds.apply(lambda x: int(x[1] - x[0]))
        
    else:
        red('Using full video representation to load dimensions (without bounds)')
        row[["N", "D"]] = row.video_representation_path.apply(robust_dimension_loading).apply(pd.Series).astype(float)
        row['N_raw'] = row['N'].astype(int)
        
    return row

def load_embeddings(row):
    
    if 'test_bounds' in row.columns:
        row["X"] = row.video_representation_path.apply(np.load)[row.test_bounds[0]:row.test_bounds[1]]
    
    else:
        row["X"] = row.video_representation_path.apply(np.load)
    
    row[["N", "D"]] = row.X.apply(np.shape).apply(pd.Series).astype(int)
    
    return row



# Question: which is the best of the two ? 

# def add_umap(df, embed_col='X'):
#     """Add 2D UMAP-reduced coordinates of `embed_col` embeddings as a new column `umap_embed`."""
#     # Flatten all embeddings to fit UMAP model
#     all_embeddings = np.vstack(df[embed_col].tolist())
#     all_embeddings = np.nan_to_num(all_embeddings)
    
#     # Fit UMAP on the combined data
#     model = umap.UMAP(n_neighbors=15, min_dist=0.2, metric='cosine')
#     reduced_embeddings = model.fit_transform(all_embeddings)
    
#     # Split reduced embeddings back to original groupings by length of original rows
#     start_idx = 0
#     def transform_embeddings(emb):
#         nonlocal start_idx
#         n_i = len(emb)
#         result = reduced_embeddings[start_idx:start_idx + n_i]
#         start_idx += n_i
#         return result
    
#     # Apply transformation and add the result to a new column
#     return df[embed_col].apply(transform_embeddings)

def add_umap_grouped(df, embed_col='X', task_col='task_name', modality_col='modality'):
    """Add UMAP coordinates per group defined by task_name and modality."""
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"

    def process_group(group):
        if embed_col not in group.columns:
            group = load_embeddings(group)
        
        
        model = umap.UMAP(n_neighbors=15, min_dist=0.2, metric='cosine')
        Z = np.vstack(group[embed_col].tolist())
        Z = np.nan_to_num(Z)
        Z_transformed = model.fit_transform(Z)
        group['umap_embed_x1'] = Z_transformed[:, 0]
        group['umap_embed_x2'] = Z_transformed[:, 1]
        
        group.drop(['X'], axis=1, inplace=True)
        return group

    return df.groupby([task_col, modality_col], group_keys=False).apply(process_group)


def add_pca_3d_grouped(df, embed_col='X', groupby_col=['task_name','modality']):
    """Add 3D PCA coordinates per group defined by task_name and modality."""
    
    def process_group(group):
        
        if embed_col not in group.columns:
            group = load_embeddings(group)
        
        all_embeddings = np.vstack(group[embed_col].tolist())
        all_embeddings = np.nan_to_num(all_embeddings)

        pca = PCA(n_components=3)
        pca_embeddings = pca.fit_transform(all_embeddings)

        group['pca_x'] = list(pca_embeddings[:, 0])
        group['pca_y'] = list(pca_embeddings[:, 1])
        group['pca_z'] = list(pca_embeddings[:, 2])
        
        group.drop(['X'], axis=1, inplace=True)
        return group

    return df.groupby(groupby_col, group_keys=False).apply(process_group)

def add_umap(df, embed_col='X', min_dist=0.2, n_neighbors=15, n_components=3):
    """Add `emb_x1` and `emb_x2` cUMAP coordinates of the `embed_col` embeddings."""
    
    #assert df.identifier.nunique() == 1, "The dataframe should have a unique identifier to fit the umap."
    
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2" 
    if embed_col not in df.columns:
        df = load_embeddings(df)

    model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine')
    
    Z = np.vstack(df[embed_col].tolist())
    Z = np.nan_to_num(Z)
    Z_transformed = model.fit_transform(Z) 
    return [Z_transformed]


def add_pca(df, embed_col='X', n_components=2):
    """Add nD PCA-reduced coordinates of `embed_col` embeddings as a new column."""
    
    if embed_col not in df.columns:
        df = load_embeddings(df)

    # Flatten all embeddings for PCA
    all_embeddings = np.vstack(df[embed_col].tolist())
    all_embeddings = np.nan_to_num(all_embeddings)

    # Fit PCA on the combined data to reduce to 3 components
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(all_embeddings)
    
    # Split reduced embeddings back to original groupings by length of original rows
    start_idx = 0
    def transform_embeddings(emb):
        nonlocal start_idx
        n_i = len(emb)
        result = pca_embeddings[start_idx:start_idx + n_i]
        start_idx += n_i
        return result

    return df[embed_col].apply(transform_embeddings)


def add_covar_label(df, covar_col, covar_label_col=None):
    """Add a '{covar}_label' column to encode the `covar` variable and broadcast it to match the shape of the embeddings."""
    if covar_label_col is None:
        covar_label_col = f'{covar_col}_label'
    
    # Generate the label for each row based on the covar column and broadcast it to the embedding shape
    
    return df.apply(lambda row: np.full(len(row['X']), row[covar_col]), axis=1)
