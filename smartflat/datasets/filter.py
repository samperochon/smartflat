"""Filters used to sample subsets of the Smartflat dataset."""

import os
import warnings

import numpy as np
import pandas as pd

from smartflat.constants import tasks_duration_lims
from smartflat.utils.utils_coding import display_safe, yellow


def define_cohort(metadata, scenario):
    """Define a subset of the dataset based on a scenario.

    .. deprecated::
        Use dataset class methods (e.g. ``SmartflatDataset.define_cohort``) instead.
    """
    warnings.warn(
        "define_cohort() is deprecated — use dataset class methods instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Filters the videos that have been processed
    if scenario == 'all' or scenario is None:
        pass    
    
    elif scenario == 'unprocessed':
        
        metadata = metadata[(~metadata['video_path'].isna()) & 
                            (metadata['flag'] == 'unprocessed') &  
                            ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()    
    
    elif scenario == 'complete':
        
        metadata = metadata[(metadata['flag'] == 'success') &  
                            ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()
        
    elif scenario == 'complete+video':
        
        metadata = metadata[(~metadata['video_path'].isna()) & 
                            (metadata['flag'] == 'success') &  
                            ((metadata['video_name'] == 'merged_video') | (metadata['n_videos'] == 1))].copy()
        
    
    elif scenario == 'collate':
        pass
    
    else:
        raise ValueError(f"Scenario {scenario} is not registered. Use one of {['unprocessed', 'complete', 'complete+video', 'collate']}")
            
    return metadata


def apply_gold_data_filtering(df_video, process=False, verbose=True):
    """
    Here we remove partitions in case ma `merged_video` is present (except when n_video == 1)
    """

    # We need here to identify the gold data, as defined by having a single video per participant, task and modality.
    if verbose:
        print(f"Applying gold data filtering (see function for info) [process={process}].")
    df_video["is_outlier"] = 0
    initial_count = len(df_video)

    # 1) first remove partitions in case a merged_video is present and recompute the n_identifiers_per_mod
    df_video["n_identifiers_per_mod"] = df_video.groupby(
        ["task_name", "participant_id", "modality"]
    )["identifier"].transform(lambda x: len(np.unique(x)))
    df_video["has_merged_video"] = df_video.groupby(
        ["task_name", "participant_id", "modality"]
    )["video_name"].transform(lambda x: 1 if "merged_video" in x.to_list() else 0)
    df_video["is_partition"] = df_video.apply(
        lambda x: (
            1
            if (
                (x["n_identifiers_per_mod"] > 1)
                & (x["has_merged_video"] == 1)
                & (x["video_name"] != "merged_video")
            )
            else 0
        ),
        axis=1,
    )

    # 2) Remove videos that have a single identifier (i.e a video), but that is a partition
    df_video["is_consolidated"] = df_video.apply(
        lambda x: (
            1
            if (x.n_videos == 1)
            | (x.video_name == "merged_video")
            | (x.percy_metadata_merge == "left_only")
            else 0
        ),
        axis=1,
    )
    df_video.loc[
        (df_video["is_consolidated"] == 0) | (df_video["is_partition"] == 1),
        "is_outlier",
    ] = 1

    # print("Summary stats of filtering")
    # display(df_video[["is_partition", "is_outlier"]].value_counts().to_frame())

    if verbose:
        final_count = len(
            df_video[
                (df_video["is_consolidated"] == 1) | (df_video["is_partition"] == 0)
            ]
        )
        if initial_count != final_count:
            print(f"Filtered {initial_count - final_count}/{initial_count} ({100*(initial_count - final_count)/initial_count:.2f}%) videos wiping partitions.")

    if process:
        df_video = df_video[df_video["is_outlier"] == 0].copy()

    # 3) Remove residual partitions (more than one video per modality), i.e when the merged_video is not present and the n_videos > 1.
    df_video["n_identifiers_per_mod"] = df_video.groupby(
        ["task_name", "participant_id", "modality"]
    )["identifier"].transform(lambda x: len(np.unique(x)))
    df_video.loc[
        (df_video["n_identifiers_per_mod"] > 1) & (df_video["is_consolidated"] == 0),
        "is_outlier",
    ] = 1

    if verbose and initial_count > 0:
        final_count = len(
            df_video[
                (df_video["n_identifiers_per_mod"] == 1)
                & (df_video["is_consolidated"] == 1)
            ]
        )
        if final_count != initial_count:
            
            print(
                f"Filtered {initial_count - final_count}/{initial_count} ({100*(initial_count - final_count)/initial_count:.2f}%) videos"
                f" residuals partitions rows (no merged_video while n_partitions>1)."
            )
        display_safe(
            df_video[
                (df_video["n_identifiers_per_mod"] == 1)
                & (df_video["is_consolidated"] == 0)
            ]
        )

    if process:
        df_video = df_video[df_video["is_outlier"] == 0].copy()

    # 4) Filter videos based on duration
    df_video = filter_duration(
        df_video, tasks_duration_lims, process=process, verbose=verbose
    )

    # 5) Filter the videos based on the duration gap between the current and the percy dataset

    # TODO: DEPRECATED ? Looks like the percy duration is not trustable enough (because of duplicated videos/renamed/tobii outputs).
    # -> A manual inspection of the folder is being done.
    # X - Filter based on duration dfferences
    # Remove the failure collated (after manually inspecting the directory)
    # verbose = False
    # df_video['abs_duration_delta'] = (df_video['duration_percy'] - df_video['duration']).abs()
    # df_video['has_duration_delay'] = df_video['abs_duration_delta'].apply(lambda x: 1 if x > 5 else 0)
    # if verbose:
    #     df_video[df_video['abs_duration_delta'] > 5][ id_cols + ['video_name_list', 'duration', 'abs_duration_delta']].duration.hist(bins=100)
    #     df_video[df_video['abs_duration_delta'] > 5].sort_values(['task_name', 'duration'])[id_cols + ['folder_path', 'has_video', 'flag_collate_video', 'video_name_list', 'duration', 'duration_percy',  'fps', 'fps_percy', 'abs_duration_delta']]
    #     # Since the duration at percy is sometimes nt trustable (multiple outputs from the same video, e.g. Tobii-extracted videos), we relax this constraint and check the folders manually when it is the case.
    #     #df_video = df_video[df_video['has_duration_delay'] == False]
    #     display(df_video['has_duration_delay'].value_counts().to_frame())
    # df_video.drop(columns=['abs_duration_delta', 'has_duration_delay'], inplace=True)

    # Ensure there are no missing fps values after merging with fps_percy
    missing_fps_count = df_video["fps"].isna().sum()
    if verbose and missing_fps_count > 0:
        # raise ValueError(f"There are {missing_fps_count} missing fps values after merging with fps_percy.")
        yellow(
            f"Note: there are {missing_fps_count} missing fps values after merging with fps_percy:"
        )
        display_safe(df_video[(df_video['fps'].isna()) & (df_video['video_path'].notna()) ][['identifier', 'fps', 'video_path']])
        
    elif verbose:
        print("All fps values are present after merging with fps_percy.")

    # 6) Remove the videos with missing fps if any TODO: make a parameter ?

    initial_count = len(df_video)
    df_video['has_fps'] = df_video.fps.notna()
    
    #display(df_video.groupby(['task_name', 'modality', 'has_video']).has_fps.value_counts().to_frame())
    #display(df_video[df_video['has_fps'] == False])
    if process: 
        df_video = df_video[df_video['has_fps'] == True]; final_count = len(df_video)
    print(f'Filtering videos with missing fps: {initial_count} -> {len(df_video)} ({initial_count-len(df_video)} removed) (TODO: take from percy if availables (<5%))')
    if verbose:
        print(f'Filtered {initial_count - final_count}/{initial_count} ({100*(initial_count - final_count)/initial_count:.2f}%) videos '
                f'videos with missing fps.')

    #df_video['is_gold'] = True # After you also check n_videos and one per modality and no failure and duration test

    return df_video


def filter_duration(df, duration_lims, process=True, verbose=True):

    for task_name, (min_duration, max_duration) in duration_lims.items():

        df.loc[
            (
                (df["task_name"] == task_name)
                & ((df["duration"] < min_duration) | (df["duration"] > max_duration))
            ),
            "is_outlier",
        ] = 1
        if verbose:
            initial_count = len(df)
            final_count = len(
                df[
                    ~(
                        (df["task_name"] == task_name)
                        & (
                            (df["duration"] < min_duration)
                            | (df["duration"] > max_duration)
                        )
                    )
                ]
            )
            if initial_count != final_count:
                print(
                    f'Filtered {initial_count - final_count}/{initial_count} ({100*(initial_count - final_count)/initial_count:.2f}%) videos from the task "{task_name}" '
                    f"with duration outside {min_duration}-{max_duration} seconds."
                )
        if process:
            df = df[df["is_outlier"] == 0].copy()
    return df


def filter_outlier_video_names(df, process=True, verbose=True):
    
    # Fix participant if if needed
    #df['participant_id'] = df.participant_id.replace(mapping_participant_id_fix)
    

    df["is_outlier"] = 0
    # Fix spaces #deprecated: done upstream when parsing.
    df["video_name"] = df["video_name"].str.strip()
    df["participant_id"] = df["participant_id"].str.strip()

    df["is_recording_video"] = df.video_name.apply(
        lambda x: True if x.startswith("Recording") else False
    )
    df["is_tobii_output"] = df.apply(
        lambda x: True if x.video_name == x.participant_id else False, axis=1
    )

    # Remove videos with non-standard video_name (not 8 characters for GoPro and 24 for Tobii or merged_video)s
    idf = df[
        (
            ((df["modality"] == "Tobii") & (df["video_name"].apply(len) != 24))
            | (
                (df["modality"] == "Tobii")
                & (df["video_name"].apply(len) == 24)
                & (
                    df["video_name"].apply(
                        lambda x: True if not x.endswith("==") else False
                    )
                )
            )
            | (
                (df["modality"].isin(["GoPro1", "GoPro2", "GoPro3"]))
                & (df["video_name"].apply(len) != 8)
            )
            | (df["is_recording_video"] == True)
            | (df["is_tobii_output"] == True)
        )
        & (df["video_name"] != "merged_video")
    ].copy()

    df.loc[df.index.isin(idf.index), "is_outlier"] = 1

    if verbose:
        initial_count = len(df)
        final_count = len(df[~df.index.isin(idf.index)])
        print(
            f"Filtered {initial_count - final_count}/{initial_count} ({100*(initial_count - final_count)/initial_count:.2f}%) "
            f" with non-standard video name."
        )
        if len(idf) > 0:
            display_safe(idf[["participant_id", "modality", "video_name"]])
    if process:
        df = df[df["is_outlier"] == 0].copy()
    # df.drop(columns=['is_recording_video', 'is_tobii_output'], inplace=True)

    df.drop(columns=['is_recording_video', 'is_tobii_output', 'is_outlier'], inplace=True)
    return df

