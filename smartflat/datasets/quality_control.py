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
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# add tools path and import our own tools


from smartflat.annotation_smartflat import AnnotationSmartflat
from smartflat.constants import (
    available_output_type,
    mapping_incorrect_modality_name,
    tasks_duration_lims,
)
from smartflat.utils.utils import check_and_convert
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


def apply_visual_inspection_results(df, verbose=False):
    """Account for the visual inspection results to swap wrong modality and identifier

    1) Add the columns to the dataset dataframe.

    2) Update `modality` and `identifier`

    3) Update the flags based on the modality: 
    
    
    Note: since this is just a soft fixing solution (not hard, we don't move data or rename data files), it means that:
        - Identification information (mostly the modality) cannot be parsed anymore from paths but from identifiers.
        - Although video blocks embedding flags and path should not be modified (to point to the correct video data), we need to recalculte whether the features extracton have to be done for those that are modality-dependent (hand and skeleton and speech-related). 
    TODO: Fix the upside down videos (transform them, keep as experiments their outputs and reset the folder to renew features extraction on fixed data).
    TODO: Experiemnt: This is a source of features computation redundancy, for the speech notably. This means it gives easy multiple-instances of extraction of the speech and gaze. 
    
    """

    cols = ["identifier",
        "task_name",
        "participant_id",
        "modality",
        'folder_modality',
        "is_fish_eyed",
        "upside_down",
        "is_swapped_modality",
        "true_modality",
        "GP3_is_sink",
        "GP2_is_wrong_buttress",
        "GP2_above",
        "GoPro1_is_wrong_buttress",
        "is_middle_range",
        "true_task_name",
        "is_old_setup",
        "is_old_recipe",
        "annot_notes",
    ]
    
    results = pd.read_csv(os.path.join(get_data_root(),'dataframes', 'quality-control', 'results_visual_inspection_2111124_last.csv'),sep=';', usecols=cols)

    if verbose:

        create_annotation_summary(results)
        cols_changed = [
            "modality",
            "flag_speech_recognition",
            "flag_speech_representation",
            "flag_video_representation",
            "flag_hand_landmarks",
            "flag_skeleton_landmarks",
            "flag_collate_video",
        ]
        value_counts_before = df[cols_changed].apply(pd.Series.value_counts)

        # Find rows in vidual annotation that are not in the dataset
        missed_matching = results.merge(
            df, left_on=['task_name', 'participant_id', 'folder_modality'], right_on=['task_name', "participant_id", "modality"], how="left", indicator=True
        )
        missed_matching = missed_matching[missed_matching["_merge"] == "left_only"]
        missed_matching = missed_matching.drop(columns="_merge")
        # print('Missing matching on the following participants and modality:')
        # display(missed_matching.drop_duplicates(['participant_id', 'modality']).groupby('participant_id')['modality'].agg(list).to_frame().transpose())
    
    df = df.merge(
        results.drop(columns=["identifier", 'modality']),
        left_on=["task_name", "participant_id", "modality"],
        right_on = ['task_name', 'participant_id', 'folder_modality'], 
        how="left",
        indicator=True,
    )
    df['folder_modality'] = df.apply(lambda x: x.folder_modality if not pd.isna(x.folder_modality) else x.modality, axis=1)
    #df["folder_identifier"] = df.apply(lambda x: "{}_{}_{}_{}".format(x.participant_id, x.task_name, x.folder_modality, x.video_name),axis=1)
    
    # Remove failed data collection and log failed modalities/participants
    failed_conditions = [
        #("upside_down", lambda row: row["upside_down"] == 1),
        ("true_task_name_nan", lambda row: row["true_task_name"] is np.nan),
        #("GP3_is_sink", lambda row: row["GP3_is_sink"] == 1),
        ("GP2_is_wrong_buttress", lambda row: row["GP2_is_wrong_buttress"] == 1),
        ("GP2_above", lambda row: row["GP2_above"] == 1),
        ("GoPro1_is_wrong_buttress", lambda row: row["GoPro1_is_wrong_buttress"] == 1),
        ("is_middle_range", lambda row: row["is_middle_range"] == 1),
    ]

    failed_logs = []
    for condition, test in failed_conditions:
        failed = df[df.apply(test, axis=1)]
        if not failed.empty:
            failed_logs.append((condition, failed[["participant_id", "modality"]]))


    # Remove failed data collection
    n = len(df)
    df = df[
        (
            #(df["upside_down"] != 1)
             (df["true_task_name"] != np.nan)
           # & (df["GP3_is_sink"] != 1)
            & (df["GP2_is_wrong_buttress"] != 1)
            & (df["GP2_above"] != 1)
            & (df["GoPro1_is_wrong_buttress"] != 1)
            & (df["is_middle_range"] != 1)
        )
    ]
    if n != len(df) and verbose:
        yellow(f"Modalities removed because of visual inspection results: M={n - len(df)}")
        for condition, failed in failed_logs:
            if not failed.empty:
                yellow(f"Condition '{condition}' failed for:")
                print(failed)
                
    elif verbose:
        green('All modalities passed the visual inspection :-).')
        
    # Filter videos with incorrect modality placement
    incorrect_videos = df[df["is_swapped_modality"] == True]

    # Iterate over each incorrect video
    for index, row in incorrect_videos.iterrows():
        # Update 'modality' column with the true modality
        df.at[index, "modality"] = row["true_modality"]
        
    # Iterate over each incorrect video
    for index, row in incorrect_videos.iterrows():
        # Generate new identifier based on true modality
        new_identifier = (
            row["participant_id"]
            + "_"
            + row["task_name"]
            + "_"
            + row["true_modality"]
            + "_"
            + row["video_name"]
        )

        # Update 'identifier' column with the new identifier
        df.at[index, "identifier"] = new_identifier

    #'sucess' or 'failure' or np.nan if un-processed

    # df["flag_video_representation"] = df.apply(
    #     lambda row: parse_flag(
    #         row["video_representation_path"],
    #         row["modality"],
    #         "flag_video_representation",
    #     ),
    #     axis=1,
    # )
    df["flag_hand_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["hand_landmarks_path"], row["modality"], "flag_hand_landmarks"
        ),
        axis=1,
    )
    df["flag_skeleton_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["skeleton_landmarks_path"], row["modality"], "flag_skeleton_landmarks"
        ),
        axis=1,
    )
    df["flag_collate_video"] = df.apply(
        lambda row: parse_flag(
            row["video_representation_path"], row["modality"], "flag_collate_video"
        ),
        axis=1,
    )    

    # Re-assess flag status
    df["flag_speech_recognition"] = df.apply(
        lambda row: parse_flag(
            row["speech_recognition_path"], row["modality"], "flag_speech_recognition"
        ),
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda row: parse_flag(
            row["speech_representation_path"],
            row["modality"],
            "flag_speech_representation",
        ),
        axis=1,
    )

    df["audio_modality"] = df.groupby(["participant_id"])["modality"].transform(
        lambda x: (
            "GoPro1"
            if "GoPro2" in x.tolist()
            else (
                "GoPro2"
                if "GoPro2" in x.tolist()
                else "GoPro3" if "GoPro3" in x.tolist() else np.nan
            )
        )
    )
    # Disabled the non-audio modality
    df["flag_speech_recognition"] = df.apply(
        lambda x: (
            x.flag_speech_recognition if x.modality == x.audio_modality else "disabled"
        ),
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda x: (
            x.flag_speech_representation
            if x.modality == x.audio_modality
            else "disabled"
        ),
        axis=1,
    )
    

    df_without_annotation = select_subset_missing_vir(df) 
    if len(df_without_annotation) > 0:

        print(f"Participants and modality missing visual inspection results (kept): N={len(df_without_annotation)}:")
        #display(df_without_annotation.head(3))
        
        display(
            df_without_annotation.drop_duplicates(["participant_id", "modality"])
            .groupby("participant_id")["modality"]
            .agg(list)
            .to_frame()
            .transpose())
        
    
    if verbose:
        # Plot proportion
        # df._merge.hist(); plt.title('Identifier from the dataset found or not in the visual inspection results')

        # print(f"Participants and modality missing visual inspection results (N={len(df_without_annotation)}):")


        # Value changes
        value_counts_after = df[cols_changed].apply(pd.Series.value_counts)
        value_counts_diff = value_counts_after - value_counts_before

        # Create a heatmap of the differences
        plt.figure(figsize=(5, 3))
        sns.heatmap(value_counts_diff, annot=True, cmap="coolwarm")
        plt.title("Changes after applying the manual annotaton results")
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.show()
        df = df.drop(columns="_merge")

    #     cols_to_change = [ 'video_representation_path',
    #        'speech_recognition_path', 'speech_representation_path',
    #        'hand_landmarks_path', 'skeleton_landmarks_path']

    #     for col in cols_to_change:
    #         df[col] = df.apply(lambda x: x[col].replace(x['modality'], x['true_modality']) if (x.is_swapped_modality == 1) else x[col], axis=1)

    return df

def select_subset_missing_vir(df):

    return df[(df["_merge"] == "left_only") & (df['video_path'].notna())]


def apply_visual_inspection_update_flags(df, verbose=False):
    # """Account for the visual inspection results to update flags for features computation.

    # 1) Add the columns to the dataset dataframe.

    # 2) Use `true_modality` to update the flags based on the modality: 


    # Note: since this is just a soft fixing solution (not hard, we don't move data or rename data files), it means that:
    #     - Identification information (mostly the modality) cannot be parsed anymore from paths but from identifiers.
    #     - Although video blocks embedding flags and path should not be modified (to point to the correct video data), we need to recalculte whether the features extracton have to be done for those that are modality-dependent (hand and skeleton and speech-related). 
    # TODO: Fix the upside down videos (transform them, keep as experiments their outputs and reset the folder to renew features extraction on fixed data).
    # TODO: Experiemnt: This is a source of features computation redundancy, for the speech notably. This means it gives easy multiple-instances of extraction of the speech and gaze. 

    # """
    cols_changed = [
        "modality",
        "flag_speech_recognition",
        "flag_speech_representation",
        "flag_video_representation",
        "flag_hand_landmarks",
        "flag_skeleton_landmarks",
        "flag_collate_video",
    ]
    cols = ["identifier",
        "task_name",
        "participant_id",
        "modality", 'folder_modality',
        "is_fish_eyed",
        "upside_down",
        "is_swapped_modality",
        "true_modality",
        "GP3_is_sink",
        "GP2_is_wrong_buttress",
        "GP2_above",
        "GoPro1_is_wrong_buttress",
        "is_middle_range",
        "true_task_name",
        "is_old_setup",
        "is_old_recipe",
        "annot_notes",
    ]

    # Init. columns
    value_counts_before = df[cols_changed].apply(pd.Series.value_counts)

    # Note: This results come from applying the visual inspection protocol to the gold folder already, having integrated the past (from results_1, reuslts_2, encapsulated in the results_all file).
    # This means that we would need to merge all files accounting for latest-file priority in the merge, to have all the results, or sum them up in the bar count table.
 
    results = pd.read_csv(os.path.join(get_data_root(),'dataframes', 'quality-control', 'results_visual_inspection_2111124_last.csv'),sep=';', usecols=cols)

    df = df.merge(
    results.drop(columns=["identifier", 'modality']),
    left_on=["task_name", "participant_id", "modality"],
    right_on = ['task_name', 'participant_id', 'folder_modality'], 
    how="left",
    indicator=verbose,
    )
    # Remove failed data collection
    n = len(df)
    df = df[
        (
            #(df["upside_down"] != 1)
             (df["true_task_name"] != np.nan)
            & (df["GP2_is_wrong_buttress"] != 1)
            & (df["GP2_above"] != 1)
            & (df["GoPro1_is_wrong_buttress"] != 1)
           # & (df["GP3_is_sink"] != 1)
            & (df["is_middle_range"] != 1)
        )
    ]
    print(f"--->Number of failed data collection (removed): {n - len(df)}")

    # df["flag_video_representation"] = df.apply(
    #     lambda row: parse_flag(
    #         row["video_representation_path"],
    #         row["modality"],
    #         "flag_video_representation",
    #     ),
    #     axis=1,
    # )
    df["flag_hand_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["hand_landmarks_path"], row["true_modality"], "flag_hand_landmarks"
        ),
        axis=1,
    )
    df["flag_skeleton_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["skeleton_landmarks_path"], row["true_modality"], "flag_skeleton_landmarks"
        ),
        axis=1,
    )
    df["flag_collate_video"] = df.apply(
        lambda row: parse_flag(
            row["video_representation_path"], row["true_modality"], "flag_collate_video"
        ),
        axis=1,
    )    

    # Re-assess flag status 
    df["flag_speech_recognition"] = df.apply(
        lambda row: parse_flag(
            row["speech_recognition_path"], row["true_modality"], "flag_speech_recognition"
        ),
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda row: parse_flag(
            row["speech_representation_path"],
            row["true_modality"],
            "flag_speech_representation",
        ),
        axis=1,
    )

    df["audio_modality"] = df.groupby(["participant_id"])["modality"].transform(
        lambda x: (
            "GoPro1"
            if "GoPro2" in x.tolist()
            else (
                "GoPro2"
                if "GoPro2" in x.tolist()
                else "GoPro3" if "GoPro3" in x.tolist() else np.nan
            )
        )
    )
    # Disabled the non-audio modality 
    df["flag_speech_recognition"] = df.apply(
        lambda x: (
            x.flag_speech_recognition if x.modality == x.audio_modality else "disabled"
        ),
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda x: (
            x.flag_speech_representation
            if x.modality == x.audio_modality
            else "disabled"
        ),
        axis=1,
    )

    if verbose:
        # Plot proportion
        # df._merge.hist(); plt.title('Identifier from the dataset found or not in the visual inspection results')

        df_without_annotation = df[df["_merge"] == "left_only"]
        print("Missing annotation on the following participants and modality:")
        display(
            df_without_annotation.drop_duplicates(["participant_id", "modality"])
            .groupby("participant_id")["modality"]
            .agg(list)
            .to_frame()
            .transpose()
        )

        # Value changes
        value_counts_after = df[cols_changed].apply(pd.Series.value_counts)
        value_counts_diff = value_counts_after - value_counts_before

        # Create a heatmap of the differences
        plt.figure(figsize=(5, 3))
        sns.heatmap(value_counts_diff, annot=True, cmap="coolwarm")
        plt.title("Changes after applying the manual annotaton results")
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.show()
        df = df.drop(columns="_merge")

    return df


def create_annotation_summary(dataset):
    # Filter relevant columns
    relevant_columns = [
        "is_swapped_modality",
        "is_fish_eyed",
        "upside_down",
        "GP3_is_sink",
        "GP2_is_wrong_buttress",
        "GP2_above",
        "GoPro1_is_wrong_buttress",
        "is_middle_range",
        "true_task_name",
        "is_old_setup",
        "is_old_recipe",
    ]   
    n_passations = dataset.participant_id.nunique()
    n_identifiers = dataset.identifier.nunique()

    df = dataset[relevant_columns]

    # Count occurrences of each validity condition
    validity_counts = df.sum()

    # Calculate proportions
    total_videos = len(dataset)
    proportions = validity_counts / total_videos

    # Combine counts and proportions for display
    summary = pd.DataFrame({
        "Count": pd.to_numeric(validity_counts, errors='coerce'),
        "Proportion": pd.to_numeric(proportions, errors='coerce'),
    }).sort_values(by="Proportion", ascending=False)

    # Plotting
    plt.figure(figsize=(6, 3))
    ax = sns.barplot(
        x=summary["Proportion"], y=summary.index,
    )

    plt.title(f"Summary of Manual Annotation Findings\n{n_passations} passations, {n_identifiers} modality folders (thumbnails)")
    plt.xlabel("Proportion")
    plt.ylabel("Validity Condition")

    # Add text labels for count and proportion on top of each bar
    for i, (count, proportion) in enumerate(zip(summary["Count"], summary["Proportion"])):
        ax.text(
            proportion + 0.01,  # Position text slightly outside each bar
            i,
            f"N={count}, {proportion:.2%}",
            ha="left",
            va="center",
        )

    plt.xlim((0, 1))
    plt.show()

