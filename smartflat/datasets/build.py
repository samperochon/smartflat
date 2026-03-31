"""Video metadata builder for the Smartflat dataset."""

import datetime
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd

from smartflat.annotation_smartflat import AnnotationSmartflat
from smartflat.constants import (
    available_modality,
    available_output_type,
    exluded_administrations,
    mapping_incorrect_modality_name,
)
from smartflat.datasets.filter import apply_gold_data_filtering, filter_outlier_video_names
from smartflat.datasets.quality_control import (
    apply_visual_inspection_results,
    apply_visual_inspection_update_flags,
)
from smartflat.datasets.utils import (
    add_test_bounds,
    append_annotations,
    append_clinical_data,
    append_percy_metadata,
    append_video_metadata,
    use_light_dataset,
)
from smartflat.utils.utils_coding import display_safe, green, red, yellow
from smartflat.utils.utils_io import (
    check_video,
    fetch_output_path,
    get_data_root,
    get_file_size_in_gb,
    parse_flag,
    parse_participant_id,
    parse_path,
    parse_task_number,
)

from smartflat.configs.smartflat_config import BaseSmartflatConfig
from smartflat.datasets.corrections import apply_manual_fixes  # noqa: F401 (re-export)

warnings.filterwarnings("ignore", category=FutureWarning)

_config = BaseSmartflatConfig()

_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".mpg", ".mpeg"]


def _discover_artifacts(root_folder, task_names, modality_to_explore):
    """Scan the data directory and collect all video artifacts into a DataFrame.

    Iterates over tasks / participants / modalities, collecting video files and
    their associated feature outputs (representations, speech, hands, skeleton).
    """
    rows = []

    for task in task_names:
        sbj_folders = glob(os.path.join(root_folder, task, "*"))
        print(f"Exploring {task} with {len(sbj_folders)} participants.")

        for sbj_folder in sbj_folders:
            participant_id = os.path.basename(sbj_folder)

            for modality in modality_to_explore:
                videos_dict = {}

                # Videos
                video_paths = glob(
                    os.path.join(root_folder, task, sbj_folder, modality, "*")
                )
                video_paths = [
                    p for p in video_paths
                    if (
                        any(p.lower().endswith(ext) for ext in _VIDEO_EXTENSIONS)
                        and "hand_landmarks" not in p
                        and "skeleton_landmarks_plot" not in p
                    )
                ]
                videos_dict = collect_videos(video_paths, videos_dict)

                # Video representations (VideoMAE-v2)
                video_representation_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "video_representations*.npy",
                ))
                videos_dict = collect_video_representations(
                    video_representation_paths, videos_dict
                )

                # Speech recognition (WhisperX)
                speech_recognition_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "speech_recognition*.json",
                ))
                videos_dict = collect_speech_recognition(
                    speech_recognition_paths, videos_dict
                )

                # Speech representations (e5-large)
                speech_representations_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "speech_representations*.npy",
                ))
                videos_dict = collect_speech_representations(
                    speech_representations_paths, videos_dict
                )

                # Hand landmarks (MediaPipe)
                hand_landmarks_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "hand_landmarks*.json",
                ))
                videos_dict = collect_hand_estimation(hand_landmarks_paths, videos_dict)

                # Skeleton landmarks (MediaPipe)
                skeleton_landmarks_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "skeleton_landmarks*.json",
                ))
                videos_dict = collect_skeleton_estimation(skeleton_landmarks_paths, videos_dict)

                # Tracked hand landmarks
                tracking_hand_landmarks_paths = glob(os.path.join(
                    root_folder, task, participant_id, modality,
                    "tracking_hand_landmarks*.json",
                ))
                videos_dict = collect_hand_processing(tracking_hand_landmarks_paths, videos_dict)

                for _video_name, video_dict in videos_dict.items():
                    rows.append(video_dict)

    df_video = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["video_path"])
    print(f'Number of unique participant_id: {df_video.participant_id.nunique() if len(df_video) else 0}')
    return df_video


def generate_video_metadata(
    root_folder,
    task_names=["cuisine", "lego"],
    modality_to_explore=["Tobii"],
    parse_video=False,
    do_check_videos=False,
    do_apply_fixation=True, 
    verbose=False,
):
    """
    All subject - modality folders can contain files of different nature (embedding, blur estimation, hands estimation, annotation, etc).

    This script first find the video names (1), and then populate a dataset containing a row per video. In case the same administration consists in multiple video, the way to go is as follows:
        - You need to hard code the order in the constants.py file
        - You need to create the mergedK embedding using the scripts in 00_dataset_construction.ipynb
        - Then at the end of this script, in case there were several videos and the "merged_*" one, we only keep the merged.

    Steps of this script are as follows:
    1) Parse all file in the modality folder looking for video (which do not have the .npy extension as videos), embdding, or blur_estimation, since these three strongly rely on the video. There fore, they are the only output to be able to create and generate new line (you only need at least one of them), and create a new line with that

    2) For each entry (if not already found) but should contain the name of the video file so that we do the maching at the end!)

    Notes:
    - For each video_name, we look for multiple ouputs potentially used by the algorithms (video path, embedding__path, blur_estimation_path, gram_outlier removal path, etc.).
    - `participant_id`: For now, whether the folder name is a trigram or a complete name (desired as it contains the date and more information), we use any of them as participant_id
    - The passation_id etc comes after filtering and sources-related additions.

    """
    if verbose:
        yellow("Entering generate_video_metadata")

    if root_folder is None:
        root_folder = get_data_root()

    # QC needs all tasks/modalities to swap e.g. GoPros, so we discover everything
    # and filter to the desired subset after QC is applied.
    desired_task_name = task_names
    desired_modality = modality_to_explore
    modality_to_explore = available_modality + [
        k
        for k, m in mapping_incorrect_modality_name.items()
        if m in modality_to_explore and k.lower() != m.lower()
    ]

    do_filter = 'refill' not in root_folder
    if verbose:
        yellow(f"Filter gold dataset (unexpected video name and gold-filtering): {do_filter}")

    # 1) Discover all video artifacts across tasks/participants/modalities
    df_video = _discover_artifacts(root_folder, task_names, modality_to_explore)

    if len(df_video) == 0:
        raise ValueError(
            f"No tasks/participant folder were found in {root_folder}.")

    df_video['folder_modality'] = df_video.modality
    
    # 2.1) Look for light versions in present
    df_video = use_light_dataset(df_video, light_by_default=False)

    # 3) Filter videos based on their video_name (non-standard names)
    df_video = filter_outlier_video_names(df_video, process=True, verbose=verbose)

    
    # 4) Utility and sanity check columns
    df_video["identifier"] = df_video.apply(
        lambda x: "{}_{}_{}_{}".format(
            x.participant_id, x.task_name, x.modality, x.video_name
        ),
        axis=1,
    )
    df_video.drop_duplicates("identifier", inplace=True)
    
    if verbose:
        n = len(df_video)
        print(f"--->Number of duplicates identifier removed: {n - len(df_video)}")
    df_video["has_video"] = df_video.video_path.apply(
        lambda x: 1 if type(x) == str else 0
    )
    df_video["folder_path"] = df_video.video_representation_path.apply(
        lambda x: os.path.dirname(x)
    )
    

    # 5) Define audio modality
    df_video["audio_modality"] = df_video.groupby(["participant_id"])[
        "modality"
    ].transform(
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

    # 6) Get feature extraction flags
    df_video = parse_flags(df_video)
    
    # 7) Merge with source dataset (Percy) dataframe and propage `n_videos` specification follow-up
    df_video = append_percy_metadata(df_video, verbose=verbose)
    df_video.loc[
        (df_video["flag_collate_video"] == "unprocessed") & (df_video["n_videos"] == 1),
        "flag_collate_video",
    ] = "disabled"  # TODO: a bit hacky
    df_video["has_collate_txt"] = df_video.apply(lambda row: (1 if (
                os.path.exists(
                    os.path.join(
                        os.path.dirname(row.video_representation_path),
                        "collate_videos.txt",
                    )
                )
                or (row.n_videos == 1)
            )
            else 0
        ),
        axis=1,
    )

    # 8) Merge with the videos metadata (fps, n_frames, duration, date, size)
    df_video = append_video_metadata(df_video)
    df_video["order"] = df_video.groupby(["task_name", "participant_id", "modality"])[
        "date"
    ].transform(lambda x: x.argsort() if pd.notnull(x).all() else [np.nan] * len(x))
    
    # 9) Append annotations
    df_video = append_annotations(df_video, verbose=verbose)

    # 9) Apply gold_data logic (remove partitions or uncollated videos or videos without fps metadata (used for calibrating the time axis).
    df_video = apply_gold_data_filtering(df_video, process=do_filter, verbose=verbose)
    #red('/!\ Passing gold data definition')

    # 10) Parse participant_id and sort per date (proxy)
    # To-fix

    df_video[["task_number", "diag_number", "trigram", "date_folder"]] = df_video["participant_id"].apply(parse_participant_id)
    df_video["task_number_int"] = df_video.task_number.apply(parse_task_number)
    print(f'Number of unique pid: {df_video.participant_id.nunique()}')

    print(f'Number of unique trigram: {df_video.trigram.nunique()}')


    # 11) Add clinical data
    df_video = append_clinical_data(df_video, verbose=verbose)
    

    # Remove potential duplicates #TODO: fix consitency since performed more upstream in the function 
    df_video.drop_duplicates("identifier", inplace=True)

    # 12) Final - Apply visual inspection results
    if do_apply_fixation:
        
        #red("/!\ Disable correction from visual inspection (update flags)")
        if verbose:
            green("/!\ Apply correction from visual inspection")
        df_video = apply_visual_inspection_results(df_video, verbose=verbose)
    else:
        red("/!\ Disable correction from visual inspection (update flags)")
        df_video = apply_visual_inspection_update_flags(df_video, verbose=verbose)
    

    # 12.5) Actually filters for task and modality (since QC need them all anyway to swap e.g.g GoPros....)
    n = len(df_video)
    #display(df_video[(~df_video.task_name.isin(desired_task_name)) & (~df_video.modality.isin(desired_modality))])
    df_video = df_video[(df_video.task_name.isin(desired_task_name)) & (df_video.modality.isin(desired_modality))]
    print(f'--->Filtering for task and modality: removed {n - len(df_video)} to final {len(df_video)} rows. ')
    
    # 13) Compute processed flag (TODO: do it with the flag_features all after the visual inspection integration ?)
    df_video["processed"] = df_video.apply(check_processed, axis=1)
    
    # 13) Remove participants without video or undetemined test bounds 
    pids_without_videos = exluded_administrations.keys()
    n = len(df_video)
    df_video = df_video[~(df_video['participant_id'].isin(pids_without_videos) & (df_video['modality'] == 'Tobii'))]
    print(f'--->Filtering for exluded administrations: removed {n - len(df_video)} to final {len(df_video)} rows. ')
    df_video = df_video[df_video['participant_id'] != 'G45_P32_CARCat_17072018']
    red('Removing participant G45_P32_CARCat_17072018 due to missing bounds (corrupted video)')
    # 14) Add test bounds 
    df_video = add_test_bounds(df_video, verbose=verbose)
    #print('Done add_test_bounds')
    # 15) Polish, sort and reset_index of the dataset metadata
    df_video.sort_values(["task_name", "task_number_int"], ascending=True, inplace=True)
    df_video["video_id"] = df_video.identifier.factorize()[0]
    
    if do_check_videos:
        df_video["is_checked"] = df_video.video_path.apply(
            lambda x: int(check_video(x)) if type(x) == str else np.nan
        )
    else:
        df_video["is_checked"] = df_video.fps.notna().astype(int)
        
    # Remove potential duplicates #TODO: fix consitency since performed more upstream in the function 
    df_video.drop_duplicates("identifier", inplace=True)
    df_video['has_fps'] = df_video.fps.notna().astype(int)
    #display(df_video.groupby(['task_name', 'modality', 'has_video']).has_fps.value_counts().to_frame())

    # # Final checks and tests
    # assert df_video.n_identifiers_per_fmod.max() == 1

    # if len(df_video[df_video['passation_id'].isna()]) > 0:
    #    print("Some passation_id are not registered.")
    #print('Is it found ? ', df_video[df_video['participant_id'].apply(lambda x: x == 'G161_AMEAmo_SDS2_P_14062024_ M24_V3_gateau')].video_path.to_list())



    return df_video



# Create all-from-one registration file functionality (allow to search for other features presences fronm a single file)

def collect_videos(video_paths, videos_dict, parse_video=False):

    for video_path in video_paths:

        video_name, modality, participant_id, folder = parse_path(video_path)

        if video_name.startswith("Recording"):
            continue

        if parse_video:
            try:
                print("Parsing video...")
                vr = video_loader(video_path)
                fps = vr.get_avg_fps()
                n_frames = vr._num_frame
                size = get_file_size_in_gb(video_path)
                timestamp = os.path.getctime(video_path)
                date = datetime.datetime.fromtimestamp(timestamp)

            except:
                print("Error parsing video {}".format(video_path))
                fps, n_frames, size, timestamp, date = 5 * [np.nan]
        else:
            fps, n_frames, size, timestamp, date = 5 * [np.nan]

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_path": video_path,
            "video_name": video_name,
            "registration_file": os.path.basename(video_path),
            #'n_frames': n_frames,
            #'fps':fps,
            # 'date': date,
            #'size': size
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(video_path)

            videos_dict[video_name].update(missing_data_dict)

    return videos_dict


def collect_video_representations(video_representation_paths, videos_dict):

    for video_representation_path in video_representation_paths:

        video_name, modality, participant_id, folder = parse_path(
            video_representation_path
        )

        update_dict = {
            "task_name": folder,  # Deprecated 'task_name': 'cuisine' if ('cuisine' in folder or 'gateau' in folder) else 'lego' if 'lego' in folder else folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(video_representation_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(video_representation_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict


def collect_speech_recognition(speech_recognition_paths, videos_dict):

    for speech_recognition_path in speech_recognition_paths:

        video_name, modality, participant_id, folder = parse_path(
            speech_recognition_path
        )

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(speech_recognition_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(speech_recognition_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict


def collect_speech_representations(speech_representations_paths, videos_dict):

    for speech_representations_path in speech_representations_paths:

        video_name, modality, participant_id, folder = parse_path(
            speech_representations_path
        )

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(speech_representations_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(speech_representations_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict


def collect_hand_estimation(hand_landmarks_paths, videos_dict):

    for hand_landmarks_path in hand_landmarks_paths:

        video_name, modality, participant_id, folder = parse_path(hand_landmarks_path)

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(hand_landmarks_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(hand_landmarks_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict

def collect_hand_processing(hand_landmarks_paths, videos_dict):

    for hand_landmarks_path in hand_landmarks_paths:

        video_name, modality, participant_id, folder = parse_path(hand_landmarks_path)

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(hand_landmarks_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(hand_landmarks_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict

def collect_skeleton_estimation(skeleton_landmarks_paths, videos_dict):

    for skeleton_landmarks_path in skeleton_landmarks_paths:

        video_name, modality, participant_id, folder = parse_path(
            skeleton_landmarks_path
        )

        update_dict = {
            "task_name": folder,
            "participant_id": participant_id,
            "modality": modality,
            "video_name": video_name,
            "registration_file": os.path.basename(skeleton_landmarks_path),
        }

        if video_name in videos_dict.keys():
            videos_dict[video_name].update(update_dict)
        else:
            videos_dict[video_name] = update_dict

            # Append features status if not registered yet
            missing_data_dict = create_features_metadata(skeleton_landmarks_path)
            videos_dict[video_name].update(missing_data_dict)

    return videos_dict


def collect_annotations(annotation_paths, videos_dict):
    """Deprecated"""
    if len(annotation_paths) > 1:
        #yellow("[WARNING] Multiple json annotation path here: {}".format(annotation_paths))
        annotation_path = annotation_paths[0]
        
        task_name = 'cuisine' if 'cuisine' in annotation_path else 'lego' if 'lego' in annotation_path else np.nan

        annotation = AnnotationSmartflat(task_name=task_name, annotation_path=annotation_path)
        # annotations_software = 'vidat'
        annotations_software = annotation.annotations_software
        has_annotation = True

    elif len(annotation_paths) == 1:

        annotation_path = annotation_paths[0]
        task_name = 'cuisine' if 'cuisine' in annotation_path else 'lego' if 'lego' in annotation_path else np.nan

        annotation = AnnotationSmartflat(task_name=task_name, annotation_path=annotation_path)
        annotations_software = annotation.annotations_software
        has_annotation = True

    else:

        annotation = np.nan
        annotation_path = np.nan
        annotations_software = np.nan
        has_annotation = False

        return videos_dict

    _, modality, participant_id, folder = parse_path(annotation_path)
    video_name = "placehol"


    update_dict = {
        "task_name": folder,
        "participant_id": participant_id,
        "modality": modality,
        "video_name": video_name,
        "registration_file": os.path.basename(annotation_path),
    }

    if len(videos_dict) == 0:

        videos_dict[video_name] = update_dict

        # Append features status if not registered yet
        missing_data_dict = create_features_metadata(annotation_path)
        videos_dict[video_name].update(missing_data_dict)

        # Add the annnotation
        videos_dict[video_name].update(
            {
                "annotation": annotation,
                "has_annotation": has_annotation,
                "annotation_path": annotation_path,
                "annotations_software": annotations_software,
            }
        )
    else:

        for video_name in videos_dict.keys():

            videos_dict[video_name].update(
                {
                    "annotation": annotation,
                    "has_annotation": has_annotation,
                    "annotation_path": annotation_path,
                    "annotations_software": annotations_software,
                }
            )

    return videos_dict


# Local datset builder utility functions

def create_features_metadata(video_path):
    # Collect metadata
    video_representation_path = fetch_output_path(video_path, _config.video_model_name)
    speech_recognition_path = fetch_output_path(
        video_path, _config.speech_recognition_model_name
    )
    speech_representation_path = fetch_output_path(
        video_path, _config.speech_embedding_model_name
    )
    hand_landmarks_path = fetch_output_path(video_path, _config.hand_landmarks_model_name)
    skeleton_landmarks_path = fetch_output_path(
        video_path, _config.skeleton_landmarks_model_name
    )
    tracking_hand_landmarks_path = fetch_output_path(
        video_path, _config.tracking_hand_landmarks_model_name
    )

    return {  # Video embedding from Video Foundation Model
        "video_representation_path": video_representation_path,
        "video_representation_computed": os.path.isfile(video_representation_path),
        # Speech recognition
        "speech_recognition_path": speech_recognition_path,
        "speech_recognition_computed": os.path.isfile(speech_recognition_path),
        # Speech representation
        "speech_representation_path": speech_representation_path,
        "speech_representation_computed": os.path.isfile(speech_representation_path),
        # Hand landmarks
        "hand_landmarks_path": hand_landmarks_path,
        "hand_landmarks_computed": os.path.isfile(hand_landmarks_path),
        # Skeleton landmarks
        "skeleton_landmarks_path": skeleton_landmarks_path,
        "skeleton_landmarks_computed": os.path.isfile(skeleton_landmarks_path),
        
        "tracking_hand_landmarks_path": tracking_hand_landmarks_path,
        "tracking_hand_landmarks_computed": os.path.isfile(tracking_hand_landmarks_path),
    }

def check_processed(x):
    """Filters feature-extracted samples"""
    # try:
    return (  # (x.in_disk) &
        ((x.flag_speech_representation in ["disabled", "success", "failure"])
        & (x.flag_speech_recognition in ["disabled", "success", "failure"])
        & (x.flag_video_representation in ["disabled", "success", "failure"])
        & (x.flag_hand_landmarks in ["disabled", "success", "failure"])  
        & (x.flag_tracking_hand_landmarks in ["disabled", "success", "failure"])  

        & (x.flag_skeleton_landmarks in ["disabled", "success", "failure"]) )|
           (x.has_light_video == True)
    )
    # except:
    #    print(x)

def parse_flags(df_video, verbose=False):
    """Add the two types of flags: (i) for the video consolidation (collate_video) process when needed and (ii) for the feature extraction process."""

    # 1) For the video consolidation (collate_video) process when needed
    df_video["flag_collate_video"] = df_video.apply(
        lambda row: parse_flag(
            row["video_representation_path"], row["modality"], "flag_collate_video"
        ),
        axis=1,
    )  # 1/2 before adding n_video == 1

    # 2) For the feature extraction process
    for output_type in available_output_type:

        df_video[f"flag_{output_type}"] = df_video.apply(
            lambda x: parse_flag(
                x[f"{output_type}_path"], x["modality"], f"flag_{output_type}"
            ),
            axis=1,
        )

    # Audio-modality specificities
    df_video["flag_speech_recognition"] = df_video.apply(
        lambda x: (
            x.flag_speech_recognition if x.modality == x.audio_modality else "disabled"
        ),
        axis=1,
    )
    df_video["flag_speech_representation"] = df_video.apply(
        lambda x: (
            x.flag_speech_representation
            if x.modality == x.audio_modality
            else "disabled"
        ),
        axis=1,
    )

    # Plot flag intermediary step
    if verbose:
        yellow(
            "Plot flag intermediary state (without accounting for presence of the output)"
        )
        display_safe(
            df_video[[f"flag_{output_type}" for output_type in available_output_type]]
            .apply(pd.Series.value_counts)
            .fillna(0)
            .astype(int)
        )

    # 3) Final flag is the combination of (i) the output is present and (ii) the feature-extraction process succeded.
    for output_type in available_output_type:

        df_video[f"flag_{output_type}"] = df_video.apply(
            lambda x: (
                "success"
                if (
                    x[f"{output_type}_computed"]
                    and (x[f"flag_{output_type}"] == "success")
                ) else
                "unprocessed" if not x[f"{output_type}_computed"]
                else x[f"flag_{output_type}"]
            ),
            axis=1,
        )

    if verbose:
        yellow("Plot flag final state")
        display_safe(
            df_video[[f"flag_{output_type}" for output_type in available_output_type]]
            .apply(pd.Series.value_counts)
            .fillna(0)
            .astype(int)
        )

    # df_video['flag_video_representation'] = df_video.apply(lambda row: parse_flag(row['video_representation_path'], row['modality'], 'flag_video_representation'), axis=1)
    # df_video['flag_hand_landmarks'] = df_video.apply(lambda row: parse_flag(row['hand_landmarks_path'], row['modality'], 'flag_hand_landmarks'), axis=1)
    # df_video['flag_skeleton_landmarks'] = df_video.apply(lambda row: parse_flag(row['skeleton_landmarks_path'], row['modality'], 'flag_skeleton_landmarks'), axis=1)
    # df_video['flag_speech_recognition'] = df_video.apply(lambda row: parse_flag(row['speech_recognition_path'], row['modality'], 'flag_speech_recognition'), axis=1)
    # df_video['flag_speech_recognition'] = df_video.apply(lambda x: x.flag_speech_recognition if x.modality == x.audio_modality else 'disabled', axis=1)
    # df_video['flag_speech_representation'] = df_video.apply(lambda row: parse_flag(row['speech_representation_path'], row['modality'], 'flag_speech_representation'), axis=1)
    # df_video['flag_speech_representation'] = df_video.apply(lambda x: x.flag_speech_representation if x.modality == x.audio_modality else 'disabled', axis=1)

    return df_video
