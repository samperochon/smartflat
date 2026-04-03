"""Dataset folder structure initialization and collation output validation.

When to run: Before feature extraction to ensure standard folder hierarchy,
    or after collation to validate merged video outputs.
Prerequisites: Metadata CSV with task_name and participant_id columns.
Outputs: Creates task/participant/modality folder structure; validation reports.
Usage: python -m smartflat.features.consolidation.main_housekeeping
"""

import argparse
import os
import re
import socket
import subprocess
import sys
from glob import glob

import numpy as np
import pandas as pd

,
)

from smartflat.constants import (
    available_modality,
    expected_folders,
    mapping_incorrect_modality_name,
)
from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_collate import remove_output_files
from smartflat.utils.utils_io import (
    check_video,
    fetch_flag_path,
    get_data_root,
    get_file_size_in_gb,
    get_gold_data_path,
    parse_participant_id,
)
from smartflat.utils.utils_visualization import print_get_last_modified_date

from smartflat.utils.utils_coding import green, select

def extract_file_paths(file_path):
    paths = []
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"file '(.*)'", line)
            if match:
                paths.append(match.group(1))
    return paths


def init_dataset_structure(path_metadata):
    """Init participants folders."""

    
    metadata = pd.read_csv(path_metadata)
    for (task_name, participant_id), _ in metadata.groupby(
        ["task_name", "participant_id"]
    ):  
        task_folder_path = os.path.join(get_data_root(), task_name)
        participant_folder_path = os.path.join(task_folder_path, participant_id)

        if not os.path.exists(task_folder_path):
            os.makedirs(task_folder_path, exist_ok=True)
            print(f"Created task folder: {task_folder_path}")
        else:
            print(f"Task folder already exists: {task_folder_path}")

        if not os.path.exists(participant_folder_path):
            os.makedirs(participant_folder_path, exist_ok=True)
            print(f"Created participant folder: {participant_folder_path}")
        else:
            print(f"Participant folder already exists: {participant_folder_path}")

        for expected_folder in expected_folders:
            folder_path = os.path.join(participant_folder_path, expected_folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created expected folder: {folder_path}")
            else:
                print(f"Expected folder already exists: {folder_path}")


def check_collate_outputs(root_dir=None, process=False):
    """Check that the collate checks are correct wrt the global metadata

    Check on the collation of the merged video. This script is used to verify that the merged video is of correct size as
    found in the global metadata registration.

    Process:
    We loop over the registered merged_video present in the machine. If the merged video is not here, we cannot check, and so the
    the outputs registering the merged video are checked using another function TODO.
    If the sizes don't match, we remove all outputs and video corresponding to the merge video.

    """

    if root_dir is None:
        root_dir = get_data_root()

    dset = get_dataset(dataset_name="base", root_dir=root_dir, scenario="all")
    df = dset.metadata
    gdf = dset.global_metadata

    commands = []
    for (participant_id, modality), group in df[
        df["video_name"] == "merged_video"
    ].groupby(["participant_id", "modality"]):

        # Compute size of the output video
        assert len(group) == 1
        merged_video_path = group.video_path.iloc[0]

        if group.video_path.isna().iloc[0]:
            print(
                f"Missing merged_video for {participant_id} and {modality} check aborted."
            )
            continue

        size_merged_video = get_file_size_in_gb(merged_video_path)

        # Compute source video sizes and check source videos metadata is correct
        cdf = gdf[
            (gdf["participant_id"] == participant_id) & (gdf["modality"] == modality)
        ]
        filepath = os.path.join(
            os.path.dirname(group["video_representation_path"].iloc[0]),
            "collate_videos.txt",
        )
        if not os.path.isfile(filepath):
            # Merged perform on another machine
            if modality != "Tobii":
                # Process the videos that have standard names (8 characters)
                order = ["GOPR", "GP01", "GP02", "GP03", "GP04", "GP05", "GP06", "GP07"]
                cdf["st"] = cdf.video_name.apply(lambda x: x[:4])
                assert np.sum([0 if st in order else 1 for st in cdf.st]) == 0
                cdf["end"] = cdf.video_name.apply(
                    lambda x: int(x[4:])
                )  # .unique()  #[['participant_id', 'modality', 'video_name']]
                cdf["st_rank"] = cdf["st"].apply(
                    lambda x: order.index(x) if x in order else len(order)
                )
                cdf = cdf.sort_values(
                    by=["participant_id", "modality", "st_rank", "end"]
                )
            else:

                cdf = cdf.sort_values(["participant_id", "modality", "order"])

            video_paths = [
                os.path.join(
                    get_data_root(),
                    group.task_name.iloc[0],
                    participant_id,
                    modality,
                    video_name,
                )
                for video_name in cdf.video_name.unique()
            ]

            if process:
                write_collate(video_paths)
            else:
                print(f"Write missing collate ?   {video_paths}")

        source_video_paths = extract_file_paths(filepath)
        assert (
            cdf["n_videos"].iloc[0] == len(cdf) == len(source_video_paths)
            and len(cdf) > 1
        )
        total_size = cdf["size"].sum()

        is_merged = np.isclose(total_size, size_merged_video, 0.05)
        if not is_merged:
            print(
                f"Report: {participant_id}-{modality} | n_videos: {cdf.n_videos.iloc[0]} | total size: {total_size} | merged size: {size_merged_video} | OKAY ?: {is_merged}"
            )
            commands = remove_output_files(
                group,
                process=process,
                include_videos_representations=True,
                include_videos=True,
            )
            with open(
                os.path.join(
                    os.path.dirname(merged_video_path), ".merged_video_collate_flag.txt"
                ),
                "w",
            ) as f:
                f.write("failure")
        else:
            with open(
                os.path.join(
                    os.path.dirname(merged_video_path), ".merged_video_collate_flag.txt"
                ),
                "w",
            ) as f:
                f.write("success")
    return commands

 
def clean_outputs_while_disabled(process=False):

    dset = get_dataset(dataset_name="smartflat", scenario="all", verbose=False)
    df = dset.metadata
    commands = []
    for path in df[
        (df["flag_skeleton_landmarks"] == "disabled")
        & (df["skeleton_landmarks_computed"])
    ].skeleton_landmarks_path.tolist():
        commands.append(["rm", path])
        commands.append(["rm", fetch_flag_path(path, "flag_skeleton_landmarks")])

    for path in df[
        (df["flag_hand_landmarks"] == "disabled") & (df["hand_landmarks_computed"])
    ].hand_landmarks_path.tolist():
        commands.append(["rm", path])
        commands.append(["rm", fetch_flag_path(path, "flag_hand_landmarks")])

    if process:
        for command in commands:
            subprocess.run(command)

    return commands


def send_hardrive_to_remote(
    metadata_remote_path,
    metadata_local_path,
    path_df_external=None,
    remote_name="cheetah",
    is_gold=True,
    inverse=False,
    process=False,
):

    df_remote = pd.read_csv(metadata_remote_path)
    df_local = pd.read_csv(metadata_local_path)
    if path_df_external is not None:
        df_external = pd.read_csv(path_df_external)
        df_external.insert(
            0,
            "min_identifier",
            df_external.apply(
                lambda x: "{}_{}_{}".format(x.participant_id, x.task_name, x.modality),
                axis=1,
            ),
        )
        print_get_last_modified_date(path_df_external)

    print('Add number of videos in case it is missing... Gold dataet is supposed to be collated already)')
    df_local['n_videos'] = df_local.groupby(['task_name', 'participant_id', 'modality'])['identifier'].transform(lambda x: len(np.unique(x)))
    n = len(df_local)
    # Send only the complete videos
    df_local = df_local[
       # ((df_local["n_videos"] == 1) | (df_local["video_name"] == "merged_video"))
        (df_local["processed"] == False)
    ]
    df_local = df_local[(df_local["task_name"] == "cuisine") & (df_local["modality"] == "Tobii")]
    print(f"TODO: Keeping non-processed cooking tobii video only {n} -> {len(df_local)}")

    df_remote.insert(
        0,
        "min_identifier",
        df_remote.apply(
            lambda x: "{}_{}_{}".format(x.participant_id, x.task_name, x.modality),
            axis=1,
        ),
    )
    df_local.insert(
        0,
        "min_identifier",
        df_local.apply(
            lambda x: "{}_{}_{}".format(x.participant_id, x.task_name, x.modality),
            axis=1,
        ),
    )

    # TOFIX
    # df_local.loc[(df_local['task_name'] == 'cuisine') & (df_local['participant_id'].apply(lambda x: True if (x.startswith('L') and x != 'Lamboust Théo soins courant') else False)), 'task_name'] = 'lego'
    # df_local.loc[df_local['is_SC'], 'participant_id'] = df_local.loc[df_local['is_SC'], 'participant_id'].apply(lambda x: 'SC_' + x)

    print_get_last_modified_date(metadata_remote_path)
    print_get_last_modified_date(metadata_local_path)

    n = len(df_local)
    df_local = df_local.dropna(subset=["video_path"])
    print(
        "Keeping only present videos: {} -> {} ()".format(
            n, len(df_local), df_local["size"].sum()
        )
    )

    #print("/!\ Reversing order of the videos to send")
    #df_local = df_local.sample(frac=1).reset_index(drop=True)
    df_local = df_local[df_local["task_name"].isin(["cuisine", "lego"])].sort_values(
        ["task_name", "modality"], ascending=[True, False]
    )
    
    print("Number of videos to send: {}".format(len(df_local)))
    print('Tobii cuisine videos: :-):')
    print(len(select(select(select(select(df_local, 'task_name', 'cuisine'), 'modality', 'Tobii'), 'video_representation_computed', False), 'flag_video_representation', 'unprocessed')))

    print("Total size to send: {} Go".format(df_local["size"].sum()))
    
    commands = []
    size_to_send = 0
    for i, row in df_local.iterrows():

        # # File already present in the target remote
        # if row.min_identifier in df_remote.min_identifier.to_list():
        #     pass#continue

        # if (
        #     path_df_external is not None
        #     and row.min_identifier in df_external.min_identifier.to_list()
        # ):
        #     continue

        # if (
        #     row.modality in mapping_incorrect_modality_name.keys()
        #     and not row.modality in available_modality
        # ):

        #     raise ValueError(
        #         f"Invalid modality folder found {row.video_path}. Consolidate the dataset first."
        #     )
        # else:
        #     modality = row.modality # We apply soft solution here: keeping videos and outputs trored in their source folder
        modality = row.modality
        cdf = df_remote[
            (df_remote["participant_id"] == row.participant_id)
            & (df_remote["modality"] == modality)
        ]
        
        print(f"Sending {row.participant_id} {modality} {row.task_name}")

        # Don't upload if a collate version exists in the remote
        # if len(cdf) > 0:# and (cdf.has_merged_video == True).all():
        #     continue

        # else:
        if is_gold:
            output_folder = os.path.join(
                get_gold_data_path(machine_name=remote_name),
                row.task_name,
                row.participant_id,
                modality,
            )

        else:
            output_folder = os.path.join(
                get_data_root(machine_name=remote_name),
                row.task_name,
                row.participant_id,
                modality,
            )

        flag_paths = glob(os.path.join(os.path.dirname(row.video_path), ".*flag*"))
        for flag_path in flag_paths:
            commands.append(
                [
                    "rsync",
                    "-ahuvzL",
                    "--progress",
                    flag_path,
                    f"{remote_name}:{output_folder}",
                ]
            )

        commands.append(
            [
                "rsync",
                "-ahuvzL",
                "--progress",
                row.video_path,
                f"{remote_name}:{output_folder}",
            ]
        )
        size_to_send += get_file_size_in_gb(row.video_path)

    if inverse:
        commands = list(reversed(commands))

    if process:
        for i, command in enumerate(commands):
            green(' '.join(command))

        for i, command in enumerate(commands):
            if os.path.isfile(os.path.join(command[-1], os.path.basename(command[-2]))):
                print(
                    "File already present:",
                    os.path.join(command[-1], os.path.basename(command[-2])),
                )
            else:
                print(
                    f"[{i}/{len(commands)}] Sending {command[-2]} to {command[-1]} ({size_to_send} Go remaining)"
                )
                print(" ".join(command))
                
                subprocess.run(command)
                # if check_video(command[-2]):
                #     subprocess.run(command)
                # else:
                #     print("[FAILURE] failed check_video: {}".format(command[-2]))
            print("Done.")
    else:
        print("Videos to copy: {}".format(len(commands)))

    return commands


def parse_args():

    parser = argparse.ArgumentParser(
        description="Clean dataset consolidation and computation"
    )
    parser.add_argument(
        "-p", "--process", action="store_true", default=False, help="Process"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Check on outputs and computation
    check_collate_outputs(process=args.process)
    commands = clean_outputs_while_disabled(process=args.process)
