"""Main scripts to be run in Percy to consolidate the dataset, clean and refill the disk, etc"""

import argparse
import os
import re
import shutil
import socket
import sys
from time import time

import numpy as np
import pandas as pd

from smartflat.constants import expected_folders
from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_io import get_data_root, get_file_size_in_gb, parse_participant_id


def send_percy_to_harddrive_to_complete_remote(
    path_df_remote, path_df_harddrive, path_metadata, output_folder, process=False
):
    """Fill the hard drive with videos to complete the missing modalities in the remote."""

    df = pd.read_csv(path_df_remote)
    drivedf = pd.read_csv(path_df_harddrive)
    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )

    video_paths = []
    for (task_name, participant_id, modality), group in df.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        has_merged_video = (group.has_merged_video == True).all()
        is_complete = (
            group[group["video_name"] != "merged_video"].is_complete == True
        ).all()

        if has_merged_video:
            continue
        elif is_complete:
            continue
        else:

            # The Percy computer don't have those sorted, so the modality is likely to be Go Pro and so no matching are pperformed since mdodality is '?'
            # Solution: we sort all folder (clinical team or technical team) in Percy, and will have to do a short script enforcing structre from the metadata

            if (
                "?"
                in metadata[
                    (metadata["participant_id"] == participant_id)
                ].modality.unique()
            ):
                continue

            cands = metadata[
                (metadata["participant_id"] == participant_id)
                & (metadata["modality"] == modality)
                & (~metadata["video_name"].isin(group.video_name.tolist()))
            ]

            if len(cands) == 0:
                continue  # raise ValueError

            elif len(cands) > 0:

                # i)
                os.makedirs(os.path.join(output_folder), exist_ok=True)
                os.makedirs(os.path.join(output_folder, task_name), exist_ok=True)
                os.makedirs(
                    os.path.join(output_folder, task_name, participant_id),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(output_folder, task_name, participant_id, modality),
                    exist_ok=True,
                )

                assert (cands.video_path.isna() == False).all()

                # ii)
                if process:
                    for video_path, size in zip(
                        cands.video_path.tolist(), cands["size"].tolist()
                    ):
                        if os.path.isfile(
                            os.path.join(
                                output_folder,
                                task_name,
                                participant_id,
                                modality,
                                os.path.basename(video_path),
                            )
                        ):
                            print(
                                "Video already in the hard drive: {}".format(video_path)
                            )
                            continue
                        try:
                            shutil.copy2(
                                video_path,
                                os.path.join(
                                    output_folder, task_name, participant_id, modality
                                ),
                            )
                            print(
                                "Copied {} to the hard drive ({:.2f} Go).".format(
                                    video_path, size
                                )
                            )
                        except:
                            print(
                                "[FAILURE] to copy {} to the hard drive.".format(
                                    video_path
                                )
                            )

                else:
                    video_paths.extend(cands.video_path.dropna().tolist())
    print("Files to process: {}".format(len(video_paths)))
    return video_paths


def send_percy_to_harddrive(
    path_df_remote, path_metadata, output_folder, other_location=None, process=True
):
    """Fill the hard drive with videos not yet represented in the remote."""

    if isinstance(path_df_remote, list):
        df = pd.concat([pd.read_csv(path) for path in path_df_remote])
    else:
        df = pd.read_csv(path_df_remote)

    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )

    video_paths = []
    total_size = 0
    ts = time()
    for (task_name, participant_id, modality), group in metadata.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        cdf = df[
            (df["participant_id"] == participant_id) & (df["modality"] == modality)
        ]

        if modality == "?":
            continue
        if len(cdf) == 0:

            os.makedirs(os.path.join(output_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, task_name), exist_ok=True)
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id, modality),
                exist_ok=True,
            )

            # ii)
            if process:
                for video_path, size in zip(
                    group.video_path.tolist(), group["size"].tolist()
                ):
                    if os.path.isfile(
                        os.path.join(
                            output_folder,
                            task_name,
                            participant_id,
                            modality,
                            os.path.basename(video_path),
                        )
                    ):
                        print("Video already in the hard drive: {}".format(video_path))
                        continue
                    try:
                        shutil.copy2(
                            video_path,
                            os.path.join(
                                output_folder, task_name, participant_id, modality
                            ),
                        )
                        print(
                            "Copied {} to the hard drive ({:.2f} Go).".format(
                                video_path, size
                            )
                        )
                        total_size += size
                    except:
                        print(
                            "[FAILURE] to copy {} to the hard drive.".format(video_path)
                        )

            else:
                video_paths.extend(group.video_path.dropna().tolist())
                total_size += group["size"].sum()
    print(
        "Done. Sent {:.2f} Go in {:.2f} seconds ({:.2f} hours).".format(
            total_size, time() - ts, (time() - ts) / 3600
        )
    )
    print("Files to process: {} for {:.0f} Go".format(len(video_paths), total_size))
    return video_paths


def send_percy_to_harddrive_vs_local(
    path_metadata, output_folder, other_locations=None, process=True
):
    """Fill the hard drive with videos not yet represented in the remote."""

    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )

    video_paths = []
    total_size = 0
    ts = time()
    for (task_name, participant_id, modality), group in metadata.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        if modality == "?":
            continue

        try:
            os.makedirs(os.path.join(output_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, task_name), exist_ok=True)
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id, modality),
                exist_ok=True,
            )

            # ii)

            for video_path, size in zip(
                group.video_path.tolist(), group["size"].tolist()
            ):

                for other_location in other_locations:
                    output_file_check = os.path.join(
                        other_location,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )

                    if os.path.isfile(output_file_check):
                        # print('Video already in the hard drive: {}'.format(output_file_check))
                        continue

                if os.path.isfile(
                    os.path.join(
                        output_folder,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )
                ):
                    # print('Video already in the output folder: {}'.format(video_path))
                    continue

                if process:
                    try:
                        shutil.copy2(
                            video_path,
                            os.path.join(
                                output_folder, task_name, participant_id, modality
                            ),
                        )
                        print(
                            "Copied {} to the hard drive ({:.2f} Go).".format(
                                video_path, size
                            )
                        )
                        total_size += size
                    except:
                        print(
                            "[FAILURE] to copy {} to the hard drive.".format(video_path)
                        )

                else:
                    video_paths.extend(group.video_path.dropna().tolist())
                    total_size += group["size"].sum()
        except:
            print(
                "Error while looping on the metadata table...",
                task_name,
                participant_id,
                modality,
            )
            continue

    print(
        "Done. Sent {:.2f} Go in {:.2f} seconds ({:.2f} hours).".format(
            total_size, time() - ts, (time() - ts) / 3600
        )
    )
    print("Files to process: {} for {:.0f} Go".format(len(video_paths), total_size))
    return video_paths


def send_percy_to_harddrive_vs_local(
    path_metadata, output_folder, other_locations=None, process=True
):
    """Fill the hard drive with videos not yet represented in the remote."""

    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )

    video_paths = []
    total_size = 0
    ts = time()
    for (task_name, participant_id, modality), group in metadata.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        if modality == "?":
            continue

        try:
            os.makedirs(os.path.join(output_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, task_name), exist_ok=True)
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id, modality),
                exist_ok=True,
            )

            # ii)

            for video_path, size in zip(
                group.video_path.tolist(), group["size"].tolist()
            ):

                for other_location in other_locations:
                    output_file_check = os.path.join(
                        other_location,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )

                    if os.path.isfile(output_file_check):
                        # print('Video already in the hard drive: {}'.format(output_file_check))
                        continue

                if os.path.isfile(
                    os.path.join(
                        output_folder,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )
                ):
                    # print('Video already in the output folder: {}'.format(video_path))
                    continue

                if process:
                    try:
                        shutil.copy2(
                            video_path,
                            os.path.join(
                                output_folder, task_name, participant_id, modality
                            ),
                        )
                        print(
                            "Copied {} to the hard drive ({:.2f} Go).".format(
                                video_path, size
                            )
                        )
                        total_size += size
                    except:
                        print(
                            "[FAILURE] to copy {} to the hard drive.".format(video_path)
                        )

                else:
                    video_paths.extend(group.video_path.dropna().tolist())
                    total_size += group["size"].sum()
        except:
            print(
                "Error while looping on the metadata table...",
                task_name,
                participant_id,
                modality,
            )
            continue

    print(
        "Done. Sent {:.2f} Go in {:.2f} seconds ({:.2f} hours).".format(
            total_size, time() - ts, (time() - ts) / 3600
        )
    )
    print("Files to process: {} for {:.0f} Go".format(len(video_paths), total_size))
    return video_paths

def send_percy_to_harddrive_vs_local(
    path_metadata, output_folder, other_locations=None, process=True
):
    """Fill the hard drive with videos not yet represented in the remote."""

    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )

    video_paths = []
    total_size = 0
    ts = time()
    for (task_name, participant_id, modality), group in metadata.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        if modality == "?":
            continue

        try:
            os.makedirs(os.path.join(output_folder), exist_ok=True)
            os.makedirs(os.path.join(output_folder, task_name), exist_ok=True)
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_folder, task_name, participant_id, modality),
                exist_ok=True,
            )

            # ii)

            for video_path, size in zip(
                group.video_path.tolist(), group["size"].tolist()
            ):

                for other_location in other_locations:
                    output_file_check = os.path.join(
                        other_location,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )

                    if os.path.isfile(output_file_check):
                        # print('Video already in the hard drive: {}'.format(output_file_check))
                        continue

                if os.path.isfile(
                    os.path.join(
                        output_folder,
                        task_name,
                        participant_id,
                        modality,
                        os.path.basename(video_path),
                    )
                ):
                    # print('Video already in the output folder: {}'.format(video_path))
                    continue

                if process:
                    try:
                        shutil.copy2(
                            video_path,
                            os.path.join(
                                output_folder, task_name, participant_id, modality
                            ),
                        )
                        print(
                            "Copied {} to the hard drive ({:.2f} Go).".format(
                                video_path, size
                            )
                        )
                        total_size += size
                    except:
                        print(
                            "[FAILURE] to copy {} to the hard drive.".format(video_path)
                        )

                else:
                    video_paths.extend(group.video_path.dropna().tolist())
                    total_size += group["size"].sum()
        except:
            print(
                "Error while looping on the metadata table...",
                task_name,
                participant_id,
                modality,
            )
            continue

    print(
        "Done. Sent {:.2f} Go in {:.2f} seconds ({:.2f} hours).".format(
            total_size, time() - ts, (time() - ts) / 3600
        )
    )
    print("Files to process: {} for {:.0f} Go".format(len(video_paths), total_size))
    return video_paths


def wipe_harddrive(path_df_remote, path_df_harddrive, output_folder, process=False):
    """Remove videos from the hard drive already presented to the remote, or having a collated video."""

    df = pd.read_csv(path_df_remote)
    drivedf = pd.read_csv(path_df_harddrive)

    video_paths = set()
    for i, row in drivedf.iterrows():

        # The video in the hard drive is in the remote
        if row["identifier"] in df.identifier.tolist() and not pd.isnull(
            row.video_path
        ):
            video_paths.add(
                os.path.join(
                    output_folder,
                    row.folder,
                    row.participant_id,
                    row.modality,
                    os.path.basename(row.video_path),
                )
            )

        cdf = df[
            (df["participant_id"] == row.participant_id)
            & (df["modality"] == row.modality)
        ]

        if (cdf.has_merged_video == True).all() and not pd.isnull(row.video_path):
            video_paths.add(
                os.path.join(
                    output_folder,
                    row.folder,
                    row.participant_id,
                    row.modality,
                    os.path.basename(row.video_path),
                )
            )

    if process:
        for video_path in video_paths:

            if not os.path.isfile(video_path):
                print("File removed already: ", video_path)
            else:
                print("Removed {}".format(video_path))
                os.remove(video_path)
    else:
        print("Videos to remove: {}".format(len(video_paths)))

    return video_paths


# Output csv to percy
def output_partitipant_folder_to_rename_percy():
    """Highlight the participant folders in Percy that do not follow standard name."""
    metadata = pd.read_csv(
        os.path.join(
            get_data_root(), "dataframes", "persistent_metadata", "metadata.csv"
        )
    )
    # metadata.participant_id = metadata.participant_id.replace(mapping_participant_id_fix)
    # metadata['is_SC'] = metadata.participant_id.apply(lambda x: True if x.startswith('SC') else False)
    # metadata['participant_id'] = metadata.apply(lambda x: x.participant_id if not x.is_SC else  x.participant_id[3:], axis=1)

    metadata[["task_number", "diag_number", "trigram", "date_folder"]] = metadata[
        "participant_id"
    ].apply(parse_participant_id)
    nonstandard_metadata = metadata[metadata["trigram"].apply(len) != 6]
    nonstandard_metadata = nonstandard_metadata.drop_duplicates(
        ["root_path", "participant_id"]
    )[["root_path", "participant_id"]]

    nonstandard_metadata.to_csv(
        os.path.join(
            get_data_root(),
            "dataframes",
            "clinical",
            "percy_participant_id_to_rename.csv",
        ),
        index=False,
    )
    print(f"{len(nonstandard_metadata)} participant folders to rename")
    print("\n\t".join(nonstandard_metadata.participant_id.tolist()))
    # print(f'[RUN] scp {socket.gethostname()}:{os.path.join(get_data_root(), 'dataframes', 'clinical', 'percy_participant_id_to_rename.csv')} ./')


def output_modality_not_found_percy(root_dir=None):
    """Deprecated. Integrated in the api.features.consolidation.main_consolidation.consolidate_dataset function
    Highlight videos in the Percy computer whose modality were not found"""

    if root_dir is None:
        root_dir = get_data_root()

    path_metadata = os.path.join(
        root_dir, "dataframes", "persistent_metadata", "metadata.csv"
    )
    metadata = pd.read_csv(path_metadata)
    metadata["task_name"] = metadata["task_name"].apply(
        lambda x: "cuisine" if x == "cooking" else x
    )
    # metadata = metadata[metadata['size'] > 0.05]
    # metadata = metadata[metadata['n_frames'] > 100]

    output_modality_not_found_percy = []
    output_path = os.path.join(
        get_data_root(), "dataframes", "clinical", "percy_participant_id_to_rename.csv"
    )

    video_paths = []
    for (task_name, participant_id, modality), group in metadata.groupby(
        ["task_name", "participant_id", "modality"]
    ):
        if modality not in expected_folders:
            for i, row in group.iterrows():
                output_modality_not_found_percy.append(
                    {
                        "task_name": task_name,
                        "participant_id": participant_id,
                        "modality": modality,
                        "video_path": row.video_path,
                    }
                )

    output_modality_not_found_percy = pd.DataFrame(output_modality_not_found_percy)
    output_modality_not_found_percy.to_csv(output_path, index=False)

    print(
        "Saved dataframe with {} rows to {}".format(
            len(output_modality_not_found_percy), output_path
        )
    )
    display(output_modality_not_found_percy)


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

    # Generate Percy output: unconventional participant folder based on global metadata
    # output_partitipant_folder_to_rename_percy()

    # Init. dataframes
    path_df_remote = os.path.join(
        get_data_root(), "dataframes", "persistent_metadata", "cheetah_dataset_df.csv"
    )
    path_df_harddrive = os.path.join(
        get_data_root(), "dataframes", "persistent_metadata", "harddrive_dataset_df.csv"
    )
    path_metadata = os.path.join(
        get_data_root(), "dataframes", "persistent_metadata", "metadata.csv"
    )
    local_harddrive_data_root = "H:\\data"  # '/Volume/harddrive/data'

    video_paths = percy_to_harddrive_to_complete_remote(
        path_df_remote,
        path_df_harddrive,
        path_metadata,
        local_harddrive_data_root,
        process=args.process,
    )
    sys.exit(0)
