"""Clean folders based on some criteria to enforce standard dataset structure."""

import datetime
import os
import subprocess
import sys
from glob import glob

import numpy as np
import pandas as pd


from smartflat.constants import (
    available_modality,
    available_tasks,
    expected_folders,
    mapping_incorrect_modality_name,
    mapping_participant_id_fix,
    video_extensions,
)
from smartflat.datasets.build import apply_manual_fixes
from smartflat.datasets.filter import filter_outlier_video_names
from smartflat.features.consolidation.main_snapshot import main as main_snapshot
from smartflat.utils.utils_io import get_data_root


def print_directories(directory):
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            print(f"\t\t\t\t| {name} |")


def find_video_files(directory):
    for root, dir, files in os.walk(directory):
        # print(extension)
        for file in files:
            for extension in video_extensions:
                if file.endswith(extension):
                    yield os.path.join(root, file)


def consolidate_dataset(root_dir=None, process=False):
    """Clean folders based on some criteria to enforce standard structure.

    This scripts explore the folders of the root data directory and the participant_id folders, and:
        (i) Verify that the `api.constants.expected_folders` are present, otherwise check if they belong to the pre-defined list
            If not, they may be added (could be e.g. `Go Pro  number 1`, to be mapped to `GoPro1`)
            If the corresponding mapping name is `` (an empty string), it means that the folder is something like `Go Pro` and
            videos inside need to be sorted first (add entry to the percy_folder_to_sort.csv shared with clinicians).

        (ii) Move videos to the correct folders if needed (and if we know which GoPro it is based on the sub-folder name), and
                rename folders if needed.

    Procedure: run the function and read output and commands ready to be launched. If they are okay, run it, otherwise
                modify the code. Run the function until the list of command is empty and the outputs satisfactory.


    """

    # Utilisation
    if root_dir is None:
        directory = get_data_root()
    else:
        directory = root_dir
    mod_name = dict()
    commands = []
    percy_folder_to_sort = []

    for task_names in available_tasks:

        folder_name_path = os.path.join(directory, task_names)
        print(f"Exploring {folder_name_path}")

        if os.path.basename(folder_name_path) in ["$RECYCLE.BIN"]:
            continue

        for participant_id_path in glob(os.path.join(folder_name_path, "*")):
            ups = [
                os.path.basename(p)
                for p in glob(f"{participant_id_path}/*")
                if os.path.isdir(p)
            ]

            for up in ups:
                if up not in expected_folders:
                    print(f"Folder: {os.path.join(participant_id_path, up)}")

                    if up in mapping_incorrect_modality_name.keys() and (
                        mapping_incorrect_modality_name[up] not in ["?", ""]
                    ):

                        commands.append(
                            [
                                "mv",
                                os.path.join(participant_id_path, up),
                                os.path.join(
                                    participant_id_path,
                                    mapping_incorrect_modality_name[up],
                                ),
                            ]
                        )

                    # if up not in mod_name and not up in expected_folders:
                    elif (up in ["unknown", "?", "Annotations"]) or (
                        mapping_incorrect_modality_name[up] == ""
                    ):  # TO BE SORTED

                        n_videos = len(
                            list(
                                find_video_files(os.path.join(participant_id_path, up))
                            )
                        )
                        print(
                            "[Additional folder]-> ({} videos)".format(
                                n_videos, up, participant_id_path
                            )
                        )
                        if n_videos == 0:
                            print(
                                "No videos found in this folder:",
                                os.path.join(participant_id_path, up),
                            )
                            print(
                                f"rm -r {os.path.join(participant_id_path, up)} ? (-> uncomment line)"
                            )
                            # subprocess.run(['rm', '-r', os.path.join(participant_id_path, up)])
                            continue
                        percy_folder_to_sort.append(
                            {
                                "path": participant_id_path,
                                #'dossier a trier': up,
                                "videos": list(
                                    find_video_files(
                                        os.path.join(participant_id_path, up)
                                    )
                                ),
                            }
                        )

                        # print_directories(os.path.join(participant_id_path, up))

                        # Case like: /Volumes/SapiroLab/gateau_2018/G49_P36_HERMic_28082018
                        #    | go pro 1 |
                        #    | go pro 2 |
                        #    | go pro 3 |
                        for subd in glob(os.path.join(participant_id_path, up, "*")):
                            if (
                                os.path.basename(subd)
                                in mapping_incorrect_modality_name.keys()
                            ):
                                commands.append(
                                    [
                                        "mv",
                                        subd,
                                        os.path.join(
                                            participant_id_path,
                                            mapping_incorrect_modality_name[
                                                os.path.basename(subd)
                                            ],
                                        ),
                                    ]
                                )
                                # print( os.path.join(participant_id_path, mapping_incorrect_modality_name[os.path.basename(subd)]))
                            elif os.path.isdir(subd):
                                print(
                                    f"/!\ {os.path.basename(subd)} is to be added to mapping_incorrect_modality_name"
                                )

                    else:
                        print(
                            f"\t\t{up} is to be added to mapping_incorrect_modality_name ? "
                        )

                    mod_name.update({up: ""})

            for video_file in find_video_files(participant_id_path):
                if os.path.basename(os.path.dirname(video_file)) not in [
                    "GoPro1",
                    "GoPro2",
                    "GoPro3",
                    "Tobii",
                ]:
                    found = False
                    # Find the corrret modality
                    for mod in expected_folders:
                        if mod in video_file:
                            print(
                                f"Potential video to move: {video_file}\n\t to folder: {os.path.join(participant_id_path, mod)}"
                            )
                            commands.append(
                                [
                                    "mv",
                                    video_file,
                                    os.path.join(participant_id_path, mod),
                                ]
                            )

                            found = True
                            continue
                    if not found:

                        if os.path.basename(video_file).startswith("Recor"):
                            commands.append(
                                [
                                    "mv",
                                    video_file,
                                    os.path.join(participant_id_path, "Tobii"),
                                ]
                            )
                        else:
                            # print('/!\ Modality not found for: ', video_file)
                            pass
                    # print(glob(f'{video_file}/*'))
    output_path = os.path.join(root_dir, "dataframes", "modality_folders_to_sort.csv")
    print(f"Saving to {output_path}")

    if process:
        pd.DataFrame(percy_folder_to_sort).to_csv(output_path, index=False)
        for command in commands:
            subprocess.run(command)
    return commands


def consolidate_metadata(input_file=None, output_file=None):
    """refactor the raw registration metadata file (created from thePercy computer).

    This dataframe brings to the local `metadata` (reflecting the dataset in a particular location)
    the video partitions lists and metadata (fps, duration, size) with sums of these variables
    for further quality control checks.

    """

    if input_file is None:
        input_file = os.path.join(
            get_data_root(), "dataframes", "persistent_metadata", "metadata_raw.csv"
        )
    if output_file is None:
        output_file = os.path.join(
            get_data_root(), "dataframes", "persistent_metadata", "metadata.csv"
        )

    gmetadata = pd.read_csv(input_file)
    # gmetadata2 = pd.read_csv(
    #     os.path.joinc(
    #         get_data_root("Smartflat"),
    #         "dataframes",
    #         "persistent_metadata",
    #         "metadata_11042024.csv",
    #     )
    # )
    # gmetadata = pd.concat([gmetadata, gmetadata2]).drop_duplicates("identifier")

    print(f'Order in metadata: {"order" in gmetadata.columns} - REMOVED ? ')
    # gmetadata.drop(columns=["order"], inplace=True)

    gmetadata["task_name"] = gmetadata.apply(parse_task_name, axis=1)
    gmetadata["participant_id"] = gmetadata.participant_id.apply(
        lambda x: (
            mapping_participant_id_fix[x]
            if x in list(mapping_participant_id_fix.keys())
            else x
        )
    )

    gmetadata["video_path"] = gmetadata["video_path"].apply(
        lambda x: x.replace("/", "\\")
    )
    gmetadata["modality"] = gmetadata.apply(parse_modality, axis=1)
    gmetadata = gmetadata[gmetadata["modality"].isin(available_modality)]

    gmetadata["identifier"] = gmetadata.apply(
        lambda x: "{}_{}_{}_{}".format(
            x.participant_id, x.task_name, x.modality, x.video_name
        ),
        axis=1,
    )

    gmetadata.drop_duplicates(["identifier"], inplace=True)

    gmetadata = apply_manual_fixes(gmetadata)

    gmetadata = filter_outlier_video_names(gmetadata, verbose=True)

    assert (
        len(gmetadata[(gmetadata["task_name"].isna() | gmetadata["modality"].isna())])
        == 0
    )  # [['participant_id', 'video_path', 'task_name', 'modality']]
    assert len(gmetadata[gmetadata["size"].isna()]) == 0
    assert len(gmetadata[gmetadata["date"].isna()]) == 0
    assert len(gmetadata[gmetadata["fps"].isna()]) == 0
    assert len(gmetadata[gmetadata["duration"].isna()]) == 0

    gmetadata["n_videos"] = gmetadata.groupby(
        ["task_name", "participant_id", "modality"]
    )["identifier"].transform(lambda x: len(np.unique(x)))

    gmetadata.sort_values(
        ["task_name", "participant_id", "modality", "video_name"], inplace=True
    )
    gmetadata = gmetadata.assign(
        passation_id=gmetadata.groupby(["task_name", "participant_id"]).ngroup()
    )
    gmetadata = gmetadata.assign(
        modality_id=gmetadata.groupby(
            ["task_name", "participant_id", "modality"]
        ).ngroup()
    )
    gmetadata = gmetadata.assign(video_id=gmetadata.video_path.factorize()[0])

    gmetadata["total_size"] = gmetadata.groupby(
        ["task_name", "participant_id", "modality"]
    )["size"].transform(np.sum)
    gmetadata["total_n_frames"] = gmetadata.groupby(
        ["task_name", "participant_id", "modality"]
    )["n_frames"].transform(np.sum)
    gmetadata["total_duration"] = gmetadata.groupby(
        ["task_name", "participant_id", "modality"]
    )["duration"].transform(np.sum)
    gmetadata["n_fps"] = gmetadata.groupby(["task_name", "participant_id", "modality"])[
        "fps"
    ].transform(lambda x: len(np.unique(x)))

    gmetadata_final = (
        gmetadata.groupby(["task_name", "participant_id", "modality"])
        .agg(
            {
                # "passation_id": lambda x: pd.Series.mode(x)[0],
                "modality_id": lambda x: pd.Series.mode(x)[0],
                "video_id": list,
                "n_videos": lambda x: pd.Series.mode(x)[0],
                "video_name": list,
                "date": list,
                "fps": lambda x: (
                    list(x)[0] if len(np.unique(x)) == 1 else list(x)
                ),  # lambda x: pd.Series.mode(x)[0],
                "n_fps": lambda x: pd.Series.mode(x)[0],
                "size": list,
                "total_size": lambda x: pd.Series.mode(x)[0],
                "n_frames": list,
                "total_n_frames": lambda x: pd.Series.mode(x)[0],
                "duration": list,
                "total_duration": lambda x: pd.Series.mode(x)[0],
            }
        )
        .rename(
            columns={
                "video_name": "video_name_list",
                "date": "dates_list",
                "size": "size_list",
                "n_frames": "n_frames_list",
                "duration": "duration_list",
                "video_id": "video_id_list",
            }
        )
        .reset_index(drop=False)
        .copy()
    )

    assert (
        gmetadata_final.apply(
            lambda x: len(x.video_name_list) == x.n_videos, axis=True
        ).mean()
        == 1
    )

    gmetadata_final["mod_identifier"] = gmetadata_final.apply(
        lambda x: "{}_{}_{}".format(x.participant_id, x.task_name, x.modality), axis=1
    )

    gmetadata_final.to_csv(
        output_file,
        index=False,
    )
    print(f"Metadata saved to {output_file}")

    return gmetadata


def parse_task_name(row):
    if row.task_name in available_tasks:
        return row.task_name
    elif row.participant_id[0] == "G":
        return "cuisine"
    elif row.participant_id[0] == "L":
        return "lego"
    else:
        return np.nan


def parse_modality(row):

    if row.modality in available_modality:
        return row.modality
    elif "GoPro1" in row.video_path:
        return "GoPro1"
    elif "GoPro2" in row.video_path:
        return "GoPro2"
    elif "GoPro3" in row.video_path:
        return "GoPro3"
    elif "Tobii" in row.video_path:
        return "Tobii"

    elif row.video_path.split("\\")[-2] in list(mapping_incorrect_modality_name.keys()):
        return mapping_incorrect_modality_name[row.video_path.split("\\")[-2]]

    else:
        print(f"/!\ Modality not found for {row.video_path}")
        print(os.path.dirname(row.video_path))
        return np.nan
