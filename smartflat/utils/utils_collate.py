import os
import subprocess
import sys
from glob import glob

import numpy as np
import pandas as pd

# add tools path and import our own tools

from smartflat.utils.utils_io import (
    fetch_flag_path,
    get_data_root,
    get_file_size_in_gb,
    get_video_output_paths,
    parse_flag,
)


def get_collate_subset(metadata):

    if not "has_merged_video" in metadata.columns:

        metadata["has_merged_video"] = metadata.groupby(
            ["task_name", "participant_id", "modality"]
        )["video_name"].transform(
            lambda x: True if "merged_video" in x.to_list() else False
        )
    #TODOFIX n(make sure this works with the data)
    print('/!\ Set as if  modality folder does not have merged_video data... checked  manually in practice!')
    metadata["has_merged_video"] = False
    metadata['n_videos'] = metadata.groupby(['task_name', 'participant_id', 'modality'])['identifier'].transform(lambda x: len(np.unique(x)))

    if not "is_complete_for_collate" in metadata.columns:
        metadata["is_complete_for_collate"] = metadata.apply(
            lambda x: (
                False
                if ((x["has_merged_video"] == True) or (x["n_videos"] == 1))
                else True
            ),
            axis=1,
        )

    print('Is complete for collate ? ')
    display(metadata["is_complete_for_collate"].value_counts().to_frame())
    
    idf = pd.concat(
        [
            sort_non_collated_df(metadata, "GoPro"),
            sort_non_collated_df(metadata, "Tobii"),
            metadata[
                (
                    (
                        (metadata["modality"] == "Tobii")
                        & (metadata["video_name"].apply(len) != 24)
                    )
                    | (
                        (metadata["modality"].isin(["GoPro1", "GoPro2", "GoPro3"]))
                        & (metadata["video_name"].apply(len) != 8)
                    )
                )
                & (metadata["video_name"] != "merged_video")
                & (metadata["n_videos"] >= 2)
            ],
        ]
    )

    nidf = idf[
        ~idf.groupby(["participant_id", "modality"]).n_videos.transform(
            lambda x: True if (len(x) == x.iloc[0]) else False
        )
    ]
    if len(nidf) > 0:
        print(
            "Non-complete modality folder: {}".format(
                nidf.drop_duplicates(
                    subset=["participant_id", "modality"]
                ).identifier.nunique()
            )
        )
        print('\n\t'.join(nidf.participant_id + ' ' +  nidf.modality  + ' ' +  nidf.video_name + ' ' + nidf.video_path + ' ' + nidf.n_videos.astype(str)))

    # Filter modality without all the videos
    print('/!\ Not Removing non-same-shape')
    display(idf[
        idf.groupby(["participant_id", "modality"]).n_videos.transform(
            lambda x: False if (len(x) == x.iloc[0]) else False
        )
    ])
    
    # Deprecated
    #print('/!\ Removing non complete ')
    # idf = idf[ 
    #     idf.groupby(["participant_id", "modality"]).n_videos.transform(
    #         lambda x: True if (len(x) == x.iloc[0]) else False
    #     )
    # ]


    #assert idf.n_videos.min() > 1

    print(
        "Modality to collate: {}".format(
            idf.drop_duplicates(
                subset=["participant_id", "modality"]
            ).video_name.nunique()
        )
    )
    print(
        "Administrations at stake: {}".format(
            idf.drop_duplicates(subset=["participant_id"]).video_name.nunique()
        )
    )

    return idf


def write_collate(video_paths, overwrite=True):
    """Write collate file read by ffmpeg to concatenate the ordered video_paths."""
    output_path = os.path.join(os.path.dirname(video_paths[0]), "collate_videos.txt")

    # Generate the content that would be written
    new_content = "".join([f"file '{path}'\n" for path in video_paths])

    # Check if the file already exists
    if os.path.exists(output_path) and not overwrite:
        with open(output_path, "r") as f:
            existing_content = f.read()
        if existing_content == new_content:
            print(f'[INFO] {output_path} already exists with identical content. No changes made.')
        else:
            print(f"[WARNING] {output_path} already exists but content differs.")
            print("[INFO] Existing content:")
            print(existing_content)
            print("[INFO] New content to be written:")
            print(new_content)
        return output_path

    # Write new content if file doesn't exist
    with open(output_path, "w") as f:
        f.write(new_content)

    print(f"[INFO] {output_path} written successfully.")
    return output_path


# def write_collate(video_paths):
#         """Write collate file read by ffmpeg to concatenate the ordered video_paths."""
#         output_path = os.path.join(os.path.dirname(video_paths[0]), 'collate_videos.txt')

#         if os.path.exists(output_path):
#             print([f'[WARNING] {output_path} already exists. Exiting.'])
#             return output_path

#         with open(output_path , 'w') as f:
#             for path in video_paths:
#                 if 'merged_video' in path:
#                     raise ValueError
#                 f.write(f"file '{path}'\n")
#         return output_path


def sort_non_collated_df(df, modality):

    if modality == "GoPro":  # Rows level filters

        # Rows level filters
        idf = (
            df[
                (df["modality"].isin(["GoPro1", "GoPro2", "GoPro3"]))
                & (~df["n_videos"].isnull())
                & (df["n_videos"] > 1)
                & (
                    df["is_complete_for_collate"]
                )  # All videos present on the folder (baseline comp: metadata file)
                &
                # (df['in_disk']) &
                (~df["video_path"].isna())
                & (df["has_merged_video"] == False)  # Not processed yet
                & (df.video_name.apply(len) == 8)
            ]
            .sort_values(["participant_id", "modality", "video_name"])
            .copy()
        )


        # Process the videos that have standard names (8 characters)
        order = ["G0PR", "GOPR", "GP01", "GP02", "GP03", "GP04", "GP05", "GP06", "GP07"]
        idf["st"] = idf.video_name.apply(lambda x: x[:4])
        assert np.sum([0 if st in order else 1 for st in idf.st]) == 0
        idf["end"] = idf.video_name.apply(
            lambda x: int(x[4:])
        )  # .unique()  #[['participant_id', 'modality', 'video_name']]
        idf["st_rank"] = idf["st"].apply(
            lambda x: order.index(x) if x in order else len(order)
        )
        idf = idf.sort_values(by=["participant_id", "modality", "st_rank", "end"])

    elif modality == "Tobii":

        idf = (
            df[
                (df["modality"] == "Tobii")
                & (~df["n_videos"].isnull())
                & (df["n_videos"] > 1)
                & (df["has_merged_video"] == False)  # Not processed yet
                & (
                    df["is_complete_for_collate"]
                )  # All videos present on the folder (baseline comp: metadata file)
                & (~df["video_path"].isna())
                & (df["video_name"].apply(len) == 24)
            ]
            .sort_values(["participant_id", "modality", "order"])
            .copy()
        )

    return idf


def remove_output_files(
    extradf, process=False, include_videos_representations=False, include_videos=False
):
    """Remove hands, skeleton, and speech outputs of the provided df"""

    commands = []
    for file_exists, hand_landmarks_path in zip(
        extradf.hand_landmarks_computed.tolist(), extradf.hand_landmarks_path.tolist()
    ):
        if file_exists:

            commands.append(["rm", hand_landmarks_path])
            commands.append(
                ["rm", fetch_flag_path(hand_landmarks_path, "flag_hand_landmarks")]
            )

    for file_exists, skeleton_landmarks_path in zip(
        extradf.skeleton_landmarks_computed.tolist(),
        extradf.skeleton_landmarks_path.tolist(),
    ):
        if file_exists:
            commands.append(["rm", skeleton_landmarks_path])
            commands.append(
                [
                    "rm",
                    fetch_flag_path(skeleton_landmarks_path, "flag_skeleton_landmarks"),
                ]
            )

    for file_exists, speech_recognition_path in zip(
        extradf.speech_recognition_computed.tolist(),
        extradf.speech_recognition_path.tolist(),
    ):
        if file_exists:
            commands.append(["rm", speech_recognition_path])
            commands.append(
                [
                    "rm",
                    fetch_flag_path(speech_recognition_path, "flag_speech_recognition"),
                ]
            )

    for file_exists, speech_representation_path in zip(
        extradf.speech_representation_computed.tolist(),
        extradf.speech_representation_path.tolist(),
    ):
        if file_exists:
            commands.append(["rm", speech_representation_path])
            commands.append(
                [
                    "rm",
                    fetch_flag_path(
                        speech_representation_path, "flag_speech_representation"
                    ),
                ]
            )

    if include_videos_representations:
        for file_exists, video_representation_path in zip(
            extradf.video_representation_computed.tolist(),
            extradf.video_representation_path.tolist(),
        ):
            if file_exists:
                commands.append(["rm", video_representation_path])
                commands.append(
                    [
                        "rm",
                        fetch_flag_path(
                            video_representation_path, "flag_video_representation"
                        ),
                    ]
                )

    if include_videos:
        for video_path in extradf.video_path.dropna().tolist():
            commands.append(["rm", video_path])

    if process:
        for command in commands:
            subprocess.run(command)

    return commands
