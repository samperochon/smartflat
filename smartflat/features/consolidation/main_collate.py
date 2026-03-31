import argparse
import logging
import os
import subprocess
import sys

import numpy as np
from IPython.display import display

,
)

from smartflat.datasets.loader import get_dataset
from smartflat.utils.utils_coding import *
from smartflat.utils.utils_collate import write_collate
from smartflat.utils.utils_io import (
    check_video,
    get_data_root,
    get_file_size_in_gb,
    parse_flag,
)


def main(
    root_dir,
    process=False,
    final_location="/Volumes/Smartflat/data",
    overwrite=True,
    n_group=3,
):

    logging.basicConfig(
        filename=os.path.join(get_data_root(), "log", "collate.log"),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Collate logging")
    logger = logging.getLogger("main")
    logger.info("Logging dataset...")
    dset = get_dataset(dataset_name="base", root_dir=root_dir, scenario="collate")

    df = dset.metadata.copy()
    # df_merged = df[(df['n_videos'] == 1) | (df['video_name'] == 'merged_video')].copy()
    df_uncollate = df[
        ~((df["n_videos"] == 1) | (df["video_name"] == "merged_video"))
    ].copy()

    logger.info(f"Collating {len(dset)} videos...")

    log_path = os.path.join(get_data_root(), "error_collate_videos.txt")
    if not os.path.isfile(log_path):
        # Si le fichier n'existe pas, le créer
        with open(log_path, "w") as file:
            pass

    commands = []
    sizes = []
    video_paths_cont = []
    n_succeeded = 0
    n = 0

    for (task_name, participant_id, modality), group in df_uncollate.groupby(
        ["task_name", "participant_id", "modality"]
    ):

        n += 1
        if n > n_group:
            continue

        video_paths = group.video_path.to_list()

        # Here we see cases: (i) whose merged_video outcome might exist abut still partiioned remained
        if not (group.video_path.isna().mean() == 0):
            print("Not all video or extra outputs: ", participant_id, modality)

            # display(group)
            # print(os.path.dirname(group.video_representation_path.iloc[0]))
            continue

        if (
            parse_flag(video_paths[0], modality, "flag_collate_video") == "failure"
        ) and not overwrite:
            continue

        # Check if a merged file exist in the `other location`
        final_output_path = os.path.join(
            final_location, task_name, participant_id, modality, "merged_video.mp4"
        )

        if os.path.isfile(final_output_path):
            print(
                "Collated output exists in  {} ({:.2f} Go). Exit.".format(
                    final_output_path, get_file_size_in_gb(final_output_path)
                )
            )
            print(
                f"These videos are removed (not expected to be transferred to data-refill)"
            )
            print("\n\t".join(video_paths))

            for video_path in video_paths:
                red("rm {}".format(video_path))
                ##subprocess.run(["rm", video_path])
            continue

        if os.path.isfile(
            os.path.join(os.path.dirname(video_paths[0]), "merged_video.mp4")
        ):
            print(
                "Collated output exists ({:.2f} Go). Exit.".format(
                    get_file_size_in_gb(
                        os.path.join(
                            os.path.dirname(video_paths[0]), "merged_video.mp4"
                        )
                    )
                )
            )

            if not (
                parse_flag(video_paths[0], modality, "flag_collate_video") == "success"
            ):
                subprocess.run(["open", os.path.dirname(video_paths[0])])
                # raise ValueError('Check collate output of: {}'.format(os.path.join(os.path.dirname(video_paths[0]), 'merged_video.mp4')))

            if (
                get_file_size_in_gb(
                    os.path.join(os.path.dirname(video_paths[0]), "merged_video.mp4")
                )
                <= 0.05
            ):
                print(
                    "Invalid merged video size size: {}".format(
                        os.path.join(os.path.dirname(video_paths[0]))
                    )
                )
                subprocess.run(["open", os.path.dirname(video_paths[0])])
            continue

        if parse_flag(video_paths[0], modality, "flag_collate_video") == "failure":
            yellow(
                f"Collated failed already (re-try=False) {participant_id} - {modality}"
            )
            continue

        # Write the collate-file
        if process:

            output_path = write_collate(video_paths)
            sizes.append(
                np.sum([get_file_size_in_gb(video_path) for video_path in video_paths])
            )
            video_paths_cont.append((video_path for video_path in video_paths))

        else:
            output_path = write_collate(video_paths)
            output_path = os.path.join(
                os.path.dirname(video_paths[0]), "collate_videos.txt"
            )

            print("\nCollating the following videos into merged_video.mp4:")
            print("\n\t".join(video_paths))

        commands.append(
            [
                "ffmpeg",
                "-probesize",
                "50M",
                "-analyzeduration",
                "100M",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                output_path,
                "-c",
                "copy",
                os.path.join(os.path.dirname(output_path), "merged_video.mp4"),
            ]
        )

        if not process:
            print(" ".join(commands[-1]))

    print("Processing: {}".format(len(commands)))
    logger.info("Processing: {}".format(len(commands)))

    if process:
        for i, command in enumerate(commands):
            print("Processing: {}".format(command[11]))
            if (
                os.path.exists(command[-1])
                and np.isclose(sizes[i], get_file_size_in_gb(command[-1]), 0.1)
                and check_video(command[-1])
            ):
                for video_path in video_paths_cont[i]:
                    logger.info("rm {}".format(video_path))
                    subprocess.run(["rm", video_path])

            # Collate videos
            subprocess.run(command)

            # Supress single video.
            if (
                os.path.exists(command[-1])
                and np.isclose(sizes[i], get_file_size_in_gb(command[-1]), 0.1)
                and check_video(command[-1])
            ):
                for video_path in video_paths_cont[i]:

                    subprocess.run(["rm", video_path])
                    logger.info("rm {}".format(video_path))
                    n_succeeded += 1

                with open(
                    os.path.join(
                        os.path.dirname(command[-1]), ".merged_video_collate_flag.txt"
                    ),
                    "w",
                ) as f:
                    f.write("success")
                green(
                    f"Success for {command[-1]} with size: {get_file_size_in_gb(command[-1])} Go"
                )

            else:
                red(f"Error collating {command[-1]}")
                red(" ".join(command))

                with open(
                    os.path.join(
                        os.path.dirname(command[-1]), ".merged_video_collate_flag.txt"
                    ),
                    "w",
                ) as f:
                    f.write("failure")

                if os.path.isfile(command[-1]):
                    os.remove(command[-1])

                # subprocess.run(['open', os.path.dirname(command[-1])])

                logger.info("[FAILED] Error {}".format(command))
                with open(log_path, "a") as file:
                    file.write("{}\n".format(command[11]))

    logger.info("Final Success: {}/{}".format(n_succeeded, len(commands)))
    return commands


def parse_args():

    parser = argparse.ArgumentParser(description="Collate dataset")
    parser.add_argument(
        "-p", "--process", action="store_true", default=False, help="Process the videos"
    )
    parser.add_argument("-r", "--root_dir", default="local", help="local or harddrive")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    if args.root_dir == "local":

        root_dir = None

    elif args.root_dir == "harddrive":

        root_dir = "/Volumes/SapiroLab"

    main(root_dir=root_dir, process=args.process)

    sys.exit(0)

    print("Done")
