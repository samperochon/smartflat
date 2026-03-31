"""Path resolution and parsing utilities for the Smartflat project.

Provides host-aware data root detection with SMARTFLAT_DATA_ROOT env var override,
plus path parsing functions for the Smartflat folder structure.

Usage:
    # Preferred: set environment variable
    export SMARTFLAT_DATA_ROOT=/Volumes/Smartflat/data-gold-final

    # In code
    from smartflat.utils.utils_paths import get_data_root
    root = get_data_root()
"""

import os
import socket

import numpy as np
import pandas as pd

from smartflat.utils.utils_coding import get_logger

logger = get_logger(__name__)

# Host names recognized as local development machines
LOCAL_MACHINE_NAMES = [
    # Current machines
    "sjp49.borelli.ens-paris-saclay.fr",
    "sjp49.home",
    "sjp49.local",
    "sjp49.Still-Gone.home",
    "macbook-pro-de-sam.home",
    "macbook-pro-de-sam.local",
    "MacBook-Pro-de-SAM.local",
    "MacBook-Pro-de-Sam.local",
    # Legacy hostnames (kept for backward compat with older data paths)
    "Mac.lan",
    "egr-sjp49-mbp.local",
    "egr-sjp49-mbp.home",
    "egr-sjp49-mbp.home.dhe.duke.edu",
    "egr-sjp49-mbp-1.home",
    "egr-sjp49-mbp-1.home.dhe.duke.edu",
    "device-58.home.dhe.duke.edu",
    "device-57.home.dhe.duke.edu",
    "device-3026.home",
    "device-3026.home.dhe.duke.edu",
    "3C-06-30-12-07-86",
    "MacOS-Sam-Perochon",
    "MacBook-Pro.local",
    "pclnrs119.biomedicale.univ-paris5.fr.dhe.duke.edu",
    "pclnrs103.biomedicale.univ-paris5.fr.dhe.duke.edu",
    "pclnrs219.biomedicale.univ-paris5.fr.dhe.duke.edu",
    "w-155-132.wfer.ens-paris-saclay.fr.dhe.duke.edu",
]


# ---------------------------------------------------------------------------
# Data root resolution
# ---------------------------------------------------------------------------

def get_data_root(machine_name=None, local=False):
    """Resolve the data root directory.

    Priority:
    1. SMARTFLAT_DATA_ROOT environment variable (if set)
    2. Host-based detection (legacy, for backward compatibility)
    """
    env_root = os.environ.get("SMARTFLAT_DATA_ROOT")
    if env_root is not None:
        if not os.path.isdir(env_root):
            logger.warning("SMARTFLAT_DATA_ROOT=%s does not exist", env_root)
        return env_root

    # Legacy host-based detection
    if machine_name is None:
        machine_name = socket.gethostname()

    if machine_name in LOCAL_MACHINE_NAMES:  # Port 8888
        if not local:
            DATA_ROOT = get_gold_data_path()
        else:
            DATA_ROOT = '/Users/samperochon/github-repositories/smartflat/data/data-gold-final'

    elif machine_name == "PCNomad-Borelli2":
        DATA_ROOT = "/media/sam/Smartflat/data-gold-final"

    elif machine_name == "pomme":  # Port 1111
        DATA_ROOT = "/home/perochon/data-gold-final"

    elif machine_name == "disque-dur":
        DATA_ROOT = "/home/perochon/data"

    elif machine_name == "asure":  # Port 3221
        DATA_ROOT = "/home01/sam/data"

    elif machine_name == "cheetah":  # Port 4372
        DATA_ROOT = '/diskA/sam_data/data-features'

    elif machine_name == "mate":  # Port 6740
        DATA_ROOT = "/home/sam/data"

    elif machine_name == "DESKTOP-PMRIG7H":
        DATA_ROOT = "C:\\Users\\admin\\samperochon\\data"

    elif "ruche" in machine_name or "node" in machine_name:
        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-final"

    elif machine_name == "smartflat":  # Port 6740
        DATA_ROOT = "/Volumes/Smartflat/data"

    elif machine_name == "gold":
        DATA_ROOT = get_gold_data_path()

    elif machine_name == "light":
        DATA_ROOT = get_light_data_path()

    else:
        raise ValueError(f"Machine name '{machine_name}' is unknown. Set SMARTFLAT_DATA_ROOT env var.")

    return DATA_ROOT


def get_gold_data_path(machine_name=None):
    env_root = os.environ.get("SMARTFLAT_DATA_ROOT_GOLD")
    if env_root is not None:
        return env_root

    if machine_name is None:
        machine_name = socket.gethostname()

    if "cheetah" in machine_name:
        DATA_ROOT = "/diskA/sam_data/data-gold-final"
    elif "ruche" in machine_name or "node" in machine_name:
        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-final"
    elif "pomme" in machine_name:
        DATA_ROOT = "/home/perochon/data-gold-final"
    elif machine_name == "PCNomad-Borelli2":
        DATA_ROOT = "/media/sam/Smartflat/data-gold-final"
    elif machine_name in LOCAL_MACHINE_NAMES:
        if os.path.isdir('/Volumes/Smartflat'):
            DATA_ROOT = "/Volumes/Smartflat/data-gold-final"
        else:
            DATA_ROOT = '/Users/samperochon/github-repositories/smartflat/data/data-gold-final'
    else:
        raise ValueError(f"Machine name '{machine_name}' is unknown. Set SMARTFLAT_DATA_ROOT_GOLD env var.")

    return DATA_ROOT


def get_light_data_path(machine_name=None):
    env_root = os.environ.get("SMARTFLAT_DATA_ROOT_LIGHT")
    if env_root is not None:
        return env_root

    if machine_name is None:
        machine_name = socket.gethostname()

    if "cheetah" in machine_name:
        DATA_ROOT = "/diskA/sam_data/data-gold-light"
    elif "ruche" in machine_name or "node" in machine_name:
        DATA_ROOT = "/gpfs/workdir/perochons/data-gold-light"
    elif "pomme" in machine_name:
        DATA_ROOT = "/home/perochon/data-gold-light"
    elif machine_name in LOCAL_MACHINE_NAMES:
        DATA_ROOT = "/Volumes/Smartflat/data-gold-light"
    elif machine_name == "PCNomad-Borelli2":
        DATA_ROOT = "/media/sam/Smartflat/data-gold-light"
    else:
        raise ValueError(f"Machine name '{machine_name}' is unknown. Set SMARTFLAT_DATA_ROOT_LIGHT env var.")

    return DATA_ROOT


def get_api_root(machine_name=None):
    env_root = os.environ.get("SMARTFLAT_API_ROOT")
    if env_root is not None:
        return env_root

    if machine_name is None:
        machine_name = socket.gethostname()

    if machine_name in LOCAL_MACHINE_NAMES:
        API_ROOT = "/Users/samperochon/github-repositories/smartflat"
    elif machine_name == "pomme":
        API_ROOT = "/home/perochon/smartflat"
    elif machine_name == "cheetah":
        API_ROOT = "/home/sam/smartflat"
    elif "ruche" in machine_name or "node" in machine_name:
        API_ROOT = "/gpfs/users/perochons/smartflat"
    elif machine_name == "PCNomad-Borelli2":
        API_ROOT = "/home/sam/smartflat"
    else:
        raise ValueError(f"Machine name '{machine_name}' is unknown. Set SMARTFLAT_API_ROOT env var.")

    return API_ROOT


def get_host_name():
    name = socket.gethostname()
    if "ruche" in name:
        return "ruche"
    elif "sjp" in name:
        return "smartflat"
    else:
        return name


# ---------------------------------------------------------------------------
# File system utilities
# ---------------------------------------------------------------------------

def get_free_space(folder):
    """Return folder free space in GB."""
    total, used, free = map(
        int, os.popen('df -k "' + folder + '"').readlines()[-1].split()[1:4]
    )
    return free / 1024 / 1024


def get_file_size_in_gb(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return size_in_bytes / (1024 * 1024 * 1024)


def check_exist(output_path, file_name):
    return os.path.isfile(os.path.join(output_path, file_name))


def fetch_root_dir(output_path):
    return os.path.dirname(os.path.dirname(os.path.dirname(output_path)))


# ---------------------------------------------------------------------------
# Output path construction
# ---------------------------------------------------------------------------

def fetch_output_path(video_path, model_name):
    """Provide output file name associated with a model for a given video sample."""
    video_name, _, _, _ = parse_path(video_path)

    if model_name == "vit_giant_patch14_224":
        name = "video_representations_VideoMAEv2"
        return os.path.join(os.path.dirname(video_path), name + "_" + video_name + ".npy")

    elif model_name == "whisperx":
        name = "speech_recognition_diarization_whisperx"
        return os.path.join(os.path.dirname(video_path), name + "_" + video_name + ".json")

    elif model_name == "multilingual-e5-large":
        name = "speech_representations_multilingual"
        return os.path.join(os.path.dirname(video_path), name + "_" + video_name + ".npy")

    elif model_name == "hand_landmarks_mediapipe":
        name = "hand_landmarks_mediapipe"
        return os.path.join(os.path.dirname(video_path), name + "_" + video_name + ".json")

    elif model_name == "skeleton_landmarks_mediapipe":
        name = "skeleton_landmarks"
        return os.path.join(os.path.dirname(video_path), name + "_" + video_name + ".json")

    elif model_name == 'tracking_hand_landmarks_v1':
        name = 'tracking_hand_landmarks'
        return os.path.join(os.path.dirname(video_path), name + '_' + video_name + '.json')

    else:
        raise NotImplementedError(f"Unknown model: {model_name}")


def fetch_flag_path(output_path, flag_type):
    """Construct the flag file path for a given output and flag type."""
    video_name, modality, participant_id, task = parse_path(output_path)

    data_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(output_path)))
    )

    if flag_type == "flag_speech_recognition":
        file_path = os.path.join(
            data_root, task, participant_id,
            f".{video_name}_speech_recognition_flag.txt",
        )
    elif flag_type == "flag_speech_representation":
        file_path = os.path.join(
            data_root, task, participant_id,
            f".{video_name}_speech_representation_flag.txt",
        )
    elif flag_type == "flag_video_representation":
        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_video_representation_flag.txt"
        )
    elif flag_type == "flag_hand_landmarks":
        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_hand_landmarks_flag.txt"
        )
    elif flag_type == "flag_tracking_hand_landmarks":
        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_tracking_hand_landmarks_flag.txt"
        )
    elif flag_type == "flag_skeleton_landmarks":
        file_path = os.path.join(
            os.path.dirname(output_path), f".{video_name}_skeleton_landmarks_flag.txt"
        )
    else:
        raise ValueError(f"Unknown flag type: {flag_type}")

    return file_path


def fetch_path(task, participant_id, modality, output_dir=None):
    """Fetch the embedding folder path for a modality of a participant."""
    if output_dir is not None:
        return os.path.join(output_dir, task, participant_id, modality)
    else:
        return os.path.join(get_data_root(), task, participant_id, modality)


def fetch_has_gaze(row, verbose=False):
    from smartflat.constants import mapping_participant_id_fix

    if row.modality != 'Tobii':
        return 0

    gaze_data_list_1 = glob(
        os.path.join(
            os.path.dirname(get_data_root()),
            'data-gaze', '*', f'*{row.participant_id}*'
        )
    )

    reversed_mapping = {v: k for k, v in mapping_participant_id_fix.items()}
    gaze_data_list_2 = glob(
        os.path.join(
            os.path.dirname(get_data_root()),
            'data-gaze', '*', '*{}*'.format(reversed_mapping.get(row.participant_id, 'ABSENT_ID_FROM_MAPPING'))
        )
    )

    gaze_data_list = set(gaze_data_list_1 + gaze_data_list_2)

    if len(gaze_data_list) == 0:
        if verbose:
            print(f"No gaze data found for participant {row.participant_id}")
        return 0
    elif len(gaze_data_list) == 1:
        if verbose:
            print(f"Found gaze data for participant {row.participant_id}: {list(gaze_data_list)}")
        return 1
    else:
        raise ValueError(f"Multiple gaze data files found for participant {row.participant_id}")


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def parse_dir_name(path):
    modality, participant_id, task = (
        path.split("/")[-1],
        path.split("/")[-2],
        path.split("/")[-3],
    )
    return task, participant_id, modality


def parse_path(path):
    """Parse data_root/task/modality from an embedding folder path.

    Example:
        video_name, modality, participant_id, task = parse_path(path)
    """
    from smartflat.constants import hard_parsed_path, available_tasks, available_modality

    if path in hard_parsed_path:
        return hard_parsed_path[path]

    if os.path.basename(path).startswith("."):
        filename, modality, participant_id, task = (
            path.split("/")[-1].split(".")[1],
            path.split("/")[-2],
            path.split("/")[-3],
            path.split("/")[-4],
        )

        if "video_representation_flag" in filename:
            video_name = filename.split("_video_representation_flag")[0]
        elif "speech_recognition_flag" in filename:
            video_name = filename.split("_speech_recognition_flag")[0]
        elif "speech_representation_flag" in filename:
            video_name = filename.split("_speech_representation_flag")[0]
        elif "skeleton_landmarks_flag" in filename:
            video_name = filename.split("_skeleton_landmarks_flag")[0]
        elif "tracking_hand_landmarks_flag" in filename:
            video_name = filename.split("_tracking_hand_landmarks_flag")[0]
        elif "hand_landmarks_flag" in filename:
            video_name = filename.split("_hand_landmarks_flag")[0]
        else:
            video_name = filename

    else:
        try:
            filename, modality, participant_id, task = (
                path.split("/")[-1].split(".")[0],
                path.split("/")[-2],
                path.split("/")[-3],
                path.split("/")[-4],
            )
        except Exception:
            print(f"Error parsing path: {path}")
            return np.nan, np.nan, np.nan, np.nan

        if filename.startswith("video_representations"):
            video_name = (
                os.path.basename(path)
                .split("video_representations_VideoMAEv2_")[1]
                .split(".npy")[0]
            )
        elif filename.startswith("speech_recognition"):
            video_name = (
                os.path.basename(path)
                .split("speech_recognition_diarization_whisperx_")[1]
                .split(".json")[0]
            )
        elif filename.startswith("speech_representations"):
            video_name = (
                os.path.basename(path)
                .split("speech_representations_multilingual_")[1]
                .split(".npy")[0]
            )
        elif filename.startswith("skeleton_landmarks"):
            video_name = (
                os.path.basename(path).split("skeleton_landmarks_")[1].split(".json")[0]
            )
        elif filename.startswith("tracking_hand_landmarks"):
            try:
                video_name = (
                    os.path.basename(path)
                    .split("tracking_hand_landmarks_")[1]
                    .split(".json")[0]
                )
            except Exception:
                logger.error("Failed to parse tracking_hand_landmarks path: %s", path)
                video_name = 'ERROR'
        elif filename.startswith("hand_landmarks"):
            video_name = (
                os.path.basename(path)
                .split("hand_landmarks_mediapipe_")[1]
                .split(".json")[0]
            )
        else:
            video_name = filename

    return video_name, modality, participant_id, task


def parse_flag(output_path, modality, flag_type):
    """'success' or 'failure' or 'unprocessed' or 'disabled' if un-processed (missing flag)."""
    from smartflat.constants import enabled_modalities

    if flag_type == "flag_collate_video":
        filepath = os.path.join(
            os.path.dirname(output_path), ".merged_video_collate_flag.txt"
        )
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                flag = f.readline().strip()
        else:
            flag = "unprocessed"
        return flag

    filepath = fetch_flag_path(output_path, flag_type)
    _, _, _, task_name = parse_path(output_path)

    task_name = (
        "cuisine"
        if ("cuisine" in task_name or "gateau" in task_name)
        else "lego" if "lego" in task_name else task_name
    )

    if modality not in enabled_modalities[task_name][flag_type]:
        return "disabled"

    if os.path.isfile(filepath):
        try:
            with open(filepath, "r") as f:
                flag = f.readline().strip()
        except Exception:
            print(f"Error reading flag: {filepath}")
            flag = "corrupted"
        return flag
    else:
        return "unprocessed"


def parse_participant_id(string):
    """Parse participant ID strings into components."""
    components = string.split("_")

    if string == 'SC_L171_PXX_240823':
        task_num, diag_num, trigram, date = 'L171', 'PXX', 'xxxxxx', '240823'
    elif len(components) == 6:
        task_num, diag_num, trigram, period, visit, date = components[:6]
        trigram = trigram.lower().strip()
        date = date[:6]
    elif len(components) == 5 and components[0] == "SC":
        task_num, diag_num, trigram, date = components[1:]
        trigram = trigram.lower().strip()
    elif len(components) == 4:
        task_num, diag_num, trigram, date = components
        trigram = trigram.lower().strip()
    elif len(components) == 7:
        task_num, diag_num, trigram, date = components[0], components[3], components[1], components[6]
    else:
        task_num, diag_num, trigram, date = np.nan, np.nan, np.nan, np.nan

    return pd.Series([task_num, diag_num, trigram, date])


def parse_identifier(identifier):
    if identifier.startswith("SC_"):
        identifier = identifier[3:]

    components = identifier.split("_")
    participant_id = "_".join(components[:4])
    task_name = components[4]
    modality = components[5]
    video_name = "_".join(components[6:])

    valid_modalities = ["GoPro1", "GoPro2", "GoPro3", "Tobii"]
    if modality not in valid_modalities:
        raise ValueError(
            f"Invalid modality: {modality}. Expected one of {valid_modalities}."
        )

    return task_name, participant_id, modality, video_name


def parse_task_number(row):
    try:
        return int(row[1:])
    except Exception:
        return -1


# Need glob for fetch_has_gaze
from glob import glob
