"""Video loading, validation, audio extraction, and metadata parsing utilities.

These functions require optional dependencies (decord, ffmpeg) that may not
be available on all machines. Functions gracefully handle missing dependencies.
"""

import datetime
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from smartflat.utils.utils_coding import get_logger
from smartflat.utils.utils_paths import get_api_root, get_file_size_in_gb

logger = get_logger(__name__)

try:
    from decord import VideoReader, bridge, cpu
    bridge.set_bridge("torch")
    _decord_available = True
except ImportError:
    _decord_available = False


def get_video_loader():
    """Return a video loader function using decord."""
    def _loader(video_path):
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr
    return _loader


def check_video(video_path):
    """Check if a video file is valid and readable."""
    try:
        vr = VideoReader(video_path, num_threads=0, ctx=cpu(0))  # noqa: F841
        return True
    except Exception:
        return False


def extract_audio_files(process=False):
    """Extract audio (.wav) files of consolidated videos."""
    from smartflat.datasets.loader import get_dataset

    dset = get_dataset(dataset_name="base", scenario="present")
    df = dset.metadata.sort_values(["modality"], ascending=True).copy()
    df = df[(df["video_name"] == "merged_video") | (df["n_videos"] == 1)]

    for video_path in df.video_path.tolist():
        output_audio_path = os.path.join(
            os.path.dirname(video_path),
            os.path.basename(video_path).split(".")[0] + ".wav",
        )
        if not os.path.isfile(output_audio_path) and process:
            subprocess.run(["ffmpeg", "-i", video_path, output_audio_path], check=False)
            logger.info("Extracted audio file: %s", output_audio_path)


def extract_audio(video_path, verbose=False):
    """Extract audio (.wav) file from a video."""
    output_audio_path = os.path.join(
        os.path.dirname(video_path), os.path.basename(video_path).split(".")[0] + ".wav"
    )
    if not os.path.isfile(output_audio_path):
        subprocess.run(["ffmpeg", "-i", video_path, output_audio_path], check=False)
        if verbose:
            logger.info("Extracted audio file: %s", output_audio_path)


def load_pca(task_name, modality, n_components, whiten=True):
    """Load a PCA model from the artifacts directory."""
    if whiten:
        filename = f"pca_{task_name}_{modality}_{n_components}_w.pkl"
    else:
        filename = f"pca_{task_name}_{modality}_{n_components}.pkl"

    logger.info("Load pca from: %s", filename)
    filepath = os.path.join(get_api_root(), "api", "models", "artifacts", filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found.")

    with open(filepath, "rb") as f:
        pca = pickle.load(f)
    return pca


def parse_video_metadata(row):
    """Extract fps, n_frames, date, and size from a video file row."""
    if not np.isnan(row.fps):
        return pd.Series([row.fps, row.n_frames, row.date, row.size])
    try:
        vr = VideoReader(row.video_path, num_threads=0, ctx=cpu(0))
    except Exception:
        logger.error(
            "Error reading video: %s - %s - %s - %s",
            row.video_path, row.task_name, row.participant_id, row.modality,
        )
        return pd.Series([np.nan] * 4)

    fps = vr.get_avg_fps()
    n_frames = vr._num_frame
    size = get_file_size_in_gb(row.video_path)
    timestamp = os.path.getctime(row.video_path)
    date = datetime.datetime.fromtimestamp(timestamp)

    return pd.Series([fps, n_frames, date, size])
