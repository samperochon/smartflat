"""Gaze feature extraction pipeline.

Extracts 56 per-segment gaze features from Tobii Pro eye-tracker recordings.
Requires change point detection to be computed first (defines the segments).

Features are computed across 4 data streams:
- gaze_event_data: fixation/saccade counts, durations, frequencies, positions
- gaze_data: 2D/3D gaze norms, path lengths, pupil diameters, validity
- accelerometric_data: head acceleration magnitude
- gyroscopic_data: head rotation magnitude

Usage:
    python -m smartflat.features.gaze.main
    python -m smartflat.features.gaze.main --cpts_config ChangePointDetectionDeploymentConfig
    python -m smartflat.features.gaze.main --annotator_id samperochon --round_number 0

Sam Perochon, April 2026.
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

from smartflat.configs.loader import import_config
from smartflat.constants import gaze_features
from smartflat.datasets.dataset_gaze import compute_segments_gaze_features
from smartflat.datasets.loader import get_dataset
from smartflat.datasets.utils import load_embedding_dimensions
from smartflat.utils.utils_io import (
    fetch_has_gaze,
    get_data_root,
    get_host_name,
    save_df,
)

log_dir = os.path.join(get_data_root(), "log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "gaze_feature_extraction.log"),
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logging.info("Gaze feature extraction")
logger = logging.getLogger("gaze_main")


def main(
    cpts_config_name="ChangePointDetectionDeploymentConfig",
    annotator_id="samperochon",
    round_number=0,
):
    """Run gaze feature extraction for all participants.

    Loads a dataset with precomputed change points, parses Tobii TSV files
    for each participant, computes 56 per-segment gaze features, and saves
    the results as a combined pickle and per-participant CSV files.

    Parameters
    ----------
    cpts_config_name : str
        Name of the change point detection config class.
    annotator_id : str
        Annotator identifier for loading change points.
    round_number : int
        Annotation round number.
    """
    start_time = time.time()
    logger.info(
        "Starting gaze extraction: cpts_config=%s, annotator=%s, round=%d",
        cpts_config_name,
        annotator_id,
        round_number,
    )

    # --- 1. Load dataset with change points ---
    cpts_config = import_config(cpts_config_name)
    dset = get_dataset(
        dataset_name=cpts_config.dataset_name, **cpts_config.dataset_params
    )
    dset.metadata = load_embedding_dimensions(dset.metadata)
    dset.load_change_point_detection(
        config_name=cpts_config_name,
        annotator_id=annotator_id,
        round_number=round_number,
    )

    logger.info("Loaded dataset: %d rows", len(dset.metadata))

    # --- 2. Prepare DataFrame ---
    df = dset.metadata.copy()
    df["cpts"] = df.cpts.apply(lambda x: x if x[-1] != x[-2] else x[:-1])
    df["cpts"] = df.apply(lambda x: np.clip(x.cpts, 0, x.N), axis=1)
    df["has_gaze"] = df.apply(fetch_has_gaze, axis=1, verbose=False)

    n_with_gaze = df["has_gaze"].sum()
    n_total = len(df)
    logger.info("Gaze availability: %d/%d rows have Tobii data", n_with_gaze, n_total)
    print(f"Gaze availability: {n_with_gaze}/{n_total} rows have Tobii data")

    if n_with_gaze == 0:
        logger.warning("No participants have gaze data. Exiting.")
        print("No participants have gaze data. Nothing to extract.")
        return df

    # --- 3. Compute gaze features ---
    logger.info("Computing gaze features...")
    df = compute_segments_gaze_features(df, cpts_col="cpts", verbose=False)
    logger.info("Gaze feature computation complete.")

    # --- 4. Save outputs ---
    output_dir = os.path.join(get_data_root(), "outputs", "gaze_features")
    os.makedirs(output_dir, exist_ok=True)

    # Combined pickle
    pickle_path = os.path.join(
        output_dir, f"{get_host_name()}_gaze_features.pkl"
    )
    save_df(df, pickle_path)
    logger.info("Saved combined pickle: %s", pickle_path)
    print(f"Saved combined pickle: {pickle_path}")

    # Per-participant CSVs and flag files
    all_feature_cols = [
        col for dtype_feats in gaze_features.values() for col in dtype_feats
    ]
    meta_cols = ["identifier", "participant_id", "task_name", "modality", "has_gaze"]
    output_cols = meta_cols + [c for c in all_feature_cols if c in df.columns]

    for pid, group in df.groupby("participant_id"):
        participant_dir = os.path.join(output_dir, pid)
        os.makedirs(participant_dir, exist_ok=True)

        # CSV
        group[output_cols].to_csv(
            os.path.join(participant_dir, f"{pid}_gaze_features.csv"), index=False
        )

        # Flag file
        flag_status = "success" if group["has_gaze"].any() else "no_gaze_data"
        flag_path = os.path.join(
            participant_dir, f".{pid}_gaze_features_flag.txt"
        )
        with open(flag_path, "w") as f:
            f.write(flag_status)

    # --- 5. Metrics ---
    elapsed = time.time() - start_time
    metrics_dir = os.path.join(get_data_root(), "dataframes", "frozen-metrics-logs")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(
        metrics_dir, f"{get_host_name()}_gaze_features_compute_time.csv"
    )

    metrics = []
    for pid, group in df.groupby("participant_id"):
        has_gaze = int(group["has_gaze"].any())
        n_segments = 0
        if has_gaze and "segments_bounds" in group.columns:
            bounds = group.iloc[0]["segments_bounds"]
            if isinstance(bounds, list):
                n_segments = len(bounds)
        metrics.append(
            {
                "participant_id": pid,
                "has_gaze": has_gaze,
                "n_segments": n_segments,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df["compute_time_total"] = elapsed
    if os.path.isfile(metrics_path):
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df])
    metrics_df.to_csv(metrics_path, index=False)

    n_participants = df["participant_id"].nunique()
    logger.info(
        "Done. %d participants, %.1f seconds elapsed. Output: %s",
        n_participants,
        elapsed,
        output_dir,
    )
    print(
        f"Done. {n_participants} participants processed in {elapsed:.1f}s. "
        f"Output: {output_dir}"
    )

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract gaze features from Tobii eye-tracker recordings"
    )
    parser.add_argument(
        "--cpts_config",
        type=str,
        default="ChangePointDetectionDeploymentConfig",
        help="Change point detection config class name",
    )
    parser.add_argument(
        "--annotator_id",
        type=str,
        default="samperochon",
        help="Annotator identifier for loading change points",
    )
    parser.add_argument(
        "--round_number",
        type=int,
        default=0,
        help="Annotation round number",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.cpts_config, args.annotator_id, args.round_number)
