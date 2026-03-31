"""Quality control for the Smartflat dataset based on visual inspection results.

Applies corrections from manual visual inspection of video recordings:
modality swaps, failed data removal, and flag reassessment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from smartflat.utils.utils_coding import display_safe, green, yellow
from smartflat.utils.utils_io import get_data_root, parse_flag


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_QC_COLUMNS = [
    "identifier", "task_name", "participant_id", "modality", "folder_modality",
    "is_fish_eyed", "upside_down", "is_swapped_modality", "true_modality",
    "GP3_is_sink", "GP2_is_wrong_buttress", "GP2_above",
    "GoPro1_is_wrong_buttress", "is_middle_range", "true_task_name",
    "is_old_setup", "is_old_recipe", "annot_notes",
]

_FLAG_COLUMNS = [
    "modality", "flag_speech_recognition", "flag_speech_representation",
    "flag_video_representation", "flag_hand_landmarks",
    "flag_skeleton_landmarks", "flag_collate_video",
]


def _load_visual_inspection_results():
    """Load the visual inspection CSV from the data root."""
    path = os.path.join(
        get_data_root(), 'dataframes', 'quality-control',
        'results_visual_inspection_2111124_last.csv',
    )
    return pd.read_csv(path, sep=';', usecols=_QC_COLUMNS)


def _merge_inspection_results(df, results, indicator=True):
    """Merge inspection results into the metadata DataFrame."""
    df = df.merge(
        results.drop(columns=["identifier", "modality"]),
        left_on=["task_name", "participant_id", "modality"],
        right_on=["task_name", "participant_id", "folder_modality"],
        how="left",
        indicator=indicator,
    )
    df['folder_modality'] = df.apply(
        lambda x: x.folder_modality if not pd.isna(x.folder_modality) else x.modality,
        axis=1,
    )
    return df


def _remove_failed_collections(df, verbose=False):
    """Remove rows flagged as failed data collection by visual inspection."""
    failed_conditions = [
        ("true_task_name_nan", lambda row: row["true_task_name"] is np.nan),
        ("GP2_is_wrong_buttress", lambda row: row["GP2_is_wrong_buttress"] == 1),
        ("GP2_above", lambda row: row["GP2_above"] == 1),
        ("GoPro1_is_wrong_buttress", lambda row: row["GoPro1_is_wrong_buttress"] == 1),
        ("is_middle_range", lambda row: row["is_middle_range"] == 1),
    ]

    failed_logs = []
    if verbose:
        for condition, test in failed_conditions:
            failed = df[df.apply(test, axis=1)]
            if not failed.empty:
                failed_logs.append((condition, failed[["participant_id", "modality"]]))

    n = len(df)
    df = df[
        (df["true_task_name"] != np.nan)
        & (df["GP2_is_wrong_buttress"] != 1)
        & (df["GP2_above"] != 1)
        & (df["GoPro1_is_wrong_buttress"] != 1)
        & (df["is_middle_range"] != 1)
    ]

    if verbose and n != len(df):
        yellow(f"Modalities removed because of visual inspection results: M={n - len(df)}")
        for condition, failed in failed_logs:
            if not failed.empty:
                yellow(f"Condition '{condition}' failed for:")
                print(failed)
    elif verbose:
        green('All modalities passed the visual inspection :-).')

    return df


def _reassess_flags(df, modality_col="modality"):
    """Reassess feature extraction flags using the given modality column.

    Parameters
    ----------
    modality_col : str
        Column to use for modality-dependent flag resolution.
        "modality" for apply_visual_inspection_results (after swap),
        "true_modality" for apply_visual_inspection_update_flags (without swap).
    """
    df["flag_hand_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["hand_landmarks_path"], row[modality_col], "flag_hand_landmarks"
        ),
        axis=1,
    )
    df["flag_skeleton_landmarks"] = df.apply(
        lambda row: parse_flag(
            row["skeleton_landmarks_path"], row[modality_col], "flag_skeleton_landmarks"
        ),
        axis=1,
    )
    df["flag_collate_video"] = df.apply(
        lambda row: parse_flag(
            row["video_representation_path"], row[modality_col], "flag_collate_video"
        ),
        axis=1,
    )
    df["flag_speech_recognition"] = df.apply(
        lambda row: parse_flag(
            row["speech_recognition_path"], row[modality_col], "flag_speech_recognition"
        ),
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda row: parse_flag(
            row["speech_representation_path"], row[modality_col], "flag_speech_representation"
        ),
        axis=1,
    )
    return df


def _reassign_audio_modality(df):
    """Assign primary audio modality per participant and disable speech flags for non-audio."""
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
    df["flag_speech_recognition"] = df.apply(
        lambda x: x.flag_speech_recognition if x.modality == x.audio_modality else "disabled",
        axis=1,
    )
    df["flag_speech_representation"] = df.apply(
        lambda x: (
            x.flag_speech_representation if x.modality == x.audio_modality else "disabled"
        ),
        axis=1,
    )
    return df


def _plot_flag_changes(df, value_counts_before):
    """Plot heatmap of flag value changes (verbose diagnostics)."""
    value_counts_after = df[_FLAG_COLUMNS].apply(pd.Series.value_counts)
    value_counts_diff = value_counts_after - value_counts_before

    plt.figure(figsize=(5, 3))
    sns.heatmap(value_counts_diff, annot=True, cmap="coolwarm")
    plt.title("Changes after applying the manual annotation results")
    plt.xlabel("Columns")
    plt.ylabel("Values")
    plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_visual_inspection_results(df, verbose=False):
    """Apply visual inspection corrections: swap modalities, update identifiers, reassess flags.

    This function modifies the modality column for swapped cameras and
    recomputes identifiers accordingly. Use when building the final dataset.
    """
    results = _load_visual_inspection_results()

    if verbose:
        create_annotation_summary(results)
        value_counts_before = df[_FLAG_COLUMNS].apply(pd.Series.value_counts)

    df = _merge_inspection_results(df, results, indicator=True)
    df = _remove_failed_collections(df, verbose=verbose)

    # Swap modality for incorrectly placed cameras
    incorrect_videos = df[df["is_swapped_modality"] == True]  # noqa: E712
    for index, row in incorrect_videos.iterrows():
        df.at[index, "modality"] = row["true_modality"]
    for index, row in incorrect_videos.iterrows():
        new_identifier = (
            f"{row['participant_id']}_{row['task_name']}"
            f"_{row['true_modality']}_{row['video_name']}"
        )
        df.at[index, "identifier"] = new_identifier

    df = _reassess_flags(df, modality_col="modality")
    df = _reassign_audio_modality(df)

    # Report participants missing visual inspection results
    df_without_annotation = select_subset_missing_vir(df)
    if len(df_without_annotation) > 0:
        print(f"Participants and modality missing visual inspection results (kept): "
              f"N={len(df_without_annotation)}:")
        display_safe(
            df_without_annotation.drop_duplicates(["participant_id", "modality"])
            .groupby("participant_id")["modality"]
            .agg(list)
            .to_frame()
            .transpose()
        )

    if verbose:
        _plot_flag_changes(df, value_counts_before)
        df = df.drop(columns="_merge")

    return df


def select_subset_missing_vir(df):
    """Return rows with missing visual inspection results but having video."""
    return df[(df["_merge"] == "left_only") & (df['video_path'].notna())]


def apply_visual_inspection_update_flags(df, verbose=False):
    """Update feature flags using true_modality, without swapping modality or identifiers.

    Use when you want to reassess which features need extraction based on the
    true camera placement, but without modifying the dataset's modality column.
    """
    results = _load_visual_inspection_results()
    value_counts_before = df[_FLAG_COLUMNS].apply(pd.Series.value_counts)

    df = _merge_inspection_results(df, results, indicator=verbose)
    df = _remove_failed_collections(df, verbose=verbose)

    df = _reassess_flags(df, modality_col="true_modality")
    df = _reassign_audio_modality(df)

    if verbose:
        df_without_annotation = df[df["_merge"] == "left_only"]
        print("Missing annotation on the following participants and modality:")
        display_safe(
            df_without_annotation.drop_duplicates(["participant_id", "modality"])
            .groupby("participant_id")["modality"]
            .agg(list)
            .to_frame()
            .transpose()
        )
        _plot_flag_changes(df, value_counts_before)
        df = df.drop(columns="_merge")

    return df


def create_annotation_summary(dataset):
    """Visualize annotation findings with bar plot of validity conditions."""
    relevant_columns = [
        "is_swapped_modality", "is_fish_eyed", "upside_down",
        "GP3_is_sink", "GP2_is_wrong_buttress", "GP2_above",
        "GoPro1_is_wrong_buttress", "is_middle_range", "true_task_name",
        "is_old_setup", "is_old_recipe",
    ]
    n_passations = dataset.participant_id.nunique()
    n_identifiers = dataset.identifier.nunique()

    df = dataset[relevant_columns]
    validity_counts = df.sum()
    total_videos = len(dataset)
    proportions = validity_counts / total_videos

    summary = pd.DataFrame({
        "Count": pd.to_numeric(validity_counts, errors='coerce'),
        "Proportion": pd.to_numeric(proportions, errors='coerce'),
    }).sort_values(by="Proportion", ascending=False)

    plt.figure(figsize=(6, 3))
    ax = sns.barplot(x=summary["Proportion"], y=summary.index)
    plt.title(
        f"Summary of Manual Annotation Findings\n"
        f"{n_passations} passations, {n_identifiers} modality folders (thumbnails)"
    )
    plt.xlabel("Proportion")
    plt.ylabel("Validity Condition")

    for i, (count, proportion) in enumerate(zip(summary["Count"], summary["Proportion"])):
        ax.text(proportion + 0.01, i, f"N={count}, {proportion:.2%}", ha="left", va="center")

    plt.xlim((0, 1))
    plt.show()
